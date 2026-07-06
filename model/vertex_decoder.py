import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import init_weights
import math


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6, include_input=True):
        """
        Args:
            num_freqs (int): Number of frequency bands for encoding.
            include_input (bool): Whether to include the raw input in the embedding.
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim = 0
        
        if include_input:
            self.out_dim += 3
        
        self.out_dim += 3 * 2 * num_freqs # 3 coords * 2 (sin/cos) * num_freqs

    def forward(self, x):
        """
        x: (..., 3)
        """
        embed = [x] if self.include_input else []
        
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            embed.append(torch.sin(freq * math.pi * x))
            embed.append(torch.cos(freq * math.pi * x))
            
        return torch.cat(embed, dim=-1)


class VertexQueryDecoder(nn.Module):
    def __init__(self, local_dim, hidden_dim=256, num_tight=1, num_free=0, num_joints=55, learnable_lbs_weights=True):
        super().__init__()
        
        self.num_gaussians = num_tight + num_free
        self.num_tight = num_tight
        self.num_joints = num_joints
        self.learnable_lbs_weights = learnable_lbs_weights

        self.pos_encoder = PositionalEncoding(num_freqs=6, include_input=True)

        # Input: Local Features (D) + Positional Encoding (3 + 3*2*6 = 39)
        input_dim = local_dim + self.pos_encoder.out_dim

        # 1. Shared MLP
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. Geometry Branch
        # Outputs: Offset (3) + Rotation (4) + Scale (3) = 10
        self.geo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10 * self.num_gaussians) 
        )

        # 3. Appearance Branch
        # Outputs: Color (3) + Opacity (1) = 4
        self.app_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 * self.num_gaussians)
        )

        # 4. LBS Weight Branch
        # Outputs: Delta logits for LBS weights (num_joints)
        if self.learnable_lbs_weights:
            self.lbs_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_joints * self.num_gaussians)
            )
        else:
            self.lbs_head = None

        self.apply(init_weights)
        self._init_special_biases()

    def _init_special_biases(self):
        # Geometry Head Initialization
        geo_last = self.geo_head[-1]
        nn.init.normal_(geo_last.weight, mean=0.0, std=0.001)
        nn.init.zeros_(geo_last.bias)

        # Reshape bias logic to handle multiple gaussians
        with torch.no_grad():
            bias = geo_last.bias.view(self.num_gaussians, 10)
            
            # Offset (0-3): Start at 0.0
            bias[:, 0:3].fill_(0.0)
            
            # If multiple gaussians, maybe slightly perturb free ones.
            # For now, keeping them at 0 is safer to avoid drift at init.
            # Rotation (3-7): Identity Quaternion (1, 0, 0, 0)
            bias[:, 3].fill_(1.0)
            bias[:, 4:7].fill_(0.0)

            # Scale (7-10): Start Small
            bias[:, 7:10].fill_(-3.0) 
            
            geo_last.bias.copy_(bias.view(-1))

        # Appearance Head Initialization
        app_last = self.app_head[-1]
        nn.init.normal_(app_last.weight, mean=0.0, std=0.001)
        nn.init.zeros_(app_last.bias)

        with torch.no_grad():
            bias = app_last.bias.view(self.num_gaussians, 4)
            # Color (0-3): Neutral Grey
            bias[:, 0:3].fill_(0.0)
            # Opacity (3): Visible
            bias[:, 3].fill_(-1.0)
            app_last.bias.copy_(bias.view(-1))

        # LBS Head Initialization
        if self.learnable_lbs_weights:
            lbs_last = self.lbs_head[-1]
            nn.init.zeros_(lbs_last.weight)
            nn.init.zeros_(lbs_last.bias)

    def forward(self, local_features, canonical_vertices, base_lbs_weights):
        """
        global_features: (B, D)
        local_features: (B, N, D)
        canonical_vertices: (N, 3) OR (B, N, 3)
        """
        B = local_features.shape[0]

        if canonical_vertices.dim() == 2:
            N = canonical_vertices.shape[0]
            verts_expanded = canonical_vertices.unsqueeze(0).expand(B, -1, -1)
        else:
            N = canonical_vertices.shape[1]
            verts_expanded = canonical_vertices

        verts_encoded = self.pos_encoder(verts_expanded)
        inp = torch.cat([local_features, verts_encoded], dim=-1)

        features = self.trunk(inp)

        # Reshape outputs to (B, N, K, D)
        # where K is num_gaussians per vertex
        raw_geo = self.geo_head(features) # (B, N, 10*K)
        raw_app = self.app_head(features) # (B, N, 4*K)

        raw_geo = raw_geo.view(B, N, self.num_gaussians, 10)
        raw_app = raw_app.view(B, N, self.num_gaussians, 4)

        # LBS Prediction
        if self.learnable_lbs_weights:
            # Predict delta in log-space
            lbs_delta = self.lbs_head(features)
            lbs_delta = lbs_delta.view(B, N, self.num_gaussians, self.num_joints)

            # Expand base weights for K gaussians
            base_weights_expanded = base_lbs_weights.unsqueeze(2) 
            base_logits = torch.log(base_weights_expanded + 1e-9)

            # Add predicted delta
            final_logits = base_logits + lbs_delta
            pred_weights = F.softmax(final_logits, dim=-1)
        else:
            # Just repeat the static weights for all K gaussians
            pred_weights = base_lbs_weights.unsqueeze(2).expand(-1, -1, self.num_gaussians, -1)

        return raw_geo, raw_app, pred_weights