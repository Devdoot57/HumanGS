import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np
import traceback
import torchvision
import os
import pickle
import math

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftGouraudShader,
    TexturesVertex,
    PerspectiveCameras
)

from utils import data_utils 
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer

from .vertex_decoder import VertexQueryDecoder
from .lbs import lbs_gaussians
from renderer.gaussian_renderer import render_predicted_gaussians

import time


class FeatureUpsampler(nn.Module):
    """
    Fuses multi-scale ViT features and upsamples the spatial resolution.
    Takes 4 intermediate feature maps (e.g., 64x64) and outputs a high-res map (256x256).
    """
    def __init__(self, in_dim, hidden_dim=256, out_dim=768):
        super().__init__()

        self.proj = nn.ModuleList([
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1) for _ in range(4)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsample 1: 64x64 -> 128x128
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsample 2: 128x128 -> 256x256
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, out_dim, kernel_size=3, stride=1, padding=1)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # features: list of 4 tensors of shape (B*V, C, H, W)
        projs = [proj(f) for proj, f in zip(self.proj, features)]
        fused = torch.cat(projs, dim=1)

        fused = self.fuse(fused)
        up1 = self.up1(fused)
        out = self.up2(up1)
        return out


class HumanGS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)
        self.use_canonical_maps = config.training.get("use_canonical_maps", False)

        # SMPL-X Neural Texture
        tex_cfg = config.model.neural_texture
        self.neural_texture_res = tex_cfg.resolution
        self.neural_texture_dim = tex_cfg.dim
        self.neural_texture = nn.Parameter(
            torch.randn(3, self.neural_texture_res, self.neural_texture_res, self.neural_texture_dim)
        )
        nn.init.normal_(self.neural_texture, mean=0.0, std=0.1)

        # Initialize tokenizers and transformer
        self._init_tokenizers()
        self._init_transformer()

        # Upsampler
        self.use_upsampler = config.model.transformer.get("use_upsampler", False)
        if self.use_upsampler:
            transformer_dim = config.model.transformer.d
            self.upsampler = FeatureUpsampler(in_dim=transformer_dim, out_dim=transformer_dim)

        self.project_all_vertices = config.model.transformer.get("project_all_vertices", True)

        # LBS Weights Buffer
        self.register_buffer('tpose_lbs_weights', torch.zeros(10475, 55))
        self._load_smpl_constants()

        # Decoder initialization
        self.use_cls_token = config.model.transformer.get("use_cls_token", False)
        transformer_dim = config.model.transformer.d

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            decoder_input_dim = transformer_dim * 2 
        else:
            decoder_input_dim = transformer_dim

        vd_cfg = config.model.vertex_decoder

        learnable_lbs_weights = vd_cfg.get('learnable_lbs_weights', True)

        self.vertex_decoder = VertexQueryDecoder(
            local_dim=decoder_input_dim,
            hidden_dim=vd_cfg.hidden_dim,
            num_tight=vd_cfg.get('num_tight_gaussians', 1),
            num_free=vd_cfg.get('num_free_gaussians', 0),
            num_joints=55,
            learnable_lbs_weights=learnable_lbs_weights
        )

        self.loss_computer = LossComputer(config)

        # Initialize Rasterizer settings
        self.raster_settings = RasterizationSettings(
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            bin_size=0
        )
        self.shader = SoftGouraudShader()

    def _load_smpl_constants(self):
        path = self.config.model.get('smpl_model_path', None)
        if path and os.path.exists(path):
            if torch.distributed.get_rank() == 0:
                print(f"Loading SMPL constants from {path}")
            if path.endswith('.npz'):
                data = np.load(path)
                if 'weights' in data:
                    self.tpose_lbs_weights.data = torch.from_numpy(data['weights']).float()
            elif path.endswith('.pkl'):
                with open(path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                if 'weights' in data:
                    self.tpose_lbs_weights.data = torch.tensor(data['weights'], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported SMPL model file format: {path}")

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)
        return tokenizer

    def _init_tokenizers(self):
        in_channels_rgb = 3
        in_channels_plucker = 6 
        in_channels_pose = 3 * self.neural_texture_dim 

        in_channels_image = in_channels_rgb + in_channels_plucker + in_channels_pose

        self.image_tokenizer = self._create_tokenizer(
            in_channels = in_channels_image,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )

        if self.use_canonical_maps:
            in_channels_target = in_channels_plucker + in_channels_pose

            self.target_pose_tokenizer = self._create_tokenizer(
                in_channels = in_channels_target,
                patch_size = self.config.model.image_tokenizer.patch_size,
                d_model = self.config.model.transformer.d
            )

    def _init_transformer(self):
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", True)
        self.n_layers = config.n_layer

        self.transformer_blocks = nn.ModuleList(
            [
                QK_Norm_TransformerBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(self.n_layers)
            ]
        )

        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_blocks):
                if config.get("depth_init", False):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * self.n_layers) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)

        self.transformer_input_layernorm = nn.LayerNorm(config.d, bias=False)
        self.norm = nn.LayerNorm(config.d)

    @staticmethod
    def _get_projection_matrix(fx, fy, cx, cy, h, w, znear=0.1, zfar=100.0):
        B = fx.shape[0]
        P = torch.zeros((B, 4, 4), device=fx.device)
        P[:, 0, 0] = 2.0 * fx / w
        P[:, 0, 2] = 2.0 * cx / w - 1.0
        P[:, 1, 1] = 2.0 * fy / h        
        P[:, 1, 2] = 2.0 * cy / h - 1.0
        P[:, 2, 2] = (zfar + znear) / (zfar - znear)
        P[:, 2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        P[:, 3, 2] = 1.0
        return P

    def _get_canonical_cameras(self, tpose_verts, batch_size, device, image_size=512):
        """
        Constructs 4 Canonical Cameras.
        """
        # Compute Bounding Boxes & Centers
        min_xyz = tpose_verts.min(dim=1)[0]
        max_xyz = tpose_verts.max(dim=1)[0]

        centers = (min_xyz + max_xyz) / 2.0

        # Size of each subject: (B,)
        scale_xyz = max_xyz - min_xyz
        max_dims = scale_xyz.max(dim=1)[0]

        # Compute Adaptive Distance for each subject: (B,)
        fov_deg = 60.0
        fov_rad = np.radians(fov_deg)
        margin = 1.2
        dists = (max_dims / 2.0 * margin) / np.tan(fov_rad / 2.0)

        # Construct Base Camera (Front View): (B, 3)
        base_eyes = centers.clone()
        base_eyes[:, 2] += dists

        ats = centers # Look at the center of each subject (B, 3)
        ups = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(batch_size, -1) # (B, 3)

        # Generate 4 Views Vectorized
        angles = [0, -np.pi/2, np.pi/2, np.pi] 
        c2w_list = []

        for angle in angles:
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            Ry = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], device=device).float()

            rel_pos = base_eyes - ats
            curr_rel_pos = torch.matmul(rel_pos, Ry.T)

            curr_eyes = ats + curr_rel_pos

            z = F.normalize(ats - curr_eyes, dim=1)            # Forward
            x = F.normalize(torch.cross(z, ups, dim=1), dim=1) # Right
            y = F.normalize(torch.cross(z, x, dim=1), dim=1)   # Down

            # Rotation Matrix
            R = torch.stack([x, y, z], dim=2)

            # Full C2W (B, 4, 4)
            c2w = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            c2w[:, :3, :3] = R
            c2w[:, :3, 3] = curr_eyes

            c2w_list.append(c2w)

        c2w_batch = torch.stack(c2w_list, dim=1)

        # Intrinsics
        f = image_size / (2.0 * np.tan(fov_rad / 2.0))
        c = image_size / 2.0

        # (B, 4, 4)
        intrinsics = torch.tensor([f, f, c, c], device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, 4, 1)

        return c2w_batch.float(), intrinsics.float()

    def _rasterize_pose_images(self, data_source):
        """
        Renders the "Position Map" using PyTorch3D.
        """
        with torch.autocast(enabled=False, device_type="cuda"):
            h, w = data_source.image_h_w
            posed_vertices = data_source.smpl_posed_vertices
            faces = data_source.smpl_faces
            c2w = data_source.c2w
            fxfycxcy = data_source.fxfycxcy

            b, v = posed_vertices.shape[:2]
            device = posed_vertices.device

            posed_vertices_f32 = rearrange(posed_vertices.float(), "b v n k -> (b v) n k")
            canonical_vertices_f32 = rearrange(data_source.smpl_canonical_vertices.float(), "b v n k -> (b v) n k")
            faces_long = rearrange(faces.long(), "b v n k -> (b v) n k")

            # Create Textures
            textures = TexturesVertex(verts_features=(canonical_vertices_f32 + 1.0) / 2.0)
            meshes = Meshes(verts=posed_vertices_f32, faces=faces_long, textures=textures)

            # Create Cameras
            w2c = torch.inverse(c2w.float())
            w2c[:, :, 0, :] *= -1.0 
            w2c[:, :, 1, :] *= -1.0 

            R = w2c[:, :, :3, :3].transpose(-2, -1) 
            T = w2c[:, :, :3, 3]

            fx = rearrange(fxfycxcy[..., 0], "b v -> (b v)")
            fy = rearrange(fxfycxcy[..., 1], "b v -> (b v)")
            cx = rearrange(fxfycxcy[..., 2], "b v -> (b v)")
            cy = rearrange(fxfycxcy[..., 3], "b v -> (b v)")

            cameras = PerspectiveCameras(
                R=rearrange(R, "b v n k -> (b v) n k"), 
                T=rearrange(T, "b v n -> (b v) n"),
                focal_length=torch.stack([fx, fy], dim=-1),
                principal_point=torch.stack([cx, cy], dim=-1),
                image_size=((h, w),), in_ndc=False, device=device
            )

            # Render
            raster_settings = self.raster_settings
            raster_settings.image_size = (h, w)

            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            shader = SoftGouraudShader(device=device, cameras=cameras, lights=None)

            # Rasterize to get fragments (Geometry + Depth)
            fragments = rasterizer(meshes)

            # Depth map
            raw_depth = fragments.zbuf[..., 0]
            depth_maps = rearrange(raw_depth, "(b v) h w -> b v 1 h w", b=b)

            # Position Map
            pos_map_render = shader(fragments, meshes)

            # Debug visualization
            if torch.distributed.get_rank() == 0 and not hasattr(self, '_debug_saved'):
                debug_st = time.time()
                os.makedirs("debug_renders", exist_ok=True)

                debug_render_img = pos_map_render[0, ..., :3].detach().cpu().permute(2, 0, 1)
                debug_gt_img = data_source.image[0, 0].detach().cpu()

                raw_depth = depth_maps[0, 0].detach().cpu() # Shape: (1, H, W)

                valid_mask = raw_depth > 0
                if valid_mask.any():
                    d_min = raw_depth[valid_mask].min()
                    d_max = raw_depth[valid_mask].max()
                    norm_depth = (raw_depth - d_min) / (d_max - d_min + 1e-8)
                    norm_depth[~valid_mask] = 0.0
                else:
                    norm_depth = torch.zeros_like(raw_depth)

                debug_depth_img = norm_depth.repeat(3, 1, 1)

                combined_img = torch.cat([debug_gt_img, debug_render_img, debug_depth_img], dim=2)                
                torchvision.utils.save_image(combined_img, "debug_renders/debug_view.png")
                print(f"DEBUG: Saved internal render to debug_renders/debug_view.png")
                self._debug_saved = True
                self.debug_overhead_1 = time.time() - debug_st

            pos_map = pos_map_render[..., :3] * 2.0 - 1.0 
            pos_map = rearrange(pos_map, "(b v) h w c -> b v h w c", b=b)

        target_dtype = self.neural_texture.dtype
        pos_map = pos_map.to(dtype=target_dtype)

        # Sample from Neural Texture
        bv = b * v
        tex_xy = self.neural_texture[0].permute(2, 0, 1).unsqueeze(0).expand(bv, -1, -1, -1)
        tex_xz = self.neural_texture[1].permute(2, 0, 1).unsqueeze(0).expand(bv, -1, -1, -1)
        tex_yz = self.neural_texture[2].permute(2, 0, 1).unsqueeze(0).expand(bv, -1, -1, -1)

        grid_xy = rearrange(pos_map[..., [0, 1]], "b v h w c -> (b v) h w c")
        grid_xz = rearrange(pos_map[..., [0, 2]], "b v h w c -> (b v) h w c")
        grid_yz = rearrange(pos_map[..., [1, 2]], "b v h w c -> (b v) h w c")

        sampled_xy = F.grid_sample(tex_xy, grid_xy, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_xz = F.grid_sample(tex_xz, grid_xz, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_yz = F.grid_sample(tex_yz, grid_yz, mode='bilinear', padding_mode='zeros', align_corners=True)

        sampled_xy = rearrange(sampled_xy, "(b v) c h w -> b v c h w", b=b)
        sampled_xz = rearrange(sampled_xz, "(b v) c h w -> b v c h w", b=b)
        sampled_yz = rearrange(sampled_yz, "(b v) c h w -> b v c h w", b=b)

        neural_features = torch.cat([sampled_xy, sampled_xz, sampled_yz], dim=2)

        return neural_features, depth_maps

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, pose_images=None, method="default_plucker"):
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d], dim=2)
        if pose_images is not None:
            pose_cond = torch.cat([pose_cond, pose_images], dim=2)
        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)

    def sample_pixel_aligned_features(self, feature_map, vertices, c2w, intrinsics, image_size, depth_maps=None):
        """
        Projects vertices onto the feature map and samples features.
        """
        B, V, N, _ = vertices.shape
        H_img, W_img = image_size

        # Flatten Batch and View dimensions
        verts_flat = rearrange(vertices, 'b v n c -> (b v) n c')
        c2w_flat = rearrange(c2w, 'b v x y -> (b v) x y').float()
        intr_flat = rearrange(intrinsics, 'b v c -> (b v) c').float()

        w2c = torch.inverse(c2w_flat)
        R = w2c[:, :3, :3]
        T = w2c[:, :3, 3].unsqueeze(1)

        verts_cam = torch.bmm(verts_flat, R.transpose(1, 2)) + T
        verts_depth = verts_cam[..., 2:3]

        # Perspective Projection
        eps = 1e-6
        z_inv = 1.0 / (verts_cam[..., 2] + eps)
        x_norm = verts_cam[..., 0] * z_inv
        y_norm = verts_cam[..., 1] * z_inv

        # Camera -> Pixel -> NDC
        fx, fy, cx, cy = intr_flat[:, 0:1], intr_flat[:, 1:2], intr_flat[:, 2:3], intr_flat[:, 3:4]

        u = fx * x_norm + cx
        v = fy * y_norm + cy

        u_ndc = (u / W_img) * 2.0 - 1.0
        v_ndc = (v / H_img) * 2.0 - 1.0

        # Stack for grid sample
        grid = torch.stack([u_ndc, v_ndc], dim=-1).unsqueeze(2)

        # Sample features
        sampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        sampled = rearrange(sampled, '(b v) n d -> b v n d', b=B, v=V)

        # Compute Visibility Mask (Occlusion Check)
        if depth_maps is not None:
            depth_flat = rearrange(depth_maps, 'b v c h w -> (b v) c h w')

            # Sample the surface depth at the vertex projection locations
            sampled_surface_depth = F.grid_sample(depth_flat, grid, mode='nearest', padding_mode='zeros', align_corners=True)
            sampled_surface_depth = sampled_surface_depth.squeeze(-1).permute(0, 2, 1)

            # Threshold for occlusion (epsilon tolerance = 5cm)
            depth_tolerance = 0.05

            is_visible = (verts_depth < (sampled_surface_depth + depth_tolerance))
            in_front = verts_depth > 0
            in_image = (u_ndc > -1.0) & (u_ndc < 1.0) & (v_ndc > -1.0) & (v_ndc < 1.0)
            in_image = in_image.unsqueeze(-1)

            valid_mask = is_visible & in_front & in_image
            valid_mask = rearrange(valid_mask, '(b v) n c -> b v n c', b=B, v=V).float()
        else:
            valid_mask = torch.ones_like(sampled[..., :1])

        # Average features from all visible views
        features_masked = sampled * valid_mask
        features_pooled = features_masked.sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

        return features_pooled

    def forward(self, data_batch, has_target_image=True):
        forward_start = torch.cuda.Event(enable_timing=True)
        pred_end = torch.cuda.Event(enable_timing=True)
        lbs_end = torch.cuda.Event(enable_timing=True)
        
        forward_start.record()

        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)
        b_size = input.image.shape[0]
        device = input.image.device
        H_img, W_img = input.image.shape[-2:]

        tpose_verts_for_scale = input.smpl_canonical_vertices[:, 0]
        min_y = tpose_verts_for_scale[..., 1].min(dim=1)[0]
        max_y = tpose_verts_for_scale[..., 1].max(dim=1)[0]
        heights = max_y - min_y
        target_height = self.config.training.get("target_height", 1.8)

        # Base scale factor (B, 1, 1)
        sf = (heights / target_height).view(b_size, 1, 1)

        if has_target_image and 'c2w' in target:
            true_target_c2w = target.c2w.clone()

        # Scale C2W translations
        input.c2w[:, :, :3, 3] = input.c2w[:, :, :3, 3] / sf

        # Scale Plucker rays origin
        sf_5d = sf.view(b_size, 1, 1, 1, 1)
        input.ray_o = input.ray_o / sf_5d

        # Scale Vertices
        sf_4d = sf.view(b_size, 1, 1, 1)
        input.smpl_posed_vertices = input.smpl_posed_vertices / sf_4d
        input.smpl_canonical_vertices = input.smpl_canonical_vertices / sf_4d

        # Tokenize Input Images
        input_pose_images, input_depth_maps = self._rasterize_pose_images(input) 
        posed_inputs = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d, pose_images=input_pose_images 
        )
        b, v_input, c, h, w = posed_inputs.size()
        input_img_tokens = self.image_tokenizer(posed_inputs)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)

        if self.use_canonical_maps:
            # Generate Canonical Targets
            tpose_verts = input.smpl_canonical_vertices[:, 0]

            canon_c2w, canon_intr = self._get_canonical_cameras(tpose_verts, b_size, device, image_size=H_img)
            canon_ray_o, canon_ray_d = self.process_data.compute_rays(canon_c2w, canon_intr, H_img, W_img, device=device)

            faces = input.smpl_faces[:, 0]

            canon_target_data = edict({
                "image_h_w": (H_img, W_img),
                "smpl_posed_vertices": tpose_verts.unsqueeze(1).expand(-1, 4, -1, -1),
                "smpl_canonical_vertices": tpose_verts.unsqueeze(1).expand(-1, 4, -1, -1),
                "smpl_faces": faces.unsqueeze(1).expand(-1, 4, -1, -1),
                "c2w": canon_c2w,
                "fxfycxcy": canon_intr
            })
            canon_pose_images, canon_depth_maps = self._rasterize_pose_images(canon_target_data)

            # Tokenize Canonical Targets
            canon_pose_cond = self.get_posed_input(ray_o=canon_ray_o, ray_d=canon_ray_d, pose_images=canon_pose_images)
            b, v_canon, _, _, _ = canon_pose_cond.size()
            canon_target_tokens = self.target_pose_tokenizer(canon_pose_cond)
            canon_target_tokens = canon_target_tokens.reshape(b, v_canon * n_patches, d)

            # Concatenate & Transform
            concat_tokens = torch.cat((input_img_tokens, canon_target_tokens), dim=1)
        
        else:
            concat_tokens = input_img_tokens
        
        # Append CLS Token if enabled
        if self.use_cls_token:
            cls_token = self.cls_token.expand(b_size, -1, -1)
            concat_tokens = torch.cat((cls_token, concat_tokens), dim=1)
        

        # Transformer Pass
        concat_tokens = self.transformer_input_layernorm(concat_tokens)

        interm_features = []

        # Extract features evenly across the network depth
        extract_layers = [self.n_layers // 4 - 1, self.n_layers // 2 - 1, 3 * self.n_layers // 4 - 1, self.n_layers - 1]

        for i, block in enumerate(self.transformer_blocks):
            concat_tokens = block(concat_tokens)
            if self.use_upsampler and i in extract_layers:
                interm_features.append(concat_tokens)
    
        concat_tokens = self.norm(concat_tokens)

        if self.use_upsampler:
            interm_features[-1] = concat_tokens 
        else:
            interm_features = [concat_tokens]

        # Extract CLS Token output
        if self.use_cls_token:
            global_feature = concat_tokens[:, 0] # (B, D)
            interm_features = [f[:, 1:] for f in interm_features]
            concat_tokens = concat_tokens[:, 1:]

        # Reshape tokens into spatial grids
        spatial_features = []
        for feat in interm_features:
            if self.use_canonical_maps:
                num_input_tokens = v_input * n_patches
                target_tokens = feat[:, num_input_tokens:, :]
                v_target = v_canon
            else:
                target_tokens = feat
                v_target = v_input

            H_feat = int(math.sqrt(n_patches))
            feat_map = rearrange(target_tokens, 'b (v h w) d -> (b v) d h w', v=v_target, h=H_feat, w=H_feat)
            spatial_features.append(feat_map)

        # Apply the CNN Upsampler
        if self.use_upsampler:
            feature_map = self.upsampler(spatial_features)
        else:
            feature_map = spatial_features[-1]

        # Get sampling vertices
        if self.use_canonical_maps:
            sampling_verts = tpose_verts.unsqueeze(1).expand(-1, 4, -1, -1)
            sampling_c2w = canon_c2w
            sampling_intr = canon_intr
            sampling_depth = canon_depth_maps
        else:
            sampling_verts = input.smpl_posed_vertices
            sampling_c2w = input.c2w
            sampling_intr = input.fxfycxcy
            sampling_depth = input_depth_maps

        # Feature Sampling and Decoding

        depth_maps = None if self.project_all_vertices else sampling_depth

        local_features = self.sample_pixel_aligned_features(
            feature_map, 
            sampling_verts, 
            sampling_c2w, 
            sampling_intr,
            (H_img, W_img),
            depth_maps=depth_maps
        )

        # Debug Visualization of Features and Vertex Projections
        if torch.distributed.get_rank() == 0 and not hasattr(self, '_debug_feats_verts_saved'):
            debug_st = time.time()
            os.makedirs("debug_renders", exist_ok=True)

            with torch.no_grad():
                # Visualize Feature Map (First batch, first view)
                feat_vis = feature_map[0].detach().float()

                if feat_vis.shape[0] > 3:
                    torch.manual_seed(42) 
                    proj_mat = torch.randn(3, feat_vis.shape[0], device=feat_vis.device)
                    feat_vis = torch.einsum('cd,dhw->chw', proj_mat, feat_vis)

                feat_vis_min = feat_vis.reshape(3, -1).min(dim=1)[0].reshape(3, 1, 1)
                feat_vis_max = feat_vis.reshape(3, -1).max(dim=1)[0].reshape(3, 1, 1)
                feat_vis = (feat_vis - feat_vis_min) / (feat_vis_max - feat_vis_min + 1e-6)

                feat_vis = F.interpolate(feat_vis.unsqueeze(0), size=(H_img, W_img), mode='nearest').squeeze(0)

                # Visualize Projected Vertices (Batch 0, View 0)
                verts_v0 = sampling_verts[0, 0] 
                c2w_v0 = sampling_c2w[0, 0]
                intr_v0 = sampling_intr[0, 0]
                depth_map_v0 = sampling_depth[0, 0, 0] # (H, W)

                w2c = torch.inverse(c2w_v0.float())
                R = w2c[:3, :3]
                T = w2c[:3, 3]
                verts_cam = (verts_v0.float() @ R.T) + T

                x_cam, y_cam, z_cam = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]
                fx, fy, cx, cy = intr_v0[0], intr_v0[1], intr_v0[2], intr_v0[3]

                u = (fx * (x_cam / (z_cam + 1e-6)) + cx).long()
                v = (fy * (y_cam / (z_cam + 1e-6)) + cy).long()

                # Frustum Check (Basic Projection)
                mask = (z_cam > 0) & (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
                u_frustum = u[mask]
                v_frustum = v[mask]
                z_frustum = z_cam[mask]

                # Occlusion Check (Depth)
                surface_z = depth_map_v0[v_frustum, u_frustum]
                is_visible = z_frustum < (surface_z + 0.05)

                # Final indices that are both in-frustum AND visible
                u_final = u_frustum[is_visible]
                v_final = v_frustum[is_visible]

                verts_canvas = torch.ones((3, H_img, W_img), device=input.image.device)
                verts_canvas[:, v_final, u_final] = 0.0

                # Save visualization
                input_img = input.image[0, 0].detach().cpu()
                feat_vis = feat_vis.cpu()
                verts_vis = verts_canvas.cpu()

                debug_grid = torch.cat([input_img, feat_vis, verts_vis], dim=2)
                torchvision.utils.save_image(debug_grid, "debug_renders/debug_features.png")
                print(f"DEBUG: Saved debug_renders/debug_features.png")
                self._debug_feats_verts_saved = True
                self.debug_overhead_2 = time.time() - debug_st

        # Concatenate Global Feature to Local Features
        if self.use_cls_token:
            global_feature_expanded = global_feature.unsqueeze(1).expand(-1, local_features.shape[1], -1)
            decoder_input = torch.cat([local_features, global_feature_expanded], dim=-1)
        else:
            decoder_input = local_features

        subject_tpose_verts = input.smpl_canonical_vertices[:, 0]
        base_lbs_weights = self.tpose_lbs_weights.unsqueeze(0).expand(b_size, -1, -1)
        raw_geo, raw_app, learnable_lbs_weights = self.vertex_decoder(decoder_input, subject_tpose_verts, base_lbs_weights)

        # Geometry Parsing
        offset = raw_geo[..., 0:3] 
        rot = F.normalize(raw_geo[..., 3:7], dim=-1)
        scale_norm = F.softplus(raw_geo[..., 7:10]) 

        # Appearance Parsing
        color = torch.sigmoid(raw_app[..., 0:3])
        opacity = torch.sigmoid(raw_app[..., 3:4])

        # Add Offsets
        means3D_norm = subject_tpose_verts.unsqueeze(2) + offset

        # Expand Means and Scales back to true Metric Space
        sf_out = sf.view(b_size, 1, 1, 1)

        flat_means3D = rearrange(means3D_norm * sf_out, 'b n k d -> b (n k) d')
        flat_rot = rearrange(rot, 'b n k d -> b (n k) d')
        flat_scale = rearrange(scale_norm * sf_out, 'b n k d -> b (n k) d')
        flat_color = rearrange(color, 'b n k d -> b (n k) d')
        flat_opacity = rearrange(opacity, 'b n k d -> b (n k) d')
        flat_weights = rearrange(learnable_lbs_weights, 'b n k j -> b (n k) j')

        pred_end.record()

        render = None
        loss_metrics = None

        # LBS & Render
        tgt_h, tgt_w = target.image.shape[-2], target.image.shape[-1]
        num_tgt_views = target.c2w.shape[1]

        tgt_c2w_flat = rearrange(true_target_c2w, 'b v x y -> (b v) x y')
        tgt_intrinsics_flat = rearrange(target.fxfycxcy, 'b v c -> (b v) c')

        fx = tgt_intrinsics_flat[:, 0]
        fy = tgt_intrinsics_flat[:, 1]
        cx = tgt_intrinsics_flat[:, 2]
        cy = tgt_intrinsics_flat[:, 3]

        proj_matrix = self._get_projection_matrix(fx, fy, cx, cy, tgt_h, tgt_w)
        view_matrix = torch.inverse(tgt_c2w_flat)

        props = {
            'means3D': flat_means3D, 'rotations': flat_rot, 'scales': flat_scale, 
            'opacity': flat_opacity, 'colors': flat_color, 'lbs_weights': flat_weights 
        }

        def expand_prop(t):
            return repeat(t, 'b ... -> (b v) ...', v=num_tgt_views)

        expanded_props = {k: expand_prop(v) for k, v in props.items()}
        tgt_transforms = rearrange(target.joint_transforms, 'b v j x y -> (b v) j x y')

        posed_props = lbs_gaussians(expanded_props, tgt_transforms)

        render_flat = render_predicted_gaussians(
            posed_props['means3D'], posed_props['rotations'], posed_props['scales'],
            posed_props['opacity'], posed_props['colors'], 
            view_matrix, proj_matrix, 
            tgt_h, tgt_w
        )

        render = rearrange(render_flat, '(b v) c h w -> b v c h w', b=b_size, v=num_tgt_views)

        if has_target_image:
            loss_metrics = self.loss_computer(render, target.image)

            # Tightness Regularization
            num_tight = self.vertex_decoder.num_tight

            if num_tight > 0:
                tight_offsets = offset[:, :, :num_tight, :]
                tightness_loss = (tight_offsets ** 2).sum(dim=-1).mean() * self.config.training.tightness_loss_weight
            else:
                tightness_loss = torch.tensor(0.0, device=offset.device)

            loss_metrics.loss += tightness_loss
            loss_metrics.tightness_loss = tightness_loss

        lbs_end.record()

        torch.cuda.synchronize()

        prediction_time = forward_start.elapsed_time(pred_end) / 1000.0
        lbs_time = pred_end.elapsed_time(lbs_end) / 1000.0

        # Subract CPU I/O overhead from debug image saving
        debug_overhead = 0.0
        if hasattr(self, 'debug_overhead_1'):
            debug_overhead += self.debug_overhead_1
            del self.debug_overhead_1
        if hasattr(self, 'debug_overhead_2'):
            debug_overhead += self.debug_overhead_2
            del self.debug_overhead_2

        prediction_time = max(0.001, prediction_time - debug_overhead)

        return edict(
            input=input,
            target=target,
            render=render, 
            loss_metrics=loss_metrics,
            prediction_time=prediction_time,
            lbs_time=lbs_time
        )

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = sorted([f for f in os.listdir(load_path) if f.endswith(".pt")])
            ckpt_paths = [os.path.join(load_path, n) for n in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
            self.load_state_dict(checkpoint["model"], strict=False) 
            print(f"Loaded checkpoint from {ckpt_paths[-1]}")
        except:
            traceback.print_exc()
            return None
        return 0

    @torch.no_grad()
    def predict_canonical_asset(self, data_batch):
        """
        Returns the Canonical Gaussian Asset based on the updated forward pass.
        """
        # Prepare Inputs
        input, _ = self.process_data(data_batch, has_target_image=False, target_has_input=False, compute_rays=True)
        b_size = input.image.shape[0]
        device = input.image.device
        H_img, W_img = input.image.shape[-2:]

        # World scaling normalization
        tpose_verts_for_scale = input.smpl_canonical_vertices[:, 0]
        min_y = tpose_verts_for_scale[..., 1].min(dim=1)[0]
        max_y = tpose_verts_for_scale[..., 1].max(dim=1)[0]
        heights = max_y - min_y
        target_height = self.config.training.get("target_height", 1.8)

        sf = (heights / target_height).view(b_size, 1, 1)

        input.c2w[:, :, :3, 3] = input.c2w[:, :, :3, 3] / sf

        sf_5d = sf.view(b_size, 1, 1, 1, 1)
        input.ray_o = input.ray_o / sf_5d

        sf_4d = sf.view(b_size, 1, 1, 1)
        input.smpl_posed_vertices = input.smpl_posed_vertices / sf_4d
        input.smpl_canonical_vertices = input.smpl_canonical_vertices / sf_4d

        # Tokenize Input Images
        input_pose_images, input_depth_maps = self._rasterize_pose_images(input) 
        posed_inputs = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d, pose_images=input_pose_images 
        )
        b, v_input, c, h, w = posed_inputs.size()
        input_img_tokens = self.image_tokenizer(posed_inputs)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)

        # Handle Canonical Target Maps (if enabled)
        if self.use_canonical_maps:
            tpose_verts = input.smpl_canonical_vertices[:, 0]
            canon_c2w, canon_intr = self._get_canonical_cameras(tpose_verts, b_size, device, image_size=H_img)
            canon_ray_o, canon_ray_d = self.process_data.compute_rays(canon_c2w, canon_intr, H_img, W_img, device=device)
            faces = input.smpl_faces[:, 0]

            canon_target_data = edict({
                "image_h_w": (H_img, W_img),
                "smpl_posed_vertices": tpose_verts.unsqueeze(1).expand(-1, 4, -1, -1),
                "smpl_canonical_vertices": tpose_verts.unsqueeze(1).expand(-1, 4, -1, -1),
                "smpl_faces": faces.unsqueeze(1).expand(-1, 4, -1, -1),
                "c2w": canon_c2w,
                "fxfycxcy": canon_intr
            })
            canon_pose_images, canon_depth_maps = self._rasterize_pose_images(canon_target_data)

            canon_pose_cond = self.get_posed_input(ray_o=canon_ray_o, ray_d=canon_ray_d, pose_images=canon_pose_images)
            b, v_canon, _, _, _ = canon_pose_cond.size()
            canon_target_tokens = self.target_pose_tokenizer(canon_pose_cond)
            canon_target_tokens = canon_target_tokens.reshape(b, v_canon * n_patches, d)

            concat_tokens = torch.cat((input_img_tokens, canon_target_tokens), dim=1)
        else:
            concat_tokens = input_img_tokens

        # Append CLS Token
        if self.use_cls_token:
            cls_token = self.cls_token.expand(b_size, -1, -1)
            concat_tokens = torch.cat((cls_token, concat_tokens), dim=1)

        # Transformer Pass with Intermediate Feature Extraction
        concat_tokens = self.transformer_input_layernorm(concat_tokens)
        interm_features = []
        extract_layers = [self.n_layers // 4 - 1, self.n_layers // 2 - 1, 3 * self.n_layers // 4 - 1, self.n_layers - 1]

        for i, block in enumerate(self.transformer_blocks):
            concat_tokens = block(concat_tokens)
            if self.use_upsampler and i in extract_layers:
                interm_features.append(concat_tokens)

        concat_tokens = self.norm(concat_tokens)

        if self.use_upsampler:
            interm_features[-1] = concat_tokens 
        else:
            interm_features = [concat_tokens]

        # Separate Global and Spatial Features
        if self.use_cls_token:
            global_feature = concat_tokens[:, 0]
            interm_features = [f[:, 1:] for f in interm_features]
            concat_tokens = concat_tokens[:, 1:]

        spatial_features = []
        for feat in interm_features:
            if self.use_canonical_maps:
                num_input_tokens = v_input * n_patches
                target_tokens = feat[:, num_input_tokens:, :]
                v_target = v_canon
            else:
                target_tokens = feat
                v_target = v_input

            H_feat = int(math.sqrt(n_patches))
            feat_map = rearrange(target_tokens, 'b (v h w) d -> (b v) d h w', v=v_target, h=H_feat, w=H_feat)
            spatial_features.append(feat_map)

        # Upsample & Sample Local Features
        if self.use_upsampler:
            feature_map = self.upsampler(spatial_features)
        else:
            feature_map = spatial_features[-1]

        if self.use_canonical_maps:
            sampling_verts = tpose_verts.unsqueeze(1).expand(-1, 4, -1, -1)
            sampling_c2w = canon_c2w
            sampling_intr = canon_intr
            sampling_depth = canon_depth_maps
        else:
            sampling_verts = input.smpl_posed_vertices
            sampling_c2w = input.c2w
            sampling_intr = input.fxfycxcy
            sampling_depth = input_depth_maps

        depth_maps = None if self.project_all_vertices else sampling_depth

        local_features = self.sample_pixel_aligned_features(
            feature_map, sampling_verts, sampling_c2w, sampling_intr, (H_img, W_img), depth_maps=depth_maps
        )

        # Decode Canonical Gaussians
        if self.use_cls_token:
            global_feature_expanded = global_feature.unsqueeze(1).expand(-1, local_features.shape[1], -1)
            decoder_input = torch.cat([local_features, global_feature_expanded], dim=-1)
        else:
            decoder_input = local_features

        subject_tpose_verts = input.smpl_canonical_vertices[:, 0]

        base_lbs_weights = self.tpose_lbs_weights.unsqueeze(0).expand(b_size, -1, -1)

        raw_geo, raw_app, learnable_lbs_weights = self.vertex_decoder(decoder_input, subject_tpose_verts, base_lbs_weights)

        # Parse Properties
        offset = raw_geo[..., 0:3] 
        rot = F.normalize(raw_geo[..., 3:7], dim=-1)
        scale_norm = F.softplus(raw_geo[..., 7:10]) 
        color = torch.sigmoid(raw_app[..., 0:3])
        opacity = torch.sigmoid(raw_app[..., 3:4])

        means3D_norm = subject_tpose_verts.unsqueeze(2) + offset

        # Un-normalization
        sf_out = sf.view(b_size, 1, 1, 1)

        flat_means3D = rearrange(means3D_norm * sf_out, 'b n k d -> b (n k) d')
        flat_scale = rearrange(scale_norm * sf_out, 'b n k d -> b (n k) d')
        flat_rot = rearrange(rot, 'b n k d -> b (n k) d')
        flat_color = rearrange(color, 'b n k d -> b (n k) d')
        flat_opacity = rearrange(opacity, 'b n k d -> b (n k) d')
        flat_weights = rearrange(learnable_lbs_weights, 'b n k j -> b (n k) j')

        return {
            'means3D': flat_means3D,
            'rotations': flat_rot,
            'scales': flat_scale,
            'colors': flat_color,
            'opacity': flat_opacity,
            'lbs_weights': flat_weights
        }