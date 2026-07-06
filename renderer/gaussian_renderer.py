import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


@torch.amp.autocast("cuda", enabled=False)
def render_predicted_gaussians(
    means3D,
    rotations,
    scales,
    opacity,
    colors,
    view_matrix,
    proj_matrix,
    img_height,
    img_width,
    bg_color=None
):
    """
    Renders a batch of Gaussians using diff-gaussian-rasterization.
    
    Args:
        view_matrix: (B, 4, 4) World-to-Camera matrix (Row-Major)
        proj_matrix: (B, 4, 4) Camera-to-Clip matrix (Row-Major)
    """
    # Cast inputs to Float32
    means3D = means3D.float()
    rotations = rotations.float()
    scales = scales.float()
    opacity = opacity.float()
    colors = colors.float()
    view_matrix = view_matrix.float()
    proj_matrix = proj_matrix.float()

    B = means3D.shape[0]
    device = means3D.device

    if bg_color is None:
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    else:
        bg_color = bg_color.float()

    rendered_batch = []

    # Pre-transpose matrices for the rasterizer (which expects Column-Major)
    # Input is Row-Major
    view_matrix_t = view_matrix.transpose(1, 2)

    for b in range(B):
        # Calculate Full Projection (P @ V) and Transpose it -> (P @ V)^T = V^T @ P^T
        full_proj = torch.matmul(proj_matrix[b], view_matrix[b])
        full_proj_t = full_proj.transpose(0, 1)
        view_matrix_t_b = view_matrix_t[b]

        # Calculate FoV from Projection Matrix diagonals
        tan_fovx = 1.0 / (proj_matrix[b, 0, 0] + 1e-7)
        tan_fovy = 1.0 / (torch.abs(proj_matrix[b, 1, 1]) + 1e-7)

        # Camera Center: Inverse of View Matrix
        c2w = torch.inverse(view_matrix[b])
        cam_center = c2w[:3, 3]

        raster_settings = GaussianRasterizationSettings(
            image_height=int(img_height),
            image_width=int(img_width),
            tanfovx=tan_fovx,
            tanfovy=tan_fovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=view_matrix_t_b,
            projmatrix=full_proj_t,
            sh_degree=0, 
            campos=cam_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize
        rendered_image, _ = rasterizer(
            means3D=means3D[b],
            means2D=torch.zeros_like(means3D[b], dtype=torch.float32, requires_grad=True),
            shs=None,
            colors_precomp=colors[b],
            opacities=opacity[b],
            scales=scales[b],
            rotations=rotations[b],
            cov3D_precomp=None
        )

        rendered_batch.append(rendered_image)

    return torch.stack(rendered_batch, dim=0)