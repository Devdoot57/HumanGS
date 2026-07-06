import os
import torch
import numpy as np
import cv2
import argparse
import math
from tqdm import tqdm
from renderer.gaussian_renderer import render_predicted_gaussians


def get_projection_matrix(znear, zfar, fx, fy, cx, cy, h, w, device):
    P = torch.zeros((4, 4), device=device)
    P[0, 0] = 2.0 * fx / w
    P[0, 2] = 2.0 * cx / w - 1.0
    P[1, 1] = 2.0 * fy / h        
    P[1, 2] = 2.0 * cy / h - 1.0
    P[2, 2] = (zfar + znear) / (zfar - znear)
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P.unsqueeze(0)

def get_look_at_matrix(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Forward (+Z) points FROM eye TO target
    z_axis = target - eye
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Right (+X)
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # Down (+Y)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye
    return c2w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_path", type=str, required=True, help="Path to .pt asset")
    parser.add_argument("--output_video", type=str, default="canonical_spin.mp4")
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument("--elevation_deg", type=float, default=10.0, help="Camera elevation angle")
    parser.add_argument("--margin", type=float, default=1.3, help="Zoom margin (1.3 = fit 1/1.3 of screen)")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load Asset
    print(f"Loading asset: {args.asset_path}")
    asset = torch.load(args.asset_path, map_location=device, weights_only=True)

    if asset['means3D'].dim() == 2:
        asset = {k: v.unsqueeze(0).to(device) for k, v in asset.items()}

    means = asset['means3D'][0] # [N, 3]
    print(f"Asset loaded. Gaussians: {means.shape[0]}")

    # Compute Bounding Box & Center
    min_xyz = means.min(dim=0)[0].cpu().numpy()
    max_xyz = means.max(dim=0)[0].cpu().numpy()
    center = (min_xyz + max_xyz) / 2.0
    height = max_xyz[1] - min_xyz[1]

    print(f"Asset Center: {center}")
    print(f"Asset Height: {height:.2f}m")

    # Auto-Calculate Radius
    fov_deg = 60.0
    fov_rad = math.radians(fov_deg)
    radius = (height * args.margin) / (2.0 * math.tan(fov_rad / 2.0))
    print(f"Auto-calculated Radius: {radius:.2f}m")

    # Intrinsics
    focal = (args.h / 2.0) / math.tan(fov_rad / 2.0)
    fx, fy = focal, focal
    cx, cy = args.w / 2.0, args.h / 2.0
    proj_matrix = get_projection_matrix(0.1, 100.0, fx, fy, cx, cy, args.h, args.w, device)

    # Render Loop
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_video, fourcc, 30.0, (args.w, args.h))

    frames = 120

    for i in tqdm(range(frames)):
        azimuth_deg = 360.0 * (i / frames)
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(args.elevation_deg)

        # Calculate Camera Position (Orbit)
        # y is up. x/z plane.
        y = center[1] + radius * math.sin(elevation)
        r_xz = radius * math.cos(elevation)
        x = center[0] + r_xz * math.sin(azimuth)
        z = center[2] + r_xz * math.cos(azimuth)

        eye = [x, y, z]
        target = center 
        up = [0, 1, 0]

        c2w_np = get_look_at_matrix(eye, target, up)
        w2c_np = np.linalg.inv(c2w_np)
        view_matrix = torch.from_numpy(w2c_np).to(device).unsqueeze(0).float()

        with torch.no_grad():
            render_flat = render_predicted_gaussians(
                asset['means3D'], 
                asset['rotations'], 
                asset['scales'],
                asset['opacity'], 
                asset['colors'],
                view_matrix, 
                proj_matrix, 
                args.h, args.w
            )

        # Format output for video writing
        img_tensor = render_flat[0].permute(1, 2, 0).detach().cpu().clamp(0, 1)
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

    writer.release()
    print(f"Saved to {args.output_video}")


if __name__ == "__main__":
    main()