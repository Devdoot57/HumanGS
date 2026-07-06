import os
import torch
import numpy as np
import json
import cv2
import argparse
import math
from tqdm import tqdm
import torch.nn.functional as F
from smplx.lbs import batch_rodrigues
import smplx
from model.lbs import lbs_gaussians
from renderer.gaussian_renderer import render_predicted_gaussians
import time


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

def compute_rigid_transforms(rot_mats, joints, parents):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    B, J, _, _ = rot_mats.shape
    tm = torch.cat([
        F.pad(rot_mats.view(-1, 3, 3), [0, 0, 0, 1]), 
        F.pad(rel_joints.view(-1, 3, 1), [0, 0, 0, 1], value=1)
    ], dim=2).view(B, J, 4, 4)

    transform_chain = [tm[:, 0]]
    for i in range(1, len(parents)):
        curr_res = torch.matmul(transform_chain[parents[i]], tm[:, i])
        transform_chain.append(curr_res)

    return torch.stack(transform_chain, dim=1)

def prep_tensor(params, key, idx, device):
    if key in params:
        t = torch.from_numpy(params[key][idx]).float().to(device)
        return t.unsqueeze(0) if t.dim() == 1 else t
    return torch.zeros(1, 3, device=device)

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
    parser = argparse.ArgumentParser(description="Animate THuman4 Asset (Sanitized Hand Pose)")

    parser.add_argument("--asset_path", type=str, required=True, help="Path to Source .pt asset")
    parser.add_argument("--raw_data_root", type=str, default="/home/devdoot/workspace/datasets/THuman4.0", help="Root of raw THuman4 dataset")
    parser.add_argument("--motion_subject", type=str, default="subject00", help="Subject ID to steal motion from")
    parser.add_argument("--smplx_model_path", type=str, default="./body_models/smplx/SMPLX_NEUTRAL.npz")

    parser.add_argument("--output_video", type=str, default="thuman4_animation.mp4")
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument("--margin", type=float, default=1.5)

    # Camera Controls
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=-30.0)
    parser.add_argument("--view_rotation", type=float, default=0.0)

    # Pose Sanitation
    parser.add_argument("--use_hand_motion", action="store_true", help="If set, uses raw hand motion. Defaults to False (Neutral Hands).")

    args = parser.parse_args()
    device = torch.device("cuda")    

    # Load Asset
    print(f"Loading Asset: {args.asset_path}")
    asset = torch.load(args.asset_path, map_location=device, weights_only=True)
    asset = {k: v.unsqueeze(0).to(device) for k, v in asset.items()}

    # Auto-Zoom
    means = asset['means3D'][0]
    min_xyz = means.min(dim=0)[0].cpu().numpy()
    max_xyz = means.max(dim=0)[0].cpu().numpy()
    h_asset = max_xyz[1] - min_xyz[1]
    center_y = (max_xyz[1] + min_xyz[1]) / 2.0

    fov_deg = 60.0
    fov_rad = math.radians(fov_deg)
    safe_radius = (h_asset * args.margin) / (2.0 * math.tan(fov_rad / 2.0))
    print(f"Asset Height: {h_asset:.4f} | Radius: {safe_radius:.4f}")

    # Load SMPL Structure
    smpl_model_data = np.load(args.smplx_model_path)
    parents = torch.from_numpy(smpl_model_data['kintree_table'][0].astype(np.int32)).long().to(device)
    J_regressor = torch.from_numpy(smpl_model_data['J_regressor']).float().to(device)
    v_template = torch.from_numpy(smpl_model_data['v_template']).float().to(device)

    J_neutral = torch.matmul(J_regressor, v_template)[:55]
    h_smpl = (J_neutral[:, 1].max() - J_neutral[:, 1].min()).item()
    scale_factor = h_asset / h_smpl

    J_rest_scaled = (J_neutral * scale_factor).unsqueeze(0) 

    # Load Motion
    smpl_path = os.path.join(args.raw_data_root, args.motion_subject, 'smpl_params.npz')
    if not os.path.exists(smpl_path):
        print(f"Error: Could not find motion data at {smpl_path}")
        return
    motion_data = np.load(smpl_path, allow_pickle=True)
    total_frames = motion_data['body_pose'].shape[0]
    frame_indices = range(0, total_frames, 1) 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_video, fourcc, 30.0, (args.w, args.h))

    az_rad = math.radians(args.azimuth)
    el_rad = math.radians(args.elevation)

    r_xz = safe_radius * math.cos(el_rad)
    cam_y = center_y + (safe_radius * math.sin(el_rad))
    cam_x = r_xz * math.sin(az_rad)
    cam_z = r_xz * math.cos(az_rad)

    eye = np.array([cam_x, cam_y, cam_z])
    target = np.array([0, center_y, 0])
    up = np.array([0, 1, 0])

    # Build Camera Matrices
    c2w = get_look_at_matrix(eye, target, up)
    w2c = np.linalg.inv(c2w)

    w2c_tensor_static = torch.from_numpy(w2c).to(device).unsqueeze(0).float()

    if args.view_rotation != 0:
        angle = math.radians(args.view_rotation)
        c, s = math.cos(angle), math.sin(angle)
        view_rot = torch.tensor([
            [1,  0,  0, 0],
            [0,  c, -s, 0],
            [0,  s,  c, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        w2c_tensor_static = torch.matmul(w2c_tensor_static, view_rot)

    focal = (args.h / 2.0) / math.tan(fov_rad / 2.0)
    proj_matrix = get_projection_matrix(0.1, 10000.0, focal, focal, args.w / 2.0, args.h / 2.0, args.h, args.w, device)

    total_start = time.time()
    frame_count = 0

    # Animation Loop
    for idx in tqdm(frame_indices, desc="Rendering"):

        if args.use_hand_motion:
            left_hand = prep_tensor(motion_data, 'left_hand_pose', idx, device).view(1, 45)
            right_hand = prep_tensor(motion_data, 'right_hand_pose', idx, device).view(1, 45)
        else:
            left_hand = torch.zeros(1, 45, device=device)
            right_hand = torch.zeros(1, 45, device=device)

        full_pose = torch.cat([
            prep_tensor(motion_data, 'global_orient', idx, device).view(1, 3),
            prep_tensor(motion_data, 'body_pose', idx, device).view(1, 63),
            prep_tensor(motion_data, 'jaw_pose', idx, device).view(1, 3),
            prep_tensor(motion_data, 'leye_pose', idx, device).view(1, 3),
            prep_tensor(motion_data, 'reye_pose', idx, device).view(1, 3),
            left_hand,
            right_hand
        ], dim=1)

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3) 
        G_posed = compute_rigid_transforms(rot_mats, J_rest_scaled, parents)

        ident_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, 55, 3, 3)
        G_rest = compute_rigid_transforms(ident_rot, J_rest_scaled, parents)

        retargeted_transforms = torch.matmul(G_posed, torch.inverse(G_rest))

        with torch.no_grad():
            posed_props = lbs_gaussians(asset, retargeted_transforms)

            render_flat = render_predicted_gaussians(
                posed_props['means3D'], posed_props['rotations'], posed_props['scales'],
                posed_props['opacity'], posed_props['colors'],
                w2c_tensor_static, proj_matrix, args.h, args.w
            )

        img_tensor = render_flat[0].permute(1, 2, 0).detach().cpu().clamp(0, 1)
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

        frame_count += 1

    writer.release()

    # Performance Report
    total_time = time.time() - total_start
    avg_time = total_time / frame_count
    fps = 1.0 / avg_time

    print("\n" + "="*30)
    print(f" PERFORMANCE REPORT")
    print("="*30)
    print(f"Total Time:      {total_time:.2f} s")
    print(f"Total Frames:    {frame_count}")
    print(f"Average Time:    {avg_time:.4f} s/frame")
    print(f"Throughput:      {fps:.2f} FPS")
    print("="*30 + "\n")
    print(f"Saved to {args.output_video}")


if __name__ == "__main__":
    main()