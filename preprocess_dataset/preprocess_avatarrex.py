import os
import json
import numpy as np
import torch
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import cv2
import smplx
from smplx.lbs import batch_rodrigues
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description="Pre-process AvatarReX (Motion + Alignment).")
    parser.add_argument("--data_root", type=str, required=True, help="Root path.")
    parser.add_argument("--output_dir", type=str, default="preprocessed_data/avatarrex", help="Output directory.")
    parser.add_argument("--smplx_model_dir", type=str, default="body_models/", help="Path to SMPL-X models.")
    parser.add_argument("--sample_rate", type=int, default=10, help="Process every Nth frame.")
    parser.add_argument("--image_size", type=int, default=512, help="Resolution.")
    return parser.parse_args()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def compute_rigid_transforms(rot_mats, joints, parents, transl=None):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    
    B, J, _, _ = rot_mats.shape
    tm = torch.cat([
        F.pad(rot_mats.view(-1, 3, 3), [0, 0, 0, 1]),
        F.pad(rel_joints.view(-1, 3, 1), [0, 0, 0, 1], value=1)
    ], dim=2).view(B, J, 4, 4)
    
    if transl is not None:
        tm[:, 0, :3, 3] += transl
        
    transform_chain = [tm[:, 0]]
    for i in range(1, len(parents)):
        curr_res = torch.matmul(transform_chain[parents[i]], tm[:, i])
        transform_chain.append(curr_res)
        
    return torch.stack(transform_chain, dim=1)

def get_alignment_transform(smpl_model, smpl_data, device):
    """
    Computes transform M using Frame 0 to standardize the subject.
    This same M will be applied to all subsequent frames.
    """
    # Use Frame 0
    def get_p(k):
        return smpl_data[k][0].to(device).unsqueeze(0) if k in smpl_data else torch.zeros(1,3).to(device)
        
    betas = smpl_data['betas'][0].to(device).unsqueeze(0)
    
    with torch.no_grad():
        output = smpl_model(
            global_orient=get_p('global_orient'),
            transl=get_p('transl'),
            body_pose=get_p('body_pose'),
            betas=betas
        )
        verts = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        
    # Rotation (Align Spine to Y+)
    true_up = (joints[12] - joints[0])
    true_up = true_up / np.linalg.norm(true_up)
    target_up = np.array([0, 1, 0])
    rotation_matrix = np.eye(4)
    v = np.cross(true_up, target_up)
    
    if np.any(v):
        c = np.dot(true_up, target_up)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        rotation_matrix[:3, :3] = R
        
    # Centering (Rotate, then find center of Frame 0)
    verts_rotated = (rotation_matrix[:3, :3] @ verts.T).T
    min_xyz = verts_rotated.min(axis=0)
    max_xyz = verts_rotated.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0

    # M = Translate(-Center) @ Rotate
    T_mat = np.eye(4)
    T_mat[:3, 3] = -center
    full_transform = T_mat @ rotation_matrix

    return full_transform

def process_subject(subject_id, args, smplx_model):
    print(f"\n--- Processing {subject_id} ---")
    subject_dir = os.path.join(args.data_root, subject_id)
    
    # Check files
    calib_path = os.path.join(subject_dir, 'calibration_full.json')
    smpl_path = os.path.join(subject_dir, 'smpl_params.npz')
    if not os.path.exists(calib_path) or not os.path.exists(smpl_path):
        return None
        
    with open(calib_path, 'r') as fp:
        cam_data = json.load(fp)
        
    # Filter for valid camera directories
    cam_keys = sorted([k for k in cam_data.keys() if os.path.isdir(os.path.join(subject_dir, k))], key=natural_keys)
    if not cam_keys:
        return None
        
    smpl_data_np = np.load(smpl_path, allow_pickle=True)
    smpl_data = {k: torch.from_numpy(v.astype(np.float32)) if isinstance(v, np.ndarray) else v for k, v in dict(smpl_data_np).items()}
    
    # Compute Alignment (Using Frame 0 stats)
    device = next(smplx_model.parameters()).device
    norm_transform = get_alignment_transform(smplx_model, smpl_data, device)
    
    # Output Dirs
    out_img_dir = Path(args.output_dir) / "images" / subject_id
    out_smpl_dir = Path(args.output_dir) / "smpl_data" / subject_id
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_smpl_dir.mkdir(parents=True, exist_ok=True)
    
    # Process Loop
    total_frames = smpl_data['body_pose'].shape[0]
    indices = range(0, total_frames, args.sample_rate)
    frames_metadata = []
    norm_inv = np.linalg.inv(norm_transform)
    
    def get_param(key, idx):
        if key in smpl_data:
            return smpl_data[key][idx].to(device).unsqueeze(0)
        return torch.zeros(1, 3).to(device)
        
    betas = smpl_data['betas'][0].to(device).unsqueeze(0)
    
    for frame_idx in tqdm(indices, desc=f"Processing {subject_id}", leave=False):
        try:
            # Load Params
            global_orient = get_param('global_orient', frame_idx)
            raw_transl = get_param('transl', frame_idx)
            body_pose = get_param('body_pose', frame_idx)
            jaw_pose = get_param('jaw_pose', frame_idx)
            leye_pose = get_param('leye_pose', frame_idx) if 'leye_pose' in smpl_data else torch.zeros(1, 3).to(device)
            reye_pose = get_param('reye_pose', frame_idx) if 'reye_pose' in smpl_data else torch.zeros(1, 3).to(device)
            left_hand_pose = get_param('left_hand_pose', frame_idx) if 'left_hand_pose' in smpl_data else torch.zeros(1, 45).to(device)
            right_hand_pose = get_param('right_hand_pose', frame_idx) if 'right_hand_pose' in smpl_data else torch.zeros(1, 45).to(device)
            expression = get_param('expression', frame_idx) if 'expression' in smpl_data else torch.zeros(1, 10).to(device)
            
            with torch.no_grad():
                # Forward Pass (Raw)
                output = smplx_model(
                    global_orient=global_orient,
                    transl=raw_transl,
                    body_pose=body_pose,
                    jaw_pose=jaw_pose,
                    betas=betas,
                    expression=expression,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose
                )

                canonical_output = smplx_model(betas=betas)
                
            v_raw = output.vertices.detach().cpu().numpy().squeeze()
            
            # Apply Norm to Vertices: V_new = M @ V_old
            v_posed = (norm_transform[:3, :3] @ v_raw.T).T + norm_transform[:3, 3]
            
            # Canonical Vertices & Joints (For Alignment)
            v_cano = canonical_output.vertices.detach().cpu().numpy().squeeze()
            J_rest_raw = canonical_output.joints[:, :55, :]
            
            # Joint Transforms
            full_pose_vec = torch.cat([global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose], dim=1)
            rot_mats = batch_rodrigues(full_pose_vec.view(-1, 3)).view(1, -1, 3, 3)
            parents = smplx_model.parents
            
            # Raw FK
            G_posed_raw = compute_rigid_transforms(rot_mats, J_rest_raw, parents, transl=raw_transl)
            ident_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 55, 3, 3).to(device)
            G_rest_raw = compute_rigid_transforms(ident_rot, J_rest_raw, parents, transl=None)
            
            # Aligned FK: T_aligned = M @ T_raw
            T_raw = torch.matmul(G_posed_raw, torch.inverse(G_rest_raw))
            M_tensor = torch.from_numpy(norm_transform).float().to(device).unsqueeze(0).unsqueeze(0)
            joint_transforms_tensor = torch.matmul(M_tensor, T_raw)
            joint_transforms = joint_transforms_tensor.squeeze(0).cpu().numpy()
            faces = smplx_model.faces
            
            # Save SMPL
            dst_smpl = out_smpl_dir / f"{frame_idx:08d}.npz"
            np.savez(dst_smpl, v_posed=v_posed, v_cano=v_cano, faces=faces, joint_transforms=joint_transforms)
            smpl_rel = str(dst_smpl.relative_to(args.output_dir))
            
            # Camera Processing
            for cam_key in cam_keys:
                img_path = os.path.join(subject_dir, cam_key, f"{frame_idx:08d}.jpg")
                msk_path = os.path.join(subject_dir, cam_key, 'mask', 'pha', f"{frame_idx:08d}.jpg")
                
                if not os.path.exists(img_path) or not os.path.exists(msk_path):
                    continue
                    
                img = cv2.imread(img_path)
                msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None or msk is None:
                    continue
                    
                # Mask BG
                alpha = (msk > 128).astype(np.float32)[..., None]
                white_bg = np.ones_like(img) * 255.0
                img = (img * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
                
                # Pad to Square
                h_orig, w_orig = img.shape[:2]
                L = max(h_orig, w_orig)
                canvas = np.ones((L, L, 3), dtype=np.uint8) * 255
                pad_h, pad_w = (L - h_orig) // 2, (L - w_orig) // 2
                canvas[pad_h:pad_h+h_orig, pad_w:pad_w+w_orig] = img
                
                img_final = cv2.resize(canvas, (args.image_size, args.image_size))
                dst_name = f"{cam_key}_{frame_idx:08d}.jpg"
                dst_path = out_img_dir / dst_name
                cv2.imwrite(str(dst_path), img_final)
                
                # Calib Update
                K_raw = np.array(cam_data[cam_key]['K'], dtype=np.float32).reshape(3, 3)
                K_raw[0, 2] += pad_w
                K_raw[1, 2] += pad_h
                scale = args.image_size / L
                fxfycxcy = [
                    float(K_raw[0,0] * scale), 
                    float(K_raw[1,1] * scale), 
                    float(K_raw[0,2] * scale), 
                    float(K_raw[1,2] * scale)
                ]
                
                # Extrinsics Alignment: W2C_new = W2C_raw @ M_inv
                R_cv = np.array(cam_data[cam_key]['R'], dtype=np.float32).reshape(3, 3)
                T_cv = np.array(cam_data[cam_key]['T'], dtype=np.float32).reshape(3)
                w2c_raw = np.eye(4)
                w2c_raw[:3, :3] = R_cv
                w2c_raw[:3, 3] = T_cv
                w2c = w2c_raw @ norm_inv
                
                frames_metadata.append({
                    "image_path": str(dst_path.relative_to(args.output_dir)),
                    "w2c": w2c.tolist(),
                    "fxfycxcy": fxfycxcy,
                    "smpl_path": smpl_rel,
                    "frame_idx": frame_idx
                })
                
        except Exception as e:
            print(f"Failed Frame {frame_idx}: {e}")
            continue
            
    if frames_metadata:
        return {
            "scene_name": subject_id, 
            "mesh_transform": norm_transform.tolist(),
            "frames": frames_metadata
        }
    return None

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        smplx_model = smplx.create(args.smplx_model_dir, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=True, ext='pkl').to(device)
    except Exception as e:
        print(f"Error loading SMPL-X: {e}")
        return
        
    subjects = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith("avatarrex_")])
    test_data = []
    
    for subj in tqdm(subjects, desc="Processing Test"):
        meta = process_subject(subj, args, smplx_model)
        if meta and len(meta['frames']) > 0:
            test_data.append(meta)
            
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    with open(output_root / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)


if __name__ == "__main__":
    main()