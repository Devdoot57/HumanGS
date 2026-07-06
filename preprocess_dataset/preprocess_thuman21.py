import trimesh
import random
import numpy as np
import pickle
import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import pyrender 
from PIL import Image
import smplx
from smplx.lbs import batch_rodrigues


def get_args():
    parser = argparse.ArgumentParser(description="Pre-process THuman2.1 dataset for HumanRAM.")
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to scans.")
    parser.add_argument("--smplx_dir", type=str, required=True, help="Path to smplx.")
    parser.add_argument("--smplx_model_dir", type=str, default="body_models/", help="Path to SMPL-X models.")
    parser.add_argument("--output_dir", type=str, default="preprocessed_data/thuman21", help="Output path.")
    
    parser.add_argument("--num_views", type=int, default=60, help="Views per subject.")
    parser.add_argument("--image_size", type=int, default=512, help="Image resolution.")
    parser.add_argument("--train_subjects", type=int, default=2300, help="Train count.")
    parser.add_argument("--test_subjects", type=int, default=145, help="Test count.")
    
    parser.add_argument("--cam_radius_min", type=float, default=2.0, help="Min distance factor (Close).")
    parser.add_argument("--cam_radius_max", type=float, default=2.0, help="Max distance factor (Far).")
    parser.add_argument("--cam_altitude_min", type=float, default=-5.0)
    parser.add_argument("--cam_altitude_max", type=float, default=15.0)

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_rigid_transforms(rot_mats, joints, parents, transl=None):
    """
    Compute rigid transforms (FK).
    """
    import torch.nn.functional as F
    
    joints = torch.unsqueeze(joints, dim=-1) # [B, J, 3, 1]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    
    B, J, _, _ = rot_mats.shape
    
    # Local transform: Rotation + Relative translation
    tm = torch.cat([
        F.pad(rot_mats.view(-1, 3, 3), [0, 0, 0, 1]), # [B*J, 4, 3]
        F.pad(rel_joints.view(-1, 3, 1), [0, 0, 0, 1], value=1) # [B*J, 4, 1]
    ], dim=2).view(B, J, 4, 4)

    if transl is not None:
        tm[:, 0, :3, 3] += transl

    # Accumulate global transforms
    transform_chain = [tm[:, 0]]
    for i in range(1, len(parents)):
        curr_res = torch.matmul(transform_chain[parents[i]], tm[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    return transforms

def get_look_at_matrix(eye, target, up):
    """Generates a Camera-to-World (C2W) matrix (OpenGL convention: -Z forward)."""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    c2w = np.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye
    return c2w

def prep_tensor(params, key):
    t = torch.tensor(params[key], dtype=torch.float32)
    return t.unsqueeze(0) if t.dim() == 1 else t

def load_smplx_data(subject_id, args, smplx_model, mesh_transform):
    pkl_path = os.path.join(args.smplx_dir, subject_id, "smplx_param.pkl")
    obj_path = os.path.join(args.smplx_dir, subject_id, "mesh_smplx.obj")

    with open(pkl_path, 'rb') as f:
        params = pickle.load(f, encoding='latin1')
    
    # Load Params
    betas = prep_tensor(params, 'betas')
    scale = torch.tensor(params['scale'], dtype=torch.float32)
    transl = prep_tensor(params, 'transl')

    # Load Posed Mesh & Apply Alignment
    posed_mesh = trimesh.load(obj_path, force='mesh')
    posed_mesh.apply_transform(mesh_transform)
    v_posed = np.array(posed_mesh.vertices)
    faces = np.array(posed_mesh.faces)

    with torch.no_grad():
        # Get Unscaled Canonical Info
        canonical_output = smplx_model(betas=betas)
        J_rest_unscaled = canonical_output.joints[:, :55, :]
        v_cano_unscaled = canonical_output.vertices.detach().cpu().numpy().squeeze()
        
        J_rest_scaled = J_rest_unscaled * scale
        transl_scaled = transl * scale
        
        # Reconstruct Poses
        kw_get = lambda k, s: prep_tensor(params, k) if k in params else torch.zeros(1, s)

        full_pose = torch.cat([
            prep_tensor(params, 'global_orient').view(1, 3),
            prep_tensor(params, 'body_pose').view(1, 63),
            kw_get('jaw_pose', 3),
            kw_get('leye_pose', 3),
            kw_get('reye_pose', 3),
            kw_get('left_hand_pose', 45),
            kw_get('right_hand_pose', 45)
        ], dim=1) 

        # Compute Rotations
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
        parents = smplx_model.parents

        # Forward Kinematics (Posed)
        G_posed = compute_rigid_transforms(rot_mats, J_rest_scaled, parents, transl=transl_scaled)
        
        # Forward Kinematics (Rest)
        ident_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 55, 3, 3).to(rot_mats.device)
        G_rest = compute_rigid_transforms(ident_rot, J_rest_scaled, parents) # No transl for rest pose
        
        # Compute LBS Matrix: T = G_posed @ G_rest^-1
        transforms_smpl = torch.matmul(G_posed, torch.inverse(G_rest))
        transforms_smpl = transforms_smpl.squeeze(0).numpy() # (55, 4, 4)
        
        # Apply Global Mesh Alignment
        transforms_aligned = np.matmul(mesh_transform[np.newaxis, ...], transforms_smpl)

    v_cano = v_cano_unscaled * scale.item()
    
    return v_posed, v_cano, faces, transforms_aligned

def create_output_paths(output_dir):
    paths = {
        "root": Path(output_dir),
        "images": Path(output_dir) / "images",
        "smpl_data": Path(output_dir) / "smpl_data"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def process_subject(subject_id, smplx_model, args, output_paths):
    obj_path = os.path.join(args.scan_dir, subject_id, f"{subject_id}.obj")
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=True)
    except Exception as e:
        print(f"Error loading GT scan {subject_id}: {e}")
        return None

    # Semantic Alignment
    pkl_path = os.path.join(args.smplx_dir, subject_id, "smplx_param.pkl")
    try:
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        
        # Load params and fix dimensions
        betas = prep_tensor(params, 'betas')
        body_pose = prep_tensor(params, 'body_pose')
        global_orient = prep_tensor(params, 'global_orient')
        transl = prep_tensor(params, 'transl')
        scale = torch.tensor(params['scale'], dtype=torch.float32)

        with torch.no_grad():
            output = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        
        joints = output.joints.detach().cpu().numpy().squeeze() * scale.item()
        true_up = (joints[12] - joints[0])
        true_up = true_up / np.linalg.norm(true_up)
        
    except Exception as e:
        print(f"Error alignment {subject_id}: {e}")
        return None

    center_offset = mesh.bounds.mean(axis=0)
    target_up = np.array([0, 1, 0])
    rotation_matrix = np.eye(4)
    v = np.cross(true_up, target_up)
    if np.any(v):
        c = np.dot(true_up, target_up)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        rotation_matrix[:3, :3] = R

    T_trans = np.eye(4)
    T_trans[:3, 3] = -center_offset
    full_transform = rotation_matrix @ T_trans
    
    mesh.apply_transform(full_transform)
    
    # Compute Bounding Sphere
    bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1))

    try:
        v_posed, v_cano, faces, joint_transforms = load_smplx_data(subject_id, args, smplx_model, full_transform)
    except Exception as e:
        print(f"Error SMPL data {subject_id}: {e}")
        return None
        
    smpl_data_path = output_paths["smpl_data"] / f"{subject_id}.npz"
    np.savez(smpl_data_path, v_posed=v_posed, v_cano=v_cano, faces=faces, joint_transforms=joint_transforms, mesh_transform=full_transform)
    
    # Setup Pyrender Scene
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh_pyrender)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    
    r = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    camera = pyrender.PerspectiveCamera(yfov=np.radians(60), aspectRatio=1.0)
    
    frames_metadata = []
    for i in range(args.num_views):
        azimuth = np.radians(np.random.uniform(0, 360))
        altitude = np.radians(np.random.uniform(args.cam_altitude_min, args.cam_altitude_max))        

        dist_factor = np.random.uniform(args.cam_radius_min, args.cam_radius_max)
        radius = dist_factor * bounding_radius
        
        y = radius * np.sin(altitude)
        r_xz = radius * np.cos(altitude)
        x = r_xz * np.cos(azimuth)
        z = r_xz * np.sin(azimuth)
        
        c2w = get_look_at_matrix(eye=[x, y, z], target=[0, 0, 0], up=[0, 1, 0])
        w2c = np.linalg.inv(c2w)
        w2c[1, :] *= -1
        w2c[2, :] *= -1
        
        cam_node = scene.add(camera, pose=c2w)
        light_node = scene.add(light, pose=c2w)
        
        color, _ = r.render(scene)
        scene.remove_node(cam_node)
        scene.remove_node(light_node)
        
        img_path = output_paths["images"] / f"{subject_id}_view{i:03d}.png"
        Image.fromarray(color).save(img_path)
        
        f = (args.image_size/2.0) / np.tan(np.radians(60)/2.0)
        frames_metadata.append({
            "image_path": img_path.relative_to(output_paths["root"]).as_posix(),
            "w2c": w2c.tolist(),
            "fxfycxcy": [f, f, args.image_size/2.0, args.image_size/2.0],
            "smpl_path": smpl_data_path.relative_to(output_paths["root"]).as_posix()
        })
    r.delete()
    
    return {"scene_name": subject_id, "mesh_transform": full_transform.tolist(), "frames": frames_metadata}
    
def main():
    args = get_args()
    set_seed(args.seed)

    output_paths = create_output_paths(args.output_dir)
    all_ids = sorted([d for d in os.listdir(args.scan_dir) if os.path.isdir(os.path.join(args.scan_dir, d))])
    np.random.shuffle(all_ids)
    
    try:
        smplx_model = smplx.create(args.smplx_model_dir, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=False)
    except:
        print("Model not found")
        return

    data_list = []
    for sid in tqdm(all_ids[:args.train_subjects + args.test_subjects]):
        res = process_subject(sid, smplx_model, args, output_paths)
        if res: data_list.append(res)
            
    with open(output_paths["root"] / "train.json", 'w') as f:
        json.dump(data_list[:args.train_subjects], f, indent=2)
    with open(output_paths["root"] / "test.json", 'w') as f:
        json.dump(data_list[args.train_subjects:], f, indent=2)


if __name__ == "__main__":
    main()