import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion


def batch_lbs(canonical_pos, weights, joint_transforms):
    """
    Applies LBS to 3D points.
    
    Args:
        canonical_pos: (B, N, 3)
        weights: (B, N, J)
        joint_transforms: (B, J, 4, 4)
    """
    # Compute per-point transformation matrix T
    T = torch.einsum('bnj,bjxy->bnxy', weights, joint_transforms)

    # Apply T to positions
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    posed_pos = torch.einsum('bnij,bnj->bni', R, canonical_pos) + t

    return posed_pos, T

def lbs_gaussians(props, joint_transforms):
    """
    Deforms Gaussian properties using LBS.
    """
    means = props['means3D']
    rotations = props['rotations'] 
    weights = props['lbs_weights']

    # Deform Positions
    posed_means, T = batch_lbs(means, weights, joint_transforms)

    # Deform Rotations
    R_lbs = T[..., :3, :3]

    # Convert Gaussian Quaternions to Matrix
    R_gauss = quaternion_to_matrix(rotations)

    # Rotate the Gaussian: R_final = R_lbs @ R_original
    R_final = torch.matmul(R_lbs, R_gauss)

    # Convert back to Quaternion
    posed_rotations = matrix_to_quaternion(R_final)

    return {
        'means3D': posed_means,
        'rotations': posed_rotations,
        'opacity': props['opacity'],
        'scales': props['scales'],
        'colors': props['colors']
    }