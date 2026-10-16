import numpy as np
import torch
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814
CORE_KEYS = ("means3D", "rotations", "scales", "opacity", "colors")
_EPS = 1e-6


def _np(t):
    return t.detach().float().cpu().numpy()


def save_gaussians_ply(asset, path):
    N = asset["means3D"].shape[0]
    cols, names, comments = [], [], []

    def add(name_prefix, arr):
        arr = arr.reshape(N, -1)
        for i in range(arr.shape[1]):
            names.append(f"{name_prefix}{i}" if name_prefix.endswith("_") else f"{name_prefix}_{i}")
            cols.append(arr[:, i])

    xyz = _np(asset["means3D"]).reshape(N, 3)
    for i, n in enumerate("xyz"):
        names.append(n); cols.append(xyz[:, i])
    for n in ("nx", "ny", "nz"):
        names.append(n); cols.append(np.zeros(N, dtype=np.float32))

    rgb = _np(asset["colors"]).reshape(N, 3)
    add("f_dc_", (rgb - 0.5) / SH_C0)

    op = np.clip(_np(asset["opacity"]).reshape(N, 1), _EPS, 1 - _EPS)
    add("opacity", np.log(op / (1 - op)))
    names[-1] = "opacity"

    add("scale_", np.log(np.clip(_np(asset["scales"]).reshape(N, 3), _EPS, None)))
    add("rot_", _np(asset["rotations"]).reshape(N, 4))

    for k in CORE_KEYS:
        comments.append("shape " + k + " " + " ".join(map(str, asset[k].shape[1:])))

    for k, v in asset.items():
        if k in CORE_KEYS:
            continue
        if v.dim() >= 1 and v.shape[0] == N:
            add(f"extra_{k}_", _np(v))
            comments.append("extra " + k + " " + " ".join(map(str, v.shape[1:])))
        else:
            print(f"[ply_io] WARNING: '{k}' {tuple(v.shape)} is not per-Gaussian, skipped")

    dtype = [(n, "f4") for n in names]
    verts = np.empty(N, dtype=dtype)
    for n, c in zip(names, cols):
        verts[n] = c.astype(np.float32)

    PlyData([PlyElement.describe(verts, "vertex")], comments=comments).write(path)


def load_gaussians_ply(path, device="cpu"):
    """Returns dict of unbatched tensors [N, ...] (same as old torch.load result)."""
    ply = PlyData.read(path)
    v = ply["vertex"].data
    N = len(v)
    f = lambda names: np.stack([v[n] for n in names], axis=1).astype(np.float32)

    shapes, extras = {}, []
    for c in ply.comments:
        p = c.split()
        if p[0] == "shape":
            shapes[p[1]] = tuple(int(x) for x in p[2:])
        elif p[0] == "extra":
            extras.append((p[1], tuple(int(x) for x in p[2:])))

    def shaped(k, arr, default):
        return torch.from_numpy(arr).reshape(N, *shapes.get(k, default))

    rgb = f([f"f_dc_{i}" for i in range(3)]) * SH_C0 + 0.5
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    n_rot = sum(1 for n in v.dtype.names if n.startswith("rot_"))

    out = {
        "means3D": shaped("means3D", f(["x", "y", "z"]), (3,)),
        "colors": shaped("colors", rgb, (3,)),
        "opacity": shaped("opacity", opacity[:, None], (1,)),
        "scales": shaped("scales", np.exp(f([f"scale_{i}" for i in range(3)])), (3,)),
        "rotations": shaped("rotations", f([f"rot_{i}" for i in range(n_rot)]), (4,)),
    }
    for k, dims in extras:
        d = int(np.prod(dims)) if dims else 1
        arr = f([f"extra_{k}_{i}" for i in range(d)])
        out[k] = torch.from_numpy(arr).reshape(N, *dims)

    return {k: t.to(device) for k, t in out.items()}
