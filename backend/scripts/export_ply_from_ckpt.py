"""Export a 3DGS .ply from a gsplat training checkpoint.

Usage:
    python scripts/export_ply_from_ckpt.py /path/to/ckpt.pt [out.ply]

gsplat's simple_trainer saves checkpoints as torch dicts holding the
splats state. We reload it, drop into the standard 3DGS PLY format
(x,y,z, opacity, scales, rots, sh DC + rest), and write it out.

This format is what every web viewer (mkkellogg/gaussian-splats-3d, Spark,
Babylon.js splats) expects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def export_ply(ckpt_path: Path, out_path: Path) -> None:
    print(f"loading {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # gsplat checkpoints typically nest splats under "splats" or come as a
    # dict with the parameter tensors directly. Try both.
    splats = ckpt.get("splats", ckpt)

    means = splats["means"].detach().cpu().numpy().astype(np.float32)
    scales = splats["scales"].detach().cpu().numpy().astype(np.float32)
    quats = splats["quats"].detach().cpu().numpy().astype(np.float32)
    opacities = splats["opacities"].detach().cpu().numpy().astype(np.float32)
    sh0 = splats["sh0"].detach().cpu().numpy().astype(np.float32)   # (N, 1, 3)
    shN = splats["shN"].detach().cpu().numpy().astype(np.float32)   # (N, K, 3)

    n = means.shape[0]
    print(f"  {n} gaussians, sh degree fragments: sh0={sh0.shape}, shN={shN.shape}")

    # Inria/3DGS PLY layout: f_dc_{0,1,2} for SH0, then f_rest_{0..3*K-1} for SH rest.
    # gsplat stores SH as (N, K, 3); the standard PLY interleaves channels by
    # SH coefficient first, then RGB:
    #   f_rest_0 = R[0], f_rest_1 = R[1], ...
    #   f_rest_K = G[0], ..., f_rest_2K-1 = G[K-1], ...
    # i.e. transpose (N,K,3) -> (N,3,K) then flatten last two dims.
    f_dc = sh0.reshape(n, -1)               # (N, 3)
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1)  # (N, 3*K)

    n_extra = f_rest.shape[1]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        *[(f"f_rest_{i}", "f4") for i in range(n_extra)],
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(n_extra):
        arr[f"f_rest_{i}"] = f_rest[:, i]
    arr["opacity"] = opacities.reshape(-1)
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    )

    el = PlyElement.describe(arr, "vertex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], text=False).write(str(out_path))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main(ckpt_str: str, out_str: str | None = None) -> int:
    ckpt = Path(ckpt_str)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 1
    out = Path(out_str) if out_str else ckpt.parent.parent / "ply" / f"{ckpt.stem}.ply"
    export_ply(ckpt, out)
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    sys.exit(main(*sys.argv[1:]))
