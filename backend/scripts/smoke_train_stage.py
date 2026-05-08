"""Train a 3D Gaussian Splat from a previous smoke_pose_stage output.

Usage:
    python scripts/smoke_train_stage.py /workspace/test5_out [iters]

Requires gsplat's `examples/simple_trainer.py` checked out at
$NUDORMS_GSPLAT_DIR (default /workspace/gsplat).

Reuses the COLMAP-format sparse model from the pose stage and shells out
to gsplat's reference MCMC trainer. The trainer writes:
    <out>/splat/ckpts/        intermediate checkpoints (latest = final)
    <out>/splat/ply/           exported .ply files
    <out>/splat/renders/       sample novel-view renders
    <out>/splat/tb/            tensorboard logs

We'll wire this into pipeline/train/gsplat_mcmc.py once the basic loop
is proven to work end-to-end on real captures.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

GSPLAT_DIR = Path(os.environ.get("NUDORMS_GSPLAT_DIR", "/workspace/gsplat"))
DEFAULT_ITERS = 30_000


def _stage_for_gsplat(pose_out: Path, stage: Path) -> None:
    """gsplat's COLMAP loader expects:
        <data_dir>/images/<frames>
        <data_dir>/sparse/0/{cameras,images,points3D}.bin

    Our pose stage writes:
        <pose_out>/frames/...
        <pose_out>/poses/sparse/0/...

    Use symlinks so we don't double the disk usage.
    """
    stage.mkdir(parents=True, exist_ok=True)
    images_link = stage / "images"
    sparse_link = stage / "sparse"
    if images_link.is_symlink() or images_link.exists():
        images_link.unlink()
    if sparse_link.is_symlink() or sparse_link.exists():
        if sparse_link.is_dir() and not sparse_link.is_symlink():
            shutil.rmtree(sparse_link)
        else:
            sparse_link.unlink()
    images_link.symlink_to(pose_out / "frames")
    sparse_link.symlink_to(pose_out / "poses" / "sparse")


def main(pose_out_str: str, iters_str: str | None = None) -> int:
    pose_out = Path(pose_out_str).resolve()
    iters = int(iters_str) if iters_str else DEFAULT_ITERS
    if not (pose_out / "poses" / "sparse" / "0" / "cameras.bin").exists():
        print(f"missing sparse model in {pose_out}/poses/sparse/0/", file=sys.stderr)
        return 1

    trainer = GSPLAT_DIR / "examples" / "simple_trainer.py"
    if not trainer.exists():
        print(f"missing gsplat trainer at {trainer}", file=sys.stderr)
        print("clone with: git clone https://github.com/nerfstudio-project/gsplat.git "
              f"{GSPLAT_DIR}", file=sys.stderr)
        return 2

    stage = pose_out / "_gsplat_input"
    _stage_for_gsplat(pose_out, stage)

    out = pose_out / "splat"
    out.mkdir(exist_ok=True)

    # MCMC strategy from gsplat's simple_trainer. Other key flags:
    #   --data_factor 1   keep full image resolution (default 4 downsamples 4x)
    #   --max_steps       total iters
    #   --save_steps      checkpoint frequency
    #   --eval_steps      eval frequency on holdout views
    #   --packed          memory-efficient packed gradients
    cmd = [
        sys.executable, str(trainer), "mcmc",
        "--data_dir", str(stage),
        "--result_dir", str(out),
        "--data_factor", "1",            # full res; avoids needing pre-downsampled images_N/
        "--max_steps", str(iters),
        "--save_steps", str(iters),       # only save final checkpoint
        "--eval_steps", str(iters // 5),  # eval 5 times during training
        "--ply_steps", str(iters),        # export final .ply
        "--init_type", "sfm",             # init from COLMAP points
        "--packed",
    ]
    print("\n── gsplat MCMC training ──")
    print(" ".join(cmd))
    print()
    res = subprocess.run(cmd, cwd=str(GSPLAT_DIR / "examples"))
    if res.returncode != 0:
        print(f"\ntrainer exited with {res.returncode}", file=sys.stderr)
        return res.returncode

    print(f"\n✔ training complete. outputs in: {out}")
    print("  - .ply files: ", out / "ply")
    print("  - renders:    ", out / "renders")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    sys.exit(main(*sys.argv[1:]))
