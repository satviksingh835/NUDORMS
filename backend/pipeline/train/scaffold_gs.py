"""Scaffold-GS trainer wrapper — Stage 2f.

Scaffold-GS (city-super/Scaffold-GS, CVPR 2024 Highlight): anchor-based
Gaussian hierarchy with view-conditioned MLP. Explicitly designed for
"intricate indoor environments with challenging observing views —
transparency, specularity, reflection, texture-less regions."

On indoor scenes Scaffold-GS achieves PSNR ~0.5 dB higher than vanilla
3DGS at the same Gaussian count because the anchor→neural-offset
representation avoids needle initialisation entirely.

This module wraps the Scaffold-GS train.py as a subprocess. Set
NUDORMS_SCAFFOLD_GS_DIR to the cloned repo path (default:
/workspace/Scaffold-GS). If the directory doesn't exist or training
fails, the orchestrator falls back to gsplat_mcmc.

Output PLY format is identical to vanilla 3DGS — the orchestrator/eval
stages treat both interchangeably.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from ..types import StageResult

log = logging.getLogger("nudorms.train.scaffold_gs")

DEFAULT_SCAFFOLD_DIR = "/workspace/Scaffold-GS"
ITERATIONS = 30_000


def _find_output_ply(model_path: Path) -> Path | None:
    """Locate the final point_cloud.ply inside Scaffold-GS model output dir."""
    # Scaffold-GS writes: <model_path>/point_cloud/iteration_<N>/point_cloud.ply
    pc_dir = model_path / "point_cloud"
    if not pc_dir.exists():
        return None
    iters = sorted(
        pc_dir.iterdir(),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.startswith("iteration_") else -1,
    )
    for d in reversed(iters):
        ply = d / "point_cloud.ply"
        if ply.exists():
            return ply
    return None


def _parse_final_psnr(log_text: str) -> float:
    """Extract last reported PSNR from Scaffold-GS stdout."""
    matches = re.findall(r"PSNR\s*[:=]\s*([\d.]+)", log_text)
    return float(matches[-1]) if matches else 0.0


def run(scan_id: str, workdir: Path, pose_artifacts: dict,
        prior_artifacts: dict | None = None, attempt: int = 1) -> StageResult:
    scaffold_dir = Path(os.environ.get("NUDORMS_SCAFFOLD_GS_DIR", DEFAULT_SCAFFOLD_DIR))
    train_script = scaffold_dir / "train.py"

    if not train_script.exists():
        return StageResult(
            ok=False, metrics={}, artifacts={},
            failure_reason=f"Scaffold-GS not found at {scaffold_dir}. "
                           "Set NUDORMS_SCAFFOLD_GS_DIR or run runpod_bootstrap.sh.",
        )

    # Scaffold-GS depends on CUDA submodules (diff-gaussian-rasterization,
    # simple-knn) that have to be built from source. If they aren't
    # importable, fail fast so the orchestrator falls through to gsplat
    # MCMC instead of paying ~2 min to spawn a subprocess that will crash
    # at import time.
    try:
        import diff_gaussian_rasterization  # noqa: F401
        import simple_knn  # noqa: F401
    except ImportError as e:
        return StageResult(
            ok=False, metrics={}, artifacts={},
            failure_reason=f"Scaffold-GS CUDA submodules missing: {e}",
        )

    sparse_dir = Path(pose_artifacts.get("sparse_dir", ""))
    if not (sparse_dir / "cameras.bin").exists():
        return StageResult(False, {}, {}, failure_reason="cameras.bin missing from pose_artifacts")

    # Scaffold-GS expects the source_path to be the *parent* of the sparse/
    # subdirectory. If sparse_dir is already named "sparse", use its parent.
    source_path = sparse_dir.parent if sparse_dir.name == "sparse" else sparse_dir
    # Scaffold-GS also needs images alongside. Symlink frames/ if needed.
    images_path = workdir / "frames"

    out_dir = workdir / f"scaffold_attempt_{attempt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    iters = ITERATIONS if attempt == 1 else int(ITERATIONS * 1.5)

    cmd = [
        sys.executable, str(train_script),
        "--source_path", str(source_path),
        "--model_path", str(out_dir),
        "--images", str(images_path),
        "--iterations", str(iters),
        "--eval",
        "--lod", "0",              # disable LoD during training; we do our own
        "--voxel_size", "0.001",   # fine enough for room-scale scenes
        "--update_until", str(int(iters * 0.8)),
        "--densify_until_iter", str(int(iters * 0.8)),
    ]

    env = {**os.environ, "PYTHONPATH": str(scaffold_dir)}

    log.info("running Scaffold-GS: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(scaffold_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,   # 2-hour hard timeout
        )
    except subprocess.TimeoutExpired:
        return StageResult(False, {}, {}, failure_reason="Scaffold-GS timed out after 2 hours")
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"Scaffold-GS launch error: {e}")

    combined_log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        log.error("Scaffold-GS failed (rc=%d):\n%s", proc.returncode, combined_log[-3000:])
        return StageResult(
            False, {}, {},
            failure_reason=f"Scaffold-GS rc={proc.returncode}: {combined_log[-500:]}",
        )

    ply_path = _find_output_ply(out_dir)
    if ply_path is None:
        return StageResult(False, {}, {}, failure_reason="Scaffold-GS finished but no PLY found")

    final_psnr = _parse_final_psnr(combined_log)
    n_gaussians = 0
    try:
        # Quick Gaussian count from PLY header
        with open(ply_path, "rb") as f:
            header = b""
            while b"end_header" not in header:
                header += f.read(256)
        m = re.search(rb"element vertex (\d+)", header)
        if m:
            n_gaussians = int(m.group(1))
    except Exception:
        pass

    # Write cameras.json compatible with eval stage (empty — eval stage reads
    # the COLMAP binary directly if cameras.json is missing)
    cameras_json = out_dir / "cameras.json"
    if not cameras_json.exists():
        cameras_json.write_text("[]")

    log.info("Scaffold-GS done: %s (%.1f MB, %d Gaussians, PSNR=%.2f)",
             ply_path.name,
             ply_path.stat().st_size / 1024 / 1024,
             n_gaussians,
             final_psnr)

    return StageResult(
        ok=True,
        metrics={
            "iterations": iters,
            "attempt": attempt,
            "n_gaussians": n_gaussians,
            "psnr_train": final_psnr,
            "trainer": "scaffold_gs",
        },
        artifacts={
            "ply_path": str(ply_path),
            "holdout_dir": str(out_dir),
            "cameras_json": str(cameras_json),
        },
    )
