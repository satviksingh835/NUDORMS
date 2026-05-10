"""PGSR mesh extractor — Stage 4a.

PGSR (Planar-based Gaussian Splatting for Efficient and High-Quality
Radiance Field Rendering, TVCG 2024): planar Gaussians + multi-view
geometric consistency loss. Best Chamfer-distance on textureless indoor;
PSNR 30.41 / SSIM 0.930 / LPIPS 0.161 on MipNeRF360 indoor.

Used here for mesh extraction alongside the primary splat, not as the
primary trainer. PGSR's planar Gaussians naturally tessellate into a
low-poly mesh with consistent normals — ideal for floor-plan generation
and AR placement.

Set NUDORMS_PGSR_DIR to the cloned hmanhng/pgsr directory
(default: /workspace/pgsr).

Output: a .ply or .obj mesh uploaded to R2 under scans/<scan_id>/mesh.ply
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from ..types import StageResult

log = logging.getLogger("nudorms.train.pgsr")

DEFAULT_PGSR_DIR = "/workspace/pgsr"


def available() -> bool:
    d = Path(os.environ.get("NUDORMS_PGSR_DIR", DEFAULT_PGSR_DIR))
    return (d / "train.py").exists()


def _find_mesh(model_path: Path) -> Path | None:
    for ext in ["*.obj", "*.ply", "*.glb"]:
        candidates = list(model_path.rglob(ext))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)
    return None


def run(scan_id: str, workdir: Path, pose_artifacts: dict,
        attempt: int = 1) -> StageResult:
    pgsr_dir = Path(os.environ.get("NUDORMS_PGSR_DIR", DEFAULT_PGSR_DIR))
    train_script = pgsr_dir / "train.py"
    if not train_script.exists():
        return StageResult(
            ok=False, metrics={}, artifacts={},
            failure_reason=f"PGSR not found at {pgsr_dir}. Set NUDORMS_PGSR_DIR.",
        )

    sparse_dir = Path(pose_artifacts.get("sparse_dir", ""))
    if not (sparse_dir / "cameras.bin").exists():
        return StageResult(False, {}, {}, failure_reason="cameras.bin missing")

    source_path = sparse_dir.parent if sparse_dir.name == "sparse" else sparse_dir
    images_path = workdir / "frames"
    out_dir = workdir / f"pgsr_attempt_{attempt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(train_script),
        "--source_path", str(source_path),
        "--model_path", str(out_dir),
        "--images", str(images_path),
        "--iterations", "15000",   # faster run for mesh only; splat quality is secondary
        "--eval",
        "--extract_mesh",          # PGSR flag to export mesh post-training
    ]

    env = {**os.environ, "PYTHONPATH": str(pgsr_dir)}
    log.info("running PGSR mesh extraction: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, cwd=str(pgsr_dir), env=env,
            capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return StageResult(False, {}, {}, failure_reason="PGSR timed out after 1 hour")
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"PGSR launch error: {e}")

    if proc.returncode != 0:
        log.error("PGSR failed (rc=%d):\n%s", proc.returncode,
                  (proc.stdout + proc.stderr)[-2000:])
        return StageResult(
            False, {}, {},
            failure_reason=f"PGSR rc={proc.returncode}: {proc.stderr[-400:]}",
        )

    mesh_path = _find_mesh(out_dir)
    if mesh_path is None:
        return StageResult(False, {}, {}, failure_reason="PGSR finished but no mesh found")

    size_mb = mesh_path.stat().st_size / 1024 / 1024
    log.info("PGSR mesh: %s (%.1f MB)", mesh_path.name, size_mb)

    return StageResult(
        ok=True,
        metrics={"mesh_size_mb": round(size_mb, 2), "mesh_method": "pgsr"},
        artifacts={"mesh_path": str(mesh_path)},
    )
