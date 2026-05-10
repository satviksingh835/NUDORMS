"""Difix3D+ diffusion artifact fixer — Stage 3a.

Difix3D+ (nv-tlabs/Difix3D, CVPR 2025 Oral + Best Paper Finalist):
  - Works on both NeRF and 3DGS outputs
  - ~2× FID improvement; 3D-consistent (no temporal flicker)
  - Single-step diffusion cleans rendered novel views, then distills
    cleaned views back into the Gaussians via fine-tuning
  - ~3–10 minutes per scene on an L40S / A100

Runs as an optional post-training polish step AFTER training passes
the PSNR gate. If Difix3D is not installed (NUDORMS_DIFIX3D_DIR not
set, or repo missing), this stage is silently skipped and the
unrefined PLY is used.

Set NUDORMS_DIFIX3D_DIR to the cloned nv-tlabs/Difix3D directory
(default: /workspace/Difix3D).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from ..types import StageResult

log = logging.getLogger("nudorms.train.difix3d")

DEFAULT_DIFIX3D_DIR = "/workspace/Difix3D"


def available() -> bool:
    d = Path(os.environ.get("NUDORMS_DIFIX3D_DIR", DEFAULT_DIFIX3D_DIR))
    return (d / "difix3d" / "infer.py").exists() or (d / "infer.py").exists()


def _find_script(difix_dir: Path) -> Path | None:
    for candidate in [
        difix_dir / "infer.py",
        difix_dir / "difix3d" / "infer.py",
        difix_dir / "scripts" / "infer.py",
        difix_dir / "run.py",
    ]:
        if candidate.exists():
            return candidate
    return None


def run(
    scan_id: str,
    workdir: Path,
    train_artifacts: dict,
    pose_artifacts: dict,
) -> StageResult:
    difix_dir = Path(os.environ.get("NUDORMS_DIFIX3D_DIR", DEFAULT_DIFIX3D_DIR))
    script = _find_script(difix_dir)
    if script is None:
        return StageResult(
            ok=False, metrics={}, artifacts={},
            failure_reason=(
                f"Difix3D not found at {difix_dir}. "
                "Set NUDORMS_DIFIX3D_DIR or run runpod_bootstrap.sh."
            ),
        )

    input_ply = Path(train_artifacts.get("ply_path", ""))
    if not input_ply.exists():
        return StageResult(False, {}, {}, failure_reason=f"input PLY not found: {input_ply}")

    sparse_dir = Path(pose_artifacts.get("sparse_dir", ""))
    frames_dir = workdir / "frames"
    out_dir = workdir / "difix3d_refined"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--input_path", str(input_ply),
        "--output_dir", str(out_dir),
        "--images_dir", str(frames_dir),
    ]
    if sparse_dir.exists():
        cmd += ["--colmap_dir", str(sparse_dir)]

    # Check for cameras.json from training stage
    cameras_json = Path(train_artifacts.get("cameras_json", ""))
    if cameras_json.exists():
        cmd += ["--cameras_path", str(cameras_json)]

    env = {**os.environ, "PYTHONPATH": str(difix_dir)}

    log.info("running Difix3D+: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(difix_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,   # 1-hour hard timeout
        )
    except subprocess.TimeoutExpired:
        return StageResult(False, {}, {}, failure_reason="Difix3D timed out after 1 hour")
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"Difix3D launch error: {e}")

    combined_log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        log.error("Difix3D failed (rc=%d):\n%s", proc.returncode, combined_log[-3000:])
        return StageResult(
            False, {}, {},
            failure_reason=f"Difix3D rc={proc.returncode}: {combined_log[-400:]}",
        )

    # Find refined PLY — Difix3D typically writes point_cloud.ply or refined.ply
    refined_ply = None
    for name in ["point_cloud.ply", "refined.ply", "output.ply"]:
        candidate = out_dir / name
        if candidate.exists():
            refined_ply = candidate
            break
    if refined_ply is None:
        candidates = list(out_dir.rglob("*.ply"))
        if candidates:
            refined_ply = max(candidates, key=lambda p: p.stat().st_mtime)

    if refined_ply is None:
        return StageResult(False, {}, {}, failure_reason="Difix3D finished but no refined PLY found")

    n_gaussians = 0
    try:
        with open(refined_ply, "rb") as f:
            header = b""
            while b"end_header" not in header:
                header += f.read(256)
        m = re.search(rb"element vertex (\d+)", header)
        if m:
            n_gaussians = int(m.group(1))
    except Exception:
        pass

    size_mb = refined_ply.stat().st_size / 1024 / 1024
    log.info("Difix3D+ done: %s (%.1f MB, %d Gaussians)", refined_ply.name, size_mb, n_gaussians)

    return StageResult(
        ok=True,
        metrics={"n_gaussians_refined": n_gaussians, "refiner": "difix3d+"},
        artifacts={
            **train_artifacts,              # carry through cameras_json etc.
            "ply_path": str(refined_ply),   # override with refined PLY
        },
    )
