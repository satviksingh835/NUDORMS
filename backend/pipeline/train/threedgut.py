"""3DGUT / 3DGRUT trainer wrapper — Stage 4b.

3DGUT (3D Gaussian Unstructured Textures, CVPR 2025, NVIDIA):
  - Handles distorted/wide-angle phone cameras via a general camera model
  - Proper ray-traced reflections on monitors, windows, polished floors
  - Eliminates the "smeared reflection" artifact common on student monitors
  - Compatible with standard COLMAP sparse models

3DGRUT = 3DGUT with Relightable Unstructured Textures (adds environment
map relighting). Used here primarily for its camera model and reflections.

Set NUDORMS_3DGUT_DIR to the cloned nv-tlabs/3DGUT repo
(default: /workspace/3DGUT).

Falls back silently — in the training ensemble this is tried after
Scaffold-GS/gsplat MCMC if PSNR < PSNR_PASS on the first pass.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from ..types import StageResult

log = logging.getLogger("nudorms.train.threedgut")

DEFAULT_3DGUT_DIR = "/workspace/3DGUT"


def available() -> bool:
    """3DGUT/3DGRUT depends on pytorch3d, which doesn't build cleanly on
    this pod's torch/CUDA combo. Verify pytorch3d is importable before
    advertising the stage; otherwise the orchestrator skips it and falls
    back to gsplat MCMC."""
    d = Path(os.environ.get("NUDORMS_3DGUT_DIR", DEFAULT_3DGUT_DIR))
    if not ((d / "train.py").exists() or (d / "threedgut" / "train.py").exists()):
        return False
    try:
        import pytorch3d  # noqa: F401
        return True
    except ImportError:
        return False


def _find_script(d: Path) -> Path | None:
    for candidate in [d / "train.py", d / "threedgut" / "train.py",
                      d / "scripts" / "train.py"]:
        if candidate.exists():
            return candidate
    return None


def _find_ply(model_path: Path) -> Path | None:
    pc_dir = model_path / "point_cloud"
    if pc_dir.exists():
        iters = sorted(
            pc_dir.iterdir(),
            key=lambda p: int(p.name.split("_")[-1]) if p.name.startswith("iteration_") else -1,
        )
        for d in reversed(iters):
            ply = d / "point_cloud.ply"
            if ply.exists():
                return ply
    candidates = list(model_path.rglob("*.ply"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _parse_psnr(log_text: str) -> float:
    matches = re.findall(r"PSNR\s*[:=]\s*([\d.]+)", log_text)
    return float(matches[-1]) if matches else 0.0


def run(scan_id: str, workdir: Path, pose_artifacts: dict,
        prior_artifacts: dict | None = None, attempt: int = 1) -> StageResult:
    gut_dir = Path(os.environ.get("NUDORMS_3DGUT_DIR", DEFAULT_3DGUT_DIR))
    script = _find_script(gut_dir)
    if script is None:
        return StageResult(
            ok=False, metrics={}, artifacts={},
            failure_reason=f"3DGUT not found at {gut_dir}. Set NUDORMS_3DGUT_DIR.",
        )

    sparse_dir = Path(pose_artifacts.get("sparse_dir", ""))
    if not (sparse_dir / "cameras.bin").exists():
        return StageResult(False, {}, {}, failure_reason="cameras.bin missing")

    source_path = sparse_dir.parent if sparse_dir.name == "sparse" else sparse_dir
    images_path = workdir / "frames"
    out_dir = workdir / f"3dgut_attempt_{attempt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    iters = 30_000 if attempt == 1 else 50_000

    cmd = [
        sys.executable, str(script),
        "--source_path", str(source_path),
        "--model_path", str(out_dir),
        "--images", str(images_path),
        "--iterations", str(iters),
        "--eval",
        # 3DGUT-specific: use the general (undistorted-compatible) camera model
        "--camera_model", "OPENCV",
    ]

    env = {**os.environ, "PYTHONPATH": str(gut_dir)}
    log.info("running 3DGUT: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, cwd=str(gut_dir), env=env,
            capture_output=True, text=True, timeout=7200,
        )
    except subprocess.TimeoutExpired:
        return StageResult(False, {}, {}, failure_reason="3DGUT timed out after 2 hours")
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"3DGUT launch error: {e}")

    combined_log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        log.error("3DGUT failed (rc=%d):\n%s", proc.returncode, combined_log[-2000:])
        return StageResult(
            False, {}, {},
            failure_reason=f"3DGUT rc={proc.returncode}: {proc.stderr[-400:]}",
        )

    ply_path = _find_ply(out_dir)
    if ply_path is None:
        return StageResult(False, {}, {}, failure_reason="3DGUT finished but no PLY found")

    psnr = _parse_psnr(combined_log)
    cameras_json = out_dir / "cameras.json"
    if not cameras_json.exists():
        cameras_json.write_text("[]")

    log.info("3DGUT done: %s (PSNR=%.2f)", ply_path.name, psnr)

    return StageResult(
        ok=True,
        metrics={"iterations": iters, "attempt": attempt, "psnr_train": psnr,
                 "trainer": "3dgut"},
        artifacts={
            "ply_path": str(ply_path),
            "holdout_dir": str(out_dir),
            "cameras_json": str(cameras_json),
        },
    )
