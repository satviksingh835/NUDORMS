"""COLMAP incremental SfM (fallback when GLOMAP underperforms).

Slower than GLOMAP, sometimes more robust on hard scenes (lots of repetition,
heavy textureless regions). Same feature DB + matcher, different mapper.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from ..types import StageResult
from .glomap import SEQUENTIAL_MATCHER_OVERLAP, USE_GPU_SIFT, _read_sparse_metrics

log = logging.getLogger("nudorms.poses.colmap")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.info("running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "database.db"
    sparse = out_dir / "sparse"
    sparse.mkdir(exist_ok=True)

    if not shutil.which("colmap"):
        return StageResult(False, {}, {}, failure_reason="colmap binary not on PATH")

    total_frames = sum(1 for _ in frames_dir.glob("*.jpg"))
    if total_frames == 0:
        return StageResult(False, {}, {}, failure_reason="no frames in frames_dir")

    try:
        _run([
            "colmap", "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(frames_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV",
            "--FeatureExtraction.use_gpu", "1" if USE_GPU_SIFT else "0",
        ])
        _run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db),
            "--FeatureMatching.use_gpu", "1" if USE_GPU_SIFT else "0",
        ])
        _run([
            "colmap", "mapper",
            "--database_path", str(db),
            "--image_path", str(frames_dir),
            "--output_path", str(sparse),
        ])
    except subprocess.CalledProcessError as e:
        return StageResult(False, {}, {},
                           failure_reason=f"{e.cmd[1]} failed: {e.stderr[-500:]}")

    sparse_0 = sparse / "0"
    if not (sparse_0 / "cameras.bin").exists():
        return StageResult(False, {}, {},
                           failure_reason="colmap mapper produced no sparse model")

    metrics = _read_sparse_metrics(sparse_0, total_frames)
    return StageResult(
        ok=True,
        metrics=metrics,
        artifacts={"sparse_dir": str(sparse_0), "database": str(db)},
    )
