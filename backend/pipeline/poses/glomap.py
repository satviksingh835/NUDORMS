"""GLOMAP pose estimation (default first try).

GLOMAP shares COLMAP's feature DB + matcher, then does global SfM on top.
For a video walkthrough we use sequential matching (much faster than
exhaustive) since consecutive frames are inherently close pairs.

Pipeline:
  1. colmap feature_extractor   → SIFT features into a SQLite db
  2. colmap sequential_matcher  → match each frame against its neighbors
  3. glomap mapper              → global SfM, writes sparse/0/

Quality is judged from the resulting sparse model:
  - registered_images / total = how many frames got a pose
  - mean track reprojection error = how well the poses fit the features
  - mean track length             = how stable the matches are
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from ..types import StageResult

log = logging.getLogger("nudorms.poses.glomap")

SEQUENTIAL_MATCHER_OVERLAP = 15   # match each frame with its 15 neighbors


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.info("running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _read_sparse_metrics(sparse_dir: Path, total_frames: int) -> dict:
    """Parse sparse/0/{cameras,images,points3D}.bin via pycolmap."""
    try:
        import pycolmap
    except ImportError:
        # Without pycolmap we can still confirm the model exists, but no quality metrics.
        return {"registered_images": -1, "reproj_error": -1, "mean_track_length": -1,
                "inlier_ratio": 0.5, "_no_pycolmap": True}

    rec = pycolmap.Reconstruction(str(sparse_dir))
    n_images = rec.num_reg_images()
    n_points = rec.num_points3D()
    if n_points == 0:
        return {"registered_images": n_images, "reproj_error": 1e9,
                "mean_track_length": 0, "inlier_ratio": 0.0}

    track_lengths = [len(p.track.elements) for p in rec.points3D.values()]
    errors = [p.error for p in rec.points3D.values()]
    return {
        "registered_images": int(n_images),
        "reproj_error": float(sum(errors) / len(errors)),
        "mean_track_length": float(sum(track_lengths) / len(track_lengths)),
        "n_points3D": int(n_points),
        # Inlier ratio: COLMAP only keeps inlier matches in points3D, so use
        # registered_images / total as a stand-in. ensemble.py treats this
        # along with reproj_error.
        "inlier_ratio": float(n_images / max(total_frames, 1)),
    }


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    """Run COLMAP feature/match + GLOMAP mapper. frames_dir holds 0001.jpg..."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "database.db"
    sparse = out_dir / "sparse"
    sparse.mkdir(exist_ok=True)

    if not _have("colmap"):
        return StageResult(False, {}, {}, failure_reason="colmap binary not on PATH")
    if not _have("glomap"):
        return StageResult(False, {}, {}, failure_reason="glomap binary not on PATH")

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
            "--SiftExtraction.use_gpu", "1",
        ])
        _run([
            "colmap", "sequential_matcher",
            "--database_path", str(db),
            "--SequentialMatching.overlap", str(SEQUENTIAL_MATCHER_OVERLAP),
            "--SequentialMatching.quadratic_overlap", "1",
            "--SiftMatching.use_gpu", "1",
        ])
        _run([
            "glomap", "mapper",
            "--database_path", str(db),
            "--image_path", str(frames_dir),
            "--output_path", str(sparse),
        ])
    except subprocess.CalledProcessError as e:
        return StageResult(False, {}, {},
                           failure_reason=f"{e.cmd[1]} failed: {e.stderr[-500:]}")

    # GLOMAP writes sparse/0/{cameras,images,points3D}.bin.
    sparse_0 = sparse / "0"
    if not (sparse_0 / "cameras.bin").exists():
        return StageResult(False, {}, {},
                           failure_reason="glomap produced no sparse model")

    metrics = _read_sparse_metrics(sparse_0, total_frames)
    return StageResult(
        ok=True,
        metrics=metrics,
        artifacts={"sparse_dir": str(sparse_0), "database": str(db)},
    )
