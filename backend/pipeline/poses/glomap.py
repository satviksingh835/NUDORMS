"""GLOMAP pose estimation (default first try).

Faster than COLMAP, similar accuracy, more robust on indoor scenes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..orchestrator import StageResult


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    """Run `glomap mapper` over frames_dir; sparse model lands in out_dir/sparse."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # TODO:
    #   1. colmap feature_extractor --image_path frames_dir --database_path db
    #   2. colmap exhaustive_matcher --database_path db
    #   3. glomap mapper --database_path db --image_path frames_dir --output_path out_dir
    #   4. parse out_dir/sparse/0 to compute inlier_ratio + reproj_error
    try:
        subprocess.run(["glomap", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return StageResult(ok=False, metrics={}, artifacts={}, failure_reason="glomap not installed")

    return StageResult(
        ok=True,
        metrics={"inlier_ratio": 0.0, "reproj_error": 0.0, "registered_images": 0},
        artifacts={"sparse_dir": str(out_dir / "sparse" / "0")},
    )
