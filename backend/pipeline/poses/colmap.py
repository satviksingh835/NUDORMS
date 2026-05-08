"""COLMAP incremental SfM (fallback)."""
from __future__ import annotations

from pathlib import Path

from ..orchestrator import StageResult


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    """Battle-tested fallback when GLOMAP fails. Slower, sometimes more robust."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # TODO: colmap automatic_reconstructor --image_path frames_dir --workspace_path out_dir
    return StageResult(
        ok=True,
        metrics={"inlier_ratio": 0.0, "reproj_error": 0.0},
        artifacts={"sparse_dir": str(out_dir / "sparse" / "0")},
    )
