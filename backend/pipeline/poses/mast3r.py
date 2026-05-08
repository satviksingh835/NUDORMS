"""MASt3R-SfM: learned pose estimation that often saves textureless captures.

Wins where classical SfM fails (white walls, repetitive patterns).
"""
from __future__ import annotations

from pathlib import Path

from ..types import StageResult


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    # TODO: load MASt3R weights, run pairwise inference + global alignment,
    #       export to a COLMAP-compatible sparse model in out_dir/sparse/0
    return StageResult(
        ok=False,
        failure_reason="mast3r stub — not yet implemented",
        metrics={},
        artifacts={},
    )
    # Placeholder to satisfy older shape; never reached.
    return StageResult(  # noqa: B019
        ok=True,
        metrics={"inlier_ratio": 0.0, "reproj_error": 0.0},
        artifacts={"sparse_dir": str(out_dir / "sparse" / "0")},
    )
