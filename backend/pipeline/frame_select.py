"""Greedy diverse frame selection.

Naive 'every Nth frame' loses quality. We score every frame and pick a subset
that is both individually high quality AND covers diverse viewpoints.
"""
from __future__ import annotations

from pathlib import Path

from .orchestrator import StageResult

TARGET_FRAMES = 300
MIN_FRAMES = 120


def run(scan_id: str, workdir: Path) -> StageResult:
    """Decode raw video, score frames, pick diverse high-quality subset."""
    # TODO: ffmpeg full-decode to workdir/all_frames/
    # TODO: per-frame score = sharpness * exposure_sanity * (1 - motion_blur)
    # TODO: feature-flow proxy between consecutive frames -> running parallax estimate
    # TODO: greedy: while |selected| < TARGET_FRAMES:
    #         pick argmax_i (score_i + lambda * min_dist_to_selected_i)
    # TODO: write selected to workdir/frames/, return paths in artifacts
    return StageResult(
        ok=True,
        metrics={"selected_frames": 0},
        artifacts={"frames_dir": str(workdir / "frames")},
    )
