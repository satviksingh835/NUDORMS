"""Held-out evaluation: PSNR / SSIM / LPIPS, plus per-region diagnostics.

Per-region diagnostics let us tell the user *which part* of the room came
out badly (e.g. 'low texture on west wall'), so a recapture is targeted.
"""
from __future__ import annotations

from pathlib import Path

from ..orchestrator import StageResult


def run(scan_id: str, workdir: Path, train_artifacts: dict, pose_artifacts: dict) -> StageResult:
    # TODO: render the splat at every holdout pose
    # TODO: compute PSNR / SSIM / LPIPS vs. the held-out frame
    # TODO: split each rendered/gt image into a 4x4 grid; report which cells
    #       have the worst PSNR -> these become 'region diagnostics'
    return StageResult(
        ok=True,
        metrics={"psnr": 30.0, "ssim": 0.92, "lpips": 0.12},  # placeholder good values
        artifacts={"region_diagnostics": {}},
    )
