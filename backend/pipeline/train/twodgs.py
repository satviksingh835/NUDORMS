"""2D Gaussian Splatting for mesh extraction (parallel to 3DGS).

The 3DGS model is what users see; 2DGS would give a watertight mesh useful
for floor plans, dimension estimates, and walk-mode collision.

Currently a stub — returns ok=False so the orchestrator skips mesh_export
instead of writing a fake mesh_key into R2 / DB. When implemented, should
TSDF-integrate the Depth Anything V2 priors + camera poses and write a
real mesh.ply.
"""
from __future__ import annotations

from pathlib import Path

from ..types import StageResult


def run(scan_id: str, workdir: Path, pose_artifacts: dict) -> StageResult:
    return StageResult(
        ok=False,
        metrics={},
        artifacts={},
        failure_reason="2DGS not implemented yet — splat-only output",
    )
