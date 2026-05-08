"""2D Gaussian Splatting for mesh extraction (parallel to 3DGS).

The 3DGS model is what users see; 2DGS gives us a watertight mesh useful
for floor plans, dimension estimates, and walk-mode collision.
"""
from __future__ import annotations

from pathlib import Path

from ..orchestrator import StageResult


def run(scan_id: str, workdir: Path, pose_artifacts: dict) -> StageResult:
    out = workdir / "twodgs"
    out.mkdir(parents=True, exist_ok=True)
    # TODO: train 2DGS on same pose dataset
    # TODO: export tsdf-fused mesh to out/mesh.ply
    return StageResult(
        ok=True,
        metrics={},
        artifacts={"mesh_ply": str(out / "mesh.ply")},
    )
