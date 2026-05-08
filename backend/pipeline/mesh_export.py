"""Mesh post-processing: 2DGS raw mesh -> watertight, decimated, floor plan."""
from __future__ import annotations

from pathlib import Path

from app.storage import scan_key

from .types import StageResult


def run(scan_id: str, workdir: Path, mesh_artifacts: dict) -> StageResult:
    # TODO: load mesh_artifacts['mesh_ply']
    # TODO: trimesh.fix_normals + fill_holes; quadric decimation -> ~50k faces
    # TODO: align floor to z=0 (RANSAC plane); compute bounding box -> dimensions
    # TODO: top-down rasterize -> floor_plan.png
    return StageResult(
        ok=True,
        metrics={"mesh_faces": 0, "floor_area_m2": 0.0},
        artifacts={"mesh_key": scan_key(scan_id, "mesh.glb")},
    )
