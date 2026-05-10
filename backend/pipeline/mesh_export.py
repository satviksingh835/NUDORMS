"""Mesh post-processing: 2DGS / PGSR raw mesh -> watertight, decimated, floor plan.

Currently a stub — only invoked when the upstream mesh stage (PGSR or 2DGS)
returns a real mesh artifact. Returns ok=False so the orchestrator records
mesh_key=None instead of pointing at non-existent R2 keys. When implemented,
should: trimesh.fix_normals + fill_holes → quadric decimate to ~50k faces →
RANSAC floor plane to align z=0 → top-down rasterize → floor_plan.png.
"""
from __future__ import annotations

from pathlib import Path

from .types import StageResult


def run(scan_id: str, workdir: Path, mesh_artifacts: dict) -> StageResult:
    return StageResult(
        ok=False,
        metrics={},
        artifacts={},
        failure_reason="mesh_export not implemented yet",
    )
