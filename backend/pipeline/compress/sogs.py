"""SOGS — Self-Organizing Gaussians compression.

Brings hundreds-of-MB raw splats down to ~10-30 MB with minimal quality loss.
"""
from __future__ import annotations

from pathlib import Path

from app.storage import put, scan_key

from ..types import StageResult


def run(scan_id: str, workdir: Path, train_artifacts: dict) -> StageResult:
    out = workdir / "compressed.splat"
    # TODO: load train_artifacts['ply_path']
    # TODO: SOGS encode -> out (single .splat file)
    # TODO: also export an uncompressed .ply for archival

    # Placeholder upload (file won't exist until SOGS is wired):
    splat_key = scan_key(scan_id, "scene.splat")
    if out.exists():
        with out.open("rb") as fh:
            put(splat_key, fh, content_type="application/octet-stream")

    return StageResult(
        ok=True,
        metrics={"splat_size_mb": 0},
        artifacts={"splat_key": splat_key, "splat_local": str(out)},
    )
