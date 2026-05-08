"""Level-of-detail pyramid for progressive streaming.

Three tiers (preview / standard / hires). Viewer loads preview first
(<500ms), then sharpens. Tiers are produced by importance-sampling
the trained splat down to fractional gaussian counts.
"""
from __future__ import annotations

from pathlib import Path

from app.storage import scan_key

from ..types import StageResult

TIERS = {"preview": 0.10, "standard": 0.40, "hires": 1.00}


def run(scan_id: str, workdir: Path, sogs_artifacts: dict) -> StageResult:
    lod_keys = {}
    # TODO: for each (name, fraction) in TIERS:
    #   - importance-sample fraction * num_gaussians from the trained splat
    #   - SOGS encode -> workdir / f'{name}.splat'
    #   - upload, record under lod_keys[name]
    for name in TIERS:
        lod_keys[name] = scan_key(scan_id, f"{name}.splat")

    return StageResult(
        ok=True,
        metrics={"lod_tiers": list(TIERS.keys())},
        artifacts={"lod_keys": lod_keys},
    )
