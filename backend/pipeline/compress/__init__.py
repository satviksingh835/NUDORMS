"""Top-level compress() — runs SOGS then builds the LOD pyramid."""
from __future__ import annotations

from pathlib import Path

from ..orchestrator import StageResult
from . import lod, sogs


def run(scan_id: str, workdir: Path, train_artifacts: dict) -> StageResult:
    sogs_result = sogs.run(scan_id, workdir, train_artifacts)
    if not sogs_result.ok:
        return sogs_result

    lod_result = lod.run(scan_id, workdir, sogs_result.artifacts)
    if not lod_result.ok:
        return lod_result

    return StageResult(
        ok=True,
        metrics={**sogs_result.metrics, **lod_result.metrics},
        artifacts={
            "splat_key": sogs_result.artifacts["splat_key"],
            "lod_keys": lod_result.artifacts["lod_keys"],
        },
    )
