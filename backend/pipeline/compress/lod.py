"""LoD pyramid for progressive streaming — tiers stream low→high on slow connections.

Primary: `build-lod` CLI from Spark 2.0 (produces .rad LoD tree).
Fallback: importance-sampling from the .splat binary produced by sogs.py,
  written as 3 separate .splat files (preview 10%, standard 40%, hires 100%).

Each tier is uploaded to R2 and its key returned in artifacts["lod_keys"].
The viewer loads the preview tier first (<500 ms), then progressively upgrades.
"""
from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from app.storage import put, scan_key

from ..types import StageResult

log = logging.getLogger("nudorms.pipeline.compress.lod")

# Fraction of total Gaussians in each tier (sorted by importance desc)
TIERS: dict[str, float] = {"preview": 0.10, "standard": 0.40, "hires": 1.00}

MAX_SPLATS_PER_TIER = 1_500_000   # rendering budget cap from plan (Spark LoD handles this)


# ---------------------------------------------------------------------------
# Spark build-lod
# ---------------------------------------------------------------------------

def _try_build_lod(splat_path: Path, out_dir: Path) -> dict[str, Path] | None:
    """Run `build-lod` CLI. Returns {tier_name: path} on success, else None."""
    build_lod = shutil.which("build-lod")
    if build_lod is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [build_lod, str(splat_path), str(out_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.warning("build-lod rc=%d: %s", result.returncode, result.stderr[:300])
            return None
    except Exception as e:
        log.warning("build-lod failed: %s", e)
        return None

    # Spark outputs something like preview.rad, standard.rad, hires.rad
    tier_files: dict[str, Path] = {}
    for tier in TIERS:
        for ext in [".rad", ".splat", f"_{tier}.splat"]:
            candidate = out_dir / f"{tier}{ext}"
            if candidate.exists():
                tier_files[tier] = candidate
                break
    if not tier_files:
        return None
    return tier_files


# ---------------------------------------------------------------------------
# Fallback: importance-sample the .splat binary
# ---------------------------------------------------------------------------

def _sample_splat(splat_path: Path, fraction: float, out_path: Path) -> None:
    """Read a .splat binary and write the top-fraction% by opacity*scale."""
    raw = splat_path.read_bytes()
    N = len(raw) // 32
    if N == 0:
        out_path.write_bytes(b"")
        return

    # Each Gaussian: 32 bytes. Byte 27 is the alpha (opacity * 255).
    # Gaussians are already sorted by importance (sogs.py does this), so we
    # just take the first k rows.
    k = min(max(1, int(N * fraction)), MAX_SPLATS_PER_TIER)
    out_path.write_bytes(raw[:k * 32])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scan_id: str, workdir: Path, sogs_artifacts: dict) -> StageResult:
    splat_local = sogs_artifacts.get("splat_local", "")
    splat_path = Path(splat_local) if splat_local else None

    if splat_path is None or not splat_path.exists():
        # Nothing to build LoD from; create placeholder keys pointing at the
        # same splat_key for all tiers (viewer can still load it).
        base_key = sogs_artifacts.get("splat_key", "")
        lod_keys = {tier: base_key for tier in TIERS}
        return StageResult(
            ok=True,
            metrics={"lod_tiers": list(TIERS.keys()), "lod_method": "passthrough"},
            artifacts={"lod_keys": lod_keys},
        )

    lod_dir = workdir / "lod"
    tier_paths: dict[str, Path] | None = None

    # Only try build-lod on .splat files (not .sogs — format mismatch)
    if splat_path.suffix == ".splat":
        tier_paths = _try_build_lod(splat_path, lod_dir)

    if tier_paths is None:
        # Importance-sampling fallback
        log.info("build-lod unavailable — importance-sampling %d tiers", len(TIERS))
        lod_dir.mkdir(parents=True, exist_ok=True)
        tier_paths = {}
        for tier, fraction in TIERS.items():
            out = lod_dir / f"{tier}.splat"
            _sample_splat(splat_path, fraction, out)
            tier_paths[tier] = out
        method = "importance_sample"
    else:
        method = "build_lod"

    lod_keys: dict[str, str] = {}
    fmt = sogs_artifacts.get("splat_format", "splat")
    for tier, path in tier_paths.items():
        if not path.exists():
            continue
        key = scan_key(scan_id, f"{tier}.{path.suffix.lstrip('.')}")
        with open(path, "rb") as fh:
            put(key, fh.read(), content_type="application/octet-stream")
        lod_keys[tier] = key
        size_mb = path.stat().st_size / 1024 / 1024
        log.info("  LoD %s: %.1f MB → %s", tier, size_mb, key)

    return StageResult(
        ok=True,
        metrics={"lod_tiers": list(lod_keys.keys()), "lod_method": method},
        artifacts={"lod_keys": lod_keys},
    )
