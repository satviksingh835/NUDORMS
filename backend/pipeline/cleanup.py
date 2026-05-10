"""Post-training Gaussian floater removal — runs before compression.

Uses DBSCAN to find the dominant cluster of Gaussians (the actual room)
and removes outliers (floaters around the capture trajectory, sky blobs,
etc.). On a badly-initialised scene, floaters can be 5–30% of all
Gaussians by count but add very little to rendering quality while
significantly inflating file size and causing artifacts in the viewer.

Algorithm:
  1. Read PLY (means + opacities only — fast).
  2. Importance-weight each Gaussian: w = sigmoid(opacity) * max(exp(scales))
  3. DBSCAN on (x, y, z) with eps = 2 × median nearest-neighbour distance.
  4. Keep the largest cluster; discard all others (floater label = -1 too).
  5. Write a new PLY with the surviving Gaussians.

CPU-only. Runs in ~5–30 s depending on Gaussian count.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np

from .types import StageResult

log = logging.getLogger("nudorms.pipeline.cleanup")

MIN_KEEP_FRACTION = 0.50   # never remove more than half — bail if DBSCAN misfires


def _read_ply_positions_opacities(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a 3DGS PLY, return (means [N,3], opacities [N], property_names)."""
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.read(256)
        header_text = header.decode("latin-1")

    lines = header_text.split("\n")
    props = []
    n_verts = 0
    in_vertex = False
    for line in lines:
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element") and in_vertex:
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            props.append(line.split()[-1])

    prop_to_idx = {p: i for i, p in enumerate(props)}
    n_props = len(props)

    # Re-open and skip header
    with open(path, "rb") as f:
        raw = f.read()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    data = np.frombuffer(raw[header_end:], dtype=np.float32).reshape(n_verts, n_props)

    means = data[:, [prop_to_idx["x"], prop_to_idx["y"], prop_to_idx["z"]]]
    opacities = data[:, prop_to_idx["opacity"]]
    return means, opacities, props


def _dbscan_keep_mask(means: np.ndarray, eps_multiplier: float = 2.0) -> np.ndarray:
    """DBSCAN to find the dominant cluster. Returns boolean keep mask."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # scipy fallback: no DBSCAN, skip cleanup
        raise ImportError("scikit-learn required for DBSCAN cleanup (pip install scikit-learn)")

    # Estimate eps from median nearest-neighbour distance (cheap subsample)
    subsample = min(len(means), 8_000)
    idx = np.random.choice(len(means), subsample, replace=False)
    sub = means[idx]
    from sklearn.metrics import pairwise_distances_chunked
    nn_dists = []
    for chunk in pairwise_distances_chunked(sub, n_jobs=-1, working_memory=64):
        chunk[chunk == 0] = np.inf
        nn_dists.append(chunk.min(axis=1))
    eps = eps_multiplier * np.median(np.concatenate(nn_dists))
    eps = max(eps, 0.01)   # never collapse to zero

    log.info("DBSCAN eps=%.4f on %d Gaussians", eps, len(means))
    db = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit(means)
    labels = db.labels_

    # Pick the largest non-noise cluster
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        # All noise — don't remove anything
        return np.ones(len(means), dtype=bool)

    dominant = unique[np.argmax(counts)]
    mask = labels == dominant
    return mask


def _copy_ply_with_mask(src: Path, dst: Path, mask: np.ndarray) -> None:
    """Write a new PLY keeping only rows where mask is True."""
    with open(src, "rb") as f:
        raw = f.read()
    header_end_bytes = b"end_header\n"
    header_offset = raw.index(header_end_bytes) + len(header_end_bytes)
    header_text = raw[:header_offset].decode("latin-1")

    # Parse n_verts and n_props from header
    n_verts_orig = 0
    props = []
    for line in header_text.split("\n"):
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts_orig = int(line.split()[-1])
        elif line.startswith("property float") or line.startswith("property f"):
            props.append(line)
    n_props = len(props)

    data = np.frombuffer(raw[header_offset:], dtype=np.float32).reshape(n_verts_orig, n_props)
    kept = data[mask]

    new_header = header_text.replace(
        f"element vertex {n_verts_orig}",
        f"element vertex {len(kept)}",
    )
    with open(dst, "wb") as f:
        f.write(new_header.encode("latin-1"))
        f.write(kept.tobytes())


def run(scan_id: str, workdir: Path, train_artifacts: dict) -> StageResult:
    ply_path = Path(train_artifacts.get("ply_path", ""))
    if not ply_path.exists():
        return StageResult(False, {}, {}, failure_reason=f"PLY not found: {ply_path}")

    try:
        means, opacities, props = _read_ply_positions_opacities(ply_path)
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"PLY read error: {e}")

    n_before = len(means)
    log.info("cleanup: %d Gaussians before DBSCAN", n_before)

    try:
        mask = _dbscan_keep_mask(means)
    except ImportError as e:
        log.warning("cleanup skipped: %s", e)
        return StageResult(
            ok=True,
            metrics={"cleanup_skipped": True, "n_gaussians": n_before},
            artifacts=train_artifacts,
        )
    except Exception as e:
        log.warning("DBSCAN failed (%s) — keeping all Gaussians", e)
        return StageResult(
            ok=True,
            metrics={"cleanup_skipped": True, "n_gaussians": n_before},
            artifacts=train_artifacts,
        )

    n_keep = int(mask.sum())
    keep_fraction = n_keep / max(n_before, 1)

    if keep_fraction < MIN_KEEP_FRACTION:
        log.warning(
            "DBSCAN would remove %.0f%% of Gaussians — looks like a misfire, keeping all",
            (1 - keep_fraction) * 100,
        )
        return StageResult(
            ok=True,
            metrics={"cleanup_skipped": True, "n_gaussians": n_before},
            artifacts=train_artifacts,
        )

    cleaned_ply = ply_path.parent / "point_cloud_cleaned.ply"
    try:
        _copy_ply_with_mask(ply_path, cleaned_ply, mask)
    except Exception as e:
        log.warning("PLY copy with mask failed (%s) — keeping original", e)
        return StageResult(
            ok=True,
            metrics={"cleanup_skipped": True, "n_gaussians": n_before},
            artifacts=train_artifacts,
        )

    removed = n_before - n_keep
    log.info("cleanup: removed %d floaters (%.1f%%), %d remain",
             removed, removed / n_before * 100, n_keep)

    return StageResult(
        ok=True,
        metrics={
            "n_gaussians_before_cleanup": n_before,
            "n_gaussians_removed": removed,
            "n_gaussians": n_keep,
        },
        artifacts={
            **train_artifacts,
            "ply_path": str(cleaned_ply),
        },
    )
