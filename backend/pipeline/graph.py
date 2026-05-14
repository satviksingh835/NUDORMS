"""Graph build stage.

Given stitched panoramas + COLMAP poses, this stage:
  1. Computes the 3D camera center for each node (centroid of its frames' centers).
  2. Builds edges:
     - Sequential: node[i] ↔ node[i+1] (the walking order from capture).
     - Proximity: any pair within PROXIMITY_M metres (handles loop closures).
  3. Computes the arrow direction (azimuth) for each directed edge A→B:
     the yaw angle of (center_B − center_A) relative to A's camera forward,
     measured clockwise in the horizontal plane, in degrees [0, 360).
  4. Uploads each pano JPEG to R2 under scans/{scan_id}/panos/{node_id}.jpg.
  5. Uploads graph.json to R2 under scans/{scan_id}/graph.json.

Returns StageResult with artifacts:
  {"graph_key": "...", "pano_keys": {"n0": "...", ...}}
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
from pathlib import Path

import numpy as np

from .types import StageResult

log = logging.getLogger("nudorms.graph")

# Env-tunable: nodes within this distance (metres) get a proximity edge.
PROXIMITY_M = float(os.environ.get("NUDORMS_PROXIMITY_M", "2.5"))


def _read_images_bin(sparse_dir: Path) -> dict[str, dict]:
    """Read COLMAP images.bin → {name: {qvec, tvec}}."""
    images_bin = Path(sparse_dir) / "images.bin"
    images: dict[str, dict] = {}
    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            struct.unpack("<I", f.read(4))[0]   # image_id
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            f.read(4)   # cam_id
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode()
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * 24)  # skip track entries
            images[name] = {"qvec": qvec, "tvec": tvec}
    return images


def _qvec_to_rotation(qvec) -> np.ndarray:
    """COLMAP (w,x,y,z) → 3×3 rotation matrix (world→camera)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _camera_center(qvec, tvec) -> np.ndarray:
    """World-space camera center: C = -R^T @ t."""
    R = _qvec_to_rotation(qvec)
    t = np.array(tvec)
    return -(R.T @ t)


def _camera_forward(qvec) -> np.ndarray:
    """Camera forward (+Z) in world space: R^T @ [0,0,1]."""
    R = _qvec_to_rotation(qvec)
    return R.T @ np.array([0.0, 0.0, 1.0])


def _azimuth_deg(direction: np.ndarray, reference_forward: np.ndarray) -> float:
    """Clockwise angle (degrees) of `direction` relative to `reference_forward`,
    measured in the horizontal (XZ) plane. Result is in [0, 360)."""
    # Project both vectors onto the horizontal plane
    d = np.array([direction[0], 0.0, direction[2]])
    fwd = np.array([reference_forward[0], 0.0, reference_forward[2]])
    norm_d = np.linalg.norm(d)
    norm_fwd = np.linalg.norm(fwd)
    if norm_d < 1e-9 or norm_fwd < 1e-9:
        return 0.0
    d = d / norm_d
    fwd = fwd / norm_fwd
    # right = fwd rotated 90° clockwise around Y
    right = np.array([fwd[2], 0.0, -fwd[0]])
    forward_comp = float(np.dot(d, fwd))
    right_comp = float(np.dot(d, right))
    azimuth = math.degrees(math.atan2(right_comp, forward_comp))
    return azimuth % 360.0


def run(
    scan_id: str,
    workdir: Path,
    stitch_artifacts: dict,
    poses_artifacts: dict,
) -> StageResult:
    from app.storage import put, scan_key

    sparse_dir = Path(poses_artifacts["sparse_dir"])
    pano_paths: dict[str, str] = stitch_artifacts["panos"]
    node_frames: dict[str, list[str]] = stitch_artifacts["node_frames"]

    if not pano_paths:
        return StageResult(False, {}, {}, failure_reason="no pano paths in stitch artifacts")

    # Load pose map
    try:
        pose_map = _read_images_bin(sparse_dir)
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"could not read images.bin: {e}")

    # Build per-node geometry (center + forward reference)
    node_ids = sorted(pano_paths.keys(), key=lambda n: int(n[1:]))
    node_centers: dict[str, np.ndarray] = {}
    node_forwards: dict[str, np.ndarray] = {}

    for nid in node_ids:
        frames = node_frames.get(nid, [])
        centers = []
        forwards = []
        for name in frames:
            if name not in pose_map:
                continue
            p = pose_map[name]
            centers.append(_camera_center(p["qvec"], p["tvec"]))
            forwards.append(_camera_forward(p["qvec"]))
        if not centers:
            # Fallback: place at origin with forward=Z
            node_centers[nid] = np.zeros(3)
            node_forwards[nid] = np.array([0.0, 0.0, 1.0])
        else:
            node_centers[nid] = np.mean(centers, axis=0)
            fwd_mean = np.mean(forwards, axis=0)
            norm = np.linalg.norm(fwd_mean)
            node_forwards[nid] = fwd_mean / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    # Build edges
    edges: list[dict] = []

    def _add_edge(a: str, b: str) -> None:
        d = node_centers[b] - node_centers[a]
        dist = float(np.linalg.norm(d))
        az_a = _azimuth_deg(d, node_forwards[a])
        az_b = _azimuth_deg(-d, node_forwards[b])
        edges.append({
            "from": a, "to": b,
            "azimuth_from": round(az_a, 1),
            "azimuth_to": round(az_b, 1),
            "distance_m": round(dist, 3),
        })

    # Sequential edges
    for i in range(len(node_ids) - 1):
        _add_edge(node_ids[i], node_ids[i + 1])
        _add_edge(node_ids[i + 1], node_ids[i])

    # Proximity edges (non-sequential pairs)
    for i, a in enumerate(node_ids):
        for b in node_ids[i + 2:]:  # skip sequential (already added)
            dist = float(np.linalg.norm(node_centers[b] - node_centers[a]))
            if dist < PROXIMITY_M:
                _add_edge(a, b)
                _add_edge(b, a)

    # Upload pano JPEGs to R2
    pano_keys: dict[str, str] = {}
    for nid, local_path in pano_paths.items():
        key = scan_key(scan_id, f"panos/{nid}.jpg")
        pano_keys[nid] = key
        with open(local_path, "rb") as fh:
            put(key, fh.read(), content_type="image/jpeg")
        log.info("uploaded pano %s → %s", nid, key)

    # Build graph.json
    nodes_json = []
    for nid in node_ids:
        c = node_centers[nid]
        nodes_json.append({
            "id": nid,
            "pano_key": pano_keys.get(nid, ""),
            "position": [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)],
        })

    graph = {"nodes": nodes_json, "edges": edges}
    graph_key = scan_key(scan_id, "graph.json")
    put(graph_key, json.dumps(graph).encode(), content_type="application/json")
    log.info("uploaded graph.json (%d nodes, %d edges)", len(nodes_json), len(edges))

    return StageResult(
        ok=True,
        metrics={"nodes": len(nodes_json), "edges": len(edges)},
        artifacts={"graph_key": graph_key, "pano_keys": pano_keys},
    )
