"""Panorama stitching stage.

For each user-defined stop window (start_s, end_s) from GuidedRecorder,
this stage:
  1. Selects the frames whose video timestamps fall inside the window.
  2. Cross-checks that each frame has a registered COLMAP pose.
  3. Verifies the frames span ≥ MIN_YAW_DEG of rotation.
  4. Runs the Hugin CLI (pto_gen → cpfind → cpclean → autooptimiser →
     pano_modify → nona → enblend) to produce an equirectangular JPEG.

Outputs a StageResult with artifacts:
  {
    "panos": { "n0": "/abs/path/to/n0.jpg", ... },
    "node_frames": { "n0": ["0001.jpg", "0042.jpg", ...], ... }
  }

Requires: hugin-tools + enblend installed on the pod
  (apt-get install -y hugin-tools enblend)
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .types import StageResult

log = logging.getLogger("nudorms.stitch")

MIN_YAW_DEG = 270.0   # require at least 270° rotation to call it a pano
MIN_FRAMES_PER_STOP = 6
JPEG_QUALITY = 90
# Panorama canvas width (height = width/2 for equirectangular 2:1)
PANO_WIDTH = 8000


def _read_images_bin(sparse_dir: Path) -> dict[str, dict]:
    """Read COLMAP images.bin → {name: {qvec, tvec, image_id}}."""
    images_bin = Path(sparse_dir) / "images.bin"
    images: dict[str, dict] = {}
    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))   # w, x, y, z
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]   # noqa: F841
            # null-terminated name
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode()
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * (8 + 8 + 8))  # skip (x, y, point3D_id) × n
            images[name] = {"image_id": image_id, "qvec": qvec, "tvec": tvec}
    return images


def _qvec_to_rotation(qvec) -> np.ndarray:
    """COLMAP (w,x,y,z) → 3×3 rotation matrix (world→camera)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _yaw_spread_deg(frames_in_stop: list[str], pose_map: dict[str, dict]) -> float:
    """Estimate yaw spread (degrees) of the camera rotations in this stop."""
    yaws = []
    for name in frames_in_stop:
        if name not in pose_map:
            continue
        R = _qvec_to_rotation(pose_map[name]["qvec"])
        # Camera forward in world: R^T @ [0,0,1]
        fwd = R.T @ np.array([0.0, 0.0, 1.0])
        yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
        yaws.append(yaw)
    if len(yaws) < 2:
        return 0.0
    # Unwrap and compute range
    yaws_arr = np.unwrap(np.radians(yaws))
    return float(math.degrees(yaws_arr.max() - yaws_arr.min()))


def _run_hugin(frames: list[Path], out_jpg: Path, tmp: Path) -> bool:
    """Stitch frames into an equirectangular JPEG using Hugin CLI tools."""
    pto = tmp / "project.pto"
    frame_args = [str(f) for f in frames]

    try:
        # 1. Generate initial project
        subprocess.run(
            ["pto_gen", "-o", str(pto)] + frame_args,
            check=True, capture_output=True,
        )

        # 2. Detect control points
        subprocess.run(
            ["cpfind", "--multirow", "-o", str(pto), str(pto)],
            check=True, capture_output=True,
        )

        # 3. Remove bad control points
        subprocess.run(
            ["cpclean", "-o", str(pto), str(pto)],
            check=True, capture_output=True,
        )

        # 4. Optimize geometry & exposure
        subprocess.run(
            ["autooptimiser", "-a", "-m", "-l", "-s", "-o", str(pto), str(pto)],
            check=True, capture_output=True,
        )

        # 5. Set equirectangular output + canvas
        canvas = f"{PANO_WIDTH}x{PANO_WIDTH // 2}"
        subprocess.run(
            ["pano_modify", f"--canvas={canvas}", "--crop=AUTO",
             "--projection=2", "-o", str(pto), str(pto)],
            check=True, capture_output=True,
        )

        # 6. Warp individual frames
        nona_prefix = str(tmp / "warp")
        subprocess.run(
            ["nona", "-m", "TIFF_m", "-o", nona_prefix, str(pto)],
            check=True, capture_output=True,
        )

        # 7. Blend
        warped = sorted(tmp.glob("warp*.tif"))
        if not warped:
            log.error("nona produced no warped images")
            return False
        tif_out = tmp / "panorama.tif"
        subprocess.run(
            ["enblend", "-o", str(tif_out)] + [str(w) for w in warped],
            check=True, capture_output=True,
        )

        # 8. Convert to JPEG
        img = cv2.imread(str(tif_out))
        if img is None:
            log.error("could not read enblend output %s", tif_out)
            return False
        cv2.imwrite(str(out_jpg), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return True

    except subprocess.CalledProcessError as e:
        log.error("Hugin step failed: %s\nstderr: %s", e.cmd, e.stderr.decode(errors="replace"))
        return False


def run(
    scan_id: str,
    workdir: Path,
    frames_artifacts: dict,
    poses_artifacts: dict,
    stops: list[dict] | None,
) -> StageResult:
    frames_dir = Path(frames_artifacts["frames_dir"])
    sparse_dir = Path(poses_artifacts["sparse_dir"])

    # Load timestamp map produced by frame_select
    ts_file = frames_dir / "frame_timestamps.json"
    if not ts_file.exists():
        return StageResult(False, {}, {}, failure_reason="frame_timestamps.json missing")
    timestamps: dict[str, float] = json.loads(ts_file.read_text())

    # Load pose map
    try:
        pose_map = _read_images_bin(sparse_dir)
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"could not read images.bin: {e}")

    all_frame_names = sorted(timestamps.keys())

    # If no stops provided, treat the entire video as one stop
    if not stops:
        log.warning("no stop windows provided — treating entire video as one stop")
        max_t = max(timestamps.values()) if timestamps else 0.0
        stops = [{"start_s": 0.0, "end_s": max_t}]

    panos_dir = workdir / "panos"
    panos_dir.mkdir(exist_ok=True)

    pano_paths: dict[str, str] = {}
    node_frames: dict[str, list[str]] = {}
    degraded_nodes: list[str] = []
    metrics: dict = {}

    for i, stop in enumerate(stops):
        node_id = f"n{i}"
        start_s = float(stop.get("start_s", 0.0))
        end_s = float(stop.get("end_s", 9999.0))

        # Collect frames in this time window
        stop_frames = [
            name for name in all_frame_names
            if start_s <= timestamps[name] <= end_s
        ]

        # Cross-check against registered poses
        registered = [n for n in stop_frames if n in pose_map]
        if len(registered) < MIN_FRAMES_PER_STOP:
            log.warning("node %s: only %d registered frames (need %d), skipping",
                        node_id, len(registered), MIN_FRAMES_PER_STOP)
            continue

        # Check yaw coverage
        yaw = _yaw_spread_deg(registered, pose_map)
        metrics[f"{node_id}_yaw_deg"] = round(yaw, 1)
        if yaw < MIN_YAW_DEG:
            log.warning("node %s: yaw spread %.1f° < %.1f°, marking degraded",
                        node_id, yaw, MIN_YAW_DEG)
            degraded_nodes.append(node_id)

        # Run Hugin stitching
        frame_paths = [frames_dir / n for n in registered]
        out_jpg = panos_dir / f"{node_id}.jpg"
        tmp = panos_dir / f"_tmp_{node_id}"
        tmp.mkdir(exist_ok=True)

        ok = _run_hugin(frame_paths, out_jpg, tmp)
        shutil.rmtree(tmp, ignore_errors=True)

        if not ok:
            log.warning("node %s: Hugin stitching failed, skipping", node_id)
            continue

        pano_paths[node_id] = str(out_jpg)
        node_frames[node_id] = registered
        log.info("node %s: stitched %d frames, yaw=%.1f°", node_id, len(registered), yaw)

    metrics["degraded_nodes"] = degraded_nodes
    metrics["total_nodes"] = len(pano_paths)

    if len(pano_paths) < 2:
        return StageResult(
            False, metrics, {},
            failure_reason=(
                f"only {len(pano_paths)} node(s) stitched successfully — "
                "need ≥2 for a navigable tour. "
                "Ensure each stop includes a full 360° rotation."
            ),
        )

    return StageResult(
        ok=True,
        metrics=metrics,
        artifacts={"panos": pano_paths, "node_frames": node_frames},
    )
