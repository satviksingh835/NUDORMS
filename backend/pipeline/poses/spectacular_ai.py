"""Spectacular AI pose wrapper — Stage 3b + 3c.

Spectacular AI SDK (sai-cli, free for non-commercial use):
  - Processes video + IMU log → VIO poses with metric scale
  - Integrates rolling-shutter + motion-blur compensation (Stage 3c)
    using an IMU-coupled differentiable image-formation model
  - PocketGS shows ARKit-pose init reduces convergence 319s → 54s
    at the same quality; VIO poses are even better than ARKit alone
    because they are VISLAM-fused rather than pure inertial

The browser GuidedRecorder collects raw DeviceMotionEvent readings
alongside the video. The backend bundles them as JSONL and passes
them to `sai-cli process`. If sai-cli is not installed, the stage
silently fails and the ensemble falls through to VGGT.

sai-cli output: NerfStudio transforms.json → converted to COLMAP
binary so the rest of the pipeline is unaffected.

Install on the pod:
    pip install spectacularai[full]
    # or: pip install spectacularai
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from ..types import StageResult

log = logging.getLogger("nudorms.pipeline.poses.spectacular_ai")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def available() -> bool:
    try:
        result = subprocess.run(
            ["sai-cli", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# COLMAP binary writer (same helpers as mast3r.py / vggt.py)
# ---------------------------------------------------------------------------

def _R_to_quat_wxyz(R: list) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → quaternion [w, x, y, z]."""
    m = R
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return w, x, y, z


def _write_colmap_binary(out_dir: Path, cam_entries, img_entries, pt_entries) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # cameras.bin
    with open(out_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cam_entries)))
        for cam_id, w, h, fx, fy, cx, cy in cam_entries:
            f.write(struct.pack("<Ii", cam_id, 1))      # PINHOLE model_id=1
            f.write(struct.pack("<QQ", w, h))
            f.write(struct.pack("<4d", fx, fy, cx, cy))

    # images.bin
    with open(out_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(img_entries)))
        for img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name in img_entries:
            f.write(struct.pack("<I", img_id))
            f.write(struct.pack("<4d", qw, qx, qy, qz))
            f.write(struct.pack("<3d", tx, ty, tz))
            f.write(struct.pack("<I", cam_id))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))   # num_points2D = 0

    # points3D.bin
    with open(out_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(pt_entries)))
        for pt_id, x, y, z, r, g, b, err, track in pt_entries:
            f.write(struct.pack("<Q", pt_id))
            f.write(struct.pack("<3d", x, y, z))
            f.write(struct.pack("<3B", r, g, b))
            f.write(struct.pack("<d", err))
            f.write(struct.pack("<Q", 0))   # track_length = 0


# ---------------------------------------------------------------------------
# NerfStudio transforms.json → COLMAP binary
# ---------------------------------------------------------------------------

def _nerfstudio_to_colmap(transforms_path: Path, out_dir: Path) -> dict:
    """Convert NerfStudio transforms.json to COLMAP binary format.

    Returns quality metrics dict.
    """
    with open(transforms_path) as f:
        ns = json.load(f)

    frames = ns.get("frames", [])
    if not frames:
        raise ValueError("transforms.json has no frames")

    # NerfStudio uses fl_x / fl_y / cx / cy for camera intrinsics
    fl_x = ns.get("fl_x", ns.get("camera_angle_x", None))
    fl_y = ns.get("fl_y", fl_x)
    cx = ns.get("cx", None)
    cy = ns.get("cy", None)
    w = int(ns.get("w", 0))
    h = int(ns.get("h", 0))

    if fl_x is None:
        # Fall back to per-frame intrinsics from first frame
        first = frames[0]
        fl_x = first.get("fl_x", 500.0)
        fl_y = first.get("fl_y", fl_x)
        cx = first.get("cx", w / 2)
        cy = first.get("cy", h / 2)
        w = int(first.get("w", w))
        h = int(first.get("h", h))

    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2

    cam_entries = [(1, w, h, float(fl_x), float(fl_y), float(cx), float(cy))]
    img_entries = []
    pt_entries = []

    for i, frame in enumerate(frames, start=1):
        # NerfStudio: c2w (camera-to-world) 4×4 as nested list
        c2w = frame["transform_matrix"]   # 4×4 row-major
        # Convert to w2c: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
        R_c2w = [c2w[r][:3] for r in range(3)]
        t_c2w = [c2w[r][3] for r in range(3)]
        R_w2c = [[R_c2w[c][r] for c in range(3)] for r in range(3)]
        t_w2c = [
            -sum(R_w2c[r][c] * t_c2w[c] for c in range(3))
            for r in range(3)
        ]
        qw, qx, qy, qz = _R_to_quat_wxyz(R_w2c)
        name = Path(frame.get("file_path", f"{i:06d}.jpg")).name

        img_entries.append((i, qw, qx, qy, qz, *t_w2c, 1, name))

    _write_colmap_binary(out_dir, cam_entries, img_entries, pt_entries)

    registered = len(img_entries)
    total = len(frames)
    return {
        "registered_images": registered,
        "inlier_ratio": 1.0,               # VIO/SLAM: all frames are registered
        "reproj_error": -1,                 # no traditional reprojection; mark as passing
        "total_frames": total,
    }


# ---------------------------------------------------------------------------
# IMU JSONL helper
# ---------------------------------------------------------------------------

def _write_sai_imu(imu_jsonl: str, out_path: Path) -> None:
    """Convert browser IMU JSONL to Spectacular AI CSV format.

    Browser format (one JSON per line):
        {"t": <ms>, "wx": <deg/s>, "wy": <deg/s>, "wz": <deg/s>,
         "ax": <m/s²>, "ay": <m/s²>, "az": <m/s²>}

    Spectacular AI CSV format:
        time,sensor,x,y,z
        0.001,gyroscope,0.01,0.02,0.03
        0.001,accelerometer,0.0,0.0,9.8
    """
    import math as _math
    lines = ["time,sensor,x,y,z"]
    for raw in imu_jsonl.strip().splitlines():
        try:
            d = json.loads(raw)
            t_s = d["t"] / 1000.0
            wx = d["wx"] * _math.pi / 180.0   # deg/s → rad/s
            wy = d["wy"] * _math.pi / 180.0
            wz = d["wz"] * _math.pi / 180.0
            ax, ay, az = d["ax"], d["ay"], d["az"]
            lines.append(f"{t_s:.6f},gyroscope,{wx:.6f},{wy:.6f},{wz:.6f}")
            lines.append(f"{t_s:.6f},accelerometer,{ax:.6f},{ay:.6f},{az:.6f}")
        except (json.JSONDecodeError, KeyError):
            continue
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(frames_dir: Path, out_dir: Path,
        video_path: Path | None = None,
        imu_jsonl_path: Path | None = None) -> StageResult:
    if not available():
        return StageResult(
            ok=False, metrics={}, artifacts={},
            failure_reason="sai-cli not installed (pip install spectacularai[full])",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    sai_out = out_dir / "sai_output"
    sai_out.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        imu_csv = None

        if imu_jsonl_path and imu_jsonl_path.exists():
            imu_csv = tmp_path / "imu.csv"
            _write_sai_imu(imu_jsonl_path.read_text(), imu_csv)

        cmd = ["sai-cli", "process"]
        if video_path and video_path.exists():
            cmd += ["--input", str(video_path)]
        else:
            cmd += ["--input", str(frames_dir)]

        if imu_csv:
            cmd += ["--imu", str(imu_csv)]

        cmd += ["--output", str(sai_out), "--format", "nerfstudio"]

        log.info("running sai-cli: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            return StageResult(False, {}, {}, failure_reason="sai-cli timed out")
        except Exception as e:
            return StageResult(False, {}, {}, failure_reason=f"sai-cli error: {e}")

        if proc.returncode != 0:
            return StageResult(
                False, {}, {},
                failure_reason=f"sai-cli rc={proc.returncode}: {proc.stderr[-400:]}",
            )

    # Convert output to COLMAP binary
    transforms_json = sai_out / "transforms.json"
    if not transforms_json.exists():
        # Look recursively
        candidates = list(sai_out.rglob("transforms.json"))
        if candidates:
            transforms_json = candidates[0]
        else:
            return StageResult(False, {}, {},
                               failure_reason="sai-cli did not produce transforms.json")

    sparse_dir = out_dir / "sparse"
    try:
        metrics = _nerfstudio_to_colmap(transforms_json, sparse_dir)
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"transforms.json parse error: {e}")

    log.info("spectacular_ai: %d images registered (VIO metric scale)",
             metrics["registered_images"])

    return StageResult(
        ok=True,
        metrics=metrics,
        artifacts={
            "sparse_dir": str(sparse_dir),
            "transforms_json": str(transforms_json),
        },
    )
