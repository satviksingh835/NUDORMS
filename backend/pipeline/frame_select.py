"""Greedy diverse frame selection.

Naive 'every Nth frame' loses quality. We:
  1. Decode the video at a moderate fps into all candidate frames.
  2. Score each frame's sharpness; drop the worst.
  3. Greedily walk forward, keeping a frame only if optical flow magnitude
     to the last-kept frame exceeds a threshold (parallax) AND the frame
     is sharp. This drops near-duplicate stationary frames AND motion-blurred
     frames in one pass.
  4. Cap at TARGET_FRAMES.

Output: workdir/frames/0001.jpg ... 0NNN.jpg
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import ffmpeg
import numpy as np

from .types import StageResult

DECODE_FPS = 6
TARGET_FRAMES = 300
MIN_FRAMES = 80
SHARPNESS_PERCENTILE_DROP = 25       # drop bottom-25% by sharpness
# px between consecutive selected frames. Lower bound is loose because Farneback
# optical flow under-reports motion on textureless surfaces (plain walls in a
# dorm) — pose estimation downstream handles those frames fine, so we shouldn't
# reject them just because flow couldn't measure their motion.
MIN_FLOW_MAGNITUDE = 3.0
MAX_FLOW_MAGNITUDE = 120.0           # px; above this user moved too fast (motion blur)


def _decode_to_frames(video_path: Path, out_dir: Path, fps: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg.input(str(video_path))
        .filter("fps", fps=fps)
        .output(str(out_dir / "%05d.jpg"), **{"qscale:v": 2})
        .global_args("-loglevel", "error")
        .overwrite_output()
        .run()
    )
    return sorted(out_dir.glob("*.jpg"))


def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_flow_magnitude(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0,
    )
    return float(np.linalg.norm(flow, axis=2).mean())


def run(scan_id: str, workdir: Path) -> StageResult:
    raw = workdir / "raw.mp4"
    if not raw.exists():
        return StageResult(False, {}, {}, failure_reason="raw video missing from workdir")

    candidates_dir = workdir / "all_frames"
    candidates = _decode_to_frames(raw, candidates_dir, DECODE_FPS)
    if len(candidates) < MIN_FRAMES:
        return StageResult(False, {"candidates": len(candidates)}, {},
                           failure_reason=f"only {len(candidates)} frames decoded, need ≥{MIN_FRAMES}")

    # Pass 1: sharpness scores.
    sharpness = []
    for p in candidates:
        img = cv2.imread(str(p))
        sharpness.append(_sharpness(img) if img is not None else 0.0)
    sharpness_arr = np.array(sharpness)
    sharpness_threshold = np.percentile(sharpness_arr, SHARPNESS_PERCENTILE_DROP)

    # Pass 2: greedy parallax-based selection.
    frames_dir = workdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    selected: list[Path] = []
    last_gray: np.ndarray | None = None
    rejections = {"too_blurry": 0, "too_close": 0, "too_far": 0}

    for path, sharp in zip(candidates, sharpness_arr):
        if len(selected) >= TARGET_FRAMES:
            break
        if sharp < sharpness_threshold:
            rejections["too_blurry"] += 1
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if last_gray is not None:
            flow_mag = _mean_flow_magnitude(last_gray, gray)
            if flow_mag < MIN_FLOW_MAGNITUDE:
                rejections["too_close"] += 1
                continue
            if flow_mag > MAX_FLOW_MAGNITUDE:
                rejections["too_far"] += 1
                continue

        out = frames_dir / f"{len(selected)+1:04d}.jpg"
        shutil.copy(path, out)
        selected.append(out)
        last_gray = gray

    if len(selected) < MIN_FRAMES:
        return StageResult(
            False,
            {"selected": len(selected), **rejections},
            {"reason": "insufficient_diverse_frames",
             "message": "Couldn't find enough sharp, well-spaced frames. "
                        "Move more deliberately and walk a full loop of the room."},
        )

    # Free disk: drop the candidates pool now that selection is final.
    shutil.rmtree(candidates_dir, ignore_errors=True)

    return StageResult(
        ok=True,
        metrics={
            "selected_frames": len(selected),
            "candidate_frames": len(candidates),
            **rejections,
        },
        artifacts={"frames_dir": str(frames_dir)},
    )
