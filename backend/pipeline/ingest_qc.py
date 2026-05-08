"""Capture QC: cheap CPU-side checks before any GPU time is spent.

Pulls the raw video, samples ~30 evenly-spaced frames, scores them, and
either passes the scan downstream or returns actionable feedback so the
user can re-record before GPU minutes are wasted.

Quality signals:
  - sharpness:  variance of Laplacian (higher = sharper)
  - brightness: mean luma (Y of YCrCb) per frame
  - exposure stability: std-dev of brightness across frames
                        (high = AE drift, kills splat training)
  - duration:   too-short clips can't cover a room

Returns ok=True with metrics on success.
Returns ok=False with a human-readable feedback dict on failure.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import ffmpeg
import numpy as np

from app.db import get_session
from app.models import Scan
from app.storage import download

from .types import StageResult

NUM_QC_FRAMES = 30
MIN_DURATION_S = 20.0
MAX_DURATION_S = 300.0
MIN_SHARPNESS = 60.0
MAX_BLURRY_FRAME_RATIO = 0.25
MIN_BRIGHTNESS = 35.0
# Variation in mean per-frame luma. Imperfect proxy for AE drift — it also
# picks up legitimate scene variation (panning from a bright window to a dim
# corner). 35 catches obvious AE failures while letting normal scene variety
# through. The trainer's per-frame appearance embeddings handle the residual.
MAX_EXPOSURE_STDDEV = 35.0


@dataclass
class FrameScore:
    idx: int
    t: float
    sharpness: float
    brightness: float


def _video_duration(path: Path) -> float:
    info = ffmpeg.probe(str(path))
    return float(info["format"]["duration"])


def _sample_timestamps(duration: float, n: int) -> list[float]:
    # Skip the first/last 5% — phones wobble at start/end of recordings.
    pad = duration * 0.05
    return list(np.linspace(pad, duration - pad, n))


def _grab_frame(video_path: Path, t: float) -> np.ndarray | None:
    """Single-frame seek via ffmpeg — much faster than decoding sequentially."""
    cmd = (
        ffmpeg.input(str(video_path), ss=t)
        .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
        .global_args("-loglevel", "error")
    )
    try:
        buf, _ = cmd.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error:
        return None
    if not buf:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _score_frame(frame: np.ndarray, idx: int, t: float) -> FrameScore:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    brightness = float(ycrcb[..., 0].mean())
    return FrameScore(idx=idx, t=t, sharpness=sharpness, brightness=brightness)


def run(scan_id: str, workdir: Path) -> StageResult:
    with get_session() as db:
        scan = db.get(Scan, scan_id)
        if scan is None or not scan.raw_video_key:
            return StageResult(False, {}, {}, failure_reason="no raw video on scan")
        raw_key = scan.raw_video_key

    raw_path = workdir / "raw.mp4"
    download(raw_key, raw_path)

    try:
        duration = _video_duration(raw_path)
    except (ffmpeg.Error, subprocess.CalledProcessError, KeyError) as e:
        return StageResult(False, {}, {"reason": "unreadable_video", "detail": str(e)})

    if duration < MIN_DURATION_S:
        return StageResult(
            False, {"duration_s": duration},
            {"reason": "too_short",
             "message": f"Recording was {duration:.0f}s. Aim for 30–90s and walk a full loop."},
        )
    if duration > MAX_DURATION_S:
        return StageResult(
            False, {"duration_s": duration},
            {"reason": "too_long",
             "message": f"Recording was {duration:.0f}s. Keep it under {MAX_DURATION_S:.0f}s."},
        )

    timestamps = _sample_timestamps(duration, NUM_QC_FRAMES)
    scores: list[FrameScore] = []
    for i, t in enumerate(timestamps):
        frame = _grab_frame(raw_path, t)
        if frame is None:
            continue
        scores.append(_score_frame(frame, i, t))

    if len(scores) < NUM_QC_FRAMES * 0.7:
        return StageResult(
            False, {"sampled": len(scores)},
            {"reason": "decode_failed",
             "message": "Could not read frames from the video. Try re-uploading."},
        )

    sharpness_arr = np.array([s.sharpness for s in scores])
    brightness_arr = np.array([s.brightness for s in scores])
    blurry_ratio = float((sharpness_arr < MIN_SHARPNESS).mean())
    mean_brightness = float(brightness_arr.mean())
    exposure_std = float(brightness_arr.std())

    metrics = {
        "duration_s": duration,
        "frames_sampled": len(scores),
        "blurry_ratio": blurry_ratio,
        "mean_brightness": mean_brightness,
        "exposure_stddev": exposure_std,
        "median_sharpness": float(np.median(sharpness_arr)),
    }

    if blurry_ratio > MAX_BLURRY_FRAME_RATIO:
        worst = sorted(scores, key=lambda s: s.sharpness)[:5]
        return StageResult(
            False, metrics,
            {"reason": "too_blurry",
             "message": f"{blurry_ratio*100:.0f}% of frames were blurry. Move slower; hold the phone steady.",
             "worst_seconds": [round(s.t, 1) for s in worst]},
        )
    if mean_brightness < MIN_BRIGHTNESS:
        return StageResult(
            False, metrics,
            {"reason": "too_dark",
             "message": "Room is too dark. Turn on lights and open blinds, then re-record."},
        )
    if exposure_std > MAX_EXPOSURE_STDDEV:
        return StageResult(
            False, metrics,
            {"reason": "exposure_drift",
             "message": "Camera auto-exposure changed mid-recording, which breaks reconstruction. "
                        "Tap-and-hold to lock exposure before recording, then try again."},
        )

    return StageResult(ok=True, metrics=metrics, artifacts={"raw_path": str(raw_path)})
