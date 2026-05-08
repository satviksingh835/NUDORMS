"""Capture QC: cheap CPU-side checks before any GPU time is spent.

Reads the raw video, samples frames, and rejects captures that are obviously
unrecoverable (too dark, too blurry, too short, no motion, etc.).
"""
from __future__ import annotations

from pathlib import Path

from .orchestrator import StageResult

MIN_DURATION_S = 20.0
MIN_SHARPNESS = 60.0          # mean Laplacian variance across sampled frames
MAX_BLURRY_FRAME_RATIO = 0.25
MIN_EXPOSURE_STDDEV = 5.0     # too low = static frozen frame; too high = AE drift


def run(scan_id: str, workdir: Path) -> StageResult:
    """Pull raw video from object store, sample ~30 frames, score them.

    Returns ok=False with a feedback dict the user can act on.
    """
    # TODO: download raw_video_key into workdir / "raw.mp4"
    # TODO: sample ~30 evenly-spaced frames with ffmpeg-python
    # TODO: for each frame compute:
    #   - cv2.Laplacian(gray, cv2.CV_64F).var()                   sharpness
    #   - mean luma                                               brightness
    #   - Sobel orientation coherence                             motion-blur axis
    # TODO: compute exposure stddev across frames
    # TODO: build region diagnostics if a region (left/right/top/bottom) is uniformly dark
    return StageResult(ok=True, metrics={}, artifacts={})
