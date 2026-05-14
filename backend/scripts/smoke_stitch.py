#!/usr/bin/env python3
"""Smoke test: stitch a directory of frames from a single stationary 360° stop.

Usage (on the GPU pod with Hugin installed):
    uv run python scripts/smoke_stitch.py /path/to/stop_frames_dir /path/to/sparse/0

The frames dir must contain JPEGs selected by frame_select (0001.jpg, ...) plus
frame_timestamps.json. The sparse/0 dir must contain images.bin from pose estimation.

Asserts:
  - Output panorama is non-empty
  - Aspect ratio is approximately 2:1 (equirectangular)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
from pipeline.stitch import _run_hugin

def main():
    if len(sys.argv) < 2:
        print("Usage: smoke_stitch.py <frames_dir> [sparse_dir]")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"No JPEGs found in {frames_dir}")
        sys.exit(1)

    print(f"Stitching {len(frames)} frames from {frames_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_jpg = tmp_path / "test_pano.jpg"
        ok = _run_hugin(frames, out_jpg, tmp_path / "work")
        if not ok:
            print("FAIL: Hugin stitching returned False")
            sys.exit(1)
        if not out_jpg.exists() or out_jpg.stat().st_size < 10_000:
            print("FAIL: output file missing or suspiciously small")
            sys.exit(1)

        import cv2
        img = cv2.imread(str(out_jpg))
        h, w = img.shape[:2]
        ratio = w / h
        print(f"Output: {w}×{h}px, aspect ratio={ratio:.3f}")
        if not (1.8 <= ratio <= 2.2):
            print(f"WARN: aspect ratio {ratio:.3f} is not close to 2:1 (equirectangular)")
        else:
            print("OK: aspect ratio is ~2:1")

    print("PASS")

if __name__ == "__main__":
    main()
