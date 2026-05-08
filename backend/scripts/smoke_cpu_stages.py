"""Run the CPU-only pipeline stages (QC + frame selection) on a local video.

Usage:
    cd backend
    uv run python scripts/smoke_cpu_stages.py /path/to/walkthrough.mp4

Skips R2 entirely — copies the video into a temp workdir and runs the same
stage code the orchestrator will run in production. Use this to validate QC
thresholds and frame selection before paying for any GPU pod time.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from pprint import pprint


def main(video_path: str) -> int:
    src = Path(video_path)
    if not src.exists():
        print(f"file not found: {src}", file=sys.stderr)
        return 1

    # Defer imports so a missing opencv/ffmpeg fails cleanly.
    from pipeline import frame_select, ingest_qc

    with tempfile.TemporaryDirectory(prefix="nudorms-smoke-") as tmp:
        workdir = Path(tmp)
        raw = workdir / "raw.mp4"
        shutil.copy(src, raw)

        # Bypass the DB lookup in ingest_qc by monkey-patching: smoke tests
        # don't need a Scan row. We feed the path directly.
        print("\n── ingest_qc ──")
        qc = _run_qc_inline(workdir, raw)
        pprint({"ok": qc.ok, "metrics": qc.metrics, "artifacts": qc.artifacts,
                "failure_reason": qc.failure_reason})
        if not qc.ok:
            return 2

        print("\n── frame_select ──")
        sel = frame_select.run("smoke", workdir)
        pprint({"ok": sel.ok, "metrics": sel.metrics, "artifacts": sel.artifacts,
                "failure_reason": sel.failure_reason})
        if not sel.ok:
            return 3

        # Persist outputs next to the input for inspection.
        out_dir = src.parent / f"{src.stem}_nudorms_out"
        out_dir.mkdir(exist_ok=True)
        frames = Path(sel.artifacts["frames_dir"])
        for f in sorted(frames.iterdir())[:6]:
            shutil.copy(f, out_dir / f.name)
        print(f"\nfirst 6 selected frames copied to: {out_dir}")
        print(f"total selected: {sel.metrics['selected_frames']}")
    return 0


def _run_qc_inline(workdir, raw):
    """Same gating as ingest_qc.run, but skips the DB lookup so we don't
    need a Scan row for smoke tests."""
    import numpy as np

    from pipeline import ingest_qc as qc

    duration = qc._video_duration(raw)
    if duration < qc.MIN_DURATION_S:
        return qc.StageResult(False, {"duration_s": duration},
                              {"reason": "too_short", "message": f"{duration:.0f}s"})
    if duration > qc.MAX_DURATION_S:
        return qc.StageResult(False, {"duration_s": duration},
                              {"reason": "too_long", "message": f"{duration:.0f}s"})

    timestamps = qc._sample_timestamps(duration, qc.NUM_QC_FRAMES)
    scores = []
    for i, t in enumerate(timestamps):
        frame = qc._grab_frame(raw, t)
        if frame is None:
            continue
        scores.append(qc._score_frame(frame, i, t))

    sharp = np.array([s.sharpness for s in scores])
    bright = np.array([s.brightness for s in scores])
    metrics = {
        "duration_s": duration,
        "frames_sampled": len(scores),
        "blurry_ratio": float((sharp < qc.MIN_SHARPNESS).mean()),
        "mean_brightness": float(bright.mean()),
        "exposure_stddev": float(bright.std()),
        "median_sharpness": float(np.median(sharp)),
    }

    if metrics["blurry_ratio"] > qc.MAX_BLURRY_FRAME_RATIO:
        worst = sorted(scores, key=lambda s: s.sharpness)[:5]
        return qc.StageResult(
            False, metrics,
            {"reason": "too_blurry",
             "message": f"{metrics['blurry_ratio']*100:.0f}% of frames are blurry. Move slower; lock exposure/focus.",
             "worst_seconds": [round(s.t, 1) for s in worst]},
        )
    if metrics["mean_brightness"] < qc.MIN_BRIGHTNESS:
        return qc.StageResult(
            False, metrics,
            {"reason": "too_dark", "message": "Turn on more lights and re-record."},
        )
    if metrics["exposure_stddev"] > qc.MAX_EXPOSURE_STDDEV:
        return qc.StageResult(
            False, metrics,
            {"reason": "exposure_drift",
             "message": "Auto-exposure shifted mid-recording. Tap-and-hold to lock exposure before recording."},
        )
    return qc.StageResult(ok=True, metrics=metrics, artifacts={"raw_path": str(raw)})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
