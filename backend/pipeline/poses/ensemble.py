"""Pose ensemble: try methods in order, pick the winner by quality metrics.

Pose estimation is the highest-leverage stage in the pipeline. A bad pose
solve makes splat training meaningless. We try the fast/best path first,
fall back to learned methods on hard scenes, and finally to slow exhaustive
COLMAP. The orchestrator records which method won so we can tune the order.
"""
from __future__ import annotations

from pathlib import Path

from ..types import StageResult
from . import colmap, glomap, mast3r, spectacular_ai, vggt

MIN_INLIER_RATIO = 0.45
MAX_REPROJ_ERROR_PX = 1.5
MIN_REGISTERED_RATIO = 0.85  # registered_images / total_frames


def _passes_quality(r: StageResult, total_frames: int) -> bool:
    if not r.ok:
        return False
    m = r.metrics
    # reproj_error of -1 means the binary fallback reader ran (no pycolmap);
    # treat it as passing since VGGT/MASt3R self-optimize and don't report it.
    reproj_ok = m.get("reproj_error", 1e9) <= MAX_REPROJ_ERROR_PX or m.get("reproj_error") == -1
    return (
        m.get("inlier_ratio", 0.0) >= MIN_INLIER_RATIO
        and reproj_ok
        and m.get("registered_images", 0) / max(total_frames, 1) >= MIN_REGISTERED_RATIO
    )


def run(scan_id: str, workdir: Path, frame_artifacts: dict,
        imu_jsonl_path=None, video_path=None) -> StageResult:
    frames_dir = Path(frame_artifacts["frames_dir"])
    total = sum(1 for _ in frames_dir.glob("*.jpg")) if frames_dir.exists() else 0

    attempts: list[tuple[str, StageResult]] = []

    # Order (from research doc, 2025):
    # 0. Spectacular AI — VIO + VISLAM with IMU, metric scale, rolling-shutter
    #    compensation. Best on casual phone capture when IMU data is available.
    # 1. VGGT — CVPR 2025 Best Paper; feed-forward, seconds, best on casual indoor.
    # 2. MASt3R-SfM — learned features + global alignment; handles textureless walls.
    # 3. GLOMAP+ALIKED — faster on careful/dense captures with good texture.
    # 4. COLMAP incremental — slowest, most exhaustive, last resort.

    if spectacular_ai.available() and (imu_jsonl_path or video_path):
        s = spectacular_ai.run(
            frames_dir, workdir / "poses_sai",
            video_path=video_path,
            imu_jsonl_path=imu_jsonl_path,
        )
        attempts.append(("spectacular_ai", s))
        if _passes_quality(s, total):
            return _wrap("spectacular_ai", s, attempts)

    v = vggt.run(frames_dir, workdir / "poses_vggt")
    attempts.append(("vggt", v))
    if _passes_quality(v, total):
        return _wrap("vggt", v, attempts)

    m = mast3r.run(frames_dir, workdir / "poses_mast3r")
    attempts.append(("mast3r", m))
    if _passes_quality(m, total):
        return _wrap("mast3r", m, attempts)

    g = glomap.run(frames_dir, workdir / "poses_glomap")
    attempts.append(("glomap", g))
    if _passes_quality(g, total):
        return _wrap("glomap", g, attempts)

    c = colmap.run(frames_dir, workdir / "poses_colmap")
    attempts.append(("colmap", c))
    if _passes_quality(c, total):
        return _wrap("colmap", c, attempts)

    # All methods underperformed. Pick the best one we got and let the trainer
    # try anyway — sometimes splatting succeeds even with mediocre poses.
    best_name, best = max(
        attempts,
        key=lambda kv: kv[1].metrics.get("inlier_ratio", 0.0) if kv[1].ok else -1.0,
    )
    if not best.ok:
        return StageResult(
            ok=False,
            metrics={"attempts": [a for a, _ in attempts]},
            artifacts={},
            failure_reason="all pose methods failed",
        )
    return _wrap(best_name, best, attempts, marginal=True)


def _wrap(winner: str, r: StageResult, attempts, marginal: bool = False) -> StageResult:
    return StageResult(
        ok=True,
        metrics={
            **r.metrics,
            "pose_winner": winner,
            "pose_marginal": marginal,
            "pose_attempts": {name: a.metrics for name, a in attempts},
        },
        artifacts=r.artifacts,
    )
