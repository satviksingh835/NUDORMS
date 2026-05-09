"""Pose ensemble: try methods in order, pick the winner by quality metrics.

Pose estimation is the highest-leverage stage in the pipeline. A bad pose
solve makes splat training meaningless. We try the fast/best path first,
fall back to learned methods on hard scenes, and finally to slow exhaustive
COLMAP. The orchestrator records which method won so we can tune the order.
"""
from __future__ import annotations

from pathlib import Path

from ..types import StageResult
from . import colmap, glomap, mast3r

MIN_INLIER_RATIO = 0.45
MAX_REPROJ_ERROR_PX = 1.5
MIN_REGISTERED_RATIO = 0.85  # registered_images / total_frames


def _passes_quality(r: StageResult, total_frames: int) -> bool:
    if not r.ok:
        return False
    m = r.metrics
    return (
        m.get("inlier_ratio", 0.0) >= MIN_INLIER_RATIO
        and m.get("reproj_error", 1e9) <= MAX_REPROJ_ERROR_PX
        and m.get("registered_images", 0) / max(total_frames, 1) >= MIN_REGISTERED_RATIO
    )


def run(scan_id: str, workdir: Path, frame_artifacts: dict) -> StageResult:
    frames_dir = Path(frame_artifacts["frames_dir"])
    total = sum(1 for _ in frames_dir.glob("*.jpg")) if frames_dir.exists() else 0

    attempts: list[tuple[str, StageResult]] = []

    # MASt3R first: handles casual capture (varied motion, textureless walls).
    # GLOMAP+ALIKED second: faster, tighter poses on careful dense captures.
    # COLMAP incremental last: slowest, most exhaustive fallback.
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
