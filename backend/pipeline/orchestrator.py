"""Pipeline orchestrator — Street-View edition.

Drives: QC → frame selection → pose estimation → pano stitching → graph build.
Each stage updates the scan row; failures are recorded before returning.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from app.db import get_session
from app.models import Scan, ScanStatus

from . import frame_select, ingest_qc
from . import stitch as stitch_stage
from . import graph as graph_stage
from .poses import ensemble as pose_ensemble
from .types import StageResult  # re-exported for backward compat

__all__ = ["run_pipeline", "StageResult"]

log = logging.getLogger("nudorms.pipeline")


def _set_status(scan_id: str, status: ScanStatus, **fields) -> None:
    with get_session() as db:
        scan = db.get(Scan, scan_id)
        if scan is None:
            raise RuntimeError(f"scan {scan_id} disappeared mid-pipeline")
        scan.status = status.value
        for k, v in fields.items():
            setattr(scan, k, v)
        db.commit()


MIN_FREE_DISK_GB = 10


def _free_disk_gb(path: str = "/tmp") -> float:
    import shutil
    return shutil.disk_usage(path).free / 1024**3


def run_pipeline(
    scan_id: str,
    task=None,
    imu_key: str | None = None,
    stops: list[dict] | None = None,
) -> dict:
    """End-to-end pipeline. Each stage updates the DB; failures are recorded."""
    from app.storage import get, scan_key as _scan_key

    free_gb = _free_disk_gb("/tmp")
    if free_gb < MIN_FREE_DISK_GB:
        log.error("only %.1f GB free in /tmp; need >=%d GB. Aborting.", free_gb, MIN_FREE_DISK_GB)
        with get_session() as db:
            scan = db.get(Scan, scan_id)
            if scan:
                scan.status = ScanStatus.FAILED.value
                scan.error = (
                    f"insufficient disk: {free_gb:.1f} GB free in /tmp "
                    f"(need ≥{MIN_FREE_DISK_GB} GB)"
                )
                db.commit()
        return {"status": "failed", "stage": "preflight", "free_gb": free_gb}

    workdir = Path(tempfile.gettempdir()) / f"scan-{scan_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        # Fetch IMU JSONL from R2 if available
        imu_jsonl_path: Path | None = None
        if imu_key:
            try:
                imu_bytes = get(imu_key)
                imu_jsonl_path = workdir / "imu.jsonl"
                imu_jsonl_path.write_bytes(imu_bytes)
                log.info("IMU data available (%d bytes)", len(imu_bytes))
            except Exception as e:
                log.warning("failed to fetch IMU data (%s) — continuing without", e)

        # Prefetch raw video for SAI (it wants the original video, not frames)
        video_path: Path | None = None
        try:
            with get_session() as db:
                scan = db.get(Scan, scan_id)
                raw_key = scan.raw_video_key if scan else None
            if raw_key:
                video_path = workdir / "raw_video.mp4"
                video_path.write_bytes(get(raw_key))
        except Exception as e:
            log.warning("could not prefetch raw video for sai-cli (%s)", e)

        _ckpt_poses = workdir / "_ckpt_poses.json"
        _ckpt_frames = workdir / "_ckpt_frames.json"

        if _ckpt_poses.exists() and _ckpt_frames.exists():
            log.info("resuming from checkpoint: skipping QC, frames, posing")
            frames = StageResult(True, {}, json.loads(_ckpt_frames.read_text()))
            poses = StageResult(True, {}, json.loads(_ckpt_poses.read_text()))
        else:
            # 1. QC -------------------------------------------------------
            _set_status(scan_id, ScanStatus.QC)
            qc = ingest_qc.run(scan_id, workdir)
            if not qc.ok:
                _set_status(scan_id, ScanStatus.NEEDS_RECAPTURE, feedback=qc.artifacts)
                return {"status": "needs_recapture", "stage": "qc"}

            # 2. Frame selection ------------------------------------------
            _set_status(scan_id, ScanStatus.FRAMES)
            frames = frame_select.run(scan_id, workdir)
            if not frames.ok:
                _set_status(scan_id, ScanStatus.FAILED, error=frames.failure_reason)
                return {"status": "failed", "stage": "frames"}
            _ckpt_frames.write_text(json.dumps(frames.artifacts))

            # 3. Pose ensemble (SAI → VGGT → MASt3R → GLOMAP → COLMAP) ----
            _set_status(scan_id, ScanStatus.POSING)
            poses = pose_ensemble.run(
                scan_id, workdir, frames.artifacts,
                imu_jsonl_path=imu_jsonl_path,
                video_path=video_path,
            )
            if not poses.ok:
                _set_status(
                    scan_id,
                    ScanStatus.NEEDS_RECAPTURE,
                    feedback={"reason": "pose_estimation_failed", "details": poses.failure_reason},
                )
                return {"status": "needs_recapture", "stage": "posing"}
            _ckpt_poses.write_text(json.dumps(poses.artifacts))

        # 4. Panorama stitching ------------------------------------------
        _set_status(scan_id, ScanStatus.STITCHING)
        stitch = stitch_stage.run(
            scan_id, workdir,
            frames_artifacts=frames.artifacts,
            poses_artifacts=poses.artifacts,
            stops=stops,
        )
        if not stitch.ok:
            _set_status(
                scan_id,
                ScanStatus.NEEDS_RECAPTURE,
                feedback={"reason": "stitching_failed", "details": stitch.failure_reason},
            )
            return {"status": "needs_recapture", "stage": "stitching"}

        # 5. Graph build + R2 upload ------------------------------------
        _set_status(scan_id, ScanStatus.GRAPH_BUILD)
        graph = graph_stage.run(
            scan_id, workdir,
            stitch_artifacts=stitch.artifacts,
            poses_artifacts=poses.artifacts,
        )
        if not graph.ok:
            _set_status(scan_id, ScanStatus.FAILED, error=graph.failure_reason)
            return {"status": "failed", "stage": "graph_build"}

        combined_metrics = {**stitch.metrics, **graph.metrics}
        _set_status(
            scan_id,
            ScanStatus.READY,
            metrics=combined_metrics,
            graph_key=graph.artifacts["graph_key"],
            pano_keys=graph.artifacts["pano_keys"],
        )
        shutil.rmtree(workdir, ignore_errors=True)
        return {"status": "ready", "metrics": combined_metrics}

    except Exception as exc:
        log.exception("unhandled exception in pipeline for scan %s", scan_id)
        try:
            _set_status(scan_id, ScanStatus.FAILED, error=str(exc)[:2000])
        except Exception:
            pass
        raise
