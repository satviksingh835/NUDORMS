"""Pipeline orchestrator.

Drives every stage of video -> 3D, picks winners between competing methods,
runs the auto-retry loop, and updates the scan row at every transition.
This is the single place where quality decisions are made.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.db import get_session
from app.models import Scan, ScanStatus

from . import compress as compress_stage
from . import frame_select, ingest_qc, mesh_export
from .poses import ensemble as pose_ensemble
from .train import eval as eval_stage
from .train import gsplat_mcmc, twodgs
from .types import StageResult  # re-exported for backward compat

__all__ = ["run_pipeline", "StageResult"]

log = logging.getLogger("nudorms.pipeline")

PSNR_PASS = 28.0
PSNR_RETRY = 24.0


def _set_status(scan_id: str, status: ScanStatus, **fields) -> None:
    with get_session() as db:
        scan = db.get(Scan, scan_id)
        if scan is None:
            raise RuntimeError(f"scan {scan_id} disappeared mid-pipeline")
        scan.status = status.value
        for k, v in fields.items():
            setattr(scan, k, v)
        db.commit()


def run_pipeline(scan_id: str, task=None) -> dict:
    """End-to-end pipeline. Each stage updates the DB; failures are recorded."""
    with tempfile.TemporaryDirectory(prefix=f"scan-{scan_id}-") as tmp:
        workdir = Path(tmp)

        # 1. QC --------------------------------------------------------------
        _set_status(scan_id, ScanStatus.QC)
        qc = ingest_qc.run(scan_id, workdir)
        if not qc.ok:
            _set_status(scan_id, ScanStatus.NEEDS_RECAPTURE, feedback=qc.artifacts)
            return {"status": "needs_recapture", "stage": "qc"}

        # 2. Frames ----------------------------------------------------------
        _set_status(scan_id, ScanStatus.FRAMES)
        frames = frame_select.run(scan_id, workdir)
        if not frames.ok:
            _set_status(scan_id, ScanStatus.FAILED, error=frames.failure_reason)
            return {"status": "failed", "stage": "frames"}

        # 3. Pose ensemble (GLOMAP -> MASt3R -> COLMAP) ---------------------
        _set_status(scan_id, ScanStatus.POSING)
        poses = pose_ensemble.run(scan_id, workdir, frames.artifacts)
        if not poses.ok:
            _set_status(
                scan_id,
                ScanStatus.NEEDS_RECAPTURE,
                feedback={"reason": "pose_estimation_failed", "details": poses.failure_reason},
            )
            return {"status": "needs_recapture", "stage": "posing"}

        # 4. Train + 5. Eval (with auto-retry) ------------------------------
        train_attempt = 0
        train_metrics: dict = {}
        while True:
            train_attempt += 1
            _set_status(scan_id, ScanStatus.TRAINING)
            train = gsplat_mcmc.run(scan_id, workdir, poses.artifacts, attempt=train_attempt)

            _set_status(scan_id, ScanStatus.EVALUATING)
            evaluation = eval_stage.run(scan_id, workdir, train.artifacts, poses.artifacts)
            train_metrics = {**train.metrics, **evaluation.metrics}

            psnr = evaluation.metrics.get("psnr", 0.0)
            if psnr >= PSNR_PASS:
                break
            if psnr >= PSNR_RETRY and train_attempt < 2:
                _set_status(scan_id, ScanStatus.RETRYING, metrics=train_metrics)
                continue
            _set_status(
                scan_id,
                ScanStatus.NEEDS_RECAPTURE,
                metrics=train_metrics,
                feedback=evaluation.artifacts.get("region_diagnostics", {}),
            )
            return {"status": "needs_recapture", "stage": "evaluating", "psnr": psnr}

        # 6. Mesh extraction (parallelizable; sequential here for simplicity)
        _set_status(scan_id, ScanStatus.MESHING)
        mesh = twodgs.run(scan_id, workdir, poses.artifacts)
        mesh_artifacts = mesh_export.run(scan_id, workdir, mesh.artifacts) if mesh.ok else None

        # 7. Compression + LOD ----------------------------------------------
        _set_status(scan_id, ScanStatus.COMPRESSING)
        compressed = compress_stage.run(scan_id, workdir, train.artifacts)

        _set_status(
            scan_id,
            ScanStatus.READY,
            metrics=train_metrics,
            splat_key=compressed.artifacts.get("splat_key"),
            mesh_key=(mesh_artifacts.artifacts.get("mesh_key") if mesh_artifacts else None),
            lod_keys=compressed.artifacts.get("lod_keys"),
        )
        return {"status": "ready", "metrics": train_metrics}
