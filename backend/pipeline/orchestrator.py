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

from . import cleanup as cleanup_stage
from . import compress as compress_stage
from . import frame_select, ingest_qc, mesh_export, priors as priors_stage
from .poses import ensemble as pose_ensemble
from .train import difix3d, eval as eval_stage
from .train import gsplat_mcmc, pgsr, scaffold_gs, threedgut, twodgs
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


def run_pipeline(scan_id: str, task=None, imu_key: str | None = None) -> dict:
    """End-to-end pipeline. Each stage updates the DB; failures are recorded."""
    from app.storage import get, scan_key as _scan_key

    with tempfile.TemporaryDirectory(prefix=f"scan-{scan_id}-") as tmp:
        workdir = Path(tmp)

        from app.storage import get as storage_get

        # Fetch IMU JSONL from R2 if available (uploaded alongside video)
        imu_jsonl_path: Path | None = None
        if imu_key:
            try:
                imu_bytes = storage_get(imu_key)
                imu_jsonl_path = workdir / "imu.jsonl"
                imu_jsonl_path.write_bytes(imu_bytes)
                log.info("IMU data available (%d bytes) — will try Spectacular AI", len(imu_bytes))
            except Exception as e:
                log.warning("failed to fetch IMU data (%s) — continuing without", e)

        # Fetch raw video path for Spectacular AI (it wants the original video, not frames)
        video_path: Path | None = None
        try:
            with get_session() as db:
                scan = db.get(Scan, scan_id)
                raw_key = scan.raw_video_key if scan else None
            if raw_key:
                video_path = workdir / "raw_video.mp4"
                video_path.write_bytes(storage_get(raw_key))
        except Exception as e:
            log.warning("could not prefetch raw video for sai-cli (%s)", e)

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

        # 3. Pose ensemble (SAI → VGGT → MASt3R → GLOMAP → COLMAP) ----------
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

        # 4. Geometric priors (Depth Anything V2 + surface normals) ----------
        _set_status(scan_id, ScanStatus.PRIORS)
        priors = priors_stage.run(scan_id, workdir, frames.artifacts)
        if not priors.ok:
            log.warning("priors stage failed (%s) — continuing with depth/normal weight=0",
                        priors.failure_reason)
        prior_artifacts = priors.artifacts if priors.ok else {}

        # 5. Train + 6. Eval (with auto-retry) ------------------------------
        # Attempt order: Scaffold-GS (CVPR 2024 Highlight, best on indoor) →
        # gsplat MCMC (fallback when Scaffold-GS is not installed or fails).
        train_attempt = 0
        train_metrics: dict = {}
        _scaffold_tried = False

        while True:
            train_attempt += 1
            _set_status(scan_id, ScanStatus.TRAINING)

            # Training order: Scaffold-GS → 3DGUT (if reflections needed) → gsplat MCMC
            if not _scaffold_tried:
                _scaffold_tried = True
                train = scaffold_gs.run(
                    scan_id, workdir, poses.artifacts,
                    prior_artifacts=prior_artifacts,
                    attempt=train_attempt,
                )
                if not train.ok:
                    log.warning("Scaffold-GS unavailable (%s)", train.failure_reason)
                    # Try 3DGUT for reflection-rich scenes (monitors, windows)
                    if threedgut.available():
                        train = threedgut.run(
                            scan_id, workdir, poses.artifacts,
                            prior_artifacts=prior_artifacts,
                            attempt=train_attempt,
                        )
                    if not train.ok:
                        log.warning("3DGUT unavailable/failed — falling back to gsplat MCMC")
                        train = gsplat_mcmc.run(
                            scan_id, workdir, poses.artifacts,
                            prior_artifacts=prior_artifacts,
                            attempt=train_attempt,
                        )
            else:
                # Retry pass uses gsplat MCMC with stronger regularization
                train = gsplat_mcmc.run(
                    scan_id, workdir, poses.artifacts,
                    prior_artifacts=prior_artifacts,
                    attempt=train_attempt,
                )

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

        # 7. Difix3D+ diffusion artifact fixer (CVPR 2025 Oral) ---------------
        # Optional polish step — skipped silently if not installed.
        if difix3d.available():
            _set_status(scan_id, ScanStatus.REFINING)
            refined = difix3d.run(scan_id, workdir, train.artifacts, poses.artifacts)
            if refined.ok:
                log.info("Difix3D+ applied — using refined PLY")
                train_artifacts_final = refined.artifacts
                train_metrics = {**train_metrics, **refined.metrics}
            else:
                log.warning("Difix3D+ failed (%s) — using unrefined PLY", refined.failure_reason)
                train_artifacts_final = train.artifacts
        else:
            train_artifacts_final = train.artifacts

        # 8. DBSCAN floater cleanup (CPU-only, ~5–30 s) ---------------------
        cleanup = cleanup_stage.run(scan_id, workdir, train_artifacts_final)
        if cleanup.ok and cleanup.metrics.get("n_gaussians_removed", 0) > 0:
            train_artifacts_final = cleanup.artifacts
            train_metrics = {**train_metrics, **cleanup.metrics}

        # 9. Mesh extraction: PGSR (best Chamfer on indoor) → 2DGS fallback -
        _set_status(scan_id, ScanStatus.MESHING)
        mesh_result = None
        if pgsr.available():
            mesh_result = pgsr.run(scan_id, workdir, poses.artifacts)
            if not mesh_result.ok:
                log.warning("PGSR mesh failed (%s) — falling back to 2DGS", mesh_result.failure_reason)
                mesh_result = None

        if mesh_result is None:
            mesh_2dgs = twodgs.run(scan_id, workdir, poses.artifacts)
            mesh_result = mesh_2dgs if mesh_2dgs.ok else None

        mesh_artifacts = (
            mesh_export.run(scan_id, workdir, mesh_result.artifacts)
            if mesh_result and mesh_result.ok else None
        )

        # 10. Compression + LOD ---------------------------------------------
        _set_status(scan_id, ScanStatus.COMPRESSING)
        compressed = compress_stage.run(scan_id, workdir, train_artifacts_final)

        _set_status(
            scan_id,
            ScanStatus.READY,
            metrics=train_metrics,
            splat_key=compressed.artifacts.get("splat_key"),
            mesh_key=(mesh_artifacts.artifacts.get("mesh_key") if mesh_artifacts else None),
            lod_keys=compressed.artifacts.get("lod_keys"),
        )
        return {"status": "ready", "metrics": train_metrics}
