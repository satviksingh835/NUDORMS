from __future__ import annotations

import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

broker_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery = Celery("nudorms", broker=broker_url, backend=broker_url)

celery.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # GPU jobs are long; never pre-grab.
    task_time_limit=60 * 60 * 2,   # 2h hard cap per scan.
    task_routes={
        "pipeline.run": {"queue": "gpu"},
    },
)


@celery.task(name="pipeline.run", bind=True)
def run_pipeline_task(self, scan_id: str, imu_key: str | None = None) -> dict:
    """Entry point for the GPU worker. Delegates to the orchestrator.

    If NUDORMS_AUTO_STOP_POD=1 is set in the env, the pod stops itself
    via RunPod API after the scan reaches a terminal state (READY,
    FAILED, NEEDS_RECAPTURE) — useful for overnight runs.
    """
    from pipeline.orchestrator import run_pipeline
    from pipeline.auto_stop import auto_stop_enabled, stop_pod_self

    try:
        result = run_pipeline(scan_id, task=self, imu_key=imu_key)
    except Exception:
        if auto_stop_enabled():
            stop_pod_self(reason="pipeline_exception")
        raise

    if auto_stop_enabled():
        # Always stop on terminal state, regardless of READY/FAILED/NEEDS_RECAPTURE.
        # The user wants the pod to power down so they don't pay for idle hours.
        stop_pod_self(reason=f"scan_{result.get('status', 'unknown')}")

    return result
