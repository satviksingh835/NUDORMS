"""Auto-stop the RunPod pod after a scan reaches a terminal state.

Call from inside the worker after the pipeline finishes (success, failure,
or needs-recapture) so an overnight run doesn't keep billing while idle.

Required env vars:
  RUNPOD_API_KEY   — from runpod.io → Settings → API Keys
  RUNPOD_POD_ID    — set automatically by the RunPod runtime
  NUDORMS_AUTO_STOP_POD=1   — opt-in flag (off by default for dev)

Stops the pod via RunPod GraphQL `podStop` mutation. The pod is paused
(not terminated): it can be restarted from the web UI without losing the
volume mount or installed packages on the container disk.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

log = logging.getLogger("nudorms.auto_stop")

GRAPHQL_URL = "https://api.runpod.io/graphql"
SHUTDOWN_GRACE_S = 30   # let logs flush + R2 uploads finalise before stopping


def auto_stop_enabled() -> bool:
    return os.environ.get("NUDORMS_AUTO_STOP_POD", "").strip() in ("1", "true", "yes")


def stop_pod_self(reason: str = "pipeline_complete") -> bool:
    """Stop this pod. Returns True on success, False if disabled or errored."""
    if not auto_stop_enabled():
        log.debug("auto-stop disabled (NUDORMS_AUTO_STOP_POD not set)")
        return False

    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")

    if not api_key:
        log.warning("auto-stop: RUNPOD_API_KEY missing — pod will keep running")
        return False
    if not pod_id:
        log.warning("auto-stop: RUNPOD_POD_ID missing — pod will keep running")
        return False

    log.info("auto-stop in %d s (reason: %s, pod=%s)", SHUTDOWN_GRACE_S, reason, pod_id)
    time.sleep(SHUTDOWN_GRACE_S)

    query = (
        "mutation { podStop(input: { podId: \""
        + pod_id
        + "\" }) { id desiredStatus } }"
    )
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        if payload.get("errors"):
            log.error("auto-stop GraphQL errors: %s", payload["errors"])
            return False
        log.info("auto-stop response: %s", payload.get("data"))
        return True
    except Exception as e:
        log.error("auto-stop request failed: %s", e)
        return False
