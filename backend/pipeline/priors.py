"""Geometric prior extraction — Stage 2.

Runs BEFORE training on every selected keyframe:
  1. Depth Anything V2 (Small) — monocular metric-relative depth
  2. Gradient-based surface normals computed from the depth map

Outputs live in workdir/priors/ as .npy files keyed by frame name:
    {frame_name}_depth.npy   float32 [H, W]  — depth (relative, not metric)
    {frame_name}_normal.npy  float32 [H, W, 3] — unit normals in camera space

Training uses these as weak supervision. If this stage fails (model not
installed, OOM), the pipeline continues with depth/normal weights forced to 0.

GPU deps:
    transformers>=4.40, torch
    Model: depth-anything/Depth-Anything-V2-Small-hf (auto-downloaded from HF)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .types import StageResult

log = logging.getLogger("nudorms.pipeline.priors")

# HuggingFace model ID — Small for VRAM efficiency; can override with Large
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
BATCH_SIZE = 8   # frames processed per forward pass


def _depth_to_normals(depth: np.ndarray) -> np.ndarray:
    """Approximate surface normals from depth-map gradients.

    Uses central differences for interior pixels, forward/backward at borders.
    Normal = normalise([-∂D/∂x, -∂D/∂y, 1]) in image-plane camera space.
    Accurate enough for planar surfaces (walls, floor, ceiling) which are
    the dominant failure mode on dorm rooms.
    """
    dz_dx = np.gradient(depth, axis=1)
    dz_dy = np.gradient(depth, axis=0)
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / (norms + 1e-8)).astype(np.float32)


def run(scan_id: str, workdir: Path, frame_artifacts: dict) -> StageResult:
    frames_dir = Path(frame_artifacts["frames_dir"])
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return StageResult(False, {}, {}, failure_reason="no frames in frames_dir")

    priors_dir = workdir / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as e:
        return StageResult(False, {}, {}, failure_reason=f"depth prior deps missing: {e}")

    try:
        log.info("loading Depth Anything V2 (%s)", DEPTH_MODEL)
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_model = depth_model.to(device).eval()
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"depth model load failed: {e}")

    processed = 0
    errors = 0

    import torch.nn.functional as F

    for i in range(0, len(frames), BATCH_SIZE):
        batch_paths = frames[i: i + BATCH_SIZE]
        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            orig_sizes = [img.size for img in images]  # (W, H)

            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = depth_model(**inputs)
            # predicted_depth: [B, H_infer, W_infer]
            pred_depth = outputs.predicted_depth

            for j, (path, (W, H)) in enumerate(zip(batch_paths, orig_sizes)):
                # Upsample back to original frame resolution
                depth_up = F.interpolate(
                    pred_depth[j].unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy().astype(np.float32)

                normals = _depth_to_normals(depth_up)

                stem = path.stem
                np.save(priors_dir / f"{stem}_depth.npy", depth_up)
                np.save(priors_dir / f"{stem}_normal.npy", normals)
                processed += 1

        except Exception as e:
            log.warning("priors batch %d failed: %s", i, e)
            errors += len(batch_paths)

    if processed == 0:
        return StageResult(False, {"errors": errors}, {},
                           failure_reason="depth prior failed for all frames")

    log.info("priors: %d/%d frames processed", processed, len(frames))
    return StageResult(
        ok=True,
        metrics={"frames_processed": processed, "frames_failed": errors},
        artifacts={"priors_dir": str(priors_dir)},
    )
