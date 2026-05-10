"""VGGT pose estimation — CVPR 2025 Best Paper (facebookresearch/vggt).

Feed-forward transformer: single pass over all frames → camera poses + dense
depth maps + 3D point maps in 30–90 s on an L40S/A100. Output is written in
COLMAP binary format so the rest of the pipeline is unchanged.

Install on the GPU worker:
    pip install git+https://github.com/facebookresearch/vggt.git

Weights download automatically on first run (HuggingFace: facebook/vggt-1B)
or point NUDORMS_VGGT_MODEL at a local checkpoint.

Env:
    NUDORMS_VGGT_MODEL  HF hub ID or local path (default: "facebook/vggt-1B")
    NUDORMS_VGGT_CHUNK  max frames per forward pass (default: 200, VRAM-limited)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from ..types import StageResult
from .glomap import _read_sparse_metrics
from .mast3r import _quat_wxyz_from_R, _write_colmap_binary

log = logging.getLogger("nudorms.poses.vggt")

VGGT_MODEL = os.environ.get("NUDORMS_VGGT_MODEL", "facebook/vggt-1B")
# IMPORTANT: VGGT outputs poses in a coordinate frame relative to the first
# frame *of each forward pass*. Splitting frames across multiple chunks gives
# poses in different coordinate systems that can't be concatenated correctly.
# So CHUNK_SIZE must be >= total frame count. Default 400 covers any
# reasonable casual capture; if VRAM is the bottleneck, use a smaller VGGT
# variant (vggt-base) instead of chunking.
CHUNK_SIZE = int(os.environ.get("NUDORMS_VGGT_CHUNK", "400"))
MAX_POINTS_PER_FRAME = 2000   # subsample dense point map to keep sparse model compact


def _load_model(device: str):
    """Load VGGT in bf16 to halve weight VRAM (~5 GB → ~2.5 GB on 1B model).

    Activations still flow through the autocast(bfloat16) block in run(), so
    the whole forward pass stays in bf16 — needed to fit 100+ frames on a
    16 GB GPU like the RTX A4000.
    """
    import torch
    from vggt.models.vggt import VGGT
    log.info("loading VGGT from %s", VGGT_MODEL)
    if os.path.isfile(VGGT_MODEL):
        model = VGGT()
        model.load_state_dict(torch.load(VGGT_MODEL, map_location="cpu"))
    else:
        model = VGGT.from_pretrained(VGGT_MODEL)
    return model.to(device, dtype=torch.bfloat16).eval()


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from vggt.models.vggt import VGGT  # noqa: F401 — import check
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as e:
        return StageResult(False, {}, {}, failure_reason=f"vggt not importable: {e}")

    frames = sorted(frames_dir.glob("*.jpg"))
    total = len(frames)
    if total < 3:
        return StageResult(False, {}, {}, failure_reason=f"need >=3 frames, got {total}")

    # VGGT outputs per-chunk coordinate systems. If we'd need to chunk,
    # bail explicitly so the ensemble falls through to MASt3R rather than
    # produce silently broken (chunk-stitched) poses.
    if total > CHUNK_SIZE:
        return StageResult(
            False, {}, {},
            failure_reason=(
                f"VGGT requires single-pass: {total} frames > CHUNK_SIZE={CHUNK_SIZE}. "
                "Raise NUDORMS_VGGT_CHUNK or pre-subsample frames."
            ),
        )

    image_paths = [str(p) for p in frames]

    try:
        model = _load_model("cuda")
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"VGGT model load failed: {e}")

    # Read original sizes for intrinsic rescaling (VGGT resizes to 224×224 internally)
    from PIL import Image
    orig_sizes: list[tuple[int, int]] = []
    for p in frames:
        with Image.open(p) as im:
            orig_sizes.append(im.size)  # (W, H)

    # VGGT processes frames in one pass (chunked if > CHUNK_SIZE to avoid OOM)
    all_extrinsics: list[np.ndarray] = []  # c2w [4, 4]
    all_intrinsics: list[np.ndarray] = []  # [3, 3] at inference resolution
    all_pts3d: list[np.ndarray] = []       # [H*W, 3] world points (subsampled)
    all_colors: list[np.ndarray] = []      # [H*W, 3] uint8 RGB

    try:
        # bf16 cast halves activation memory and matches the bf16 model weights.
        imgs = load_and_preprocess_images(image_paths).to("cuda", dtype=torch.bfloat16)
        infer_H, infer_W = imgs.shape[-2], imgs.shape[-1]
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"image preprocessing failed: {e}")

    # Process in chunks to respect VRAM limits
    chunk_extrinsics = []
    chunk_intrinsics = []
    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk = imgs[start:end]
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    preds = model(chunk)
            ext, intr = pose_encoding_to_extri_intri(
                preds["pose_enc"], chunk.shape[-2:]
            )
            # ext: [n, 3, 4] or [n, 4, 4] — cam_from_world (w2c)
            # intr: [n, 3, 3] at inference resolution
            chunk_extrinsics.append(ext.float().cpu().numpy())
            chunk_intrinsics.append(intr.float().cpu().numpy())

            # Extract world points from depth/point maps for sparse model
            if "world_points" in preds:
                wpts = preds["world_points"].float().cpu().numpy()  # [n, H, W, 3]
            elif "depth" in preds and "intrinsics" in preds:
                # Back-project depth using predicted intrinsics
                wpts = None
            else:
                wpts = None

            if wpts is not None:
                for i in range(wpts.shape[0]):
                    pts_flat = wpts[i].reshape(-1, 3)  # [H*W, 3]
                    # Load corresponding image for colors
                    frame_idx = start + i
                    img_np = (chunk[i].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    cols_flat = img_np.reshape(-1, 3)

                    # Subsample to keep sparse model manageable
                    if pts_flat.shape[0] > MAX_POINTS_PER_FRAME:
                        sel = np.random.default_rng(frame_idx).choice(
                            pts_flat.shape[0], MAX_POINTS_PER_FRAME, replace=False
                        )
                        pts_flat = pts_flat[sel]
                        cols_flat = cols_flat[sel]

                    # Filter points behind camera (z <= 0 in camera space)
                    ext_i = chunk_extrinsics[-1][i]  # [3, 4] or [4, 4]
                    R = ext_i[:3, :3]
                    t = ext_i[:3, 3]
                    cam_z = (pts_flat @ R.T + t)[..., 2]
                    valid = cam_z > 1e-3
                    all_pts3d.append(pts_flat[valid])
                    all_colors.append(cols_flat[valid])

        except Exception as e:
            return StageResult(False, {}, {}, failure_reason=f"VGGT inference failed on chunk {start}–{end}: {e}")

    extrinsics = np.concatenate(chunk_extrinsics, axis=0)  # [N, 3|4, 4]
    intrinsics = np.concatenate(chunk_intrinsics, axis=0)  # [N, 3, 3]

    if extrinsics.shape[0] != total:
        log.warning("VGGT returned %d poses for %d frames", extrinsics.shape[0], total)

    n_imgs = extrinsics.shape[0]

    # Build COLMAP camera entry (single shared camera, rescaled to original size)
    W_orig, H_orig = orig_sizes[0]
    sx = W_orig / infer_W
    sy = H_orig / infer_H
    K0 = intrinsics[0]
    cam_entry = {
        "id": 1, "model_id": 1,  # PINHOLE
        "w": W_orig, "h": H_orig,
        "params": [K0[0, 0] * sx, K0[1, 1] * sy, K0[0, 2] * sx, K0[1, 2] * sy],
    }

    # Build image entries — extrinsics are w2c [3, 4] or [4, 4]
    img_entries = []
    for i in range(n_imgs):
        ext = extrinsics[i]  # [3, 4] or [4, 4]
        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        img_entries.append({
            "id": i + 1,
            "name": Path(image_paths[i]).name,
            "qvec": _quat_wxyz_from_R(R_w2c),
            "tvec": t_w2c.astype(np.float64),
            "cam_id": 1,
        })

    # Build point entries from world point maps
    pt_entries = []
    for pts, cols in zip(all_pts3d, all_colors):
        for m in range(pts.shape[0]):
            pt_entries.append({
                "xyz": pts[m].astype(np.float64),
                "rgb": np.clip(cols[m], 0, 255).astype(np.uint8),
            })

    sparse_0 = out_dir / "sparse" / "0"
    sparse_0.mkdir(parents=True, exist_ok=True)
    _write_colmap_binary(sparse_0, cam_entry, img_entries, pt_entries)

    if not (sparse_0 / "cameras.bin").exists():
        return StageResult(False, {}, {}, failure_reason="VGGT: binary write produced no cameras.bin")

    metrics = _read_sparse_metrics(sparse_0, total)
    log.info("VGGT done: %d/%d registered, %d points", metrics.get("registered_images", 0), total, len(pt_entries))
    return StageResult(ok=True, metrics=metrics, artifacts={"sparse_dir": str(sparse_0)})
