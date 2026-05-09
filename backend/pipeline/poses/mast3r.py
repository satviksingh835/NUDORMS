"""MASt3R pose estimation via Sparse Global Alignment.

We previously routed MASt3R pairwise matches into COLMAP's incremental
mapper. That preserved the very thing the strategic shift was meant to
escape: the COLMAP mapper is the brittle stage that drops frames on
casual capture (test6: 7/300 frames, test5: 175/258 with MASt3R matches).

This implementation skips the COLMAP mapper entirely. It runs MASt3R's
own Sparse Global Alignment (cloud_opt.sparse_ga), which optimizes per-
frame poses and a sparse 3D point cloud directly from MASt3R's pairwise
predictions. The output is repacked into a COLMAP sparse model so the
gsplat trainer downstream consumes it unchanged.

Pipeline:
  1. Load frames (dust3r load_images, long edge 512, patch-aligned crop)
  2. Sliding-window pairs (each frame paired with WIN_SIZE neighbors)
  3. sparse_global_alignment → SparseGA with cam2w, intrinsics, pts3d
  4. Scale intrinsics from MASt3R's true_shape back to original frame size
  5. Write cameras.bin / images.bin / points3D.bin via pycolmap

Config (env):
  NUDORMS_MAST3R_MODEL  path to .pth weights
  NUDORMS_MAST3R_WIN    sliding-window size (default 20)
  NUDORMS_MAST3R_SIZE   long-edge size used during inference (default 512)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import StageResult
from .glomap import _read_sparse_metrics

log = logging.getLogger("nudorms.poses.mast3r")

MAST3R_MODEL = os.environ.get(
    "NUDORMS_MAST3R_MODEL", "/workspace/mast3r_models/MASt3R.pth"
)
WIN_SIZE = int(os.environ.get("NUDORMS_MAST3R_WIN", "20"))
INFER_SIZE = int(os.environ.get("NUDORMS_MAST3R_SIZE", "512"))
# Anchor density for the SGA optimizer. Lower = more anchors = more VRAM.
# Upstream default is 8; on a 24 GB GPU we OOM around ~200 frames at 8,
# so we ship 16 by default and let users tighten it on bigger GPUs.
SUBSAMPLE = int(os.environ.get("NUDORMS_MAST3R_SUBSAMPLE", "16"))

# SGA writes per-pair torch.save zips and reads them back across the run.
# On RunPod, /workspace is MooseFS (network FS) which races against torch's
# zip reopen and corrupts files mid-run. Keep the cache on local disk by
# default; override to point at fast SSD if local disk is tight.
CACHE_ROOT = os.environ.get("NUDORMS_MAST3R_CACHE", "/tmp/nudorms_mast3r_cache")

# Cap anchors written per frame so a 300-frame scan doesn't produce a
# multi-million-point sparse model that bloats gsplat init.
MAX_ANCHORS_PER_FRAME = 4000

_model_cache: Optional[object] = None


def _load_model(device: str):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    from mast3r.model import AsymmetricMASt3R
    log.info("loading MASt3R weights from %s", MAST3R_MODEL)
    _model_cache = AsymmetricMASt3R.from_pretrained(MAST3R_MODEL).to(device)
    _model_cache.eval()
    return _model_cache


def _quat_xyzw_from_R(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> quaternion in [x, y, z, w] (pycolmap convention)."""
    from scipy.spatial.transform import Rotation as ScipyR
    return ScipyR.from_matrix(R).as_quat().astype(np.float64)


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(MAST3R_MODEL).exists():
        return StageResult(False, {}, {},
                           failure_reason=f"MASt3R weights not found: {MAST3R_MODEL}")

    try:
        import torch  # noqa: F401
        import pycolmap
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from mast3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
    except ImportError as e:
        return StageResult(False, {}, {},
                           failure_reason=f"mast3r/dust3r/pycolmap not importable: {e}")

    frames = sorted(frames_dir.glob("*.jpg"))
    total_frames = len(frames)
    if total_frames < 3:
        return StageResult(False, {}, {},
                           failure_reason=f"need >=3 frames, got {total_frames}")

    try:
        model = _load_model("cuda")
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"MASt3R model load failed: {e}")

    # Read original frame sizes once — needed to rescale MASt3R-space
    # intrinsics into the original-image coordinate system that gsplat
    # will read from disk.
    from PIL import Image
    orig_sizes: list[tuple[int, int]] = []  # (W, H) per frame
    for p in frames:
        with Image.open(p) as im:
            orig_sizes.append(im.size)

    # dust3r load_images returns dicts with 'true_shape' = [[H, W]] at the
    # resized resolution. The order matches sorted(frames_dir.glob('*.jpg'))
    # because load_images sorts internally.
    image_paths = [str(p) for p in frames]
    try:
        imgs_dicts = load_images(image_paths, size=INFER_SIZE, verbose=False)
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"load_images failed: {e}")

    win = min(WIN_SIZE, total_frames - 1)
    pairs = make_pairs(imgs_dicts, scene_graph=f"swin-{win}-noncyclic",
                       prefilter=None, symmetrize=True)
    if not pairs:
        return StageResult(False, {}, {}, failure_reason="no pairs formed")

    # Cache lives on local disk (NUDORMS_MAST3R_CACHE), keyed by an
    # out_dir-derived name so concurrent runs don't collide. We deliberately
    # do NOT put this under out_dir on /workspace — see the comment on
    # CACHE_ROOT for the network-FS corruption story.
    cache_dir = Path(CACHE_ROOT) / f"sga_{out_dir.name}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene = sparse_global_alignment(
            image_paths, pairs, str(cache_dir), model,
            shared_intrinsics=True,  # one phone, one camera — share intrinsics
            subsample=SUBSAMPLE,
            device="cuda",
        )
    except Exception as e:
        return StageResult(False, {}, {},
                           failure_reason=f"sparse_global_alignment failed: {e}")

    # Pull tensors off GPU once; subsequent ops are numpy.
    cam2w_all = scene.get_im_poses().detach().cpu().numpy()           # [N, 4, 4]
    intrinsics_all = [K.detach().cpu().numpy() for K in scene.intrinsics]  # list of [3,3]
    pts3d_all = [p.detach().cpu().numpy() for p in scene.get_sparse_pts3d()]
    colors_all = scene.get_pts3d_colors()  # list of [Mi, 3] in [0,1]
    # imgs returned by SGA are at resized resolution; use true_shape from loaded dicts
    true_shapes = [d["true_shape"][0] for d in imgs_dicts]  # each [H, W]

    n_imgs = cam2w_all.shape[0]
    if n_imgs != total_frames:
        log.warning("SGA returned %d poses for %d frames", n_imgs, total_frames)

    # ---- Build COLMAP sparse model ----
    rec = pycolmap.Reconstruction()

    # Single shared camera (PINHOLE: fx, fy, cx, cy) at original resolution.
    # SGA writes the same K into every intrinsics[i] when shared_intrinsics=True,
    # so any index works.
    K_resized = intrinsics_all[0]
    H_res, W_res = int(true_shapes[0][0]), int(true_shapes[0][1])
    # All originals are same-resolution video frames; use the first.
    W_orig, H_orig = orig_sizes[0]
    sx = W_orig / W_res
    sy = H_orig / H_res
    fx = float(K_resized[0, 0]) * sx
    fy = float(K_resized[1, 1]) * sy
    cx = float(K_resized[0, 2]) * sx
    cy = float(K_resized[1, 2]) * sy

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=W_orig,
        height=H_orig,
        params=[fx, fy, cx, cy],
        camera_id=1,
    )
    rec.add_camera(camera)

    # Per-image extrinsics. cam_from_world = inv(cam2w).
    image_points2D: dict[int, list[tuple[float, float, int]]] = {}
    for i in range(n_imgs):
        cam2w = cam2w_all[i]
        w2c = np.linalg.inv(cam2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        rotation = pycolmap.Rotation3d(_quat_xyzw_from_R(R))
        rigid = pycolmap.Rigid3d(rotation=rotation, translation=t.astype(np.float64))

        image = pycolmap.Image(
            image_id=i + 1,
            name=Path(image_paths[i]).name,
            camera_id=1,
        )
        image.cam_from_world = rigid
        rec.add_image(image)
        image_points2D[i + 1] = []

    # Project per-image sparse anchors into 2D, register points3D with
    # length-1 tracks. Tracks are short because SGA's per-image anchors
    # don't come pre-linked across views — gsplat's init only needs the
    # 3D positions + colors, so this is sufficient for downstream.
    for i in range(n_imgs):
        pts = pts3d_all[i]
        cols = colors_all[i]
        if pts.shape[0] == 0:
            continue
        if pts.shape[0] > MAX_ANCHORS_PER_FRAME:
            sel = np.random.default_rng(i).choice(pts.shape[0],
                                                   MAX_ANCHORS_PER_FRAME, replace=False)
            pts = pts[sel]
            cols = cols[sel]

        # Project into image i (using the *resized* intrinsics K_resized
        # because pts2d at true_shape are easy; then scale to original).
        cam2w = cam2w_all[i]
        w2c = np.linalg.inv(cam2w)
        cam_pts = (w2c[:3, :3] @ pts.T + w2c[:3, 3:4]).T   # [Mi, 3]
        z = cam_pts[:, 2]
        valid = z > 1e-3
        if not valid.any():
            continue
        cam_pts = cam_pts[valid]
        pts = pts[valid]
        cols = cols[valid]

        K = intrinsics_all[i]
        proj = (K @ cam_pts.T).T
        uv_resized = proj[:, :2] / proj[:, 2:3]            # in true_shape px
        uv = uv_resized * np.array([sx, sy])               # in original px

        # Filter to those falling within original frame bounds (tiny margin).
        in_frame = ((uv[:, 0] >= 0) & (uv[:, 0] < W_orig) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < H_orig))
        if not in_frame.any():
            continue
        uv = uv[in_frame]
        pts = pts[in_frame]
        cols = cols[in_frame]

        rgb = np.clip(np.asarray(cols) * 255.0, 0, 255).astype(np.uint8)

        for m in range(pts.shape[0]):
            track = pycolmap.Track()
            point2D_idx = len(image_points2D[i + 1])
            track.add_element(
                pycolmap.TrackElement(image_id=i + 1, point2D_idx=point2D_idx)
            )
            p3d_id = rec.add_point3D(
                xyz=pts[m].astype(np.float64),
                track=track,
                color=rgb[m],
            )
            image_points2D[i + 1].append((float(uv[m, 0]), float(uv[m, 1]), p3d_id))

    # Attach points2D to each image and register the image as posed.
    for i in range(n_imgs):
        pts_list = image_points2D.get(i + 1, [])
        img = rec.images[i + 1]
        if pts_list:
            xys = np.array([[u, v] for (u, v, _) in pts_list], dtype=np.float64)
            p3d_ids = np.array([pid for (_, _, pid) in pts_list], dtype=np.int64)
            img.points2D = pycolmap.ListPoint2D([
                pycolmap.Point2D(xy=xys[k], point3D_id=int(p3d_ids[k]))
                for k in range(len(pts_list))
            ])
        rec.register_image(i + 1)

    # Write cameras.bin / images.bin / points3D.bin to sparse/0/.
    sparse_0 = out_dir / "sparse" / "0"
    sparse_0.mkdir(parents=True, exist_ok=True)
    try:
        rec.write_binary(str(sparse_0))
    except AttributeError:
        # Older pycolmap exposes this as `write` without _binary suffix.
        rec.write(str(sparse_0))

    if not (sparse_0 / "cameras.bin").exists():
        return StageResult(False, {}, {},
                           failure_reason="pycolmap wrote no sparse model")

    metrics = _read_sparse_metrics(sparse_0, total_frames)
    return StageResult(
        ok=True,
        metrics=metrics,
        artifacts={"sparse_dir": str(sparse_0)},
    )
