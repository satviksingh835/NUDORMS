"""MASt3R-SfM pose estimation — direct Python API.

Uses the mast3r Python package in-process (no subprocess shim). Must be
installed into the worker environment; bootstrap.sh handles this on RunPod.

Pipeline:
  1. Load model (cached after first call — stays in VRAM)
  2. Build kapture scene from frames_dir (single shared camera for video)
  3. Generate sliding-window image pairs (window=WIN_SIZE)
  4. Run MASt3R pairwise inference → export matches to COLMAP db
  5. Geometric verification (pycolmap.verify_matches)
  6. glomap mapper → reconstruction/0/{cameras,images,points3D}.bin

Configure via env vars:
  NUDORMS_MAST3R_MODEL  path to .pth weights (default /workspace/mast3r_models/MASt3R.pth)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional
import os

from ..types import StageResult
from .glomap import _read_sparse_metrics

log = logging.getLogger("nudorms.poses.mast3r")

MAST3R_MODEL = os.environ.get("NUDORMS_MAST3R_MODEL", "/workspace/mast3r_models/MASt3R.pth")
WIN_SIZE = 20  # each frame matches its WIN_SIZE temporal neighbors

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


def _make_pairs(image_paths: list, win_size: int) -> list:
    """Sliding-window pairs: frame i matches i+1 .. i+win_size."""
    pairs = []
    n = len(image_paths)
    for i in range(n):
        for j in range(i + 1, min(i + win_size + 1, n)):
            pairs.append((image_paths[i], image_paths[j]))
    return pairs


def run(frames_dir: Path, out_dir: Path) -> StageResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(MAST3R_MODEL).exists():
        return StageResult(False, {}, {},
                           failure_reason=f"MASt3R weights not found: {MAST3R_MODEL}")
    if not shutil.which("glomap"):
        return StageResult(False, {}, {}, failure_reason="glomap binary not on PATH")

    try:
        from mast3r.colmap.mapping import (
            kapture_import_image_folder_or_list,
            run_mast3r_matching,
            glomap_run_mapper,
        )
        from kapture.converter.colmap.database_extra import kapture_to_colmap
        from kapture.converter.colmap.database import COLMAPDatabase
        import pycolmap
    except ImportError as e:
        return StageResult(False, {}, {},
                           failure_reason=f"mast3r/kapture not importable: {e}")

    frames = sorted(frames_dir.glob("*.jpg"))
    total_frames = len(frames)
    if total_frames == 0:
        return StageResult(False, {}, {}, failure_reason="no frames in frames_dir")

    try:
        model = _load_model("cuda")
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"MASt3R model load failed: {e}")

    maxdim = max(model.patch_embed.img_size)
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]

    kdata = kapture_import_image_folder_or_list(str(frames_dir), use_single_camera=True)
    image_paths = kdata.records_camera.data_list()
    pairs = _make_pairs(image_paths, WIN_SIZE)
    if not pairs:
        return StageResult(False, {}, {}, failure_reason="not enough frames to form pairs")

    db_path = out_dir / "colmap.db"
    recon_path = out_dir / "reconstruction"
    pairs_txt = out_dir / "pairs.txt"
    recon_path.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    colmap_db = COLMAPDatabase.connect(str(db_path))
    try:
        # None as kapture root: safe because keypoints_type=None so nothing is
        # read from disk; kapture_to_colmap only needs kdata in-memory here.
        kapture_to_colmap(kdata, None, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None,
                          export_two_view_geometry=False)
        colmap_image_pairs = run_mast3r_matching(
            model, maxdim, patch_size, "cuda",
            kdata, str(frames_dir), pairs, colmap_db,
            False, 0, 1.001, False, 5,
        )
    except Exception as e:
        colmap_db.close()
        return StageResult(False, {}, {}, failure_reason=f"MASt3R matching failed: {e}")

    colmap_db.close()

    if not colmap_image_pairs:
        return StageResult(False, {}, {}, failure_reason="MASt3R produced no matching pairs")

    with open(pairs_txt, "w") as f:
        for p1, p2 in colmap_image_pairs:
            f.write(f"{p1} {p2}\n")
    pycolmap.verify_matches(str(db_path), str(pairs_txt))

    try:
        glomap_run_mapper("glomap", str(db_path), str(recon_path), str(frames_dir))
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"glomap mapper failed: {e}")

    # glomap writes the sparse model to reconstruction/0/
    sparse_0 = recon_path / "0"
    if not (sparse_0 / "cameras.bin").exists():
        return StageResult(False, {}, {},
                           failure_reason="glomap produced no sparse model")

    metrics = _read_sparse_metrics(sparse_0, total_frames)
    return StageResult(
        ok=True,
        metrics=metrics,
        artifacts={"sparse_dir": str(sparse_0)},
    )
