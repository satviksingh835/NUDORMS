"""Held-out evaluation: PSNR / SSIM / LPIPS + per-region diagnostics.

Renders the trained splat at every holdout camera pose, compares rendered
frames against the actual held-out images, and computes aggregate metrics.

Per-region diagnostics: each rendered image is divided into a 4×4 grid.
The cell with the worst PSNR is tagged in region_diagnostics — this tells
the user which part of the room to recapture (e.g. "NE corner, eye height").

Metrics:
  psnr  — average PSNR over holdout views (primary gate; threshold 28 dB)
  ssim  — average SSIM
  lpips — average LPIPS (lower is better; requires the lpips package)

GPU deps:
  torch, gsplat>=1.4, Pillow, numpy
  Optional: lpips (pip install lpips)
"""
from __future__ import annotations

import json
import logging
import math
import struct
from pathlib import Path

import numpy as np

from ..types import StageResult

log = logging.getLogger("nudorms.train.eval")

GRID_ROWS = 4
GRID_COLS = 4


# ---------------------------------------------------------------------------
# PLY reader (fast — only reads means, scales, quats, opacities, sh0)
# ---------------------------------------------------------------------------

def _load_splat_params(ply_path: Path, device):
    """Load a 3DGS PLY into torch tensors for rasterization."""
    import torch
    import torch.nn.functional as F

    with open(ply_path, "rb") as f:
        raw = f.read()

    end_tag = b"end_header\n"
    header_end = raw.index(end_tag) + len(end_tag)
    header = raw[:header_end].decode("latin-1")

    props, n_verts, in_vertex = [], 0, False
    for line in header.split("\n"):
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element") and in_vertex:
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            props.append(line.split()[-1])

    data = np.frombuffer(raw[header_end:], dtype=np.float32).reshape(n_verts, len(props))
    p = {name: i for i, name in enumerate(props)}

    def _t(*names):
        for n in names:
            if n in p:
                return torch.from_numpy(data[:, p[n]].copy()).to(device)
        return None

    means    = torch.stack([_t("x"), _t("y"), _t("z")], dim=-1)
    scales   = torch.stack([_t("scale_0"), _t("scale_1"), _t("scale_2")], dim=-1)
    quats    = torch.stack([_t("rot_0"), _t("rot_1"), _t("rot_2"), _t("rot_3")], dim=-1)
    opacities = _t("opacity")

    # SH DC coefficients → treat as constant colour for eval (sh_degree=0)
    dc0, dc1, dc2 = _t("f_dc_0"), _t("f_dc_1"), _t("f_dc_2")
    sh0 = torch.stack([dc0, dc1, dc2], dim=-1).unsqueeze(1)  # [N, 1, 3]

    # f_rest_* (all SH rest coefficients)
    rest_cols = sorted([k for k in p if k.startswith("f_rest_")],
                       key=lambda k: int(k.split("_")[-1]))
    if rest_cols:
        shN = torch.stack([_t(c) for c in rest_cols], dim=-1).reshape(n_verts, -1, 3)
    else:
        shN = torch.zeros(n_verts, 0, 3, device=device)

    return {
        "means": means,
        "scales": scales,
        "quats": F.normalize(quats, dim=-1),
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR in dB. pred/gt: float32 [H, W, 3] in [0, 1]."""
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Quick SSIM approximation (11×11 window, no torch needed)."""
    from scipy.ndimage import uniform_filter
    C1, C2 = 0.01**2, 0.03**2
    mu1 = uniform_filter(pred, size=11, axes=(0, 1))
    mu2 = uniform_filter(gt,   size=11, axes=(0, 1))
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    s1 = uniform_filter(pred**2, size=11, axes=(0, 1)) - mu1_sq
    s2 = uniform_filter(gt**2,   size=11, axes=(0, 1)) - mu2_sq
    s12 = uniform_filter(pred * gt, size=11, axes=(0, 1)) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-8)))


def _try_lpips(pred_np: np.ndarray, gt_np: np.ndarray, device) -> float | None:
    """LPIPS/VGG if available, else None. pred/gt: [H, W, 3] float32 in [0,1]."""
    try:
        import lpips, torch
        if not hasattr(_try_lpips, "_fn"):
            _try_lpips._fn = lpips.LPIPS(net="vgg").to(device)
            _try_lpips._fn.eval()
        fn = _try_lpips._fn
        to_t = lambda a: torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            return float(fn(to_t(pred_np), to_t(gt_np)).mean().cpu())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Region diagnostics (4×4 grid PSNR map)
# ---------------------------------------------------------------------------

def _region_diagnostics(pred: np.ndarray, gt: np.ndarray) -> dict:
    H, W, _ = pred.shape
    cell_h, cell_w = H // GRID_ROWS, W // GRID_COLS
    grid = {}
    worst_psnr, worst_cell = float("inf"), None
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell_pred = pred[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            cell_gt   = gt[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            p = _psnr(cell_pred, cell_gt)
            key = f"r{r}c{c}"
            grid[key] = round(p, 2)
            if p < worst_psnr:
                worst_psnr = p
                worst_cell = key
    return {"grid_psnr": grid, "worst_cell": worst_cell, "worst_psnr": round(worst_psnr, 2)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scan_id: str, workdir: Path, train_artifacts: dict, pose_artifacts: dict) -> StageResult:
    ply_path = Path(train_artifacts.get("ply_path", ""))
    cameras_json = Path(train_artifacts.get("cameras_json", ""))

    if not ply_path.exists():
        return StageResult(False, {}, {}, failure_reason=f"PLY not found: {ply_path}")
    if not cameras_json.exists():
        return StageResult(False, {}, {}, failure_reason=f"cameras.json not found: {cameras_json}")

    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image as PILImage
        from gsplat import rasterization
    except ImportError as e:
        return StageResult(False, {}, {}, failure_reason=f"GPU deps missing: {e}")

    try:
        from scipy.ndimage import uniform_filter  # noqa: F401 (validate scipy available)
    except ImportError:
        return StageResult(False, {}, {}, failure_reason="scipy missing (pip install scipy)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cameras
    cam_list = json.loads(cameras_json.read_text())
    holdout = [c for c in cam_list if c.get("split") == "holdout"]
    if not holdout:
        log.warning("no holdout cameras in cameras.json — falling back to all cameras")
        holdout = cam_list[:max(1, len(cam_list) // 10)]

    log.info("eval: rendering %d holdout views from %s", len(holdout), ply_path.name)

    # Load splat
    try:
        splat = _load_splat_params(ply_path, device)
    except Exception as e:
        return StageResult(False, {}, {}, failure_reason=f"PLY load error: {e}")

    sh_feats = torch.cat([splat["sh0"], splat["shN"]], dim=1)
    scales_exp = torch.exp(splat["scales"])
    ops_sig    = torch.sigmoid(splat["opacities"])

    frames_dir = workdir / "frames"

    psnrs, ssims, lpips_vals = [], [], []
    region_diag_worst: dict = {}

    for cam_info in holdout:
        frame_path = frames_dir / cam_info["name"]
        if not frame_path.exists():
            # Search workdir recursively
            candidates = list(workdir.rglob(cam_info["name"]))
            if not candidates:
                log.warning("holdout frame not found: %s — skipping", cam_info["name"])
                continue
            frame_path = candidates[0]

        W, H = cam_info["W"], cam_info["H"]

        # Load ground-truth frame
        gt_img = PILImage.open(frame_path).convert("RGB").resize((W, H), PILImage.BILINEAR)
        gt_np = np.array(gt_img, dtype=np.float32) / 255.0

        w2c = torch.tensor(cam_info["w2c"], dtype=torch.float32, device=device).unsqueeze(0)
        K   = torch.tensor(cam_info["K"],   dtype=torch.float32, device=device).unsqueeze(0)

        try:
            with torch.no_grad():
                renders, _, _ = rasterization(
                    means=splat["means"],
                    quats=splat["quats"],
                    scales=scales_exp,
                    opacities=ops_sig,
                    colors=sh_feats,
                    viewmats=w2c,
                    Ks=K,
                    width=W,
                    height=H,
                    render_mode="RGB",
                    sh_degree=0,   # use DC only for speed
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    rasterize_mode="antialiased",
                )
        except Exception as e:
            log.warning("rasterization failed for %s: %s", cam_info["name"], e)
            continue

        pred_np = renders[0].clamp(0, 1).cpu().numpy().astype(np.float32)

        p = _psnr(pred_np, gt_np)
        s = _ssim(pred_np, gt_np)
        l = _try_lpips(pred_np, gt_np, device)

        psnrs.append(p)
        ssims.append(s)
        if l is not None:
            lpips_vals.append(l)

        # Track worst-PSNR region for user feedback
        if not region_diag_worst or p < region_diag_worst.get("worst_psnr", float("inf")):
            region_diag_worst = _region_diagnostics(pred_np, gt_np)
            region_diag_worst["frame"] = cam_info["name"]

    if not psnrs:
        return StageResult(False, {}, {}, failure_reason="no holdout frames could be rendered")

    avg_psnr = float(np.mean(psnrs))
    avg_ssim = float(np.mean(ssims))
    avg_lpips = float(np.mean(lpips_vals)) if lpips_vals else None

    log.info("eval: PSNR=%.2f  SSIM=%.4f  LPIPS=%s  (%d views)",
             avg_psnr, avg_ssim,
             f"{avg_lpips:.4f}" if avg_lpips is not None else "n/a",
             len(psnrs))

    metrics: dict = {"psnr": round(avg_psnr, 3), "ssim": round(avg_ssim, 4)}
    if avg_lpips is not None:
        metrics["lpips"] = round(avg_lpips, 4)

    return StageResult(
        ok=True,
        metrics=metrics,
        artifacts={"region_diagnostics": region_diag_worst},
    )
