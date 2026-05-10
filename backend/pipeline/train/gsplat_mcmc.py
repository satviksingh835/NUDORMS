"""3DGS trainer — Stage 1 + 2 implementation.

Stage 1 regularizers:
  - Scale regularization: penalises max/min scale ratio > 10 (needle killer).
  - Absolute scale cap: penalises max scale > 0.1 world units.
  - Mip-Splatting: rasterize_mode='antialiased' (3D smoothing + 2D Mip filter).
  - Aggressive opacity culling: cull_alpha_thresh=0.005.
  - SH degree warm-up: degree 0→3 over first 3000 iterations.

Stage 2 additions (active when prior_artifacts supplied):
  - DropGaussian (CVPR 2025): random Gaussian dropout each iteration.
    Prevents any single Gaussian from becoming a load-bearing needle.
  - Effective Rank (NeurIPS 2024): entropy of normalised squared scales.
    Pushes Gaussians toward isotropy, mathematically defeating needles.
  - Monocular depth supervision (DN-Splatter WACV 2025): scale-invariant
    L1 between rendered depth and Depth Anything V2 prior depth.
  - Normal supervision: L1 between gradient-normals from rendered depth
    and prior normals from priors.py.
  - Apple WD-R perceptual loss (apple/ml-perceptual-3dgs, 2.3× human
    preference vs L1+SSIM). Falls back to LPIPS→SSIM if not installed.

GPU deps:
    torch, gsplat>=1.4, Pillow, numpy
    Optional: perceptual-3dgs (Apple WD-R), lpips
"""
from __future__ import annotations

import json
import logging
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import StageResult

log = logging.getLogger("nudorms.train.gsplat_mcmc")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    iterations: int = 30_000
    sh_degree: int = 3
    use_mip: bool = True               # Mip-Splatting anti-aliasing
    scale_ratio_cap: float = 10.0      # penalise max/min scale ratio above this
    scale_abs_cap: float = 0.10        # penalise max scale above this (world units)
    scale_reg_weight: float = 0.01
    eff_rank_weight: float = 0.005     # Effective Rank regularization (NeurIPS 2024)
    drop_rate: float = 0.05            # DropGaussian fraction per iteration (CVPR 2025)
    mono_depth_weight: float = 0.01    # depth supervision via Depth Anything V2
    normal_weight: float = 0.05        # normal supervision via gradient normals
    perceptual_weight: float = 0.1     # WD-R / LPIPS perceptual loss (Apple 2025)
    holdout_fraction: float = 0.10
    cap_max: int = 1_000_000           # MCMCStrategy Gaussian cap
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN_factor: float = 20.0        # shN lr = lr_sh0 / factor


def config_for_attempt(attempt: int) -> TrainConfig:
    if attempt == 1:
        return TrainConfig()
    return TrainConfig(
        iterations=50_000,
        normal_weight=0.02,
        mono_depth_weight=0.02,
        scale_reg_weight=0.02,
        eff_rank_weight=0.01,
        perceptual_weight=0.1,
    )


# ---------------------------------------------------------------------------
# COLMAP binary reader
# ---------------------------------------------------------------------------

def _read_colmap_cameras(path: Path) -> dict:
    """Read cameras.bin → {cam_id: {w, h, fx, fy, cx, cy}}"""
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id = struct.unpack("<Ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            # Parameter count per model: SIMPLE_PINHOLE=1, PINHOLE=2(fy,fx)or4, OPENCV=8
            # We only write PINHOLE (model_id=1) with 4 params (fx,fy,cx,cy)
            n_params = {0: 3, 1: 4, 2: 4, 3: 8, 4: 5, 5: 8, 6: 8}.get(model_id, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {"w": int(w), "h": int(h),
                               "fx": params[0], "fy": params[1],
                               "cx": params[2], "cy": params[3]}
    return cameras


def _read_colmap_images(path: Path) -> list[dict]:
    """Read images.bin → list of {id, name, R [3×3], t [3], cam_id}"""
    images = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            # null-terminated name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)  # skip points2D (x, y, point3D_id each 8 bytes)

            # Convert quaternion [w,x,y,z] → rotation matrix (w2c)
            R = _quat_to_R(qw, qx, qy, qz)
            images.append({"id": img_id, "name": name, "R": R,
                           "t": np.array([tx, ty, tz]), "cam_id": cam_id})
    return images


def _read_colmap_points(path: Path) -> np.ndarray:
    """Read points3D.bin → float32 [N, 6] (xyz + rgb/255)."""
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                          # point3D_id (uint64)
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            f.read(8)                          # error (double)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)              # track elements
            pts.append([x, y, z, r / 255.0, g / 255.0, b / 255.0])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 6), dtype=np.float32)


def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP quaternion [w,x,y,z] → 3×3 rotation matrix."""
    n = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1 - 2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1 - 2*(qx**2+qy**2)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _ssim_loss(pred: "torch.Tensor", gt: "torch.Tensor") -> "torch.Tensor":
    """Simple window-based SSIM (11×11 Gaussian kernel), returns 1 - SSIM."""
    import torch
    import torch.nn.functional as F
    C1, C2 = 0.01**2, 0.03**2
    # pred/gt: [B, 3, H, W]
    kernel_size = 11
    sigma = 1.5
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1,1,11,11]
    kernel = kernel / kernel.sum()
    kernel = kernel.repeat(3, 1, 1, 1).to(pred.device)
    mu1 = F.conv2d(pred, kernel, padding=kernel_size//2, groups=3)
    mu2 = F.conv2d(gt, kernel, padding=kernel_size//2, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(pred**2, kernel, padding=kernel_size//2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt**2, kernel, padding=kernel_size//2, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * gt, kernel, padding=kernel_size//2, groups=3) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


def _effective_rank_loss(scales: "torch.Tensor") -> "torch.Tensor":
    """Effective Rank regularization (NeurIPS 2024).

    Effective rank = exp(H(p)) where p_i = s_i^2 / Σ s_j^2.
    Maximum = 3 (sphere), minimum = 1 (needle).
    Minimising -mean(effective_rank) pushes Gaussians toward isotropy.
    """
    import torch
    s_sq = torch.exp(scales) ** 2                                      # [N, 3]
    s_sq_norm = s_sq / (s_sq.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -torch.sum(s_sq_norm * torch.log(s_sq_norm + 1e-8), dim=-1)  # [N]
    return -torch.exp(entropy).mean()                                  # minimise


def _scale_invariant_depth_loss(
    rendered: "torch.Tensor", prior: "torch.Tensor"
) -> "torch.Tensor":
    """Scale-invariant L1: normalise both maps then compare.

    Depth Anything V2 outputs relative depth (not metric), so we can't
    compare absolute values — only structure. Normalising removes scale/shift.
    """
    import torch
    r = rendered.reshape(-1)
    p = prior.reshape(-1)
    r_n = (r - r.mean()) / (r.std() + 1e-8)
    p_n = (p - p.mean()) / (p.std() + 1e-8)
    return torch.mean(torch.abs(r_n - p_n))


def _build_perceptual_loss():
    """Build best available perceptual loss: WD-R → LPIPS → None.

    Apple WD-R (apple/ml-perceptual-3dgs, 2025): 2.3× human preference
    over L1+SSIM in a 39k-rating study. Falls back to LPIPS/VGG which
    is still much better than SSIM alone for novel-view quality.

    Returns a callable (pred, gt) → scalar with pred/gt in [0, 1] BCHW,
    or None if no perceptual library is available.
    """
    try:
        import torch
        from perceptual_3dgs import WDRLoss  # apple/ml-perceptual-3dgs
        loss_fn = WDRLoss().to(torch.device("cuda"))
        loss_fn.eval()
        log.info("perceptual loss: Apple WD-R (ml-perceptual-3dgs)")

        def _wdr(pred, gt):
            return loss_fn(pred, gt).mean()

        return _wdr
    except (ImportError, Exception):
        pass

    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net="vgg").to(torch.device("cuda"))
        loss_fn.eval()
        log.info("perceptual loss: LPIPS/VGG (Apple WD-R not available)")

        def _lpips_fn(pred, gt):
            # LPIPS expects [-1, 1]
            return loss_fn(pred * 2 - 1, gt * 2 - 1).mean()

        return _lpips_fn
    except (ImportError, Exception):
        pass

    log.warning("perceptual loss: none (install perceptual-3dgs or lpips for WD-R/LPIPS)")
    return None


def run(scan_id: str, workdir: Path, pose_artifacts: dict,
        prior_artifacts: dict | None = None, attempt: int = 1) -> StageResult:
    cfg = config_for_attempt(attempt)
    out = workdir / f"splat_attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torch.nn.functional as F
        from gsplat import rasterization
        from gsplat.strategy import MCMCStrategy
        from PIL import Image as PILImage
    except ImportError as e:
        return StageResult(False, {}, {}, failure_reason=f"GPU deps missing: {e}")

    sparse_dir = Path(pose_artifacts["sparse_dir"])
    if not (sparse_dir / "cameras.bin").exists():
        return StageResult(False, {}, {}, failure_reason="cameras.bin missing from pose_artifacts")

    # ------------------------------------------------------------------
    # Load COLMAP sparse model
    # ------------------------------------------------------------------
    log.info("loading COLMAP sparse model from %s", sparse_dir)
    cameras_db = _read_colmap_cameras(sparse_dir / "cameras.bin")
    images_list = _read_colmap_images(sparse_dir / "images.bin")
    pts_np = _read_colmap_points(sparse_dir / "points3D.bin")
    log.info("  %d cameras, %d images, %d points", len(cameras_db), len(images_list), len(pts_np))

    if len(images_list) == 0:
        return StageResult(False, {}, {}, failure_reason="no images in sparse model")

    # ------------------------------------------------------------------
    # Build camera dataset
    # ------------------------------------------------------------------
    # Find frame images — they live next to the sparse model's parent dirs
    # The frames_dir is workdir/frames/ (set by frame_select stage)
    frames_dir = workdir / "frames"

    all_cams = []
    for img_info in images_list:
        frame_path = frames_dir / img_info["name"]
        if not frame_path.exists():
            # Try alternate: the sparse_dir may be deeper, look relative
            candidates = list(workdir.rglob(img_info["name"]))
            if candidates:
                frame_path = candidates[0]
            else:
                continue

        cam = cameras_db.get(img_info["cam_id"], cameras_db.get(1))
        if cam is None:
            continue

        w2c = np.eye(4)
        w2c[:3, :3] = img_info["R"]
        w2c[:3, 3] = img_info["t"]

        K = np.array([
            [cam["fx"], 0, cam["cx"]],
            [0, cam["fy"], cam["cy"]],
            [0, 0, 1],
        ], dtype=np.float32)

        all_cams.append({
            "path": frame_path,
            "w2c": torch.from_numpy(w2c.astype(np.float32)),
            "K": torch.from_numpy(K),
            "W": cam["w"],
            "H": cam["h"],
        })

    if len(all_cams) < 3:
        return StageResult(False, {}, {}, failure_reason=f"only {len(all_cams)} usable cameras after path lookup")

    random.shuffle(all_cams)
    n_holdout = max(1, int(len(all_cams) * cfg.holdout_fraction))
    holdout_cams = all_cams[:n_holdout]
    train_cams = all_cams[n_holdout:]
    log.info("  %d train / %d holdout cameras", len(train_cams), len(holdout_cams))

    # ------------------------------------------------------------------
    # Initialise Gaussians from sparse points
    # ------------------------------------------------------------------
    device = torch.device("cuda")

    if pts_np.shape[0] > 0:
        means_init = torch.from_numpy(pts_np[:, :3]).to(device)
        rgb_init = torch.from_numpy(pts_np[:, 3:]).to(device)  # [N, 3] in [0,1]
    else:
        # Fallback: unit cube cloud
        log.warning("no sparse points — initialising random cloud in unit cube")
        means_init = (torch.rand(10_000, 3, device=device) - 0.5) * 2.0
        rgb_init = torch.ones(10_000, 3, device=device) * 0.5

    N = means_init.shape[0]

    # Estimate initial scale from nearest-neighbour distances (cheap k=3 approx)
    from torch import cdist
    with torch.no_grad():
        subsample = min(N, 5000)
        idx = torch.randperm(N)[:subsample]
        dists = cdist(means_init[idx], means_init[idx])
        dists.fill_diagonal_(float("inf"))
        nn_dist = dists.min(dim=1).values.median().clamp(min=1e-4)
    init_scale = nn_dist.item()

    # Convert RGB to SH DC coefficient (SH DC = RGB / 0.28209 - 0.5)
    sh0_init = (rgb_init / 0.28209479177387814 - 0.5).unsqueeze(1)  # [N, 1, 3]
    n_sh_rest = (cfg.sh_degree + 1) ** 2 - 1

    splats: dict[str, torch.nn.Parameter] = {
        "means":      torch.nn.Parameter(means_init),
        "scales":     torch.nn.Parameter(torch.log(torch.full((N, 3), init_scale, device=device))),
        "quats":      torch.nn.Parameter(torch.cat([torch.ones(N,1,device=device),
                                                     torch.zeros(N,3,device=device)], dim=1)),
        "opacities":  torch.nn.Parameter(torch.full((N,), -3.0, device=device)),  # sigmoid(-3)≈0.047
        "sh0":        torch.nn.Parameter(sh0_init),
        "shN":        torch.nn.Parameter(torch.zeros(N, n_sh_rest, 3, device=device)),
    }

    optimizers = {
        "means":     torch.optim.Adam([splats["means"]],     lr=cfg.lr_means,     eps=1e-15),
        "scales":    torch.optim.Adam([splats["scales"]],    lr=cfg.lr_scales,    eps=1e-15),
        "quats":     torch.optim.Adam([splats["quats"]],     lr=cfg.lr_quats,     eps=1e-15),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=cfg.lr_opacities, eps=1e-15),
        "sh0":       torch.optim.Adam([splats["sh0"]],       lr=cfg.lr_sh0,       eps=1e-15),
        "shN":       torch.optim.Adam([splats["shN"]],       lr=cfg.lr_sh0 / cfg.lr_shN_factor, eps=1e-15),
    }

    # Compute scene scale for MCMC noise amplitude (median distance from origin)
    scene_scale = float(means_init.norm(dim=-1).median().clamp(min=0.1))

    strategy = MCMCStrategy(
        cap_max=cfg.cap_max,
        noise_lr=5e-3 * scene_scale,
        refine_start_iter=500,
        refine_stop_iter=int(cfg.iterations * 0.8),
        refine_every=100,
        min_opacity=0.005,           # cull_alpha_thresh from the doc
    )
    strategy_state = strategy.initialize_state()

    # Build perceptual loss (WD-R → LPIPS → None)
    perceptual_loss_fn = _build_perceptual_loss() if cfg.perceptual_weight > 0 else None

    # Prior depth/normal maps: {frame_stem -> npy path}
    priors_dir = Path(prior_artifacts["priors_dir"]) if prior_artifacts else None
    has_priors = priors_dir is not None and priors_dir.exists()
    if has_priors:
        log.info("depth/normal priors available at %s", priors_dir)
    else:
        log.info("no priors — depth/normal weights will be 0")

    # Image + prior cache (load on-demand)
    img_cache: dict[str, torch.Tensor] = {}
    prior_cache: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    def _load_image(path: Path, W: int, H: int) -> torch.Tensor:
        key = str(path)
        if key not in img_cache:
            img = PILImage.open(path).convert("RGB").resize((W, H), PILImage.BILINEAR)
            img_cache[key] = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).to(device)
            if len(img_cache) > 50:
                img_cache.pop(next(iter(img_cache)))
        return img_cache[key]

    def _load_prior(path: Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        key = str(path)
        if key not in prior_cache:
            depth_p = priors_dir / f"{path.stem}_depth.npy"  # type: ignore[operator]
            norm_p  = priors_dir / f"{path.stem}_normal.npy"
            depth_t = torch.from_numpy(np.load(str(depth_p))).to(device) if depth_p.exists() else None
            norm_t  = torch.from_numpy(np.load(str(norm_p))).to(device) if norm_p.exists() else None
            prior_cache[key] = (depth_t, norm_t)
            if len(prior_cache) > 30:
                prior_cache.pop(next(iter(prior_cache)))
        return prior_cache[key]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log.info("starting training: %d iterations, %d Gaussians, priors=%s",
             cfg.iterations, N, has_priors)
    losses_window: list[float] = []

    for step in range(cfg.iterations):
        cam = random.choice(train_cams)
        W, H = cam["W"], cam["H"]
        gt = _load_image(cam["path"], W, H)  # [H, W, 3]

        viewmat = cam["w2c"].unsqueeze(0).to(device)   # [1, 4, 4]
        K_mat = cam["K"].unsqueeze(0).to(device)       # [1, 3, 3]

        # DropGaussian (CVPR 2025): randomly drop fraction of Gaussians each step.
        # Forces every Gaussian to contribute independently — prevents needles from
        # becoming load-bearing load-bearing across multiple views.
        N_cur = splats["means"].shape[0]
        if cfg.drop_rate > 0 and N_cur > 100:
            keep = torch.rand(N_cur, device=device) > cfg.drop_rate
            render_means    = splats["means"][keep]
            render_quats    = splats["quats"][keep]
            render_scales   = splats["scales"][keep]
            render_opacities = splats["opacities"][keep]
            render_sh0      = splats["sh0"][keep]
            render_shN      = splats["shN"][keep]
        else:
            keep = None
            render_means, render_quats = splats["means"], splats["quats"]
            render_scales, render_opacities = splats["scales"], splats["opacities"]
            render_sh0, render_shN = splats["sh0"], splats["shN"]

        # SH degree warm-up: advance every 1000 steps
        active_sh = min(cfg.sh_degree, step // 1000)
        sh_feats = torch.cat([render_sh0, render_shN], dim=1)

        renders, alphas, info = rasterization(
            means=render_means,
            quats=F.normalize(render_quats, dim=-1),
            scales=torch.exp(render_scales),
            opacities=torch.sigmoid(render_opacities),
            colors=sh_feats,
            viewmats=viewmat,
            Ks=K_mat,
            width=W,
            height=H,
            render_mode="RGB+ED",
            sh_degree=active_sh,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            absgrad=True,
            rasterize_mode="antialiased" if cfg.use_mip else "classic",  # Mip-Splatting
        )
        # When DropGaussian is active, info is for the kept subset — remap for strategy
        if keep is not None:
            info["dropped_mask"] = keep   # strategy ignores unknown keys

        rendered_rgb   = renders[0, ..., :3]   # [H, W, 3]
        rendered_depth = renders[0, ..., 3]    # [H, W]

        # Photometric loss: L1 + SSIM + WD-R/LPIPS perceptual (Apple 2025)
        l1 = torch.mean(torch.abs(rendered_rgb - gt))
        pred_bchw = rendered_rgb.permute(2, 0, 1).unsqueeze(0)
        gt_bchw   = gt.permute(2, 0, 1).unsqueeze(0)
        ssim_val  = _ssim_loss(pred_bchw, gt_bchw)
        photo_loss = 0.8 * l1 + 0.2 * ssim_val

        if perceptual_loss_fn is not None:
            perc_val = perceptual_loss_fn(pred_bchw, gt_bchw)
            photo_loss = photo_loss + cfg.perceptual_weight * perc_val

        # Scale regularization — primary needle killer
        scales_exp = torch.exp(splats["scales"])
        s_max = scales_exp.max(dim=-1).values
        s_min = scales_exp.min(dim=-1).values
        ratio_loss = torch.mean(F.relu(s_max / (s_min + 1e-8) - cfg.scale_ratio_cap))
        abs_loss   = torch.mean(F.relu(s_max - cfg.scale_abs_cap))
        scale_reg  = cfg.scale_reg_weight * (ratio_loss + abs_loss)

        # Effective Rank (NeurIPS 2024) — mathematical needle repulsion
        eff_rank_reg = cfg.eff_rank_weight * _effective_rank_loss(splats["scales"])

        # Monocular depth supervision (DN-Splatter WACV 2025)
        depth_loss = torch.tensor(0.0, device=device)
        normal_loss = torch.tensor(0.0, device=device)
        if has_priors and cfg.mono_depth_weight > 0:
            prior_depth, prior_normal = _load_prior(cam["path"])
            if prior_depth is not None:
                # Resize prior to match rendered resolution if needed
                if prior_depth.shape != rendered_depth.shape:
                    prior_depth = F.interpolate(
                        prior_depth.unsqueeze(0).unsqueeze(0),
                        size=(H, W), mode="bilinear", align_corners=False,
                    ).squeeze()
                depth_loss = _scale_invariant_depth_loss(rendered_depth, prior_depth)

                # Normal supervision: compare depth-gradient normals
                if prior_normal is not None and cfg.normal_weight > 0:
                    if prior_normal.shape[:2] != (H, W):
                        prior_normal = F.interpolate(
                            prior_normal.permute(2,0,1).unsqueeze(0),
                            size=(H, W), mode="bilinear", align_corners=False,
                        ).squeeze(0).permute(1,2,0)
                    # Compute normals from rendered depth gradients
                    rd = rendered_depth
                    dz_dx = torch.gradient(rd, dim=1)[0]
                    dz_dy = torch.gradient(rd, dim=0)[0]
                    rendered_normal = F.normalize(
                        torch.stack([-dz_dx, -dz_dy, torch.ones_like(rd)], dim=-1), dim=-1
                    )
                    normal_loss = torch.mean(torch.abs(rendered_normal - prior_normal))

        loss = (
            photo_loss
            + scale_reg
            + eff_rank_reg
            + cfg.mono_depth_weight * depth_loss
            + cfg.normal_weight * normal_loss
        )

        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        _lr = optimizers["means"].param_groups[0]["lr"]
        strategy.step_post_backward(splats, optimizers, strategy_state, step, info, _lr)

        losses_window.append(loss.item())
        if len(losses_window) > 200:
            losses_window.pop(0)

        if step % 2000 == 0:
            log.info("  step %d/%d  loss=%.4f  N=%d  depth=%.4f  normal=%.4f",
                     step, cfg.iterations, sum(losses_window)/len(losses_window),
                     splats["means"].shape[0], depth_loss.item(), normal_loss.item())

    # ------------------------------------------------------------------
    # Export PLY
    # ------------------------------------------------------------------
    log.info("exporting PLY")
    ply_path = out / "point_cloud.ply"
    _export_ply(splats, ply_path)

    # Save camera JSON for eval stage
    cam_json = []
    for c in all_cams:
        cam_json.append({
            "name": c["path"].name,
            "w2c": c["w2c"].tolist(),
            "K": c["K"].tolist(),
            "W": c["W"],
            "H": c["H"],
            "split": "holdout" if c in holdout_cams else "train",
        })
    (out / "cameras.json").write_text(json.dumps(cam_json, indent=2))

    final_loss = sum(losses_window) / max(len(losses_window), 1)
    return StageResult(
        ok=True,
        metrics={"iterations": cfg.iterations, "attempt": attempt,
                 "final_loss": final_loss, "n_gaussians": splats["means"].shape[0]},
        artifacts={
            "ply_path": str(ply_path),
            "holdout_dir": str(out),
            "cameras_json": str(out / "cameras.json"),
        },
    )


# ---------------------------------------------------------------------------
# PLY export (same layout as export_ply_from_ckpt.py)
# ---------------------------------------------------------------------------

def _export_ply(splats: dict, out_path: Path) -> None:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        means  = splats["means"].detach().cpu().numpy().astype(np.float32)
        scales = splats["scales"].detach().cpu().numpy().astype(np.float32)
        quats  = F.normalize(splats["quats"], dim=-1).detach().cpu().numpy().astype(np.float32)
        ops    = splats["opacities"].detach().cpu().numpy().astype(np.float32)
        sh0    = splats["sh0"].detach().cpu().numpy().astype(np.float32)
        shN    = splats["shN"].detach().cpu().numpy().astype(np.float32)

    n = means.shape[0]
    f_dc = sh0.reshape(n, -1)
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1)
    n_extra = f_rest.shape[1]

    dtype = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
        ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
        *[(f"f_rest_{i}","f4") for i in range(n_extra)],
        ("opacity","f4"),
        ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
        ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
    ]
    arr = np.empty(n, dtype=dtype)
    arr["x"],arr["y"],arr["z"] = means[:,0],means[:,1],means[:,2]
    arr["nx"]=arr["ny"]=arr["nz"]=0.0
    arr["f_dc_0"],arr["f_dc_1"],arr["f_dc_2"] = f_dc[:,0],f_dc[:,1],f_dc[:,2]
    for i in range(n_extra):
        arr[f"f_rest_{i}"] = f_rest[:,i]
    arr["opacity"] = ops.reshape(-1)
    arr["scale_0"],arr["scale_1"],arr["scale_2"] = scales[:,0],scales[:,1],scales[:,2]
    arr["rot_0"],arr["rot_1"],arr["rot_2"],arr["rot_3"] = quats[:,0],quats[:,1],quats[:,2],quats[:,3]

    try:
        from plyfile import PlyData, PlyElement
        PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out_path))
    except ImportError:
        # Fallback: write binary PLY manually
        _write_ply_manual(arr, out_path)

    mb = out_path.stat().st_size / 1024 / 1024
    log.info("  wrote %s (%.1f MB, %d Gaussians)", out_path.name, mb, n)


def _write_ply_manual(arr: np.ndarray, out_path: Path) -> None:
    """Minimal binary PLY writer (no plyfile dep needed on GPU worker)."""
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(arr)}",
    ]
    for name, dtype_char in arr.dtype.descr:
        type_map = {"f4": "float", "f8": "double", "i4": "int", "u1": "uchar"}
        header_lines.append(f"property {type_map.get(dtype_char, 'float')} {name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"
    with open(out_path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())
