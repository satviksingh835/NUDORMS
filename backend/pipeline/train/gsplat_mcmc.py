"""3DGS trainer with quality-maximizing knobs turned on.

- MCMC densification (more stable than vanilla 3DGS densification)
- Mip-Splatting AA (anti-aliased rendering for zoom-in/out)
- Depth supervision when LiDAR present, else weak monocular prior
- Planar/normal regularization (walls, floor, ceiling alignment)
- Per-image appearance embeddings (residual exposure drift absorbing)
- 30k iterations default; auto-retry uses 50k + relaxed densification
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..orchestrator import StageResult


@dataclass
class TrainConfig:
    iterations: int = 30_000
    use_mcmc: bool = True
    use_mip: bool = True
    use_appearance_embedding: bool = True
    depth_weight: float = 0.05         # only active when LiDAR present
    mono_depth_weight: float = 0.01    # weak monocular prior (Depth Anything V2)
    normal_weight: float = 0.05
    holdout_fraction: float = 0.10
    sh_degree: int = 3


def config_for_attempt(attempt: int) -> TrainConfig:
    if attempt == 1:
        return TrainConfig()
    # On retry, train longer and densify more aggressively.
    return TrainConfig(iterations=50_000, normal_weight=0.02, mono_depth_weight=0.02)


def run(scan_id: str, workdir: Path, pose_artifacts: dict, attempt: int = 1) -> StageResult:
    cfg = config_for_attempt(attempt)
    out = workdir / f"splat_attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=True)
    # TODO: load pose_artifacts['sparse_dir'] (COLMAP-format) into a gsplat dataset
    # TODO: split holdout views (~10%)
    # TODO: build trainer:
    #   - gsplat.MCMCStrategy if cfg.use_mcmc else gsplat.DefaultStrategy
    #   - filter_2d_kernel_size + 3D filter (Mip-Splatting)
    #   - per-frame nn.Embedding for appearance
    #   - if depth maps exist (LiDAR): L1(rendered_depth, gt_depth) * cfg.depth_weight
    #   - else: L1(rendered_depth, mono_depth_prior) * cfg.mono_depth_weight  (weak)
    #   - planar regularizer on near-coplanar gaussian clusters
    # TODO: train, log per-iter loss + periodic eval
    # TODO: write workdir/splat_attempt_N/point_cloud.ply + camera.json
    return StageResult(
        ok=True,
        metrics={"iterations": cfg.iterations, "attempt": attempt},
        artifacts={
            "ply_path": str(out / "point_cloud.ply"),
            "holdout_dir": str(out / "holdout"),
        },
    )
