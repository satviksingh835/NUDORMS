# Plan: Replace COLMAP/GLOMAP pose estimation with MASt3R-SfM

## Context

The current pipeline uses COLMAP+ALIKED+LightGlue for pose estimation. It works
when users follow a strict capture recipe (slow, continuous, dense overlap),
but fails on casual home-tour videos:

- **test5** (careful sidestep capture, 254 frames): 254/258 registered ✅
- **test6** (varied-motion casual capture, 300 frames): 7/300 then 2/300 ❌

Real users will not follow a 4-minute capture recipe. NUDORMS only ships
when the pipeline works on a casual handheld home tour.

## Goal

Swap the pose stage to a learned model that handles casual capture:
**MASt3R-SfM** (Naver Labs). Outputs COLMAP-format sparse models, so the
existing gsplat trainer consumes them unchanged.

Why MASt3R-SfM specifically (vs VGGT, DUSt3R, Fast3R):
- **Designed as a COLMAP replacement** — its export format is what gsplat
  already eats. Zero downstream changes.
- **Public weights, public code**: <https://github.com/naver/mast3r>
- **Pre-tested in the gsplat ecosystem** — Nerfstudio integrations exist.
- VGGT is faster but newer; integration is less mature. Add as a second
  candidate in the ensemble after MASt3R-SfM lands.

## Success criteria

1. **test6 (the bad varied capture) registers >90% of frames.** This was the
   original validation case. If MASt3R-SfM can't fix test6, the whole
   approach fails.
2. **test5 still registers >95% of frames.** Regression check — we shouldn't
   regress on the easy case.
3. **End-to-end splat quality on test6 is at least as good as test5_v3.**
   PSNR ≥ 28, recognizable room.
4. **No capture instructions required for users.** Frontend capture page
   becomes "hit record, walk around for 60-180 seconds, hit stop."

## Files to create / modify

### New: `infra/runpod_bootstrap.sh` additions

Add to the bootstrap script (after Python deps install):

```bash
# MASt3R-SfM model + code
git clone --recursive https://github.com/naver/mast3r /workspace/mast3r
cd /workspace/mast3r/dust3r
pip install -r requirements.txt
pip install -r requirements_optional.txt  # for sparse SfM
cd /workspace/mast3r
pip install -r requirements.txt

# Model weights (~5 GB)
mkdir -p /workspace/mast3r_models
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -O /workspace/mast3r_models/MASt3R.pth
```

### Replace: `backend/pipeline/poses/mast3r.py`

Currently a stub returning `ok=False`. Replace with real implementation:

```python
"""MASt3R-SfM pose estimation — handles casual capture.

Uses Naver's MASt3R model to estimate per-frame poses + sparse 3D points
in a single learned pass. Outputs COLMAP-format sparse model (cameras.bin,
images.bin, points3D.bin) that gsplat's simple_trainer reads directly.

Robust to:
  - Discontinuous motion (height/direction changes mid-capture)
  - Sparse temporal overlap between frames
  - Textureless surfaces (it's learned end-to-end on real video)

The model lives at $NUDORMS_MAST3R_MODEL (default
/workspace/mast3r_models/MASt3R.pth).
"""

# Pseudocode — concrete impl uses:
#   from mast3r.model import AsymmetricMASt3R
#   from mast3r.sfm.kinematic_chain import KinematicChain
#   from mast3r.sfm.export_colmap import export_to_colmap

def run(frames_dir: Path, out_dir: Path) -> StageResult:
    # 1. Load model (cached after first call)
    # 2. Pairwise inference: for each pair within sequential window of ~30,
    #    run MASt3R to get 3D matches + relative pose
    # 3. Build kinematic chain across all pairs (global alignment)
    # 4. Export to COLMAP format at out_dir/sparse/0/
    # 5. Use existing _read_sparse_metrics() from glomap.py to score
```

Reference implementation: <https://github.com/naver/mast3r/blob/main/sparse_global_inference.py>
which already runs the full SfM pipeline and exports COLMAP format.
We can shell out to it via subprocess to start, then port to a direct
Python integration in v2.

**MVP impl: shell out to MASt3R's `sparse_global_inference.py`.** Same
pattern we use for `colmap mapper`. Zero ML model loading code in our
pipeline; we just orchestrate.

### Modify: `backend/pipeline/poses/ensemble.py`

Reorder so MASt3R is the *primary* method, COLMAP+ALIKED is fallback:

```python
attempts: list[tuple[str, StageResult]] = []

m = mast3r.run(frames_dir, workdir / "poses_mast3r")
attempts.append(("mast3r", m))
if _passes_quality(m, total):
    return _wrap("mast3r", m, attempts)

g = glomap.run(frames_dir, workdir / "poses_glomap")
attempts.append(("glomap", g))
if _passes_quality(g, total):
    return _wrap("glomap", g, attempts)

c = colmap.run(frames_dir, workdir / "poses_colmap")
attempts.append(("colmap", c))
if _passes_quality(c, total):
    return _wrap("colmap", c, attempts)
```

Rationale: MASt3R first because it's robust on casual capture (the common
case). COLMAP+ALIKED stays as fallback because for *very careful* captures
it sometimes produces a slightly tighter pose graph.

### Modify: `frontend/src/pages/Capture.tsx` and `GuidedRecorder.tsx`

Drop the strict capture coaching once MASt3R pose is validated. New UX:

- Pre-capture screen: simple "record a 60-90 second walkthrough of your
  room" prompt. Drop the bullet list of strict rules.
- Live HUD: keep the **exposure/focus lock** (still matters — both classical
  and learned methods hate AE drift mid-capture). Drop the speed warning
  and coverage map (no longer mission-critical).
- Stop gate: minimum 30s duration, no coverage requirement.

Capture quality still matters (lighting, sharpness) — but motion path
doesn't.

## Validation plan

1. **Implement MVP `mast3r.py`** that shells out to MASt3R's reference
   `sparse_global_inference.py`.
2. **Re-run test5 pose smoke** — must register ≥95% of frames. Regression
   check.
3. **Re-run test6 pose smoke** — must register ≥90% of frames. Win condition.
4. **Train splat on both, compare to v3 baseline** — visual + PSNR check.
5. **Capture a deliberately casual home tour** (no recipe, just film
   normally) — confirm pose works on a true real-world capture.
6. **Drop the strict capture coaching from the frontend** — once 3-5
   different casual captures all work, ship this.

## What carries forward unchanged

- `pipeline/ingest_qc.py` — sharpness/exposure/duration QC stays valuable.
- `pipeline/frame_select.py` — frame selection stays valuable (drop bottom
  25% by sharpness, dedupe near-stationary frames).
- `pipeline/train/gsplat_mcmc.py` — splat training is unchanged. Reads
  COLMAP-format sparse, doesn't care how it was produced.
- `pipeline/poses/{glomap,colmap}.py` — kept as ensemble fallbacks.
- `infra/RUNPOD.md` workflow.

## Cost & time estimate

- **Engineering time**: ~3 days for a competent dev to wire MASt3R-SfM in,
  validate on test5/test6, and tune the ensemble. ~1 week including the
  frontend simplification and capture UX rework.
- **Compute**: MASt3R-SfM inference is GPU-bound (~30s for 300 frames on a
  4090), much faster than COLMAP incremental mapper (10-30 min). Per-scan
  cost goes *down*.
- **Disk**: ~5 GB for MASt3R weights, persistent on the pod.

## Open questions

- **Quality cap**: MASt3R-SfM may produce slightly less geometrically
  accurate poses than well-tuned COLMAP on careful captures. Is the
  splat training quality good enough to compensate? Validate empirically.
- **Casual-capture floor**: even with learned pose, *very* bad captures
  (lots of motion blur, all dark) still won't reconstruct. The QC stage
  needs to gate those before they reach pose.
- **Model size on pod cold start**: ~5 GB download adds 1-2 min to first
  bootstrap. Mitigation: bake into pod template or cache on a Network
  Volume.

## Filed-but-deferred work

- **VGGT** as second ML candidate in the ensemble. Add after MASt3R-SfM is
  shipped and stable.
- **End-to-end NoPoSplat / Splatt3R** evaluation. These skip pose entirely
  and predict the splat directly. Frontier research; revisit in 6 months.
- **iPhone ARKit native capture app**. Best long-term path for quality and
  bulletproof UX. ~4 weeks of native dev.
