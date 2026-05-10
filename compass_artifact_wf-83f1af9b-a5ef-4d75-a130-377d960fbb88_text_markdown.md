# State-of-the-Art Indoor Scene Reconstruction from Casual Phone Video for Web Viewing (2025–2026)

## TL;DR
- **Best current pipeline as of mid-2026:** capture with **Spectacular Rec (or any iOS app that exports ARKit poses)**, run **VGGT** for dense pose/point initialization (replacing COLMAP/MASt3R-SfM), train a **Scaffold-GS or PGSR / 2DGS variant with DN-Splatter-style monocular depth + normal regularization** (using Depth Anything V2 / UniDepth-V2), apply **Difix3D+** as a diffusion-based artifact fixer, then export to **SOG** and serve via **Spark 2.0 (sparkjs.dev)** in the browser — this combination, not any single repo, is what the current literature suggests will fix the "needle/streak/mush" failure mode the user is seeing.
- **Root cause of the user's artifacts:** vanilla 3DGS in indoor scenes has *no surface or scale prior* on Gaussians, so under sparse parallax and textureless walls it overfits training views by growing anisotropic, "needle-like" splats whose PSNR is fine on training rays but explode in novel views. Both the geometry literature (Effective Rank, AbsGS, PGSR, 2DGS, Trim2DGS, GeoGaussian) and the indoor-specific literature (DN-Splatter, GaussianRoom, LighthouseGS, AGS-Mesh) identify this as a *shape/scale regularization* + *missing geometric prior* problem, not primarily a pose problem.
- **Realistic expectation for a 30–90 s casual dorm walkthrough:** with the recommended stack you can get *Polycam/Luma-class* photorealism (sharp, recognizable furniture and posters; readable monitor screens within the captured arc) but you should **not** expect crisp novel-view extrapolation more than ~30–60 cm outside the captured trajectory, and reflective windows / mirrors / monitors will remain failure cases. Expect 1.5–3 M Gaussians per dorm, 5–30 minutes of training on an A100/L40S, and a ~30–80 MB SOG file for the web.

---

## Key Findings

### 1. Why the user is seeing needle splats and painterly mush — the consensus diagnosis
Three independent lines of recent work converge on the same diagnosis:

- **Effective Rank (NeurIPS 2024, "Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting")** explicitly shows that vanilla 3DGS Gaussians "converge into anisotropic Gaussians with one dominant variance" — i.e. mathematical needles — and that this directly produces the "needle-like artifacts, especially in novel and extreme views far from the training images" the user describes.
- **AbsGS (ACM MM 2024)** identifies *gradient collision* in the densification heuristic as the cause of "blurry rendered images" in textureless / over-reconstructed regions — exactly the painterly mush.
- **DN-Splatter (WACV 2025)**, **GaussianRoom**, **AGS-Mesh**, **TIDI-GS**, and **LighthouseGS (ICCV 2025)** all open with the observation that 3DGS "performance on scenes commonly seen in indoor datasets is poor due to the lack of geometric constraints during optimization," producing floaters above textureless planes (walls, ceilings, floors) and structural noise.

The user's switch from COLMAP to MASt3R-SfM, while it improves *pose* quality, does not address any of this — pose was not the dominant problem. The dominant problems are (a) Gaussian shape/scale, (b) missing geometric supervision in textureless regions, and (c) under-constrained novel views from limited parallax.

### 2. Current SOTA — Gaussian Splatting variants relevant to indoor

| Method | What it adds | Indoor relevance |
|---|---|---|
| **2DGS** (SIGGRAPH 2024, hbb1/2d-gaussian-splatting) | Disk-shaped 2D Gaussians; multi-view consistent depth | Strong on flat surfaces; clean mesh extraction via TSDF |
| **PGSR** (TVCG 2024) | Planar Gaussians + multi-view geometric consistency loss | Best Chamfer-distance results on textureless indoor scenes; reaches PSNR 30.41 / SSIM 0.930 / LPIPS 0.161 on indoor MipNeRF360 (best LPIPS in class) |
| **GOF — Gaussian Opacity Fields** (SIGGRAPH 2024) | Level-set surface extraction, marching tetrahedra | Best LPIPS 0.167 on indoor MipNeRF360, no Poisson/TSDF needed |
| **RaDe-GS** (2024) | Closed-form ray-Gaussian intersection depth | Used as backbone in GSPlane and elsewhere; better surface depth |
| **Mip-Splatting** (CVPR 2024 Best Student Paper) | 3D smoothing filter + 2D Mip filter | Eliminates scale-mismatch aliasing when web viewer zooms; **should be applied on top of any backbone for web viewing** |
| **Scaffold-GS** (CVPR 2024 Highlight) | Anchor-based hierarchy, view-conditioned MLP | Explicitly motivated by "intricate indoor environments with challenging observing views, e.g. transparency, specularity, reflection, texture-less regions"; significantly fewer floaters |
| **AbsGS** (ACM MM 2024) | Homodirectional gradient densification | Recovers fine details where vanilla 3DGS goes mushy |
| **TrimGS / Trim2DGS** (NeurIPS 2024) | Contribution-based pruning + small-scale regularization | Reduces floaters and "spikey" Gaussians |
| **3DGUT** (CVPR 2025, NVIDIA) | Unscented-Transform projection | Handles distorted/wide-angle phone cameras and lets the same model be ray-traced for reflections |
| **DropGaussian** (CVPR 2025) | Random Gaussian dropout during training | Prior-free regularizer that mitigates sparse-view overfitting |
| **DN-Splatter** (WACV 2025, maturk/dn-splatter) | Monocular depth + normal supervision + smoothness | Designed *specifically* for casual iPhone indoor capture; demonstrated on MuSHRoom/ScanNet++ |
| **AGS-Mesh** (2025) | Adaptive joint depth/normal refinement | Out-performs DN-Splatter for smartphone indoor capture |
| **GaussianRoom** (2025) | SDF-guided Gaussians + monocular normals | Directly targets ScanNet/ScanNet++ texture-less indoor failures |
| **LighthouseGS** (ICCV 2025) | Indoor-structure-aware GS for "panorama-style" mobile capture | Targets exactly the user's use case |
| **LongSplat** (2025) | MASt3R-init octree-anchored 3DGS for long unposed casual videos | Built for long handheld walkthroughs |
| **FreeSplat++** (2025) | Feed-forward whole-scene indoor reconstruction with floater removal | Indoor benchmark on ScanNet/Replica |

### 3. Pose-free / feed-forward priors (the new replacement for COLMAP/MASt3R-SfM)

- **VGGT — Visual Geometry Grounded Transformer (CVPR 2025 Best Paper, facebookresearch/vggt).** Single feed-forward transformer outputs poses, depth, point maps, and tracks for hundreds of frames in seconds. Crucially, the official repo since June 2025 ships a script that exports VGGT predictions in COLMAP format and feeds gsplat directly. *In benchmarks (E3D-Bench, CARVE), VGGT outperforms DUSt3R, MASt3R, Spann3R, Fast3R, MonST3R on pose, depth, and pointmap accuracy.* This is the obvious replacement for the user's current MASt3R-SfM step.
- **π³ / Pi3 (2025).** Permutation-equivariant, fully feed-forward; on Sintel, ATE drops from VGGT's 0.167 to 0.074, and runs at 57 FPS vs VGGT's 43 FPS. A drop-in upgrade once a stable repo is available.
- **VGG-T³ (NVIDIA, 2025).** Distills VGGT into a fixed-size MLP via test-time training; reconstructs 1k images in 54 s vs >10 minutes for global-alignment DUSt3R/MASt3R. Useful if scaling to many phones.
- **InstantSplat (CVPR 2025, instantsplat.github.io).** End-to-end DUSt3R-init + 3DGS in <1 min, +32% SSIM and -80% ATE vs prior pose-free baselines. Reference reproduction on rerun.io.
- **NoPoSplat (ICLR 2025).** Pose-free *feed-forward* Gaussian generator from sparse unposed images; outputs Gaussians directly in a canonical frame. Best for very-sparse (2–4 view) settings, less for dense walkthroughs.
- **Gesplat (2025).** Combines VGGT init + GS optimization; explicit demonstration that VGGT-init beats COLMAP-init on sparse/unposed inputs.
- **MonST3R, Spann3R, CUT3R, Fast3R.** Useful primarily when scenes are dynamic or streaming; for static dorm rooms VGGT/Pi3 dominate.

### 4. Diffusion-based "artifact fixers" — a 2025 game-changer

For sparse-view casually captured scenes, the field has converged on running a **single-step diffusion prior on rendered novel views, then distilling back into the splat representation**. This is the closest thing to a silver bullet for the painterly mush / streak problem:

- **Difix3D+ (CVPR 2025 Oral & Best Paper Finalist, NVIDIA — nv-tlabs/Difix3D).** Single model works on both NeRF and 3DGS, runs on 8 GB VRAM, achieves ~2× FID improvement over baselines while preserving 3D consistency. **Code is released and is the most mature option.**
- **GSFixer, GSFix3D, FixingGS, ArtifactWorld, GenFusion (all 2025).** Newer iterations — most use video diffusion priors and are heavier (40+ GB VRAM, ~2 hours per scene).
- **FlowR (CVPR 2025).** Multi-view flow matching that turns sparse-view renders into "as-if-dense" renders; highest quality but heavy.

For a student-product pipeline on RunPod, Difix3D+ is the right starting point.

### 5. Mesh-based and hybrid approaches

If you want geometry as well as photorealism (e.g. for a measured floor plan, AR placement, or a fallback web representation that uses standard glTF):

- **SuGaR, 2DGS-mesh, PGSR-mesh, GOF marching-tetrahedra mesh** all extract clean meshes; PGSR currently leads on indoor F-score.
- **Neuralangelo / BakedSDF / NeuS2** still produce the best raw geometry but are an order of magnitude slower than GS variants; not recommended for a per-scene-in-minutes product.
- **Hybrid GS + mesh (HMGS, GSDF, GS-SR collection).** GSPlane (2025) on top of 2DGS produces a low-vertex-count mesh with consistent normals on ScanNet, F-score 0.689. Useful for a mesh fallback.

### 6. Capture-side best practices (the lowest-hanging fruit by far)

- **Use ARKit / ARCore poses as priors, not just video.** Spectacular AI's `sai-cli process` (free for non-commercial use) takes Spectacular Rec recordings and outputs Nerfstudio-formatted data with VIO poses + IMU-registered images, and a sparse VISLAM point cloud — eliminating COLMAP/MASt3R entirely and giving metric scale. This is what Polycam, Scaniverse and Luma do under the hood. PocketGS (2025) shows that ARKit-pose initialization alone reduces convergence time from 319 s to 54 s for similar quality.
- **Compensate for motion blur and rolling shutter.** "Gaussian Splatting on the Move" (Spectacular AI, 2024) integrates IMU data into a differentiable image-formation model; the difference on casual iPhone footage is dramatic. A user reporting "streaks" should suspect untreated rolling shutter and motion blur first.
- **Capture pattern.** LighthouseGS (ICCV 2025) shows that *panorama-style motion at multiple stations* (stand, rotate with half-stretched arms, then translate) outperforms the typical "walk in a circle" pattern for non-experts in indoor rooms. The Polyvia3D, Niantic Scaniverse, and forensic-NeRF papers agree: ~3–5 m/min walking speed, overcast or all-artificial-light to keep illumination constant, three height passes (knee, eye, overhead), and specifically *rotate around* glass/screens rather than capturing them head-on.
- **Frame selection.** Drop frames with high motion blur (Laplacian variance threshold) and re-densify around chairs / desks / textured artwork.
- **Capture practical limits.** 30–90 s of phone video at 30 fps gives 900–2700 frames; subsample to ~150–400 keyframes before training. This is the sweet spot for VGGT memory and per-scene 3DGS training time.

### 7. Commercial pipelines — what they're known to do
- **Polycam:** photogrammetry + LiDAR fusion on Pro iPhones; cloud SfM, then 3DGS on the server. Best for textureless indoor capture because LiDAR depth bypasses the textureless-correspondence problem.
- **Scaniverse (Niantic, free):** LiDAR-only fast TSDF + a 3DGS pass; lowest barrier; ~3–5 minutes processing. Highest quality on a *LiDAR* iPhone for room capture.
- **Luma AI:** photogrammetry + NeRF/3DGS hybrid; best web sharing UX; tends to be cleanest on reflective surfaces (uses appearance embeddings).
- **KIRI Engine:** flexible across NeRF/3DGS/photogrammetry; weaker on rooms, stronger on objects.
- All four converge on cloud processing in 3–10 minutes, output 1.5–3M Gaussians, and (per Polyvia3D's benchmarking) "produce results difficult to distinguish from workstation-trained counterparts in typical viewing conditions" for room-scale captures.

### 8. Web rendering — the viewer side is essentially solved in 2026

- **Spark 2.0 (sparkjs.dev, World Labs, MIT-licensed, April 2026)** is the current state of the art. Streaming `.RAD` LoD format, 100M+ splat scenes on phones, foveated rendering, 98% WebGL2 device coverage, Three.js integration. Released after a USD 1B World Labs round and named by GitHub as one of 2025's most influential repositories. **This should be the default web viewer choice.**
- **PlayCanvas SuperSplat / SuperSplat Editor** — the dominant *editor* (cropping, cleaning, exporting). Use this in your post-processing step before serving.
- **Three.js GaussianSplats3D (mkkellogg)** — historically popular but the author has explicitly recommended migrating to Spark.
- **antimatter15/splat** — minimal WebGL viewer; good for MVP.
- **gsplat.js, KeKsBoTer/web-splat, MarcusAndreasSvensson/gaussian-splatting-webgpu** — all viable; WebGPU variants are now feasible since Apple shipped WebGPU in iOS 26 / macOS 26 (Sept 2025).
- **Brush (Arthur Brussee, Apache 2.0)** — Rust+WebGPU engine that even *trains* in browser; useful as a preview tool.
- **Compression formats (most-to-least compressed):** **SOG (PlayCanvas, May 2025)** ≈ 15–20× smaller than PLY using WebP-encoded, spatially-ordered Gaussians (Morton order, palette-quantized SH, single .sog container) — open-spec; **SPZ** ≈ half of SOG; **compressed PLY** then raw PLY. SOG is the right delivery format for a college-student web product: a 4M-Gaussian scene that was 1 GB as PLY becomes ~42 MB as SOG, loadable instantly on mobile.

### 9. Where the field is still failing for this exact use case

- **Mirrors, monitors, glass, polished floors.** Even 3DGRUT/3DGUT (which adds ray-traced reflections) has limitations. Current consensus: rotate around them, don't shoot head-on.
- **Closets / under-beds / behind monitors.** No diffusion prior fully fills these; expect ghosts/floaters. A semantic-mask cleanup pass (Clean-GS, PointNuker) is recommended.
- **Far-field walls in tight spaces.** Limited parallax = under-determined depth. DN-Splatter's monocular-depth regularizer largely fixes this in the median case but produces visibly worse novel views than dense capture.
- **Small text/labels (book spines, posters).** Vanilla 3DGS produces "painterly" output here. AbsGS-style densification + a perceptual loss like Apple's WD-R (`apple/ml-perceptual-3dgs`, 2025 paper showing 2.3× human preference vs L1+SSIM) materially helps.

---

## Details — Concrete Recommended Pipeline

### Capture side (in the student app)
1. Capture 30–90 s of 1080p or 4K video via a Spectacular Rec-style recorder that also writes ARKit/ARCore poses, accelerometer, gyroscope, and intrinsics.
2. Instruct the user (in-app overlay) to do a *modified panorama-style* pattern: 3 stations × 360° rotation at varying heights, plus one slow translational loop. Avoid head-on captures of mirrors, screens, windows.
3. Reject blurry frames client-side (Laplacian variance) and upload ~200–400 keyframes + IMU log.

### Cloud side (RunPod, A100 80 GB or L40S 48 GB)

**Step A — Pose / point-cloud initialization (replace your COLMAP+MASt3R step)**
- Run **VGGT** (`facebookresearch/vggt`) with the included COLMAP-export script on the keyframes. Optionally enable bundle-adjustment refinement.
- Sanity-check ATE against ARKit/ARCore poses; if VGGT diverges, fall back to ARKit poses + sparse VISLAM point cloud from Spectacular AI's `sai-cli process`. (This is what serious indoor pipelines do.)
- This step is seconds-to-a-minute and should completely replace the part of your pipeline that's giving you bad MASt3R-SfM-induced poses.

**Step B — Geometric priors**
- Run **Depth Anything V2** or **UniDepth-V2** monocular metric depth on every keyframe.
- Run a monocular surface-normal estimator (e.g. StableNormal, Metric3D-v2 normals).
- These will be consumed as supervision in the next step.

**Step C — Train the splat (the part most likely to be different from what you tried)**

The single biggest bang-for-buck stack for a dorm room is:
- **Backbone:** Scaffold-GS (anchor-based, fewer floaters in textureless regions) **or** PGSR (planar Gaussians, best on indoor flat surfaces) **or** 2DGS (cleanest mesh extraction).
- **+ Mip-Splatting filters** (3D smoothing + 2D Mip) — drop-in addition that fixes the scale-mismatch artifacts the user will otherwise see when zooming in the web viewer.
- **+ DN-Splatter / AGS-Mesh-style depth + normal regularization** using the priors from Step B. This is the specific module that kills the "needle splats over white walls" pathology.
- **+ Effective-Rank or AbsGS densification** to eliminate needle-shaped Gaussians and allow proper splitting in mushy regions.
- **+ DropGaussian** for sparse-view robustness (free regularizer, no extra priors).
- **Optionally: Scaffold-GS + Difix3D+ post-distillation** — Difix3D+ runs as a final refinement pass that single-step-diffusion-cleans rendered novel views and distills the cleaned views back into the Gaussians. This is the closest available technique to "make it look like a Polycam/Luma capture."

Implementation-wise this is most easily done by starting from `nerfstudio` Splatfacto-big or `gsplat` 1.3+, enabling `--pipeline.model.use_scale_regularization=True` and `--pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False` (Nerfstudio's own recommended quality settings), and adding the depth/normal losses. Several 2025 forks of Nerfstudio implement DN-Splatter as a Splatfacto plugin.

**Step D — Cleanup**
- Run a semantic outlier removal pass — e.g. PointNuker, Clean-GS, or simple DBSCAN on the .ply — to delete obvious floaters around the capture trajectory.
- (Optional) Train a **2DGS or PGSR mesh** from the same data and store it as an FBX/glTF fallback for clients without WebGL2.

**Step E — Compress and serve**
- Convert PLY → SOG via `playcanvas/splat-transform` (`splat-transform input.ply output.sog`). Expected size: 30–80 MB for a dorm.
- Optionally precompute a Spark `.RAD` LoD tree (`build-lod` CLI) for streaming on slow networks.
- Serve from any CDN. No GPU required server-side at view time.

### Viewer side
- Embed Spark 2.0 (Three.js + WebGL2). 30 lines of HTML; works on iOS Safari 26+, Android Chrome, desktop. Fixed rendering budget (e.g. 1.5 M splats) keeps the framerate steady on phones; Spark's LoD tree handles the rest.
- For richer UX (dollhouse view, hotspots, multi-room) use SuperSplat-edited scenes plus a thin wrapper.

### Expected end-to-end cost & latency on RunPod
- VGGT init: 30–90 s on an L40S/A100 for 300 frames.
- Depth/normal priors: ~30 s.
- 30k-iteration Scaffold-GS+DN-Splatter+Mip-Splatting train: 8–25 minutes on A100 for a dorm.
- Difix3D+ refinement (optional): +3–10 minutes.
- Total: **~15–40 minutes per scene** on a single A100 — fits the user's "minutes to a few hours" budget with plenty of headroom.

---

## Recommendations (concrete and staged)

**Stage 1 (this week): the cheapest fix that will already remove most of the user's artifacts**
- Replace MASt3R-SfM with **VGGT** (`facebookresearch/vggt`). Use its COLMAP exporter and feed `gsplat`'s `simple_trainer` directly. This alone often eliminates the worst pose-induced streaks.
- Enable Nerfstudio Splatfacto's `use_scale_regularization=True` (this is exactly the PhysGaussian scale regularizer and is documented to "reduce huge spikey gaussians" — i.e. the user's needles).
- Switch to **gsplat with Mip-Splatting's 3D smoothing + 2D Mip filter enabled** — this kills the "spikes when zoomed in / mush when zoomed out" web-viewer pathology.
- Trigger to escalate to Stage 2: if the renders are still painterly on textureless walls.

**Stage 2 (next 2–4 weeks): the indoor-specific stack**
- Add **DN-Splatter** (or AGS-Mesh) regularization with **Depth Anything V2** monocular depth and a normals network. Repo: `maturk/dn-splatter`. This is the single largest quality jump documented in the indoor literature for casually-captured iPhone data.
- Switch backbone from vanilla 3DGS to **Scaffold-GS** (`city-super/Scaffold-GS`) or **PGSR** (planar). Scaffold-GS is the safer first choice because it handles transparency/specularity better and was designed for "intricate indoor environments."
- Trigger to escalate to Stage 3: if novel-view extrapolation outside the captured arc is still poor.

**Stage 3 (1–2 months, for "feels like Luma" quality): diffusion + capture aids**
- Plug **Difix3D+** (`nv-tlabs/Difix3D`) in as the final refinement pass. ~8 GB VRAM, tens of minutes per scene; CVPR 2025 oral paper showed an average 2× FID improvement and is the most mature option.
- On the capture side, integrate the **Spectacular AI SDK / Spectacular Rec** so you get ARKit poses, IMU, and the **"Gaussian Splatting on the Move"** rolling-shutter+motion-blur compensation for free. Spectacular AI is free for non-commercial use; this is the same VIO foundation that Polycam and Scaniverse rely on.
- Add a perceptual loss: Apple's **WD-R** (`apple/ml-perceptual-3dgs`, 2025) — drop-in replacement for L1+SSIM, measured 2.3× human preference vs default 3DGS loss in a 39k-rating study.

**Stage 4 (long-term differentiation): hybrid mesh + splat for AR/VR**
- Train a **PGSR mesh** alongside the splat for a measured floor plan or AR placement. PGSR currently has the best Chamfer accuracy on indoor scenes among GS-based methods.
- Consider **3DGUT/3DGRUT** for proper reflections on monitors and windows once the basic pipeline is stable.

**Web viewer (do this once and forget):**
- Use **Spark 2.0** (`sparkjs.dev`). Convert outputs to **SOG** with `playcanvas/splat-transform`. Optionally pre-build `.RAD` LoD trees with `build-lod`. Expect 30–80 MB scenes that load instantly on a student's phone.
- For an editing/cleanup UX, ship **SuperSplat** (browser-based, no install).

**Benchmarks/thresholds that should change the recommendation:**
- If on your captures Stage-1 VGGT+Mip-Splatting hits **PSNR ≥ 28 / LPIPS ≤ 0.20** on held-out frames, ship it; the additional stages give diminishing returns.
- If LPIPS is still > 0.30 and walls show streaks, **the missing piece is monocular geometric supervision (Stage 2), not pose**.
- If LPIPS is fine but novel views > 1 m off-trajectory look hallucinated, **the missing piece is generative inpainting (Difix3D+, Stage 3)**.
- If users ask "can I see a floor plan?" or "can I drop a virtual desk in?", you need the **mesh** branch (Stage 4).

---

## Caveats and honest limits

- **VGGT and Pi3 are very recent (2025).** Both have sharp accuracy advantages on indoor benchmarks (7-Scenes, NRGBD, Sintel) but real-world phone scenes are out-of-distribution; sanity-check against ARKit poses and have COLMAP as a fallback. The "VGGT outperforms COLMAP" claim is benchmark-specific, not universal — for very large scenes COLMAP's bundle adjustment is still more accurate.
- **Diffusion fixers (Difix3D+, GSFixer, FlowR) sometimes hallucinate.** They improve perceptual metrics dramatically but can fabricate plausible-but-wrong textures (e.g. invented book covers) — fine for a dorm-tour product, problematic if students rely on the model for spatial decisions.
- **Indoor benchmarks favor specific datasets.** PSNR/LPIPS numbers cited above are mostly from MipNeRF360-indoor, ScanNet++, MuSHRoom, Replica, and Tanks & Temples *indoor*; numbers on a real dorm-room walkthrough will be 1–3 dB worse. Treat published numbers as relative orderings, not absolute targets.
- **WebGPU adoption is partial.** Apple shipped WebGPU in iOS 26/macOS 26 (Sept 2025) and Firefox added it in version 141 on Windows; for broadest compatibility, *prefer WebGL2-based viewers (Spark, antimatter15) over pure-WebGPU viewers (web-splat, MarcusAndreasSvensson)* until rollout is complete.
- **Commercial app pipelines are partly opaque.** Statements about Polycam/Luma/Scaniverse pipelines come from third-party reverse-engineering, blog posts, and marketing pages; the exact training recipes are not public.
- **The "World Labs", "Spark 2.0", and "1B raise" claims** come from Radiance Fields (radiancefields.com), VP-Land, and World Labs' own blog (worldlabs.ai); these are press/marketing sources rather than peer-reviewed. The repository (`sparkjsdev/spark`) and SOG specification (`playcanvas/splat-transform`) are real and MIT-licensed, so the *technology* claims are verifiable; the *impact* claims (e.g. "100 M splats on a phone") are demonstrable on the project's own demos but may not generalize.
- **PocketGS, LighthouseGS, FreeSplat++, Gesplat, LongSplat** are 2025 papers; mature open implementations may still be evolving. Treat as research references for technique selection rather than as drop-in dependencies.
- **Per-scene time and quality vary widely by capture.** A 30-second well-lit dorm video gets near-Polycam quality with the recommended stack; a 30-second dim, motion-blurred, mostly-white-walls video may still produce mush even with all of Stage 3 enabled. Capture quality remains the dominant variable, which is why Stage 3's capture-side improvements (Spectacular AI VIO + deblur + rolling-shutter compensation) are arguably more important than any single algorithm choice.