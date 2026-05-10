# NUDORMS pipeline + RunPod bootstrap — bugs and fixes

A working notes file from the May 2026 session that took the pipeline from
"crashes on first scan" to "produces a real splat end-to-end on a 16 GB
RTX A4000 pod". Each entry: **symptom → root cause → fix**. Keep this up to
date — most of these will bite again on the next pod or next variant.

---

## Bootstrap / pod setup

### 1. `colmap` cmake: `CMAKE_CUDA_COMPILER could not be found`
- **Cause:** RunPod images ship CUDA at `/usr/local/cuda` but don't put `nvcc` on `PATH`.
- **Fix:** Top of `runpod_bootstrap.sh`:
  ```bash
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDACXX="/usr/local/cuda/bin/nvcc"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  ```

### 2. COLMAP cmake: "Disabling download support (Curl/Crypto not found)"
- **Cause:** `libssl-dev` / `libcurl4-openssl-dev` aren't in the apt list.
- **Fix:** Add both to the apt install line.

### 3. GLOMAP cmake: `Target glomap links to colmap::colmap but the target was not found`
- **Cause:** GLOMAP's CMakeLists.txt does `find_package(COLMAP)` and expects
  the imported target `colmap::colmap`. A plain `make install` of standalone
  COLMAP only installs the static libs, not the CMake config files that
  expose that target.
- **Fix:** Don't pass `-DFETCH_COLMAP=OFF` to GLOMAP cmake. Let it
  FetchContent COLMAP itself (its CMakeLists pulls a specific commit and
  builds it as a CMake target the right way). Same logic for
  `-DFETCH_POSELIB=OFF` — fails with "PoseLib not found" because COLMAP
  links PoseLib statically and doesn't install its config files.
- **Cost:** ~15 min slower bootstrap, but reliable. We mitigate by
  pre-caching the FetchContent deps (next entry).

### 4. GLOMAP download stalls for hours on RunPod's network
- **Symptom:** `colmap-populate-gitclone.cmake` and `download-faiss-populate.cmake`
  crawl at ~60 KB/s. Bootstrap stuck at "Configuring COLMAP..." for 6+ hours.
- **Cause:** RunPod's egress is ~100–300 KB/s on single TCP connections;
  cmake's `file(DOWNLOAD)` uses one connection.
- **Fix:** Pre-stage the FetchContent sources at `/workspace/_glomap_deps/{colmap,
  poselib,faiss}` (volume disk, survives container rebuilds) and pass
  `-DFETCHCONTENT_SOURCE_DIR_<NAME>=...` to glomap's cmake. Download via
  `aria2c -j N -x 16 -s 16` from `github.com/<repo>/archive/<commit>.tar.gz` —
  ~50 MiB/s instead of 60 KB/s.
- **Source URLs to keep current:**
  - `https://github.com/colmap/colmap/archive/<commit>.tar.gz`
  - `https://github.com/PoseLib/PoseLib/archive/<commit>.tar.gz`
  - `https://github.com/ahojnnes/faiss/archive/<commit>.zip`

### 5. Single-connection HTTPS downloads are crippling slow
- **Symptom:** wget on the 2.5 GB MASt3R weights ETAs at 1h47m.
- **Fix:** `aria2c -x 16 -s 16` over the same URL → 9.1 MiB/s avg, 5 min.
  Drop wget for anything >50 MB on this pod.

### 6. `pip install -e /workspace/mast3r` fails with "neither setup.py nor pyproject.toml"
- **Cause:** Upstream `naver/mast3r` and `naverlabs/dust3r` ship neither.
  They're meant to be added to `PYTHONPATH`.
- **Fix:** Drop the editable install. Add their roots to `PYTHONPATH` in
  the env file: `PYTHONPATH=/workspace/nudorms/backend:/workspace/mast3r:/workspace/mast3r/dust3r`.
  Worker boot log + Python imports both work after this.

### 7. Bootstrap pip-installs only run on fresh clone
- **Symptom:** Pre-staged tarball-extracted repos at `/workspace/<name>` make
  the `if [ ! -d ... ]` clone block skip — but the `pip install -e <name>`
  was *inside* that if-block too, so dependencies never got installed. VGGT,
  Scaffold-GS, Difix3D, PGSR, 3DGUT all silently unimportable at worker boot.
- **Fix:** Restructure each block: clone gated on dir existence, pip install
  runs unconditionally (no-op if already satisfied).

### 8. Wrong GitHub paths for PGSR and 3DGUT
- **Symptom:** Both 404'd. `hmanhng/pgsr` and `nv-tlabs/3DGUT` don't exist.
- **Fix:** Correct upstream paths:
  - PGSR → `zju3dv/PGSR`
  - 3DGUT → `nv-tlabs/3dgrut` (the relightable variant subsumes 3DGUT)

### 9. tmux / aria2c / unzip not preinstalled
- **Fix:** First-run `apt-get install -y tmux aria2 unzip` before any
  background-running commands.

### 10. Container disk wipe on volume upgrade
- **Symptom:** Adding 250 GB container disk via RunPod console wipes
  `/usr/local/bin/colmap`, all pip packages, `/tmp/*` — but `/workspace/`
  survives because it's a separate volume.
- **Fix:** Make the bootstrap idempotent so it can rebuild from scratch
  in ~25 min when the container disk is fresh. Keep big assets
  (MASt3R weights, repo clones, glomap deps cache) under `/workspace/`.

---

## Pose stage

### 11. VGGT poses produced in disjoint coordinate systems when chunking
- **Symptom:** With `CHUNK_SIZE=100` and 152 frames, VGGT ran in 9 seconds
  but failed the `registered_images / total >= 0.85` quality gate.
- **Cause:** VGGT outputs poses relative to the first frame of *each
  forward pass*. Concatenating chunk 1's poses with chunk 2's poses yields
  nonsense because they're in different coordinate frames.
- **Fix (interim):** Force single-pass — bail with `failure_reason` if
  `total > CHUNK_SIZE`. Default `CHUNK_SIZE` raised to 400 to cover any
  reasonable casual capture.
- **Fix (proper):** Load model + inputs in `bfloat16` so 100+ frames fit
  in 16 GB VRAM; `model.to(device, dtype=torch.bfloat16)` halves weight
  memory (~5 GB → ~2.5 GB on the 1B model).
- **Fix (long-term):** Multi-chunk pose merging via overlapping frames
  + relative-pose chaining. Not yet implemented.

### 12. MASt3R SGA cache fills container disk
- **Symptom:** SUBSAMPLE=16 + 152 frames + sliding window pairs grew the
  per-pair torch.save cache to 39 GB at 25% completion (linear extrapolation
  → 150 GB total). 50 GB container disk hit 100% mid-run, all SQLite writes
  started failing with "database or disk is full" downstream.
- **Fix #1:** Container disk → 250 GB (orchestrator preflight already
  enforces ≥30 GB free in `/tmp`).
- **Fix #2:** `WIN_SIZE` 20 → 10 halves total pairs and halves both runtime
  (75 → 38 min) and cache size with negligible quality loss for forward-walking
  captures. `NUDORMS_MAST3R_WIN=10` is now the default.
- **Note:** `NUDORMS_MAST3R_CACHE` *must* stay on local disk (e.g. `/tmp`).
  `/workspace` is MooseFS on RunPod and races against torch's zip reopen,
  silently corrupting cache files mid-run.

---

## Pipeline stages

### 13. `2DGS` and `mesh_export` were stubs returning `ok=True` with fake artifacts
- **Symptom:** Scan rows in the DB ended up with `mesh_key=scans/<id>/mesh.glb`
  but no such object existed in R2; presigned URLs 404. The orchestrator
  treated the stages as successes.
- **Fix:** Both stubs now `ok=False`. Orchestrator already handled None
  mesh, so `mesh_key=None` is now truthfully recorded.
- **Real fix (long-term):** TSDF-integrate Depth Anything V2 priors with
  the trained pose camera positions to produce a real mesh.

### 14. Difix3D wrapper assumed a 3DGS-aware CLI; upstream is image-level
- **Symptom:** Wrapper passes `--input_path PLY --colmap_dir ...` but
  upstream `src/inference_difix.py` takes `--input_image / --prompt`.
- **Cause:** Full Difix3D+ flow needs render → diffuse-clean → distill,
  not a one-shot subprocess.
- **Fix:** `difix3d.available()` always returns False until the multi-step
  refinement loop is implemented. Orchestrator skips the refining stage
  cleanly.

### 15. Scaffold-GS / PGSR / 3DGUT depend on CUDA submodules that don't pip-install
- **Symptom:** Subprocess crashes immediately on `import diff_gaussian_rasterization`.
- **Cause:** Upstream Scaffold-GS expects `submodules/diff-gaussian-rasterization`
  + `submodules/simple-knn` to be cloned via git submodule and pip-installed
  separately. PGSR is the same plus `fused-ssim`. 3DGUT needs `pytorch3d`,
  which doesn't have a wheel for this torch/CUDA combo.
- **Fix (interim):** All three now import-check their CUDA deps via
  `available()` (or at top of `run()` for Scaffold-GS) and return
  `ok=False` when missing, so the orchestrator falls through to gsplat
  MCMC + skips PGSR mesh entirely instead of paying ~2 min to spawn a
  doomed subprocess.
- **Fix (long-term):** Add the submodule clone + pip install steps to
  `runpod_bootstrap.sh`, gated behind a flag because they take ~5 min
  each to build from source.

### 16. `eval.py` was returning hardcoded `psnr=30.0`
- **Symptom:** Every scan "passed" the PSNR ≥ 28 gate regardless of actual
  rendering quality. The retry loop and `NEEDS_RECAPTURE` decisions were
  meaningless.
- **Fix:** Rewrote `eval.py` to render the trained PLY at every holdout
  camera using gsplat's rasterizer, compute PSNR / SSIM (scipy) / LPIPS
  (optional via `lpips` package), and produce a 4×4 grid of region-PSNR
  diagnostics for the user feedback page.

---

## Worker / DB / queue

### 17. Celery rejects `rediss://` without `ssl_cert_reqs`
- **Symptom:** `ValueError("A rediss:// URL must have parameter ssl_cert_reqs ...")`
  on worker boot when broker URL has no query string.
- **Fix:** Append `?ssl_cert_reqs=CERT_REQUIRED` to the Upstash Redis URL
  in both `/tmp/nudorms_pod.env` and `backend/.env`. Otherwise the worker
  crashes during banner emit.

### 18. Worker holds stale DB connection if `scans` table didn't exist at start
- **Symptom:** Inserted scan row via Python script *after* worker booted →
  worker still saw `sqlite3.OperationalError: no such table: scans`.
- **Cause:** SQLAlchemy connection pool keeps the schema cached at first
  connection time. If the file existed but was empty, the engine cached
  that empty schema.
- **Fix:** Always run `init_db()` (Base.metadata.create_all) before the
  worker boots. Easiest: have `app.celery_app` import call `init_db()` at
  module load. (Workaround for now: kill + restart worker after manually
  populating the DB.)

### 19. Mac API + pod worker have separate SQLite files
- **Symptom:** `POST /scans` on Mac wrote to `backend/nudorms.db`. Worker
  on pod looked at `/workspace/nudorms.db`. Different files. Worker can't
  see the scan row that the API wrote.
- **Fix (interim):** Insert the scan row directly on the pod via Python
  script before dispatching the celery task. Both sides know
  `raw_video_key=scans/<id>/raw.mp4` deterministically since the API
  generates it from the scan_id.
- **Fix (production):** Move both sides to a shared Postgres via
  `DATABASE_URL` (Neon free tier or similar). The current architecture
  comment in CLAUDE.md says this; the bootstrap should set it up.

### 20. Celery task wrapper missing kwarg after orchestrator change
- **Symptom:** `TypeError: run_pipeline_task() got an unexpected keyword
  argument 'imu_key'`.
- **Cause:** I added `imu_key` to `run_pipeline()` signature but forgot to
  add it to the `@celery.task` wrapper.
- **Fix:** `def run_pipeline_task(self, scan_id, imu_key=None)` and pass
  through to `run_pipeline`. Lesson: when the orchestrator's signature
  grows, `app/celery_app.py` is the matching file to update.

---

## Auto-stop / RunPod API

### 21. RunPod GraphQL API returns 403 for `rpa_*` keys
- **Symptom:** `https://api.runpod.io/graphql` with `Authorization: Bearer rpa_...`
  returns HTTP 403 Forbidden.
- **Cause:** New `rpa_*` keys go through the REST API, not the legacy
  GraphQL endpoint.
- **Fix (TBD):** Switch `auto_stop.py` from GraphQL `podStop` mutation to
  REST `POST https://rest.runpod.io/v1/pods/{pod_id}/stop` with the same
  `Authorization: Bearer rpa_...` header. Not yet implemented; the user
  manually stops the pod in the meantime.

---

## Patterns / lessons

- **Pre-cache anything large** on `/workspace` (MASt3R weights, FetchContent
  deps) so a container-disk wipe doesn't force a re-download. The volume
  survives.
- **Idempotent bootstrap.** Every "if [ ! -d ... ]" should *only* gate the
  download. The pip install / configure step runs every time — it's a
  fast no-op when satisfied, and a real install when the container disk
  was wiped.
- **Stubs must fail loudly.** `ok=True` with a fake artifact path is
  worse than `ok=False` because the orchestrator records a non-existent
  R2 key into the DB and the user gets a 404 from the presigned URL.
- **Validate before dispatching long-running tasks.** A pre-flight that
  imports every module the pipeline needs + does a R2 read + checks
  `/tmp` free space takes ~10 seconds and saves hours of waiting only to
  crash at training stage.
- **`task_acks_late=True`** + a properly drained queue + careful
  validation = safe to restart workers without losing tasks.
- **Single-connection HTTPS is unusably slow on RunPod.** Use aria2c.
- **Always confirm RUNPOD_POD_ID** before touching the auto-stop logic.
  It's at `/proc/1/environ`, never in interactive shell env.
- **Volume vs container disk:** `/workspace` is the persistent volume.
  `/`, `/tmp`, `/usr/local/bin` are container disk and get wiped when
  the pod is stopped/upgraded.
- **The MASt3R SGA cache is the worst single offender** for pod disk
  budgeting. ~150 GB at 152 frames + WIN=10 + SUBSAMPLE=16. Plan for it.

---

## Open / unresolved

Things still broken or papered-over. Each is a real gap; the pipeline runs
end-to-end despite them but quality / cost / reliability all suffer.

### A. Auto-stop pod via RunPod REST API
- **Status:** Disabled (`NUDORMS_AUTO_STOP_POD=0`). User stops the pod
  manually after each run, paying for ~5–15 min of idle billing while
  noticing the run is done.
- **Why:** GraphQL `podStop` mutation in `auto_stop.py` returns 403 with
  `rpa_*` keys (entry 21).
- **Fix needed:** Switch `auto_stop.py` to
  `POST https://rest.runpod.io/v1/pods/{pod_id}/stop` with the same
  `Authorization: Bearer rpa_...` header. Verify the auth works with a
  `GET /v1/pods/{pod_id}` first.
- **Impact:** ~$0.10–0.30 per scan in idle billing.

### B. Real Difix3D+ refinement loop
- **Status:** `difix3d.available()` hardcoded to False.
- **Why:** Upstream nv-tlabs/Difix3D ships an image-level diffusion CLI;
  the full Difix3D+ flow needs render → diffuse-clean → distill back into
  3DGS via fine-tuning (entry 14).
- **Fix needed:** Wrapper that
  1. Renders novel views from the trained PLY at, say, 16 random poses
  2. Calls `inference_difix.py --input_image <rendered.png> --prompt "..." --output_dir ...`
  3. Spawns a fine-tune subprocess that adds the cleaned views as
     additional supervision and re-trains for ~5k iterations.
- **Impact:** Plan doc claims ~2× FID improvement. Real but optional.

### C. Real 2DGS / mesh stage
- **Status:** Both stubs return `ok=False`. `mesh_key` is always None.
- **Why:** Was punted; the splat is what users actually navigate.
- **Fix needed:** TSDF-integrate the Depth Anything V2 priors with the
  trained pose camera positions (Open3D's `ScalableTSDFVolume`), run
  marching cubes, decimate to ~50k faces, RANSAC-align floor to z=0,
  export glTF.
- **Impact:** No floor plan / dimensions / AR placement / glTF fallback
  for non-WebGL2 clients.

### D. CUDA submodule installs for Scaffold-GS / PGSR / 3DGUT
- **Status:** All three `available()` functions import-check the missing
  CUDA exts and return False on this pod, so the orchestrator falls
  through to gsplat MCMC + skip mesh.
- **Why:** Bootstrap doesn't build `submodules/diff-gaussian-rasterization`,
  `submodules/simple-knn`, `fused-ssim`, or fetch a working `pytorch3d`
  wheel.
- **Fix needed:**
  ```bash
  cd /workspace/Scaffold-GS && git submodule update --init --recursive
  pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
  pip install fused-ssim
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  ```
  Each takes ~5 min to compile from source. Gate behind a flag because
  failures are common on torch/CUDA combinations.
- **Impact:** ~0.5 dB PSNR loss vs Scaffold-GS, no PGSR mesh, no 3DGUT
  reflection handling.

### E. VGGT multi-chunk pose merging for >400 frames
- **Status:** Bails with `failure_reason` if `total > CHUNK_SIZE` (entry 11).
- **Why:** Per-chunk poses live in independent coordinate frames; naively
  concatenating gives nonsense.
- **Fix needed:** Process overlapping chunks (e.g. 5-frame overlap),
  compute the rigid transform aligning chunk 2's overlap frames to chunk
  1's poses, apply to all of chunk 2. Or: pose-graph optimization across
  chunk boundaries.
- **Impact:** Captures with >400 keyframes silently fall through to
  MASt3R (~10× slower). Today's 152-frame test6 fits, so unblocked.

### F. Shared DB for Mac API + pod worker
- **Status:** Each side has its own `nudorms.db` (`backend/nudorms.db` on
  Mac, `/workspace/nudorms.db` on pod). Manually inserting the scan row
  on the pod is the workaround (entry 19).
- **Why:** No shared Postgres set up.
- **Fix needed:** Free Neon Postgres → set `DATABASE_URL=postgres://...`
  on both sides. Update bootstrap to install `psycopg2-binary` (it's
  already in `pyproject.toml`).
- **Impact:** Every scan needs a manual sync step right now. Production
  blocker.

### G. `init_db()` at celery worker boot
- **Status:** SQLAlchemy connection pool caches an empty schema if the
  DB file existed but had no tables when the worker started. Workaround
  is to run `init_db()` from a separate Python script before booting the
  worker (entry 18).
- **Fix needed:** Call `init_db()` at module load in `app.celery_app`,
  so it runs in every worker process before the connection pool warms up.
  Or use `pool_pre_ping` + `pool_recycle` to force reconnect.
- **Impact:** First scan after a fresh container disk requires a manual
  init step.

### H. Bootstrap container-disk size sentinel
- **Status:** Hard-failed on the original 50 GB container disk because
  the MASt3R SGA cache hit ~150 GB. User had to manually upsize to 250 GB.
- **Fix needed:** First-line check in `runpod_bootstrap.sh`:
  ```bash
  free=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
  [ "$free" -lt 200 ] && { echo "ERROR: need >=200 GB container disk"; exit 1; }
  ```
- **Impact:** Silent OOMs mid-run on undersized pods.

### I. Pre-flight validation script committed to repo
- **Status:** We ran the ~9 import + R2 + Redis + RunPod-API + disk
  checks ad-hoc via SSH heredocs.
- **Fix needed:** `backend/scripts/preflight.py` that the celery worker
  runs at boot time before announcing ready, or that `runpod_bootstrap.sh`
  invokes after deps are installed but before `exec celery worker`.
- **Impact:** Each fresh pod requires manually re-deriving the validation
  steps.

### J. MASt3R SGA cache cleanup between runs
- **Status:** Cache at `/tmp/nudorms_mast3r_cache/<scan_id>` accumulates
  ~150 GB per scan and is never cleaned up. Eventually fills the
  container disk again.
- **Fix needed:** Either
  - Use a workdir-scoped path that gets removed when the temp dir does
    (move into `/tmp/scan-<id>/sga_cache/`), or
  - Add a cleanup at end of pose stage: `rm -rf /tmp/nudorms_mast3r_cache/<scan_id>`.
- **Impact:** Long-running pod will hit disk-full again after ~3–4 scans.

### K. End-to-end smoke test
- **Status:** Only `scripts/smoke_cpu_stages.py` exists (QC + frame
  selection). No GPU-stages smoke test that runs on a tiny synthetic
  scene.
- **Fix needed:** A test that uses ~10 frames + tiny VGGT inference
  + 100-iteration gsplat MCMC, asserts a PLY is produced. Saves a full
  pod-bootstrap-and-test loop when iterating on pipeline code.
- **Impact:** Discovering pipeline bugs costs ~30 min per iteration on
  the pod; should cost ~30 sec locally with mocked GPU.

### L. Frontend integration with new artifacts
- **Status:** SplatViewer migrated to optional Spark 2.0 + mkkellogg
  fallback, but the capture page hasn't been verified end-to-end with
  the new IMU upload + multi-station guidance flow.
- **Fix needed:** Manual end-to-end test on a phone, plus a `/demo`
  page that loads a known-good `.splat` and verifies it renders.
- **Impact:** Real-user capture flow not exercised.

### M. Mac local API loads `.env` from CWD, not project root
- **Status:** Running `uvicorn app.main:app` from `backend/` works because
  `.env` is in CWD. Running from repo root or anywhere else picks up the
  wrong env.
- **Fix needed:** `load_dotenv(find_dotenv())` in `app/main.py` so the
  CWD doesn't matter.
- **Impact:** Surprise debugging when the wrong creds are used.

### N. `pytorch3d` install path is brittle
- **Status:** No matching wheel; pip falls back to source build which
  needs CUDA + a long compile, often failing.
- **Fix needed:** Detect torch + CUDA versions, pull the right
  pre-built wheel from Facebook's release page (e.g.
  `https://anaconda.org/pytorch3d/pytorch3d/...`). Or vendor a working
  pip path in `requirements.txt`.
- **Impact:** 3DGUT permanently unavailable until this is fixed.

### O. RoPE2D CUDA kernel not compiled
- **Status:** dust3r prints `Warning, cannot find cuda-compiled version of
  RoPE2D, using a slow pytorch version instead`. Real cost on MASt3R SGA.
- **Fix needed:** Build the CUDA op:
  ```bash
  cd /workspace/mast3r/dust3r/croco/models/curope && python setup.py build_ext --inplace
  ```
  And add to bootstrap.
- **Impact:** Estimated 20–30% speedup on MASt3R pose stage. Today's run
  is paying the slow-path cost (~38 min instead of ~28 min).
