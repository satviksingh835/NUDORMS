# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NUDORMS is a platform where students record a phone walkthrough of their dorm and viewers navigate a 3D Gaussian Splat recreation in the browser.

- Students film with a phone → upload → pipeline reconstructs → viewers explore via splat renderer
- The Mac runs the API + frontend. GPU work runs on a rented pod (RunPod / Lambda Labs).
- They communicate via managed Upstash Redis (task broker) and Cloudflare R2 (artifact storage) — the two sides never reach each other directly.

## Running locally

### Prerequisites
Copy `.env.example` → `backend/.env` and fill in `REDIS_URL`, `S3_*`, and optionally `DATABASE_URL` (SQLite default is fine for dev).

### Local infra (MinIO + Redis, instead of R2 + Upstash)
```bash
cd infra && docker compose up -d
```
Uses MinIO on `:9000` (console `:9001`) and Redis on `:6379`. The `minio-init` container creates the `nudorms` bucket automatically.

### API
```bash
cd backend
set -a && source .env && set +a
uv run uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Vite proxies `/api/*` → `http://localhost:8000`, so the frontend never makes direct API calls with a hardcoded port.

### Frontend build / type-check
```bash
cd frontend
npm run build    # tsc -b && vite build
```

### GPU worker (on the rented pod)
See `infra/RUNPOD.md`. The bootstrap script installs COLMAP + GLOMAP from source (~10 min on first boot). GPU-only Python deps (`torch`, `gsplat`, `pycolmap`, `lpips`) are in `pyproject.toml [gpu]`.

## Smoke testing pipeline stages

Run CPU-only stages (QC + frame selection) locally on a video file without needing a Scan DB row or R2:
```bash
cd backend
uv run python scripts/smoke_cpu_stages.py /path/to/walkthrough.mp4
```

Other smoke scripts for the GPU stages:
- `scripts/smoke_pose_stage.py` — runs pose ensemble on a frames dir
- `scripts/smoke_train_stage.py` — runs gsplat training

## Architecture

### Backend (`backend/`)

**`app/`** — FastAPI API server:
- `main.py` — two real endpoints: `POST /scans` (uploads video to R2, enqueues Celery task) and `GET /scans/{id}` (returns status + presigned URLs for splat/mesh/LOD outputs).
- `celery_app.py` — Celery config. One queue named `gpu`; the worker runs on the GPU pod.
- `models.py` — `Scan` SQLAlchemy model + `ScanStatus` enum (the full status lifecycle). `ScanResponse` is the Pydantic API shape.
- `storage.py` — thin boto3 wrapper; `S3_ENDPOINT` env var switches between MinIO and R2/S3.
- `db.py` — SQLAlchemy setup; defaults to SQLite, switches to Postgres via `DATABASE_URL`.

**`pipeline/`** — runs inside the Celery worker on the GPU pod:
- `orchestrator.py` — end-to-end driver. Calls each stage in order, updates the DB at every transition, handles the PSNR-gated auto-retry loop (PSNR ≥ 28 passes, 24–28 retries once, <24 asks for recapture).
- `types.py` — `StageResult(ok, metrics, artifacts, failure_reason)`. Every stage returns one.
- `ingest_qc.py` — CPU-only pre-flight: duration, sharpness, brightness, exposure stability on 30 sampled frames.
- `frame_select.py` — greedy parallax-based frame selection: decode at 6 fps, drop blurry frames, keep frames whose optical-flow magnitude to the previous kept frame falls in [3, 120] px.
- `poses/ensemble.py` — tries MASt3R → GLOMAP → COLMAP in that order, picks the first that clears quality thresholds (`inlier_ratio ≥ 0.45`, `reproj_error ≤ 1.5 px`, `registered_images / total ≥ 0.85`). If none clears, picks best and continues.
- `poses/mast3r.py` — MASt3R's own Sparse Global Alignment (`cloud_opt.sparse_ga`), skipping COLMAP's incremental mapper entirely. Outputs are repacked into COLMAP binary format so the trainer is unaffected.
- `train/gsplat_mcmc.py` — MCMC Gaussian Splat training via `gsplat`.
- `train/twodgs.py` — 2DGS for mesh extraction.
- `train/eval.py` — PSNR evaluation gating the retry loop.
- `compress/` — LOD generation (`lod.py`) and SOGS compression (`sogs.py`).
- `mesh_export.py` — exports mesh artifact.

### Frontend (`frontend/`)

React + Vite + TypeScript. No component library.

**Routes** (`src/main.tsx`):
- `/capture` → `CapturePage` — guided phone capture
- `/scans/:id` → `StatusPage` — polls `GET /scans/:id` and shows pipeline status
- `/scans/:id/view` → `ViewerPage` — 3D splat viewer once READY
- `/demo` → `DemoPage` — loads a local `.ply` or `.splat` file for testing the viewer

**Key components**:
- `capture/GuidedRecorder.tsx` — streams 4K30, locks exposure/focus via `applyConstraints`, tracks yaw coverage via device gyro, warns on fast motion. Calls `onComplete(blob)` when done.
- `capture/CoverageMap.ts` — accumulates device motion events into a yaw-coverage score.
- `viewer/SplatViewer.tsx` — wraps `@mkkellogg/gaussian-splats-3d`. Takes `lodUrls[]` (streaming from R2) or `file` (local demo). The library is excluded from Vite's `optimizeDeps` to avoid ESM issues.
- `src/api.ts` — all API calls (`uploadScan`, `getScan`, `getFeedback`). Calls go to `/api/*` which Vite proxies to `:8000`.

### Infra (`infra/`)
- `docker-compose.yml` — Redis + MinIO for local dev.
- `gpu-worker.Dockerfile` — worker image.
- `runpod_bootstrap.sh` — installs COLMAP, GLOMAP, clones the repo, starts the Celery worker. Run once per pod session.

## Key environment variables

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + backend |
| `S3_ENDPOINT` | `http://localhost:9000` | MinIO (local) or R2/S3 (prod) |
| `S3_BUCKET` | `nudorms` | Object storage bucket |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `minioadmin` | Storage credentials |
| `DATABASE_URL` | `sqlite:///./nudorms.db` | SQLAlchemy DB connection |
| `NUDORMS_MAST3R_MODEL` | `/workspace/mast3r_models/MASt3R.pth` | Path to MASt3R weights |
| `NUDORMS_MAST3R_WIN` | `20` | Sliding-window size for SGA |
| `NUDORMS_MAST3R_SUBSAMPLE` | `16` | Anchor density for SGA (lower = more VRAM) |
| `NUDORMS_MAST3R_CACHE` | `/tmp/nudorms_mast3r_cache` | SGA pair cache; use local disk on network-FS pods |

## Pose estimation notes

MASt3R's SGA runs first because SIFT/GLOMAP register very few frames on textureless dorm walls. MASt3R uses learned features and its own global optimizer, bypassing COLMAP's incremental mapper which is the brittle stage for casual capture. Quality thresholds gating advancement are defined in `poses/ensemble.py`.

## Pipeline status flow

```
QUEUED → QC → FRAMES → POSING → TRAINING → EVALUATING ⟶ MESHING → COMPRESSING → READY
                                                      ↓           ↑
                                                   RETRYING ──────┘
                              NEEDS_RECAPTURE (from QC, POSING, or EVALUATING)
                              FAILED (from FRAMES or unrecoverable errors)
```
