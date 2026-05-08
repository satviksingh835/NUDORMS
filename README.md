# NUDORMS

Platform where students record a phone walkthrough of their dorm and viewers navigate a 3D recreation in the browser.

This repo currently contains the **video → 3D reconstruction pipeline** scaffold. Plan: `~/.claude/plans/can-you-start-desgining-hazy-forest.md`.

## Layout

- `frontend/` — React + Vite + TypeScript. Guided capture UI and splat viewer.
- `backend/` — FastAPI API + Celery worker. Hosts the reconstruction pipeline.
- `infra/` — docker-compose for local dev (Redis, MinIO, API).

## Status

Scaffold complete. Pipeline stages are stubbed — the orchestrator runs end-to-end but every ML stage has a `# TODO` body. Implementation order: `ingest_qc` → `frame_select` → `poses/glomap` → `train/gsplat_mcmc` → `eval` → `compress`.

## Setup (RunPod / Lambda Labs path)

GPU work runs on a rented pod; the Mac runs the API and frontend. Both connect to managed Upstash (Redis broker) and Cloudflare R2 (artifact storage). See `infra/RUNPOD.md` for step-by-step.

```bash
# 1. one-time: create Upstash + R2, copy .env.example → backend/.env, fill in.

# 2. local API + frontend (Mac is fine)
cd backend && set -a && source .env && set +a && uv run uvicorn app.main:app --reload
cd frontend && npm install && npm run dev

# 3. GPU worker on a rented pod — see infra/RUNPOD.md
```
