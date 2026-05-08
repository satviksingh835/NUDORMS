# Running the GPU worker on RunPod / Lambda Labs

The Mac runs the API and the frontend. The rented pod runs the Celery worker
that does pose estimation + splat training. They communicate via managed
Redis (broker) and managed S3-compatible storage (artifacts) — neither side
needs to reach the other directly.

```
┌──────────────────────┐   Redis (Upstash)   ┌────────────────────────┐
│ Mac dev box          │ ◀──── jobs ────────▶│ RunPod / Lambda Labs   │
│  - Vite dev server   │                     │  - Celery worker       │
│  - FastAPI uvicorn   │                     │  - GLOMAP + gsplat     │
└──────────┬───────────┘                     └─────────────┬──────────┘
           │                                               │
           └──────────── R2 / S3 (raw video, splats) ──────┘
```

## One-time setup

1. **Upstash Redis** — create a database, copy the `rediss://...` connection string.
2. **Cloudflare R2** — create a bucket called `nudorms`, mint an API token with
   read/write on that bucket. Note the account ID, access key, secret.
3. Copy `.env.example` → `backend/.env` and fill in those values.
4. Push this repo to a private GitHub repo so the pod can clone it.

## Per-session: rent a pod and start the worker

1. **RunPod**: pick a community-cloud RTX 4090 or A40 (cheap), CUDA 12.x template
   ("RunPod PyTorch 2.4" works). **Lambda Labs**: an A10 or A100.
2. In the pod's "Environment variables" panel, paste the same values from your
   `.env`, plus:
   ```
   NUDORMS_REPO=https://USERNAME:TOKEN@github.com/USERNAME/NUDORMS.git
   NUDORMS_REF=main
   ```
3. SSH in (or paste into "On-start command"):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/USERNAME/NUDORMS/main/infra/runpod_bootstrap.sh | bash
   ```
   First boot installs COLMAP + GLOMAP from source (~10 min). Subsequent boots
   skip those steps and start the worker in seconds.

## Per-session: start the API + frontend on the Mac

```bash
# in one shell
cd backend && set -a && source .env && set +a && uv run uvicorn app.main:app --reload

# in another shell
cd frontend && npm install && npm run dev
```

Open the Vite URL on your phone (same Wi-Fi or via `--host`), record a
walkthrough, and the job hops to the GPU pod via Upstash within ~1s.

## Cost shape

- Upstash Redis: free tier (10k commands/day) is plenty for dev.
- R2: free 10 GB stored, free egress to anywhere.
- 4090 on RunPod: ~$0.40/hr; a typical scan trains in ~15 min, so ~$0.10/scan.
  Stop the pod when not actively iterating; spin up only when needed.
