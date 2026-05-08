from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile

load_dotenv()  # picks up backend/.env (REDIS_URL, S3_*, DATABASE_URL)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .celery_app import celery
from .db import get_session, init_db
from .models import Scan, ScanResponse, ScanStatus
from .storage import presigned_get, put, scan_key


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="NUDORMS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db() -> Session:
    db = get_session()
    try:
        yield db
    finally:
        db.close()


@app.post("/scans", response_model=ScanResponse)
async def create_scan(video: UploadFile, db: Session = Depends(_db)) -> ScanResponse:
    scan_id = uuid.uuid4().hex
    raw_key = scan_key(scan_id, "raw.mp4")
    put(raw_key, await video.read(), content_type=video.content_type or "video/mp4")

    scan = Scan(id=scan_id, status=ScanStatus.QUEUED.value, raw_video_key=raw_key)
    db.add(scan)
    db.commit()

    celery.send_task("pipeline.run", args=[scan_id], queue="gpu")
    return ScanResponse(id=scan_id, status=ScanStatus.QUEUED)


@app.get("/scans/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: str, db: Session = Depends(_db)) -> ScanResponse:
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(404, "scan not found")

    return ScanResponse(
        id=scan.id,
        status=ScanStatus(scan.status),
        metrics=scan.metrics,
        error=scan.error,
        splat_url=presigned_get(scan.splat_key) if scan.splat_key else None,
        mesh_url=presigned_get(scan.mesh_key) if scan.mesh_key else None,
        lod_urls={k: presigned_get(v) for k, v in (scan.lod_keys or {}).items()} or None,
    )


@app.get("/scans/{scan_id}/feedback")
def get_feedback(scan_id: str, db: Session = Depends(_db)) -> dict:
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(404, "scan not found")
    return scan.feedback or {}


@app.delete("/scans/{scan_id}", status_code=204)
def delete_scan(scan_id: str, db: Session = Depends(_db)) -> None:
    scan = db.get(Scan, scan_id)
    if scan is not None:
        db.delete(scan)
        db.commit()
