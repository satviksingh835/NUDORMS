from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, File
from typing import Optional

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
async def create_scan(
    video: UploadFile,
    imu: Optional[UploadFile] = File(default=None),
    stops: Optional[str] = Form(default=None),
    db: Session = Depends(_db),
) -> ScanResponse:
    import json as _json
    scan_id = uuid.uuid4().hex
    raw_key = scan_key(scan_id, "raw.mp4")
    put(raw_key, await video.read(), content_type=video.content_type or "video/mp4")

    imu_key = None
    if imu is not None:
        imu_data = await imu.read()
        if imu_data:
            imu_key = scan_key(scan_id, "imu.jsonl")
            put(imu_key, imu_data, content_type="application/jsonl")

    stops_list = None
    if stops:
        try:
            stops_list = _json.loads(stops)
        except Exception:
            stops_list = None

    scan = Scan(
        id=scan_id,
        status=ScanStatus.QUEUED.value,
        raw_video_key=raw_key,
        metrics={"stops": stops_list} if stops_list else None,
    )
    db.add(scan)
    db.commit()

    celery.send_task(
        "pipeline.run",
        args=[scan_id],
        kwargs={"imu_key": imu_key, "stops": stops_list},
        queue="gpu",
    )
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
        graph_url=presigned_get(scan.graph_key) if scan.graph_key else None,
        pano_urls={k: presigned_get(v) for k, v in (scan.pano_keys or {}).items()} or None,
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
