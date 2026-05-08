from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ScanStatus(str, Enum):
    QUEUED = "queued"
    QC = "qc"
    FRAMES = "frames"
    POSING = "posing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    RETRYING = "retrying"
    MESHING = "meshing"
    COMPRESSING = "compressing"
    READY = "ready"
    NEEDS_RECAPTURE = "needs_recapture"
    FAILED = "failed"


class Base(DeclarativeBase):
    pass


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default=ScanStatus.QUEUED.value)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Pipeline outputs / artifacts (S3 keys).
    raw_video_key: Mapped[str | None] = mapped_column(String, nullable=True)
    splat_key: Mapped[str | None] = mapped_column(String, nullable=True)
    mesh_key: Mapped[str | None] = mapped_column(String, nullable=True)
    lod_keys: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Quality + telemetry.
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    feedback: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ScanResponse(BaseModel):
    id: str
    status: ScanStatus
    progress: float = 0.0
    metrics: dict | None = None
    error: str | None = None
    splat_url: str | None = None
    mesh_url: str | None = None
    lod_urls: dict | None = None
