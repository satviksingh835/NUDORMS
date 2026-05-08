"""S3-compatible object storage wrapper.

Local dev points at MinIO via env vars; prod points at R2 or S3.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import BinaryIO

import boto3
from botocore.client import Config


@dataclass(frozen=True)
class StorageConfig:
    endpoint: str = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
    bucket: str = os.environ.get("S3_BUCKET", "nudorms")
    access_key: str = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    secret_key: str = os.environ.get("S3_SECRET_KEY", "minioadmin")
    region: str = os.environ.get("S3_REGION", "us-east-1")


def _client(cfg: StorageConfig | None = None):
    cfg = cfg or StorageConfig()
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name=cfg.region,
        config=Config(signature_version="s3v4"),
    )


def put(key: str, body: BinaryIO | bytes, content_type: str | None = None) -> str:
    cfg = StorageConfig()
    extra = {"ContentType": content_type} if content_type else {}
    _client(cfg).put_object(Bucket=cfg.bucket, Key=key, Body=body, **extra)
    return key


def download(key: str, dest) -> None:
    """Download an object to a local path."""
    cfg = StorageConfig()
    _client(cfg).download_file(cfg.bucket, key, str(dest))


def presigned_get(key: str, expires: int = 3600) -> str:
    cfg = StorageConfig()
    return _client(cfg).generate_presigned_url(
        "get_object",
        Params={"Bucket": cfg.bucket, "Key": key},
        ExpiresIn=expires,
    )


def scan_key(scan_id: str, *parts: str) -> str:
    return "/".join(("scans", scan_id, *parts))
