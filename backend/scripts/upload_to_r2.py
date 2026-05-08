"""Upload a local file to the R2 bucket configured in .env.

Usage:
    cd backend && set -a && source ../.env && set +a
    python scripts/upload_to_r2.py /path/to/file.mp4 [key]

If `key` is omitted, the basename of the file is used as the R2 key.
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.storage import put


def main(local: str, key: str | None = None) -> int:
    src = Path(local)
    if not src.exists():
        print(f"file not found: {src}", file=sys.stderr)
        return 1
    key = key or src.name
    with src.open("rb") as f:
        put(key, f, content_type="video/mp4")
    print(f"uploaded {src} -> r2://{key}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    sys.exit(main(*sys.argv[1:]))
