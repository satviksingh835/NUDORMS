"""Upload a .ply file to R2 under a given key (default: ply basename)."""
from __future__ import annotations

import os
import sys

from app.storage import put


def main(local: str, key: str | None = None) -> int:
    if not os.path.exists(local):
        print(f"file not found: {local}", file=sys.stderr)
        return 1
    key = key or os.path.basename(local)
    print(f"uploading {os.path.getsize(local) / 1024 / 1024:.1f} MB to r2://{key} ...")
    with open(local, "rb") as f:
        put(key, f, content_type="application/octet-stream")
    print("done")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    sys.exit(main(*sys.argv[1:]))
