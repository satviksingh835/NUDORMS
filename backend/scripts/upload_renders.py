"""Upload the first N render PNGs to R2 under debug/ for visual inspection."""
from __future__ import annotations

import glob
import os
import sys

from app.storage import put


def main(renders_dir: str = "/workspace/test5_out/splat/renders", n: int | str = 6) -> int:
    n = int(n)
    paths = sorted(glob.glob(os.path.join(renders_dir, "*.png")))[:n]
    if not paths:
        print(f"no PNGs found in {renders_dir}", file=sys.stderr)
        return 1
    for p in paths:
        name = os.path.basename(p)
        with open(p, "rb") as f:
            put(f"debug/{name}", f, content_type="image/png")
        print(f"uploaded {name}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(*args))
