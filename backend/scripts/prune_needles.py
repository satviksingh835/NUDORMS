"""Remove needle Gaussians from a 3DGS PLY by scale-ratio filtering.

Needle artifacts occur when the optimizer stretches Gaussians to explain a
scene point as seen from inconsistent camera poses. They have an extreme
scale ratio: one axis is huge, the two others are tiny.

The scale properties in a 3DGS PLY are stored in log space, so the ratio
between the largest and smallest axis is exp(max_s - min_s).

Usage:
    cd backend
    uv run python scripts/prune_needles.py /path/to/input.ply [out.ply] [--ratio 10] [--opacity 0.05]

Defaults:
    --ratio   10   Keep Gaussians where max_scale / min_scale < 10
    --opacity 0.05 Also drop near-invisible Gaussians (sigmoid(opacity) < 0.05)
    out.ply        input stem + '_pruned.ply' next to input file

The script prints a histogram of ratio buckets so you can tune the threshold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prune(input_path: Path, output_path: Path, ratio_threshold: float, opacity_threshold: float) -> None:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("plyfile not installed. Run: uv add plyfile", file=sys.stderr)
        sys.exit(1)

    print(f"reading {input_path} ({input_path.stat().st_size / 1024 / 1024:.1f} MB) …")
    ply = PlyData.read(str(input_path))
    v = ply["vertex"]
    n_total = len(v)
    print(f"  {n_total:,} Gaussians")

    # Scales are stored as log(scale); reconstruct ratio in log space.
    s = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)  # (N, 3)
    log_ratio = s.max(axis=1) - s.min(axis=1)  # log(max/min)
    ratio = np.exp(log_ratio)

    # Print distribution so user can tune the threshold.
    thresholds = [3, 5, 10, 15, 20, 50, 100]
    print("\nScale-ratio distribution:")
    prev = 0
    for t in thresholds:
        count = int((ratio < t).sum()) - prev
        print(f"  ratio < {t:>4}: {count:>8,} Gaussians  ({count/n_total*100:5.1f}%)")
        prev = int((ratio < t).sum())
    count = int((ratio >= thresholds[-1]).sum())
    print(f"  ratio ≥ {thresholds[-1]:>4}: {count:>8,} Gaussians  ({count/n_total*100:5.1f}%)")

    # Opacity filter.
    op = sigmoid(np.asarray(v["opacity"], dtype=np.float32))

    keep_ratio = ratio < ratio_threshold
    keep_opacity = op >= opacity_threshold
    keep = keep_ratio & keep_opacity

    n_keep = int(keep.sum())
    n_drop_ratio = int((~keep_ratio).sum())
    n_drop_opacity = int((keep_ratio & ~keep_opacity).sum())
    print(f"\nThreshold: ratio < {ratio_threshold}, opacity >= {opacity_threshold}")
    print(f"  Dropped by ratio:   {n_drop_ratio:>8,} ({n_drop_ratio/n_total*100:.1f}%)")
    print(f"  Dropped by opacity: {n_drop_opacity:>8,} ({n_drop_opacity/n_total*100:.1f}%)")
    print(f"  Kept:               {n_keep:>8,} ({n_keep/n_total*100:.1f}%)")

    if n_keep == 0:
        print("ERROR: threshold too aggressive — no Gaussians would remain. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Rebuild the PLY with only the kept rows.
    dtype = v.data.dtype
    arr = np.empty(n_keep, dtype=dtype)
    for name in dtype.names:
        arr[name] = np.asarray(v[name])[keep]

    el = PlyElement.describe(arr, "vertex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], text=False).write(str(output_path))
    print(f"\nwrote {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", nargs="?", help="Output .ply file (default: input_pruned.ply)")
    parser.add_argument("--ratio", type=float, default=10.0, help="Max scale ratio to keep (default: 10)")
    parser.add_argument("--opacity", type=float, default=0.05, help="Min sigmoid(opacity) to keep (default: 0.05)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"file not found: {inp}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else inp.parent / f"{inp.stem}_pruned.ply"
    prune(inp, out, args.ratio, args.opacity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
