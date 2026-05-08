"""Run QC + frame_select + GLOMAP on a local video. Designed for the GPU pod.

Usage:
    python scripts/smoke_pose_stage.py /path/to/walkthrough.mp4 [out_dir]

Prereqs (from infra/runpod_bootstrap.sh):
    colmap, glomap, python deps installed.

Output:
    out_dir/
      frames/        selected frames
      poses/sparse/0 COLMAP-format sparse reconstruction (cameras, images, points)
      metrics.json   stage metrics + winning pose method
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from pprint import pprint


def main(video: str, out: str | None = None) -> int:
    src = Path(video).resolve()
    if not src.exists():
        print(f"file not found: {src}", file=sys.stderr)
        return 1

    out_root = Path(out).resolve() if out else src.parent / f"{src.stem}_pose_out"
    out_root.mkdir(exist_ok=True)

    from pipeline import frame_select
    from pipeline.poses import ensemble as pose_ensemble

    # Inline QC (skips DB), reusing the smoke_cpu_stages helper.
    sys.path.insert(0, str(Path(__file__).parent))
    from smoke_cpu_stages import _run_qc_inline  # type: ignore

    with tempfile.TemporaryDirectory(prefix="nudorms-pose-") as tmp:
        workdir = Path(tmp)
        raw = workdir / "raw.mp4"
        shutil.copy(src, raw)

        print("\n── ingest_qc ──")
        qc = _run_qc_inline(workdir, raw)
        pprint({"ok": qc.ok, "metrics": qc.metrics, "artifacts": qc.artifacts})
        if not qc.ok:
            return 2

        print("\n── frame_select ──")
        sel = frame_select.run("smoke", workdir)
        pprint({"ok": sel.ok, "metrics": sel.metrics})
        if not sel.ok:
            return 3

        # Persist frames to out so they're available for any retry without re-decoding.
        frames_out = out_root / "frames"
        if frames_out.exists():
            shutil.rmtree(frames_out)
        shutil.copytree(sel.artifacts["frames_dir"], frames_out)

        print("\n── pose ensemble ──")
        poses = pose_ensemble.run("smoke", workdir, {"frames_dir": str(frames_out)})
        pprint({"ok": poses.ok, "metrics": poses.metrics,
                "failure_reason": poses.failure_reason})
        # Surface per-attempt failures (the ensemble swallows them otherwise).
        from pipeline.poses import colmap as colmap_mod, glomap as glomap_mod
        for name in ("glomap", "colmap"):
            log_path = workdir / f"poses_{name}" / "sparse"
            print(f"  {name} sparse dir exists: {log_path.exists()} -> {log_path}")
        if not poses.ok:
            (out_root / "metrics.json").write_text(json.dumps({
                "qc": qc.metrics, "frame_select": sel.metrics, "pose_failure": poses.failure_reason,
            }, indent=2))
            return 4

        # Copy the winning sparse model into out so we can inspect / hand to the trainer.
        winning_sparse = Path(poses.artifacts["sparse_dir"])
        sparse_out = out_root / "poses" / "sparse" / "0"
        sparse_out.parent.mkdir(parents=True, exist_ok=True)
        if sparse_out.exists():
            shutil.rmtree(sparse_out)
        shutil.copytree(winning_sparse, sparse_out)

        (out_root / "metrics.json").write_text(json.dumps({
            "qc": qc.metrics, "frame_select": sel.metrics, "poses": poses.metrics,
        }, indent=2))

        print(f"\nfull artifacts in: {out_root}")
        print(f"sparse model:      {sparse_out}")
        print(f"pose winner:       {poses.metrics.get('pose_winner')}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    sys.exit(main(*sys.argv[1:]))
