"""SOG / .splat compression — reduces PLY size 15–20× before upload.

Primary: playcanvas/splat-transform via `npx splat-transform` (Node.js required).
  Produces a .sogs file: ~15–20× smaller than raw PLY, Spark-native format.

Fallback: write a .splat binary (widely supported by most GS viewers).
  Format: 32 bytes/Gaussian, sorted descending by opacity × max-scale.
  Not as compressed as SOG but compatible with @mkkellogg and SuperSplat.

Both outputs are uploaded to R2 under scans/<scan_id>/scene.sogs (or .splat).
"""
from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from app.storage import put, scan_key

from ..types import StageResult

log = logging.getLogger("nudorms.pipeline.compress.sogs")


# ---------------------------------------------------------------------------
# PLY reader (minimal — only reads fields needed for .splat encoding)
# ---------------------------------------------------------------------------

def _read_splat_fields(ply_path: Path) -> dict[str, np.ndarray]:
    """Read a 3DGS PLY. Returns dict of field_name -> ndarray [N]."""
    with open(ply_path, "rb") as f:
        raw = f.read()

    end_tag = b"end_header\n"
    header_end = raw.index(end_tag) + len(end_tag)
    header = raw[:header_end].decode("latin-1")

    props = []
    n_verts = 0
    in_vertex = False
    for line in header.split("\n"):
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element") and in_vertex:
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            props.append(line.split()[-1])

    n_props = len(props)
    data = np.frombuffer(raw[header_end:], dtype=np.float32).reshape(n_verts, n_props)
    prop_idx = {p: i for i, p in enumerate(props)}

    def _get(*names):
        for n in names:
            if n in prop_idx:
                return data[:, prop_idx[n]]
        return None

    return {
        "x": _get("x"), "y": _get("y"), "z": _get("z"),
        "scale_0": _get("scale_0"), "scale_1": _get("scale_1"), "scale_2": _get("scale_2"),
        "opacity": _get("opacity"),
        "f_dc_0": _get("f_dc_0"), "f_dc_1": _get("f_dc_1"), "f_dc_2": _get("f_dc_2"),
        "rot_0": _get("rot_0"), "rot_1": _get("rot_1"),
        "rot_2": _get("rot_2"), "rot_3": _get("rot_3"),
    }


# ---------------------------------------------------------------------------
# .splat binary writer
# ---------------------------------------------------------------------------

def _sh_dc_to_rgb(dc: float) -> int:
    """SH DC coefficient → 0–255 sRGB byte."""
    linear = dc * 0.28209479177387814 + 0.5
    return max(0, min(255, int(linear * 255)))


def _write_splat_binary(fields: dict, out_path: Path) -> None:
    """Write .splat binary: each Gaussian = 32 bytes, sorted by importance."""
    N = len(fields["x"])

    x  = fields["x"]
    y  = fields["y"]
    z  = fields["z"]
    s0 = np.exp(fields["scale_0"])
    s1 = np.exp(fields["scale_1"])
    s2 = np.exp(fields["scale_2"])
    op = 1.0 / (1.0 + np.exp(-fields["opacity"]))   # sigmoid

    importance = op * np.maximum(np.maximum(s0, s1), s2)
    order = np.argsort(-importance)

    r0, r1 = fields["rot_0"], fields["rot_1"]
    r2, r3 = fields["rot_2"], fields["rot_3"]
    norms = np.sqrt(r0**2 + r1**2 + r2**2 + r3**2) + 1e-8
    r0, r1, r2, r3 = r0/norms, r1/norms, r2/norms, r3/norms

    dc0, dc1, dc2 = fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]

    buf = bytearray(N * 32)
    view = memoryview(buf)
    for i, idx in enumerate(order):
        off = i * 32
        struct.pack_into("<fff", view, off,
                         float(x[idx]), float(y[idx]), float(z[idx]))
        struct.pack_into("<fff", view, off + 12,
                         float(s0[idx]), float(s1[idx]), float(s2[idx]))
        struct.pack_into("<BBBB", view, off + 24,
                         _sh_dc_to_rgb(float(dc0[idx])),
                         _sh_dc_to_rgb(float(dc1[idx])),
                         _sh_dc_to_rgb(float(dc2[idx])),
                         max(0, min(255, int(op[idx] * 255))))

        def _q(v):
            return max(0, min(255, int(float(v) * 128 + 128)))
        struct.pack_into("<BBBB", view, off + 28,
                         _q(r0[idx]), _q(r1[idx]), _q(r2[idx]), _q(r3[idx]))

    out_path.write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
# SOG via splat-transform (npx)
# ---------------------------------------------------------------------------

def _try_splat_transform(ply_path: Path, out_path: Path) -> bool:
    npx = shutil.which("npx")
    if npx is None:
        return False
    try:
        result = subprocess.run(
            [npx, "--yes", "splat-transform", str(ply_path), str(out_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True
        log.warning("splat-transform rc=%d: %s", result.returncode, result.stderr[:300])
    except Exception as e:
        log.warning("splat-transform failed: %s", e)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scan_id: str, workdir: Path, train_artifacts: dict) -> StageResult:
    ply_path = Path(train_artifacts.get("ply_path", ""))
    if not ply_path.exists():
        return StageResult(False, {}, {}, failure_reason=f"PLY not found: {ply_path}")

    size_mb_before = ply_path.stat().st_size / 1024 / 1024

    sogs_out = workdir / "scene.sogs"
    if _try_splat_transform(ply_path, sogs_out):
        out_file = sogs_out
        ext = "sogs"
    else:
        log.info("splat-transform unavailable — writing .splat binary")
        splat_out = workdir / "scene.splat"
        try:
            fields = _read_splat_fields(ply_path)
            _write_splat_binary(fields, splat_out)
        except Exception as e:
            return StageResult(False, {}, {}, failure_reason=f"PLY encode error: {e}")
        out_file = splat_out
        ext = "splat"

    size_mb_after = out_file.stat().st_size / 1024 / 1024
    ratio = size_mb_before / max(size_mb_after, 0.001)
    log.info("%.1f MB → %.1f MB (%.1f× ratio, %s)", size_mb_before, size_mb_after, ratio, ext)

    key = scan_key(scan_id, f"scene.{ext}")
    with open(out_file, "rb") as fh:
        put(key, fh.read(), content_type="application/octet-stream")

    return StageResult(
        ok=True,
        metrics={"splat_size_mb": round(size_mb_after, 2), "compression_ratio": round(ratio, 1),
                 "format": ext},
        artifacts={"splat_key": key, "splat_local": str(out_file), "splat_format": ext},
    )
