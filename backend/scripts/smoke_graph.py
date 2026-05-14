#!/usr/bin/env python3
"""Smoke test: build a graph from two stitched panos + a sparse_dir.

Usage:
    uv run python scripts/smoke_graph.py <sparse_dir> <pano1.jpg> <pano2.jpg>

Asserts:
  - graph.json has 2 nodes, 2 directed edges
  - all azimuths are in [0, 360)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    if len(sys.argv) < 4:
        print("Usage: smoke_graph.py <sparse_dir> <pano1.jpg> <pano2.jpg>")
        sys.exit(1)

    sparse_dir = Path(sys.argv[1])
    pano1 = Path(sys.argv[2])
    pano2 = Path(sys.argv[3])

    stitch_artifacts = {
        "panos": {"n0": str(pano1), "n1": str(pano2)},
        "node_frames": {"n0": [], "n1": []},
    }
    poses_artifacts = {"sparse_dir": str(sparse_dir)}

    # Monkey-patch storage.put so we don't need a real R2 connection
    import app.storage as storage_mod
    uploads = {}
    def fake_put(key, data, **kw):
        uploads[key] = data
    storage_mod.put = fake_put
    storage_mod.scan_key = lambda scan_id, suffix: f"scans/{scan_id}/{suffix}"

    from pipeline.graph import run
    result = run("smoketest", Path(tempfile.mkdtemp()), stitch_artifacts, poses_artifacts)

    if not result.ok:
        print(f"FAIL: graph build returned not ok: {result.failure_reason}")
        sys.exit(1)

    graph_key = result.artifacts["graph_key"]
    graph = json.loads(uploads[graph_key])
    nodes = graph["nodes"]
    edges = graph["edges"]

    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    assert len(nodes) == 2, f"expected 2 nodes, got {len(nodes)}"
    assert len(edges) == 2, f"expected 2 directed edges, got {len(edges)}"
    for e in edges:
        az = e["azimuth_from"]
        assert 0 <= az < 360, f"azimuth {az} out of [0,360)"
    print("PASS")

if __name__ == "__main__":
    main()
