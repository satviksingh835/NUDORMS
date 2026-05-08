import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getScan, type ScanResponse } from "../api";

const STAGE_ORDER = [
  "queued", "qc", "frames", "posing", "training",
  "evaluating", "retrying", "meshing", "compressing", "ready",
] as const;

const STAGE_LABEL: Record<string, string> = {
  queued: "Queued",
  qc: "Checking capture quality",
  frames: "Selecting frames",
  posing: "Estimating camera positions",
  training: "Training 3D model",
  evaluating: "Evaluating quality",
  retrying: "Retraining at higher quality",
  meshing: "Building mesh",
  compressing: "Compressing for streaming",
  ready: "Ready",
  needs_recapture: "Needs recapture",
  failed: "Failed",
};

export function StatusPage() {
  const { id } = useParams<{ id: string }>();
  const [scan, setScan] = useState<ScanResponse | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await getScan(id);
        if (!cancelled) setScan(s);
        if (!cancelled && s.status !== "ready" && s.status !== "failed" && s.status !== "needs_recapture") {
          setTimeout(tick, 2000);
        }
      } catch {
        if (!cancelled) setTimeout(tick, 4000);
      }
    };
    tick();
    return () => { cancelled = true; };
  }, [id]);

  if (!scan) return <div style={{ padding: 24 }}>Loading…</div>;

  const idx = STAGE_ORDER.indexOf(scan.status as (typeof STAGE_ORDER)[number]);

  return (
    <div style={{ maxWidth: 560, margin: "8vh auto", padding: 24 }}>
      <h1>{STAGE_LABEL[scan.status]}</h1>

      <ol style={{ paddingLeft: 16, lineHeight: 1.8 }}>
        {STAGE_ORDER.slice(0, -1).map((s, i) => (
          <li key={s} style={{ opacity: i <= idx ? 1 : 0.35 }}>
            {STAGE_LABEL[s]}
          </li>
        ))}
      </ol>

      {scan.status === "ready" && (
        <Link to={`/scans/${id}/view`} style={{ fontSize: 18 }}>Open viewer →</Link>
      )}
      {scan.status === "needs_recapture" && (
        <p>The capture wasn't usable. <Link to="/capture">Try again</Link>.</p>
      )}
      {scan.status === "failed" && <pre style={{ color: "#f88" }}>{scan.error}</pre>}
    </div>
  );
}
