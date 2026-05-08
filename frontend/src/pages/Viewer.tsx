import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { getScan, type ScanResponse } from "../api";
import { SplatViewer } from "../viewer/SplatViewer";

export function ViewerPage() {
  const { id } = useParams<{ id: string }>();
  const [scan, setScan] = useState<ScanResponse | null>(null);

  useEffect(() => {
    if (!id) return;
    getScan(id).then(setScan);
  }, [id]);

  if (!scan) return <div style={{ padding: 24 }}>Loading…</div>;
  if (scan.status !== "ready") return <div style={{ padding: 24 }}>Scan isn't ready yet.</div>;

  // Order: preview -> standard -> hires for progressive sharpening.
  const lodUrls = [
    scan.lod_urls?.preview,
    scan.lod_urls?.standard,
    scan.lod_urls?.hires ?? scan.splat_url ?? null,
  ].filter((u): u is string => Boolean(u));

  return <SplatViewer lodUrls={lodUrls} />;
}
