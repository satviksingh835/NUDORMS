export type ScanStatus =
  | "queued"
  | "qc"
  | "frames"
  | "posing"
  | "stitching"
  | "graph_build"
  | "ready"
  | "needs_recapture"
  | "failed";

export interface ScanResponse {
  id: string;
  status: ScanStatus;
  progress: number;
  metrics?: Record<string, unknown> | null;
  error?: string | null;
  graph_url?: string | null;
  pano_urls?: Record<string, string> | null;
}

export interface Stop {
  start_s: number;
  end_s: number;
}

const API = "/api";

export async function uploadScan(video: Blob, imu?: Blob, stops?: Stop[]): Promise<ScanResponse> {
  const fd = new FormData();
  fd.append("video", video, "capture.mp4");
  if (imu) fd.append("imu", imu, "imu.jsonl");
  if (stops && stops.length > 0) fd.append("stops", JSON.stringify(stops));
  const res = await fetch(`${API}/scans`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getScan(id: string): Promise<ScanResponse> {
  const res = await fetch(`${API}/scans/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getFeedback(id: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}/scans/${id}/feedback`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
