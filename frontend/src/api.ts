export type ScanStatus =
  | "queued"
  | "qc"
  | "frames"
  | "posing"
  | "training"
  | "evaluating"
  | "retrying"
  | "refining"
  | "meshing"
  | "compressing"
  | "ready"
  | "needs_recapture"
  | "failed";

export interface ScanResponse {
  id: string;
  status: ScanStatus;
  progress: number;
  metrics?: Record<string, number> | null;
  error?: string | null;
  splat_url?: string | null;
  mesh_url?: string | null;
  lod_urls?: Record<string, string> | null;
}

const API = "/api";

export async function uploadScan(video: Blob, imu?: Blob): Promise<ScanResponse> {
  const fd = new FormData();
  fd.append("video", video, "capture.mp4");
  if (imu) fd.append("imu", imu, "imu.jsonl");
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
