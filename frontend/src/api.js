const API = "/api";
export async function uploadScan(video, imu, stops) {
    const fd = new FormData();
    fd.append("video", video, "capture.mp4");
    if (imu)
        fd.append("imu", imu, "imu.jsonl");
    if (stops && stops.length > 0)
        fd.append("stops", JSON.stringify(stops));
    const res = await fetch(`${API}/scans`, { method: "POST", body: fd });
    if (!res.ok)
        throw new Error(await res.text());
    return res.json();
}
export async function getScan(id) {
    const res = await fetch(`${API}/scans/${id}`);
    if (!res.ok)
        throw new Error(await res.text());
    return res.json();
}
export async function getFeedback(id) {
    const res = await fetch(`${API}/scans/${id}/feedback`);
    if (!res.ok)
        throw new Error(await res.text());
    return res.json();
}
