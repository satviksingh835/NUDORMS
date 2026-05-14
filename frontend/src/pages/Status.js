import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getScan } from "../api";
const STAGE_ORDER = [
    "queued", "qc", "frames", "posing", "stitching", "graph_build", "ready",
];
const STAGE_LABEL = {
    queued: "Queued",
    qc: "Checking capture quality",
    frames: "Selecting frames",
    posing: "Estimating camera positions",
    stitching: "Stitching panoramas",
    graph_build: "Building tour graph",
    ready: "Ready",
    needs_recapture: "Needs recapture",
    failed: "Failed",
};
export function StatusPage() {
    const { id } = useParams();
    const [scan, setScan] = useState(null);
    useEffect(() => {
        if (!id)
            return;
        let cancelled = false;
        const tick = async () => {
            try {
                const s = await getScan(id);
                if (!cancelled)
                    setScan(s);
                if (!cancelled && s.status !== "ready" && s.status !== "failed" && s.status !== "needs_recapture") {
                    setTimeout(tick, 2000);
                }
            }
            catch {
                if (!cancelled)
                    setTimeout(tick, 4000);
            }
        };
        tick();
        return () => { cancelled = true; };
    }, [id]);
    if (!scan)
        return _jsx("div", { style: { padding: 24 }, children: "Loading\u2026" });
    const idx = STAGE_ORDER.indexOf(scan.status);
    return (_jsxs("div", { style: { maxWidth: 560, margin: "8vh auto", padding: 24 }, children: [_jsx("h1", { children: STAGE_LABEL[scan.status] }), _jsx("ol", { style: { paddingLeft: 16, lineHeight: 1.8 }, children: STAGE_ORDER.slice(0, -1).map((s, i) => (_jsx("li", { style: { opacity: i <= idx ? 1 : 0.35 }, children: STAGE_LABEL[s] }, s))) }), scan.status === "ready" && (_jsx(Link, { to: `/scans/${id}/view`, style: { fontSize: 18 }, children: "Open viewer \u2192" })), scan.status === "needs_recapture" && (_jsxs("p", { children: ["The capture wasn't usable. ", _jsx(Link, { to: "/capture", children: "Try again" }), "."] })), scan.status === "failed" && _jsx("pre", { style: { color: "#f88" }, children: scan.error })] }));
}
