import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { getScan } from "../api";
import { PanoramaViewer } from "../viewer/PanoramaViewer";
/* ─── Loading screen ─────────────────────────────────────────── */
function LoadingScreen() {
    return (_jsxs("div", { style: styles.overlay, children: [_jsxs("div", { style: styles.loadingInner, children: [_jsx("div", { style: styles.ring, children: _jsx("div", { style: styles.ringInner }) }), _jsx("p", { style: styles.loadingText, children: "Building your tour" }), _jsx("p", { style: styles.loadingSubtext, children: "This may take a moment" })] }), _jsx("style", { children: ringAnim })] }));
}
/* ─── Not-ready screen ───────────────────────────────────────── */
function NotReadyScreen({ status }) {
    return (_jsx("div", { style: styles.overlay, children: _jsxs("div", { style: styles.notReadyCard, children: [_jsx("div", { style: styles.notReadyDot }), _jsx("p", { style: styles.notReadyTitle, children: "Scan in progress" }), _jsxs("p", { style: styles.notReadyBody, children: ["Your space is still being processed.", _jsx("br", {}), "Current stage: ", _jsx("span", { style: styles.statusBadge, children: status })] }), _jsx("p", { style: styles.notReadyHint, children: "This page will update automatically." })] }) }));
}
/* ─── Share button ───────────────────────────────────────────── */
function ShareButton() {
    const [copied, setCopied] = useState(false);
    const copy = () => {
        navigator.clipboard.writeText(window.location.href);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };
    return (_jsx("button", { onClick: copy, style: styles.shareBtn, children: copied ? (_jsxs(_Fragment, { children: [_jsx("svg", { width: "13", height: "13", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2.5", children: _jsx("polyline", { points: "20 6 9 17 4 12" }) }), "Copied"] })) : (_jsxs(_Fragment, { children: [_jsxs("svg", { width: "13", height: "13", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", children: [_jsx("circle", { cx: "18", cy: "5", r: "3" }), _jsx("circle", { cx: "6", cy: "12", r: "3" }), _jsx("circle", { cx: "18", cy: "19", r: "3" }), _jsx("line", { x1: "8.59", y1: "13.51", x2: "15.42", y2: "17.49" }), _jsx("line", { x1: "15.41", y1: "6.51", x2: "8.59", y2: "10.49" })] }), "Share"] })) }));
}
/* ─── Main viewer page ───────────────────────────────────────── */
export function ViewerPage() {
    const { id } = useParams();
    const [scan, setScan] = useState(null);
    const [graph, setGraph] = useState(null);
    const pollRef = useRef(null);
    useEffect(() => {
        if (!id)
            return;
        const load = () => getScan(id).then(setScan).catch(() => { });
        load();
        pollRef.current = setInterval(load, 5000);
        return () => { if (pollRef.current)
            clearInterval(pollRef.current); };
    }, [id]);
    // Stop polling once ready
    useEffect(() => {
        if (scan?.status === "ready" && pollRef.current) {
            clearInterval(pollRef.current);
        }
    }, [scan]);
    // Fetch graph.json when graph_url is available
    useEffect(() => {
        if (!scan?.graph_url)
            return;
        fetch(scan.graph_url)
            .then((r) => r.json())
            .then(setGraph)
            .catch(() => { });
    }, [scan?.graph_url]);
    const label = id ? `Scan ${id.slice(0, 8).toUpperCase()}` : "Untitled space";
    const panoUrls = scan?.pano_urls ?? {};
    const ready = scan?.status === "ready" && graph !== null;
    return (_jsxs("div", { style: styles.root, children: [ready && (_jsx(PanoramaViewer, { graph: graph, panoUrls: panoUrls })), !scan && _jsx(LoadingScreen, {}), scan && scan.status !== "ready" && _jsx(NotReadyScreen, { status: scan.status }), scan?.status === "ready" && !graph && _jsx(LoadingScreen, {}), ready && (_jsxs(_Fragment, { children: [_jsx("div", { style: styles.wordmark, children: _jsx("span", { style: styles.wordmarkText, children: "NUDORMS" }) }), _jsxs("div", { style: styles.bottomBar, children: [_jsx("span", { style: styles.roomLabel, children: label }), _jsx("span", { style: styles.hint, children: "drag to look \u00B7 click arrows to move" }), _jsx(ShareButton, {})] })] })), _jsx("style", { children: ringAnim })] }));
}
/* ─── Styles ─────────────────────────────────────────────────── */
const styles = {
    root: {
        position: "fixed",
        inset: 0,
        background: "#0a0a0a",
        fontFamily: "'DM Mono', 'Courier New', monospace",
    },
    overlay: {
        position: "fixed",
        inset: 0,
        background: "#0a0a0a",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 10,
    },
    loadingInner: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 20,
    },
    ring: {
        width: 64,
        height: 64,
        borderRadius: "50%",
        border: "1.5px solid rgba(255,255,255,0.08)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
    },
    ringInner: {
        position: "absolute",
        inset: 0,
        borderRadius: "50%",
        border: "1.5px solid transparent",
        borderTopColor: "rgba(255,255,255,0.6)",
        animation: "nudorms-spin 1.1s linear infinite",
    },
    loadingText: {
        margin: 0,
        color: "rgba(255,255,255,0.85)",
        fontSize: 15,
        letterSpacing: "0.06em",
        fontWeight: 500,
    },
    loadingSubtext: {
        margin: 0,
        color: "rgba(255,255,255,0.28)",
        fontSize: 12,
        letterSpacing: "0.04em",
    },
    notReadyCard: {
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: 16,
        padding: "36px 40px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
        maxWidth: 340,
        textAlign: "center",
    },
    notReadyDot: {
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: "#f59e0b",
        boxShadow: "0 0 12px #f59e0b88",
        animation: "nudorms-pulse 2s ease-in-out infinite",
    },
    notReadyTitle: {
        margin: 0,
        color: "rgba(255,255,255,0.9)",
        fontSize: 16,
        fontWeight: 600,
        letterSpacing: "0.02em",
    },
    notReadyBody: {
        margin: 0,
        color: "rgba(255,255,255,0.45)",
        fontSize: 13,
        lineHeight: 1.6,
    },
    statusBadge: {
        color: "rgba(255,255,255,0.75)",
        fontFamily: "'DM Mono', monospace",
        fontSize: 11,
        background: "rgba(255,255,255,0.07)",
        padding: "1px 6px",
        borderRadius: 4,
        letterSpacing: "0.04em",
    },
    notReadyHint: {
        margin: 0,
        color: "rgba(255,255,255,0.2)",
        fontSize: 11,
        letterSpacing: "0.03em",
    },
    wordmark: {
        position: "fixed",
        top: 20,
        left: 20,
        zIndex: 20,
        background: "rgba(10,10,10,0.55)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 999,
        padding: "7px 14px",
    },
    wordmarkText: {
        color: "rgba(255,255,255,0.9)",
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: "0.18em",
        fontFamily: "'DM Mono', monospace",
    },
    bottomBar: {
        position: "fixed",
        bottom: 24,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 20,
        background: "rgba(10,10,10,0.55)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 999,
        padding: "10px 16px 10px 20px",
        display: "flex",
        alignItems: "center",
        gap: 20,
        whiteSpace: "nowrap",
    },
    roomLabel: {
        color: "rgba(255,255,255,0.75)",
        fontSize: 12,
        fontWeight: 600,
        letterSpacing: "0.1em",
        fontFamily: "'DM Mono', monospace",
    },
    hint: {
        color: "rgba(255,255,255,0.35)",
        fontSize: 11,
        letterSpacing: "0.04em",
    },
    shareBtn: {
        display: "flex",
        alignItems: "center",
        gap: 5,
        background: "rgba(255,255,255,0.08)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: 999,
        color: "rgba(255,255,255,0.7)",
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.06em",
        fontFamily: "'DM Mono', monospace",
        padding: "5px 12px",
        cursor: "pointer",
    },
};
const ringAnim = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
  @keyframes nudorms-spin { to { transform: rotate(360deg); } }
  @keyframes nudorms-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }
`;
