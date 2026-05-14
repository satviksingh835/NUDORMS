import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useState } from "react";
import { PanoramaViewer } from "../viewer/PanoramaViewer";
export function DemoPage() {
    const [file, setFile] = useState(null);
    const [objectUrl, setObjectUrl] = useState(null);
    const [dragging, setDragging] = useState(false);
    const accept = useCallback((f) => {
        if (f.name.endsWith(".jpg") || f.name.endsWith(".jpeg") || f.name.endsWith(".png")) {
            if (objectUrl)
                URL.revokeObjectURL(objectUrl);
            const url = URL.createObjectURL(f);
            setObjectUrl(url);
            setFile(f);
        }
    }, [objectUrl]);
    const onDrop = useCallback((e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        if (f)
            accept(f);
    }, [accept]);
    const onFileInput = useCallback((e) => {
        const f = e.target.files?.[0];
        if (f)
            accept(f);
    }, [accept]);
    if (file && objectUrl) {
        // Wrap single image as a one-node, no-edge graph for the viewer
        const demoGraph = {
            nodes: [{ id: "demo", pano_key: "", position: [0, 0, 0] }],
            edges: [],
        };
        const demoPanoUrls = { demo: objectUrl };
        return (_jsxs("div", { style: { position: "fixed", inset: 0, background: "#0a0a0a" }, children: [_jsx(PanoramaViewer, { graph: demoGraph, panoUrls: demoPanoUrls, initialNodeId: "demo" }), _jsxs("button", { onClick: () => { setFile(null); URL.revokeObjectURL(objectUrl); setObjectUrl(null); }, style: styles.backBtn, children: [_jsx("svg", { width: "14", height: "14", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2", children: _jsx("polyline", { points: "15 18 9 12 15 6" }) }), "Back"] }), _jsxs("div", { style: styles.fileLabel, children: [_jsx("span", { style: styles.fileLabelDot }), file.name] }), _jsx("style", { children: fonts })] }));
    }
    return (_jsxs("div", { style: styles.root, children: [_jsx("style", { children: fonts }), _jsx("div", { style: styles.wordmark, children: "NUDORMS" }), _jsxs("div", { style: { ...styles.dropZone, ...(dragging ? styles.dropZoneActive : {}) }, onDragOver: (e) => { e.preventDefault(); setDragging(true); }, onDragLeave: () => setDragging(false), onDrop: onDrop, children: [_jsx("div", { style: styles.dropIcon, children: _jsxs("svg", { width: "32", height: "32", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.2", strokeLinecap: "round", children: [_jsx("path", { d: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" }), _jsx("polyline", { points: "17 8 12 3 7 8" }), _jsx("line", { x1: "12", y1: "3", x2: "12", y2: "15" })] }) }), _jsx("p", { style: styles.dropTitle, children: "Drop an equirectangular image" }), _jsx("p", { style: styles.dropSub, children: ".jpg \u00B7 .png" }), _jsxs("label", { style: styles.browseBtn, children: ["Browse files", _jsx("input", { type: "file", accept: ".jpg,.jpeg,.png", style: { display: "none" }, onChange: onFileInput })] })] }), _jsx("p", { style: styles.footer, children: "Local preview \u00B7 no upload \u00B7 no backend required" })] }));
}
const styles = {
    root: {
        position: "fixed",
        inset: 0,
        background: "#0a0a0a",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'DM Mono', monospace",
    },
    wordmark: {
        position: "fixed",
        top: 24,
        left: 24,
        color: "rgba(255,255,255,0.9)",
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: "0.18em",
        fontFamily: "'DM Mono', monospace",
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: 999,
        padding: "7px 14px",
    },
    dropZone: {
        width: 380,
        padding: "56px 40px",
        border: "1px dashed rgba(255,255,255,0.12)",
        borderRadius: 20,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
        background: "rgba(255,255,255,0.015)",
        transition: "border-color 0.2s, background 0.2s",
        cursor: "default",
    },
    dropZoneActive: {
        borderColor: "rgba(255,255,255,0.4)",
        background: "rgba(255,255,255,0.04)",
    },
    dropIcon: { color: "rgba(255,255,255,0.25)", marginBottom: 8 },
    dropTitle: { margin: 0, color: "rgba(255,255,255,0.75)", fontSize: 15, fontWeight: 500 },
    dropSub: { margin: 0, color: "rgba(255,255,255,0.25)", fontSize: 11, letterSpacing: "0.08em" },
    browseBtn: {
        marginTop: 16,
        background: "rgba(255,255,255,0.07)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 999,
        color: "rgba(255,255,255,0.65)",
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.08em",
        padding: "8px 20px",
        cursor: "pointer",
        fontFamily: "'DM Mono', monospace",
    },
    footer: {
        position: "fixed",
        bottom: 28,
        margin: 0,
        color: "rgba(255,255,255,0.15)",
        fontSize: 10,
        letterSpacing: "0.06em",
    },
    backBtn: {
        position: "fixed",
        top: 20,
        left: 20,
        zIndex: 30,
        display: "flex",
        alignItems: "center",
        gap: 6,
        background: "rgba(10,10,10,0.55)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 999,
        color: "rgba(255,255,255,0.7)",
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.06em",
        fontFamily: "'DM Mono', monospace",
        padding: "7px 14px",
        cursor: "pointer",
    },
    fileLabel: {
        position: "fixed",
        top: 20,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 30,
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "rgba(10,10,10,0.55)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 999,
        color: "rgba(255,255,255,0.55)",
        fontSize: 11,
        letterSpacing: "0.06em",
        fontFamily: "'DM Mono', monospace",
        padding: "7px 16px",
        maxWidth: 320,
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
    },
    fileLabelDot: {
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: "#4ade80",
        flexShrink: 0,
        boxShadow: "0 0 8px #4ade8099",
    },
};
const fonts = `@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');`;
