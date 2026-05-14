import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { CoverageMap } from "./CoverageMap";
const FAST_MOTION_DEG_S = 90;
const MIN_STOP_COVERAGE = 0.85; // 85% yaw coverage per stop
const MIN_STOPS = 2; // need at least 2 nodes for navigation
export function GuidedRecorder({ onComplete }) {
    const videoRef = useRef(null);
    const recorderRef = useRef(null);
    const chunksRef = useRef([]);
    const imuRef = useRef([]);
    const recordingRef = useRef(false);
    const recordingStartRef = useRef(0);
    const coverage = useRef(new CoverageMap());
    const stopsRef = useRef([]);
    const stopStartRef = useRef(null);
    const [recording, setRecording] = useState(false);
    const [mode, setMode] = useState("walking");
    const [stopCoverage, setStopCoverage] = useState(0);
    const [tooFast, setTooFast] = useState(false);
    const [completedStops, setCompletedStops] = useState(0);
    const [elapsedS, setElapsedS] = useState(0);
    const currentVideoTime = () => recording ? (performance.now() - recordingStartRef.current) / 1000 : 0;
    useEffect(() => {
        let stream = null;
        (async () => {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 3840 },
                    height: { ideal: 2160 },
                    frameRate: { ideal: 30 },
                },
                audio: false,
            });
            const track = stream.getVideoTracks()[0];
            const caps = (track.getCapabilities?.() ?? {});
            const constraints = {};
            if (caps.exposureMode?.includes("manual"))
                constraints.exposureMode = "manual";
            if (caps.focusMode?.includes("manual"))
                constraints.focusMode = "manual";
            if (Object.keys(constraints).length)
                await track.applyConstraints(constraints).catch(() => { });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }
        })();
        const onMotion = (ev) => {
            const yawRate = Math.abs(ev.rotationRate?.alpha ?? 0);
            setTooFast(yawRate > FAST_MOTION_DEG_S);
            if (recordingRef.current) {
                imuRef.current.push({
                    t: ev.timeStamp,
                    wx: ev.rotationRate?.alpha ?? 0,
                    wy: ev.rotationRate?.beta ?? 0,
                    wz: ev.rotationRate?.gamma ?? 0,
                    ax: ev.accelerationIncludingGravity?.x ?? 0,
                    ay: ev.accelerationIncludingGravity?.y ?? 0,
                    az: ev.accelerationIncludingGravity?.z ?? 0,
                });
                coverage.current.ingestMotion(ev);
                setStopCoverage(coverage.current.overallCoverage());
            }
        };
        window.addEventListener("devicemotion", onMotion);
        return () => {
            window.removeEventListener("devicemotion", onMotion);
            stream?.getTracks().forEach((t) => t.stop());
        };
    }, []);
    useEffect(() => {
        if (!recording)
            return;
        const start = performance.now();
        const id = setInterval(() => setElapsedS((performance.now() - start) / 1000), 250);
        return () => clearInterval(id);
    }, [recording]);
    const startRecording = () => {
        const stream = videoRef.current?.srcObject;
        if (!stream)
            return;
        stopsRef.current = [];
        chunksRef.current = [];
        imuRef.current = [];
        recordingRef.current = true;
        recordingStartRef.current = performance.now();
        setCompletedStops(0);
        setMode("walking");
        const rec = new MediaRecorder(stream, { mimeType: "video/mp4" });
        rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
        rec.onstop = () => {
            recordingRef.current = false;
            const videoBlob = new Blob(chunksRef.current, { type: "video/mp4" });
            let imuBlob;
            if (imuRef.current.length > 0) {
                const jsonl = imuRef.current.map((s) => JSON.stringify(s)).join("\n");
                imuBlob = new Blob([jsonl], { type: "application/jsonl" });
            }
            onComplete(videoBlob, imuBlob, stopsRef.current);
        };
        rec.start(1000);
        recorderRef.current = rec;
        setRecording(true);
    };
    const enterStop = () => {
        coverage.current.reset();
        setStopCoverage(0);
        stopStartRef.current = currentVideoTime();
        setMode("at_stop");
    };
    const leaveStop = () => {
        if (stopStartRef.current === null)
            return;
        stopsRef.current.push({
            start_s: stopStartRef.current,
            end_s: currentVideoTime(),
        });
        stopStartRef.current = null;
        setCompletedStops(stopsRef.current.length);
        setMode("walking");
        coverage.current.reset();
        setStopCoverage(0);
    };
    const stopReady = stopCoverage >= MIN_STOP_COVERAGE;
    const canFinish = completedStops >= MIN_STOPS;
    const finish = () => recorderRef.current?.stop();
    return (_jsxs("div", { style: { position: "relative", width: "100%", height: "100vh" }, children: [_jsx("video", { ref: videoRef, muted: true, playsInline: true, style: { width: "100%", height: "100%", objectFit: "cover" } }), tooFast && (_jsx("div", { style: overlay("rgba(220,40,40,0.85)"), children: "SLOW DOWN \u2014 moving too fast" })), recording && (_jsx("div", { style: modeBanner(mode), children: mode === "walking" ? (_jsxs(_Fragment, { children: [_jsx("div", { style: { fontSize: 16, fontWeight: 700 }, children: "Walking between stops" }), _jsx("div", { style: { fontSize: 13, opacity: 0.8, marginTop: 4 }, children: "Move to your next spot, then tap \"At a stop\"" })] })) : (_jsxs(_Fragment, { children: [_jsx("div", { style: { fontSize: 16, fontWeight: 700 }, children: "At a stop \u2014 rotate 360\u00B0" }), _jsx("div", { style: { fontSize: 13, opacity: 0.8, marginTop: 4 }, children: "Hold steady and rotate slowly all the way around at eye level" })] })) })), _jsxs("div", { style: hud, children: [_jsxs("div", { children: ["stops: ", completedStops] }), _jsxs("div", { children: ["elapsed: ", elapsedS.toFixed(0), "s"] }), mode === "at_stop" && (_jsxs("div", { children: ["rotation: ", Math.round(stopCoverage * 100), "%"] }))] }), !recording && (_jsxs("div", { style: preCaptureTips, children: [_jsx("div", { style: { fontWeight: 700, marginBottom: 8 }, children: "How to capture" }), _jsx("div", { children: "1. Tap \"Start\" and walk to your first spot" }), _jsx("div", { children: "2. Tap \"At a stop\" and do a slow 360\u00B0 rotation" }), _jsx("div", { children: "3. Tap \"Done with stop\" and walk to the next spot" }), _jsx("div", { children: "4. Repeat for every viewpoint you want (\u2265 2 stops)" }), _jsx("div", { children: "5. Tap \"Finish\" when done" }), _jsx("div", { style: { marginTop: 8, opacity: 0.7, fontSize: 12 }, children: "Keep phone at eye level. Move slowly. Good lighting helps." })] })), _jsx("div", { style: controls, children: !recording ? (_jsx("button", { onClick: startRecording, style: btn, children: "Start capture" })) : mode === "walking" ? (_jsxs("div", { style: { display: "flex", gap: 12 }, children: [_jsx("button", { onClick: enterStop, style: btn, children: "At a stop" }), _jsxs("button", { onClick: finish, disabled: !canFinish, style: { ...btn, ...btnSecondary, opacity: canFinish ? 1 : 0.35 }, children: ["Finish (", completedStops, " stops)"] })] })) : (_jsx("button", { onClick: leaveStop, disabled: !stopReady, style: { ...btn, opacity: stopReady ? 1 : 0.4 }, children: stopReady
                        ? "Done with stop"
                        : `Keep rotating (${Math.round(stopCoverage * 100)}%)` })) })] }));
}
const hud = {
    position: "absolute", top: 16, left: 16, padding: "8px 12px",
    background: "rgba(0,0,0,0.6)", borderRadius: 8, fontVariantNumeric: "tabular-nums",
    color: "#fff", fontSize: 13,
};
const controls = {
    position: "absolute", bottom: 32, left: 0, right: 0, display: "flex", justifyContent: "center",
};
const btn = {
    padding: "14px 28px", fontSize: 17, borderRadius: 999, border: 0,
    background: "#fff", color: "#000", fontWeight: 600, cursor: "pointer",
};
const btnSecondary = {
    background: "rgba(255,255,255,0.2)", color: "#fff",
    border: "1px solid rgba(255,255,255,0.4)",
};
const overlay = (bg) => ({
    position: "absolute", top: "50%", left: 0, right: 0, transform: "translateY(-50%)",
    textAlign: "center", padding: "16px", background: bg, fontSize: 22, fontWeight: 700, color: "#fff",
});
const modeBanner = (mode) => ({
    position: "absolute", top: 0, left: 0, right: 0,
    padding: "12px 16px",
    background: mode === "at_stop" ? "rgba(79,70,229,0.85)" : "rgba(0,0,0,0.65)",
    color: "#fff",
});
const preCaptureTips = {
    position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
    padding: "20px 24px", background: "rgba(0,0,0,0.75)", borderRadius: 12, color: "#fff",
    lineHeight: 1.8, minWidth: 280,
};
