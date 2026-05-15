import { useEffect, useRef, useState } from "react";

import { type Stop } from "../api";
import { CoverageMap } from "./CoverageMap";

const FAST_MOTION_DEG_S = 90;
const MIN_STOP_COVERAGE = 0.85;   // 85% yaw coverage per stop
const MIN_STOPS = 2;              // need at least 2 nodes for navigation

interface ImuSample {
  t: number;
  wx: number; wy: number; wz: number;
  ax: number; ay: number; az: number;
}

type Mode = "walking" | "at_stop";

interface Props {
  onComplete: (video: Blob, imu?: Blob, stops?: Stop[]) => void;
}

export function GuidedRecorder({ onComplete }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const imuRef = useRef<ImuSample[]>([]);
  const recordingRef = useRef(false);
  const recordingStartRef = useRef<number>(0);
  const coverage = useRef(new CoverageMap());
  const stopsRef = useRef<Stop[]>([]);
  const stopStartRef = useRef<number | null>(null);

  const [recording, setRecording] = useState(false);
  const [mode, setMode] = useState<Mode>("walking");
  const [stopCoverage, setStopCoverage] = useState(0);
  const [tooFast, setTooFast] = useState(false);
  const [completedStops, setCompletedStops] = useState(0);
  const [elapsedS, setElapsedS] = useState(0);

  const currentVideoTime = () =>
    recording ? (performance.now() - recordingStartRef.current) / 1000 : 0;

  useEffect(() => {
    let stream: MediaStream | null = null;

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
      const caps = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & {
        exposureMode?: string[]; focusMode?: string[];
      };
      const constraints: MediaTrackConstraints = {};
      if (caps.exposureMode?.includes("manual")) (constraints as any).exposureMode = "manual";
      if (caps.focusMode?.includes("manual")) (constraints as any).focusMode = "manual";
      if (Object.keys(constraints).length) await track.applyConstraints(constraints).catch(() => {});

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    })();

    const onMotion = (ev: DeviceMotionEvent) => {
      const yawRate = Math.abs(ev.rotationRate?.alpha ?? 0);
      setTooFast(yawRate > FAST_MOTION_DEG_S);

      if (recordingRef.current) {
        imuRef.current.push({
          t: ev.timeStamp,
          wx: ev.rotationRate?.alpha ?? 0,
          wy: ev.rotationRate?.beta  ?? 0,
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
    if (!recording) return;
    const start = performance.now();
    const id = setInterval(() => setElapsedS((performance.now() - start) / 1000), 250);
    return () => clearInterval(id);
  }, [recording]);

  const startRecording = () => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    if (!stream) return;
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
      let imuBlob: Blob | undefined;
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
    if (stopStartRef.current === null) return;
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

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh" }}>
      <video ref={videoRef} muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} />

      {tooFast && (
        <div style={overlay("rgba(220,40,40,0.85)")}>SLOW DOWN — moving too fast</div>
      )}

      {/* Mode banner */}
      {recording && (
        <div style={modeBanner(mode)}>
          {mode === "walking" ? (
            <>
              <div style={{ fontSize: 16, fontWeight: 700 }}>Walking to Next Node</div>
              <div style={{ fontSize: 13, opacity: 0.8, marginTop: 4 }}>
                Walk 5-10 feet to your next spot, then tap "Capture Node"
              </div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 16, fontWeight: 700 }}>Capturing Node — rotate 360°</div>
              <div style={{ fontSize: 13, opacity: 0.8, marginTop: 4 }}>
                Hold steady in place and rotate slowly all the way around
              </div>
            </>
          )}
        </div>
      )}

      {/* HUD */}
      <div style={hud}>
        <div>nodes: {completedStops}</div>
        <div>elapsed: {elapsedS.toFixed(0)}s</div>
        {mode === "at_stop" && (
          <div>rotation: {Math.round(stopCoverage * 100)}%</div>
        )}
      </div>

      {/* Pre-capture tips */}
      {!recording && (
        <div style={preCaptureTips}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>How to capture</div>
          <div>1. Tap "Start" and walk to your first spot</div>
          <div>2. Tap "Capture Node" and do a slow 360° rotation</div>
          <div>3. Tap "Done with Node" and walk 5-10ft to the next location</div>
          <div>4. Repeat for every spot (e.g. Center, Kitchen, Desk)</div>
          <div>5. Tap "Finish" when done</div>
          <div style={{ marginTop: 8, opacity: 0.7, fontSize: 12 }}>
            Keep phone at eye level. Move slowly. Good lighting helps.
          </div>
        </div>
      )}

      {/* Controls */}
      <div style={controls}>
        {!recording ? (
          <button onClick={startRecording} style={btn}>Start capture</button>
        ) : mode === "walking" ? (
          <div style={{ display: "flex", gap: 12 }}>
            <button onClick={enterStop} style={btn}>Capture Node</button>
            <button
              onClick={finish}
              disabled={!canFinish}
              style={{ ...btn, ...btnSecondary, opacity: canFinish ? 1 : 0.35 }}
            >
              Finish ({completedStops} nodes)
            </button>
          </div>
        ) : (
          <button
            onClick={leaveStop}
            disabled={!stopReady}
            style={{ ...btn, opacity: stopReady ? 1 : 0.4 }}
          >
            {stopReady
              ? "Done with Node"
              : `Keep rotating (${Math.round(stopCoverage * 100)}%)`}
          </button>
        )}
      </div>
    </div>
  );
}

const hud: React.CSSProperties = {
  position: "absolute", top: 16, left: 16, padding: "8px 12px",
  background: "rgba(0,0,0,0.6)", borderRadius: 8, fontVariantNumeric: "tabular-nums",
  color: "#fff", fontSize: 13,
};
const controls: React.CSSProperties = {
  position: "absolute", bottom: 32, left: 0, right: 0, display: "flex", justifyContent: "center",
};
const btn: React.CSSProperties = {
  padding: "14px 28px", fontSize: 17, borderRadius: 999, border: 0,
  background: "#fff", color: "#000", fontWeight: 600, cursor: "pointer",
};
const btnSecondary: React.CSSProperties = {
  background: "rgba(255,255,255,0.2)", color: "#fff",
  border: "1px solid rgba(255,255,255,0.4)",
};
const overlay = (bg: string): React.CSSProperties => ({
  position: "absolute", top: "50%", left: 0, right: 0, transform: "translateY(-50%)",
  textAlign: "center", padding: "16px", background: bg, fontSize: 22, fontWeight: 700, color: "#fff",
});
const modeBanner = (mode: "walking" | "at_stop"): React.CSSProperties => ({
  position: "absolute", top: 0, left: 0, right: 0,
  padding: "12px 16px",
  background: mode === "at_stop" ? "rgba(79,70,229,0.85)" : "rgba(0,0,0,0.65)",
  color: "#fff",
});
const preCaptureTips: React.CSSProperties = {
  position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
  padding: "20px 24px", background: "rgba(0,0,0,0.75)", borderRadius: 12, color: "#fff",
  lineHeight: 1.8, minWidth: 280,
};
