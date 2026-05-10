import { useEffect, useRef, useState } from "react";

import { CoverageMap } from "./CoverageMap";

const FAST_MOTION_DEG_S = 90;
const MIN_OVERALL_COVERAGE = 0.85;
const MIN_DURATION_S = 30;

// Multi-station guidance steps (LighthouseGS / Scaniverse consensus, 2025)
const STEPS = [
  { id: "knee",      label: "Station 1 — knee height",   hint: "Hold phone low, rotate slowly 360°" },
  { id: "eye",       label: "Station 2 — eye height",    hint: "Hold phone at eye level, rotate 360°" },
  { id: "raised",    label: "Station 3 — arm raised",    hint: "Hold phone high, rotate 360°" },
  { id: "loop",      label: "Connecting loop",           hint: "Slow walk connecting all 3 stations" },
] as const;
interface ImuSample {
  t: number;
  wx: number; wy: number; wz: number;
  ax: number; ay: number; az: number;
}

interface Props {
  onComplete: (video: Blob, imu?: Blob) => void;
}

export function GuidedRecorder({ onComplete }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const imuRef = useRef<ImuSample[]>([]);
  const recordingRef = useRef(false);
  const coverage = useRef(new CoverageMap());

  const [recording, setRecording] = useState(false);
  const [coveragePct, setCoveragePct] = useState(0);
  const [heightPasses, setHeightPasses] = useState<[boolean, boolean, boolean]>([false, false, false]);
  const [currentHeight, setCurrentHeight] = useState<string>("eye");
  const [tooFast, setTooFast] = useState(false);
  const [elapsedS, setElapsedS] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);

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
      coverage.current.ingestMotion(ev);
      const overall = coverage.current.overallCoverage();
      const hp = coverage.current.heightPasses.slice() as [boolean, boolean, boolean];
      setCoveragePct(overall);
      setHeightPasses(hp);
      setCurrentHeight(coverage.current.currentHeightLabel());

      const yawRate = Math.abs(ev.rotationRate?.alpha ?? 0);
      setTooFast(yawRate > FAST_MOTION_DEG_S);

      // Auto-advance step based on height coverage
      const hCount = coverage.current.heightPassCount();
      const yaw = coverage.current.coverage();
      if (hCount === 0) setStepIdx(0);
      else if (hCount === 1 && yaw > 0.6) setStepIdx(1);
      else if (hCount === 2 && yaw > 0.6) setStepIdx(2);
      else if (hCount === 3) setStepIdx(3);

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

  const start = () => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    if (!stream) return;
    coverage.current.reset();
    chunksRef.current = [];
    imuRef.current = [];
    recordingRef.current = true;
    setStepIdx(0);

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
      onComplete(videoBlob, imuBlob);
    };
    rec.start(1000);
    recorderRef.current = rec;
    setRecording(true);
  };

  const canStop = coveragePct >= MIN_OVERALL_COVERAGE && elapsedS >= MIN_DURATION_S;
  const stop = () => recorderRef.current?.stop();

  const currentStep = STEPS[Math.min(stepIdx, STEPS.length - 1)];

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh" }}>
      <video ref={videoRef} muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} />

      {tooFast && (
        <div style={overlay("rgba(220,40,40,0.85)")}>SLOW DOWN — moving too fast</div>
      )}

      {/* Step guidance */}
      {recording && (
        <div style={stepBanner}>
          <div style={{ fontSize: 13, opacity: 0.7, marginBottom: 2 }}>
            Step {stepIdx + 1} / {STEPS.length}
          </div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{currentStep.label}</div>
          <div style={{ fontSize: 14, opacity: 0.85, marginTop: 4 }}>{currentStep.hint}</div>
        </div>
      )}

      <div style={hud}>
        <div>coverage: {Math.round(coveragePct * 100)}%</div>
        <div>elapsed: {elapsedS.toFixed(0)}s</div>
        <div style={{ marginTop: 4, fontSize: 12 }}>
          height passes:{" "}
          {(["knee", "eye", "arm↑"] as const).map((h, i) => (
            <span
              key={h}
              style={{ marginRight: 6, opacity: heightPasses[i] ? 1 : 0.35,
                       fontWeight: currentHeight === (["knee","eye","arm-raised"])[i] ? 700 : 400 }}
            >
              {h}
            </span>
          ))}
        </div>
      </div>

      {!recording && (
        <div style={preCaptureTips}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Capture tips</div>
          <div>1. Rotate 360° at knee height</div>
          <div>2. Rotate 360° at eye height</div>
          <div>3. Rotate 360° with arm raised</div>
          <div>4. Slow loop connecting all stations</div>
          <div style={{ marginTop: 8, opacity: 0.7, fontSize: 12 }}>
            Use overcast or all-artificial light. Avoid moving mirrors/screens.
          </div>
        </div>
      )}

      <div style={controls}>
        {!recording ? (
          <button onClick={start} style={btn}>Start capture</button>
        ) : (
          <button onClick={stop} disabled={!canStop} style={{ ...btn, opacity: canStop ? 1 : 0.4 }}>
            {canStop
              ? "Done — upload"
              : `Keep going (${Math.round(coveragePct * 100)}% / ${elapsedS.toFixed(0)}s)`}
          </button>
        )}
      </div>
    </div>
  );
}

const hud: React.CSSProperties = {
  position: "absolute", top: 16, left: 16, padding: "8px 12px",
  background: "rgba(0,0,0,0.6)", borderRadius: 8, fontVariantNumeric: "tabular-nums",
};
const controls: React.CSSProperties = {
  position: "absolute", bottom: 32, left: 0, right: 0, display: "flex", justifyContent: "center",
};
const btn: React.CSSProperties = {
  padding: "14px 28px", fontSize: 18, borderRadius: 999, border: 0,
  background: "#fff", color: "#000", fontWeight: 600,
};
const overlay = (bg: string): React.CSSProperties => ({
  position: "absolute", top: "50%", left: 0, right: 0, transform: "translateY(-50%)",
  textAlign: "center", padding: "16px", background: bg, fontSize: 22, fontWeight: 700,
});
const stepBanner: React.CSSProperties = {
  position: "absolute", top: 0, left: 0, right: 0,
  padding: "12px 16px", background: "rgba(0,0,0,0.65)", color: "#fff",
};
const preCaptureTips: React.CSSProperties = {
  position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
  padding: "20px 24px", background: "rgba(0,0,0,0.75)", borderRadius: 12, color: "#fff",
  lineHeight: 1.7, minWidth: 260,
};
