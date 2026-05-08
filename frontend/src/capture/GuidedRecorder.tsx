import { useEffect, useRef, useState } from "react";

import { CoverageMap } from "./CoverageMap";

const FAST_MOTION_DEG_S = 90; // angular speed above which we warn the user
const MIN_COVERAGE = 0.85;
const MIN_DURATION_S = 30;

interface Props {
  onComplete: (video: Blob) => void;
}

/**
 * In-browser guided capture. Streams 4K30 if supported, locks exposure +
 * focus, tracks yaw coverage via gyro, and warns when the user moves too
 * fast. Capture quality is the single biggest lever on reconstruction
 * quality, so this component is doing real work — not just wrapping a
 * <input type="file"> picker.
 */
export function GuidedRecorder({ onComplete }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const coverage = useRef(new CoverageMap());

  const [recording, setRecording] = useState(false);
  const [coveragePct, setCoveragePct] = useState(0);
  const [tooFast, setTooFast] = useState(false);
  const [elapsedS, setElapsedS] = useState(0);

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

      // Lock exposure + focus where supported. Auto-exposure mid-walkthrough
      // is one of the most common silent killers of splat training quality.
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
      setCoveragePct(coverage.current.coverage());
      const yawRate = Math.abs(ev.rotationRate?.alpha ?? 0);
      setTooFast(yawRate > FAST_MOTION_DEG_S);
    };

    // iOS requires explicit motion permission; assumes a parent CTA already requested it.
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
    const rec = new MediaRecorder(stream, { mimeType: "video/mp4" });
    rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
    rec.onstop = () => onComplete(new Blob(chunksRef.current, { type: "video/mp4" }));
    rec.start(1000);
    recorderRef.current = rec;
    setRecording(true);
  };

  const canStop = coveragePct >= MIN_COVERAGE && elapsedS >= MIN_DURATION_S;
  const stop = () => recorderRef.current?.stop();

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh" }}>
      <video ref={videoRef} muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} />

      {tooFast && (
        <div style={overlay("rgba(220,40,40,0.85)")}>SLOW DOWN — moving too fast</div>
      )}

      <div style={hud}>
        <div>coverage: {Math.round(coveragePct * 100)}%</div>
        <div>elapsed: {elapsedS.toFixed(0)}s</div>
      </div>

      <div style={controls}>
        {!recording ? (
          <button onClick={start} style={btn}>Start capture</button>
        ) : (
          <button onClick={stop} disabled={!canStop} style={{ ...btn, opacity: canStop ? 1 : 0.4 }}>
            {canStop ? "Done" : `Keep going (${Math.round(coveragePct * 100)}% / ${elapsedS.toFixed(0)}s)`}
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
