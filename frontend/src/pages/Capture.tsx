import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { uploadScan, type Stop } from "../api";
import { GuidedRecorder } from "../capture/GuidedRecorder";

export function CapturePage() {
  const [coachAccepted, setCoachAccepted] = useState(false);
  const [uploading, setUploading] = useState(false);
  const nav = useNavigate();

  const handleComplete = async (video: Blob, imu?: Blob, stops?: Stop[]) => {
    setUploading(true);
    const { id } = await uploadScan(video, imu, stops);
    nav(`/scans/${id}`);
  };

  if (!coachAccepted) {
    return (
      <div style={{ maxWidth: 520, margin: "10vh auto", padding: 24, lineHeight: 1.5 }}>
        <h1>Capture your dorm</h1>
        <p>You'll create a Street View-style tour by stopping at viewpoints and doing a 360° rotation at each one.</p>
        <ul>
          <li>Walk to a spot → tap <strong>At a stop</strong> → slowly rotate 360° at eye level.</li>
          <li>Tap <strong>Done with stop</strong> → walk to the next spot and repeat.</li>
          <li>You need at least 2 stops. More stops = richer tour.</li>
          <li>Turn lights on. Keep the phone steady. Rotate slowly (about 30 seconds per rotation).</li>
          <li>Don't zoom or switch lenses mid-recording.</li>
        </ul>
        <p>The app tracks rotation coverage per stop and won't let you mark a stop done until you've covered enough.</p>
        <button
          onClick={async () => {
            // iOS requires user-gesture-bound motion permission.
            const anyDeviceMotion = (DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> });
            if (typeof anyDeviceMotion.requestPermission === "function") {
              await anyDeviceMotion.requestPermission().catch(() => {});
            }
            setCoachAccepted(true);
          }}
          style={{ padding: "12px 20px", fontSize: 16, borderRadius: 8 }}
        >
          Got it — start
        </button>
      </div>
    );
  }

  if (uploading) {
    return <div style={{ padding: 24 }}>Uploading…</div>;
  }

  return <GuidedRecorder onComplete={handleComplete} />;
}
