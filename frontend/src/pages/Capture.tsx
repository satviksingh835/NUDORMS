import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { uploadScan } from "../api";
import { GuidedRecorder } from "../capture/GuidedRecorder";

export function CapturePage() {
  const [coachAccepted, setCoachAccepted] = useState(false);
  const [uploading, setUploading] = useState(false);
  const nav = useNavigate();

  const handleComplete = async (video: Blob) => {
    setUploading(true);
    const { id } = await uploadScan(video);
    nav(`/scans/${id}`);
  };

  if (!coachAccepted) {
    return (
      <div style={{ maxWidth: 520, margin: "10vh auto", padding: 24, lineHeight: 1.5 }}>
        <h1>Capture your dorm</h1>
        <p>Before you start, a few things that make a big difference:</p>
        <ul>
          <li>Hold the phone at chest height. Move slowly — about one step every two seconds.</li>
          <li>Walk a full loop of the room and end where you started.</li>
          <li>Turn lights on. Open the blinds.</li>
          <li>Don't switch lenses or zoom mid-recording.</li>
          <li>Plan to record 30–90 seconds.</li>
        </ul>
        <p>The app will warn you if you're moving too fast. It won't let you finish until you've covered the whole room.</p>
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
