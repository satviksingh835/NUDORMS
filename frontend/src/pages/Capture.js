import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadScan } from "../api";
import { GuidedRecorder } from "../capture/GuidedRecorder";
export function CapturePage() {
    const [coachAccepted, setCoachAccepted] = useState(false);
    const [uploading, setUploading] = useState(false);
    const nav = useNavigate();
    const handleComplete = async (video, imu, stops) => {
        setUploading(true);
        const { id } = await uploadScan(video, imu, stops);
        nav(`/scans/${id}`);
    };
    if (!coachAccepted) {
        return (_jsxs("div", { style: { maxWidth: 520, margin: "10vh auto", padding: 24, lineHeight: 1.5 }, children: [_jsx("h1", { children: "Capture your dorm" }), _jsx("p", { children: "You'll create a Street View-style tour by stopping at viewpoints and doing a 360\u00B0 rotation at each one." }), _jsxs("ul", { children: [_jsxs("li", { children: ["Walk to a spot \u2192 tap ", _jsx("strong", { children: "At a stop" }), " \u2192 slowly rotate 360\u00B0 at eye level."] }), _jsxs("li", { children: ["Tap ", _jsx("strong", { children: "Done with stop" }), " \u2192 walk to the next spot and repeat."] }), _jsx("li", { children: "You need at least 2 stops. More stops = richer tour." }), _jsx("li", { children: "Turn lights on. Keep the phone steady. Rotate slowly (about 30 seconds per rotation)." }), _jsx("li", { children: "Don't zoom or switch lenses mid-recording." })] }), _jsx("p", { children: "The app tracks rotation coverage per stop and won't let you mark a stop done until you've covered enough." }), _jsx("button", { onClick: async () => {
                        // iOS requires user-gesture-bound motion permission.
                        const anyDeviceMotion = DeviceMotionEvent;
                        if (typeof anyDeviceMotion.requestPermission === "function") {
                            await anyDeviceMotion.requestPermission().catch(() => { });
                        }
                        setCoachAccepted(true);
                    }, style: { padding: "12px 20px", fontSize: 16, borderRadius: 8 }, children: "Got it \u2014 start" })] }));
    }
    if (uploading) {
        return _jsx("div", { style: { padding: 24 }, children: "Uploading\u2026" });
    }
    return _jsx(GuidedRecorder, { onComplete: handleComplete });
}
