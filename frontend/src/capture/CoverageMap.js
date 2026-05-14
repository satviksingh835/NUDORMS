/**
 * Tracks capture coverage across 3 stations × 3 height passes.
 *
 * Multi-station pattern (from LighthouseGS / Polyvia3D / Scaniverse consensus):
 *   Station A: knee height, 360° rotation
 *   Station B: eye height, 360° rotation
 *   Station C: arm-raised height, 360° rotation
 *   Plus one slow translational loop connecting the stations
 *
 * Yaw coverage is tracked per-station via DeviceMotionEvent (gyro integration).
 * Height pass is tracked via DeviceMotionEvent accelerometer (gravity projection).
 */
export class CoverageMap {
    numBins;
    bins;
    heading = 0;
    lastT = 0;
    // Height-pass state: tracks which of the 3 height ranges has been covered
    // 0 = knee (<0.8m tilt), 1 = eye (~level), 2 = arm-raised (>0.6m tilt)
    heightPasses = [false, false, false];
    _currentHeight = 1; // 0 | 1 | 2
    constructor(numBins = 36) {
        this.numBins = numBins;
        this.bins = new Uint8Array(numBins);
    }
    reset() {
        this.bins.fill(0);
        this.heading = 0;
        this.lastT = 0;
        this.heightPasses[0] = this.heightPasses[1] = this.heightPasses[2] = false;
        this._currentHeight = 1;
    }
    /** Feed a DeviceMotionEvent — yaw rate is rotationRate.alpha (deg/s). */
    ingestMotion(ev) {
        const now = ev.timeStamp ?? performance.now();
        const dt = this.lastT ? (now - this.lastT) / 1000 : 0;
        this.lastT = now;
        const yawRateDeg = ev.rotationRate?.alpha ?? 0;
        this.heading = (this.heading + yawRateDeg * dt + 360) % 360;
        const bin = Math.floor((this.heading / 360) * this.numBins) % this.numBins;
        this.bins[bin] = 1;
        // Estimate height pass from gravity vector (accelerationIncludingGravity.y).
        // On a phone held flat: ay ≈ -9.8. Tilted down (knee): ay more negative.
        // Tilted up (arm-raised): ay less negative / x-axis tilts.
        const ay = ev.accelerationIncludingGravity?.y ?? -9.8;
        if (ay < -8.5) {
            this._currentHeight = 0; // knee: phone tilted down
        }
        else if (ay > -5.0) {
            this._currentHeight = 2; // arm-raised: phone tilted toward horizontal / up
        }
        else {
            this._currentHeight = 1; // eye: phone roughly vertical
        }
        this.heightPasses[this._currentHeight] = true;
    }
    /** Fraction of yaw bins covered (0–1). */
    coverage() {
        let covered = 0;
        for (const b of this.bins)
            if (b)
                covered++;
        return covered / this.numBins;
    }
    /** Number of height passes completed (0–3). */
    heightPassCount() {
        return this.heightPasses.filter(Boolean).length;
    }
    /** Overall combined score: 70% yaw coverage + 30% height coverage. */
    overallCoverage() {
        return 0.7 * this.coverage() + 0.3 * (this.heightPassCount() / 3);
    }
    currentHeightLabel() {
        return ["knee", "eye", "arm-raised"][this._currentHeight];
    }
    snapshot() {
        return this.bins.slice();
    }
}
