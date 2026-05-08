/**
 * Tracks which yaw angles the camera has covered during a guided capture.
 *
 * We integrate gyro angular velocity (z-axis = yaw) over time to estimate
 * heading, and bin coverage into N angular sectors. The capture screen uses
 * coverage% to gate the "stop recording" button — we don't let users finish
 * until they've actually looped the room.
 */
export class CoverageMap {
  private readonly bins: Uint8Array;
  private heading = 0;
  private lastT = 0;

  constructor(public readonly numBins = 36) {
    this.bins = new Uint8Array(numBins);
  }

  reset(): void {
    this.bins.fill(0);
    this.heading = 0;
    this.lastT = 0;
  }

  /** Feed a DeviceMotionEvent — yaw rate is rotationRate.alpha (deg/s). */
  ingestMotion(ev: DeviceMotionEvent): void {
    const now = ev.timeStamp ?? performance.now();
    const dt = this.lastT ? (now - this.lastT) / 1000 : 0;
    this.lastT = now;

    const yawRateDeg = ev.rotationRate?.alpha ?? 0;
    this.heading = (this.heading + yawRateDeg * dt + 360) % 360;

    const bin = Math.floor((this.heading / 360) * this.numBins) % this.numBins;
    this.bins[bin] = 1;
  }

  coverage(): number {
    let covered = 0;
    for (const b of this.bins) if (b) covered++;
    return covered / this.numBins;
  }

  snapshot(): Uint8Array {
    return this.bins.slice();
  }
}
