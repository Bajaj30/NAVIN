/**
 * Step detector — peak detection on acceleration magnitude.
 *
 * |a| is posture-independent, so it works however the phone is held. Two low-pass filters
 * (time-constant based, so the result doesn't depend on the device's event rate):
 *   fast  → removes sensor jitter
 *   slow  → tracks the gravity baseline (≈ 9.8, but not exactly, per device)
 * A step is the bounce (fast − slow) rising through HIGH. The detector then stays disarmed
 * until the bounce falls back under LOW, so one heel strike can't count twice.
 *
 * Checked on the recorded sessions in ml/data/raw (10 Hz, no ground-truth step counts):
 * ≈1.4 steps/s while WALKING, ≈1.1 on STAIRS, ≈0.02 while STATIONARY.
 */
const FAST_TAU_S = 0.05
const SLOW_TAU_S = 1.0
const HIGH = 1.0          // m/s² above baseline → step
const LOW = 0.3           // m/s² → re-arm
const MIN_INTERVAL_MS = 300   // > 3.3 steps/s is not walking

export class StepDetector {
  constructor() {
    this.reset()
  }

  reset() {
    this.fast = null
    this.slow = null
    this.lastT = null
    this.lastStepT = -Infinity
    this.armed = true
  }

  /** Feed one sample (|a| in m/s², time in ms). Returns true when it completes a step. */
  update(mag, t) {
    if (this.fast === null || t - this.lastT > 1000) {   // first sample, or sensors paused
      this.fast = this.slow = mag
      this.lastT = t
      this.armed = true
      return false
    }
    const dt = Math.max(0, t - this.lastT) / 1000
    this.lastT = t
    this.fast += (1 - Math.exp(-dt / FAST_TAU_S)) * (mag - this.fast)
    this.slow += (1 - Math.exp(-dt / SLOW_TAU_S)) * (mag - this.slow)
    const bounce = this.fast - this.slow

    if (!this.armed) {
      if (bounce < LOW) this.armed = true
      return false
    }
    if (bounce > HIGH && t - this.lastStepT >= MIN_INTERVAL_MS) {
      this.armed = false
      this.lastStepT = t
      return true
    }
    return false
  }
}
