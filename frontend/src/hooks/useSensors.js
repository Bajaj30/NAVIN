import { useState, useEffect, useCallback, useRef } from 'react'
import { StepDetector } from '../utils/stepDetector'
import { headingFromEvent, smoothHeading } from '../utils/heading'

const PENDING_STEP_MS = 2500   // steps seen while gated are credited if the gate opens this soon after

/**
 * useSensors — live motion + orientation from the phone.
 *
 *   DeviceMotionEvent       → accelerometer (incl. gravity) + gyroscope (rotationRate)
 *   DeviceOrientationEvent  → heading (clockwise from north) + tilt angles
 *
 * Nothing is uploaded from here: the CNN gets its 10 Hz windows from useMLClassifier,
 * which samples `latestRef`.
 *
 * Steps are only counted while `setStepsAllowed(true)` (the default). Navigate closes the gate
 * while the CNN says the user is standing still, so bumps and gestures don't move the map dot.
 */
export default function useSensors(enabled = true) {
  const [sensors, setSensors] = useState({
    // Orientation
    compass: 0,      // heading, degrees clockwise from north (raw)
    heading: 0,      // smoothed heading used for dead reckoning
    headingSource: null,   // 'compass' (absolute) | 'relative' (gyro-only, arbitrary zero) | null
    tilt_fb: 0,      // beta:  tilt front/back -180 to 180°
    tilt_lr: 0,      // gamma: tilt left/right -90 to 90°
    // Acceleration (DeviceMotionEvent.accelerationIncludingGravity)
    accel_x: 0,
    accel_y: 0,
    accel_z: 0,
    accel_mag: 0,    // sqrt(x²+y²+z²), ~9.8 when still
    // Gyroscope (DeviceMotionEvent.rotationRate) — angular velocity in deg/s
    gyro_alpha: 0,
    gyro_beta: 0,
    gyro_gamma: 0,
    // Derived
    steps: 0,
    timestamp: Date.now(),
  })
  const [granted, setGranted] = useState(false)
  const [error, setError] = useState(null)

  const stepsRef = useRef(0)
  const detectorRef = useRef(new StepDetector())
  const stepsAllowedRef = useRef(true)
  const pendingStepsRef = useRef([])
  const lastUpdateRef = useRef(0)
  const headingRef = useRef(null)
  const hasAbsoluteRef = useRef(false)
  // Latest raw motion reading (unthrottled) — sampled at 10 Hz by useMLClassifier
  const latestRef = useRef(null)

  const setStepsAllowed = useCallback((allowed) => {
    if (allowed && !stepsAllowedRef.current) {
      // The classifier needs ~2 s to notice walking has started — don't lose those first steps
      const cutoff = Date.now() - PENDING_STEP_MS
      stepsRef.current += pendingStepsRef.current.filter(t => t >= cutoff).length
    }
    pendingStepsRef.current = []
    stepsAllowedRef.current = allowed
  }, [])

  const handleMotion = useCallback((event) => {
    const a = event.accelerationIncludingGravity || {}
    const accel_x = a.x ?? 0
    const accel_y = a.y ?? 0
    const accel_z = a.z ?? 0
    const accel_mag = Math.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)

    const r = event.rotationRate || {}
    const gyro_alpha = r.alpha ?? 0
    const gyro_beta = r.beta ?? 0
    const gyro_gamma = r.gamma ?? 0

    const now = Date.now()
    if (detectorRef.current.update(accel_mag, now)) {
      if (stepsAllowedRef.current) stepsRef.current += 1
      else pendingStepsRef.current = [...pendingStepsRef.current.filter(t => now - t < PENDING_STEP_MS), now]
    }
    latestRef.current = { accel_x, accel_y, accel_z, gyro_alpha, gyro_beta, gyro_gamma, t: now }

    // Throttle UI updates to ~15fps
    if (now - lastUpdateRef.current < 66) return
    lastUpdateRef.current = now

    setSensors(prev => ({
      ...prev,
      accel_x: +accel_x.toFixed(2),
      accel_y: +accel_y.toFixed(2),
      accel_z: +accel_z.toFixed(2),
      accel_mag: +accel_mag.toFixed(2),
      gyro_alpha: +gyro_alpha.toFixed(1),
      gyro_beta: +gyro_beta.toFixed(1),
      gyro_gamma: +gyro_gamma.toFixed(1),
      steps: stepsRef.current,
      timestamp: now,
    }))
  }, [])

  const handleOrientation = useCallback((event) => {
    const h = headingFromEvent(event)
    if (!h) return
    // Android fires both `deviceorientation` (relative) and `deviceorientationabsolute`.
    // Once a real compass bearing has been seen, ignore the relative stream.
    if (h.absolute) hasAbsoluteRef.current = true
    else if (hasAbsoluteRef.current) return

    headingRef.current = smoothHeading(headingRef.current, h.heading)
    setSensors(prev => ({
      ...prev,
      compass: +h.heading.toFixed(1),
      heading: +headingRef.current.toFixed(1),
      headingSource: h.absolute ? 'compass' : 'relative',
      tilt_fb: +(event.beta ?? 0).toFixed(1),
      tilt_lr: +(event.gamma ?? 0).toFixed(1),
    }))
  }, [])

  const requestPermission = useCallback(async () => {
    try {
      if (typeof DeviceMotionEvent !== 'undefined' &&
          typeof DeviceMotionEvent.requestPermission === 'function') {
        const res = await DeviceMotionEvent.requestPermission()
        if (res !== 'granted') {
          setError('Motion sensor permission denied')
          return
        }
      }
      if (typeof DeviceOrientationEvent !== 'undefined' &&
          typeof DeviceOrientationEvent.requestPermission === 'function') {
        await DeviceOrientationEvent.requestPermission()
      }
      setGranted(true)
      setError(null)
    } catch (err) {
      setError(`Sensor permission error: ${err.message}`)
    }
  }, [])

  useEffect(() => {
    if (!granted || !enabled) return
    window.addEventListener('devicemotion', handleMotion)
    window.addEventListener('deviceorientation', handleOrientation)
    window.addEventListener('deviceorientationabsolute', handleOrientation)
    return () => {
      window.removeEventListener('devicemotion', handleMotion)
      window.removeEventListener('deviceorientation', handleOrientation)
      window.removeEventListener('deviceorientationabsolute', handleOrientation)
    }
  }, [granted, enabled, handleMotion, handleOrientation])

  useEffect(() => {
    if (typeof DeviceMotionEvent === 'undefined') {
      setGranted(true)
      return
    }
    if (typeof DeviceMotionEvent.requestPermission !== 'function') {
      setGranted(true)
    }
  }, [])

  return { sensors, granted, requestPermission, error, latestRef, setStepsAllowed }
}
