import { useState, useEffect, useRef } from 'react'
import { predictActivity } from '../services/api'

const SAMPLE_MS = 100        // 10 Hz — same rate the training data was collected at
const WINDOW = 20            // 2-second window
const PREDICT_EVERY = 10     // samples → one prediction per second (50% overlap, as in training)
const SENSOR_STALE_MS = 1000 // no devicemotion for this long → no sensors (desktop)

/**
 * useMLClassifier — streams 2-second motion windows to the 1D-CNN on the backend.
 *
 * Samples useSensors' latest raw reading every 100 ms (zero-order hold, exactly like the
 * collector page), so live windows match the training data. On any API failure it reports
 * mode 'fallback' and backs off; the caller then uses the on-device heuristic.
 *
 * Returns { mode: 'disabled' | 'waiting' | 'ml' | 'fallback' | 'no-sensors', prediction, latency, error }
 */
export default function useMLClassifier(latestRef, sessionId, enabled = true) {
  const [state, setState] = useState({ mode: 'waiting', prediction: null, latency: null, error: null })
  const bufferRef = useRef([])
  const inFlightRef = useRef(false)
  const failuresRef = useRef(0)
  const retryAtRef = useRef(0)

  useEffect(() => {
    if (!enabled) return
    let ticks = 0
    let alive = true

    const id = setInterval(() => {
      const s = latestRef.current
      if (!s || Date.now() - s.t > SENSOR_STALE_MS) {
        bufferRef.current = []
        setState(prev => (prev.mode === 'no-sensors' ? prev : { ...prev, mode: 'no-sensors', prediction: null }))
        return
      }

      bufferRef.current.push([s.accel_x, s.accel_y, s.accel_z, s.gyro_alpha, s.gyro_beta, s.gyro_gamma])
      if (bufferRef.current.length > WINDOW) bufferRef.current.shift()
      ticks += 1

      if (bufferRef.current.length < WINDOW) {
        setState(prev => (prev.mode === 'no-sensors' ? { ...prev, mode: 'waiting' } : prev))
        return
      }
      if (ticks % PREDICT_EVERY !== 0 || inFlightRef.current || Date.now() < retryAtRef.current) return

      inFlightRef.current = true
      const t0 = performance.now()
      predictActivity(bufferRef.current.slice(), sessionId)
        .then(res => {
          if (!alive) return
          failuresRef.current = 0
          setState({ mode: 'ml', prediction: { ...res.data, at: Date.now() }, latency: Math.round(performance.now() - t0), error: null })
        })
        .catch(err => {
          if (!alive) return
          failuresRef.current += 1
          retryAtRef.current = Date.now() + Math.min(30000, 2000 * failuresRef.current)
          setState(prev => ({ ...prev, mode: 'fallback', error: err.response?.data?.detail || err.message }))
        })
        .finally(() => { inFlightRef.current = false })
    }, SAMPLE_MS)

    return () => { alive = false; clearInterval(id) }
  }, [latestRef, sessionId, enabled])

  // Sensors not granted yet (iOS and recent Chrome need a tap on "Enable Sensors")
  return enabled ? state : { ...state, mode: 'disabled', prediction: null }
}
