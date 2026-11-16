import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' }
})

// ── Backend readiness ───────────────────────────────────────────────
// Render's free tier sleeps after 15 min idle and takes ~30–60 s to wake.
// One shared poll of /api/health so every caller waits on the same wake-up.
let backendReady = null
export function waitForBackend(maxWaitMs = 150000) {
  if (!backendReady) {
    backendReady = (async () => {
      const deadline = Date.now() + maxWaitMs
      while (Date.now() < deadline) {
        try {
          await api.get('/api/health', { timeout: 20000 })
          return true
        } catch {
          await new Promise(r => setTimeout(r, 2000))
        }
      }
      backendReady = null   // let a later call try again
      return false
    })()
  }
  return backendReady
}

// ── Sessions ────────────────────────────────────────────────────────
export const createSession  = (payload)  => api.post('/api/sessions/entry', payload)
export const getSession     = (id)       => api.get(`/api/sessions/${id}`)
export const parkSession    = (id, bay)  => api.post(`/api/sessions/${id}/park`, { bay_id: bay })
export const exitSession    = (id)       => api.post(`/api/sessions/${id}/exit`)

// ── Occupancy ───────────────────────────────────────────────────────
export const getOccupancy   = (structureId) => api.get(`/api/occupancy/${structureId}`)
export const getOccSummary  = (structureId) => api.get(`/api/occupancy/${structureId}/summary`)

// ── Sensors ─────────────────────────────────────────────────────────
export const getPosition    = (sessionId) => api.get(`/api/sensors/position/${sessionId}`)

// ── ML (1D-CNN served by the backend via ONNX Runtime) ─────────────
// samples: 20 × [accel_x, accel_y, accel_z, rotation_alpha, rotation_beta, rotation_gamma]
export const predictActivity = (samples, sessionId) =>
  api.post('/api/ml/predict', { samples, session_id: sessionId }, { timeout: 4000 })
export const getMLModel = () => api.get('/api/ml/model')

// ── QR ──────────────────────────────────────────────────────────────
export const getQRUrl = (structureId, type = 'entry') =>
  `${import.meta.env.VITE_API_BASE_URL || ''}/api/qr/${structureId}${type === 'exit' ? '/exit' : ''}`

// ── Admin ───────────────────────────────────────────────────────────
export const getAdminSessions = () => api.get('/api/admin/sessions')
export const getAdminLogs     = () => api.get('/api/admin/logs')
export const getAdminStats    = () => api.get('/api/admin/stats')
export const getTrainingData  = () => api.get('/api/admin/training-data')
export const getTrainingDataUrl = (filename) =>
  `${import.meta.env.VITE_API_BASE_URL || ''}/api/admin/training-data/${encodeURIComponent(filename)}`

// ── Floor Plan ──────────────────────────────────────────────────────
export const uploadFloorPlan = (formData) =>
  api.post('/api/admin/floorplan', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
export const getFloorPlanMeta = (structureId, floor = 0) =>
  api.get(`/api/admin/floorplan/${structureId}/meta?floor=${floor}`)
export const getFloorPlanUrl = (structureId, floor = 0) =>
  `${import.meta.env.VITE_API_BASE_URL || ''}/api/admin/floorplan/${structureId}?floor=${floor}`

export default api
