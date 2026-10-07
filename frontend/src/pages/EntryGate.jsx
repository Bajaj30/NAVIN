import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import QRScanner from '../components/QRScanner'
import { createSession, waitForBackend } from '../services/api'
import useBackendStatus from '../hooks/useBackendStatus'

export default function EntryGate() {
  const navigate = useNavigate()
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [scanned, setScanned] = useState(false)
  const { status: serverStatus, retry: retryServer } = useBackendStatus()

  const handleDecode = useCallback(async (data) => {
    if (scanned || loading) return
    setScanned(true)
    setLoading(true)
    setError(null)

    try {
      // A QR scan can arrive while the free-tier server is still waking up
      if (!(await waitForBackend())) throw new Error('Server unreachable — check your connection and retry')
      const res = await createSession({
        structure_id: data.structure_id,
        gate: data.gate || 1,
      })
      const sessionId = res.data.session_id
      // Persist session for page refreshes
      localStorage.setItem('navin_session_id', sessionId)
      localStorage.setItem('navin_structure_id', data.structure_id)
      navigate(`/navigate/${sessionId}`)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
      setScanned(false)
    } finally {
      setLoading(false)
    }
  }, [navigate, scanned, loading])

  // Demo: skip QR if no camera (desktop testing)
  const handleDemoEntry = async () => {
    handleDecode({ structure_id: 'struct_demo_001', gate: 1, type: 'entry' })
  }

  return (
    <div className="min-h-screen bg-mesh flex flex-col items-center justify-center p-6">
      <div className="glass-card p-8 max-w-md w-full animate-fade-in">
        {/* Logo */}
        <div className="text-center mb-6">
          <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-navin-500 to-purple-600 flex items-center justify-center glow-blue">
            <span className="text-4xl">🅿️</span>
          </div>
          <h1 className="text-3xl font-bold text-white">NAVIN</h1>
          <p className="text-white/50 text-sm mt-1">Indoor Parking Navigation</p>
        </div>

        {/* QR Scanner */}
        {!scanned && (
          <div className="mb-6">
            <QRScanner
              onDecode={handleDecode}
              onError={setError}
            />
          </div>
        )}

        {/* Loading state */}
        {loading && (
          <div className="text-center py-8">
            <div className="w-10 h-10 mx-auto mb-3 rounded-full border-2 border-navin-400 border-t-transparent animate-spin" />
            <p className="text-white/60 text-sm">
              {serverStatus === 'ready' ? 'Creating your session…' : 'Waiting for the server to wake up…'}
            </p>
          </div>
        )}

        {/* Server status (Render free tier cold start) */}
        {serverStatus === 'waking' && !loading && (
          <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center gap-3">
            <div className="w-4 h-4 shrink-0 rounded-full border-2 border-amber-400 border-t-transparent animate-spin" />
            <p className="text-amber-300 text-xs">
              Waking up the server — free hosting sleeps when idle, this takes up to a minute.
            </p>
          </div>
        )}
        {serverStatus === 'offline' && (
          <div className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center justify-between gap-3">
            <p className="text-red-400 text-xs">Server unreachable.</p>
            <button onClick={retryServer} className="text-xs text-white/70 underline">Retry</button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Instructions */}
        <p className="text-white/40 text-xs text-center mb-4">
          Point your camera at the entry QR code
        </p>

        {/* Demo button for desktop testing */}
        <button
          onClick={handleDemoEntry}
          disabled={loading || serverStatus !== 'ready'}
          className="w-full btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {serverStatus === 'ready' ? '🧪 Demo Entry (No QR)' : '⏳ Connecting to server…'}
        </button>
        {serverStatus === 'ready' && (
          <p className="text-emerald-400/70 text-[10px] text-center mt-2">● Server online</p>
        )}
      </div>
    </div>
  )
}
