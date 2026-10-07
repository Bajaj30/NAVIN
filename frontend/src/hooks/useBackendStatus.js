import { useState, useEffect, useCallback } from 'react'
import { waitForBackend } from '../services/api'

/**
 * useBackendStatus — tracks whether the (possibly sleeping) Render backend is reachable.
 * Returns { status: 'checking' | 'waking' | 'ready' | 'offline', retry }
 */
export default function useBackendStatus() {
  const [status, setStatus] = useState('checking')
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    // Healthy servers answer in < 1 s; anything slower is a cold start
    const slow = setTimeout(() => {
      if (!cancelled) setStatus(s => (s === 'checking' ? 'waking' : s))
    }, 2500)
    waitForBackend().then(ok => {
      if (!cancelled) setStatus(ok ? 'ready' : 'offline')
    })
    return () => { cancelled = true; clearTimeout(slow) }
  }, [attempt])

  const retry = useCallback(() => {
    setStatus('checking')
    setAttempt(a => a + 1)
  }, [])

  return { status, retry }
}
