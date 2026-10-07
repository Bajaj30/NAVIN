import { useEffect, useMemo, useRef, useState } from 'react'
import { buildFingerprintMap, PSOSimulation, WORLD, CELL, DT, ENTRY, LANES, BAY_ROWS } from '../utils/pso'

/**
 * PSOSimulator — visualises PSO magnetic-fingerprint positioning correcting dead-reckoning drift.
 * Runs entirely in the browser on a synthetic fingerprint map (see utils/pso.js).
 */
const PX = 20                                  // pixels per metre
const W = WORLD.w * PX, H = WORLD.h * PX
const COLORS = { truth: 'rgba(255,255,255,0.75)', dr: '#d95926', pso: '#3987e5', particle: 'rgba(165,180,252,0.75)' }

function drawPath(ctx, pts, color, width) {
  if (pts.length < 2) return
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.lineJoin = 'round'
  ctx.beginPath()
  ctx.moveTo(pts[0].x * PX, pts[0].y * PX)
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * PX, pts[i].y * PX)
  ctx.stroke()
}

function dot(ctx, p, r, fill, ring) {
  ctx.beginPath()
  ctx.arc(p.x * PX, p.y * PX, r, 0, Math.PI * 2)
  ctx.fillStyle = fill
  ctx.fill()
  if (ring) { ctx.strokeStyle = ring; ctx.lineWidth = 2; ctx.stroke() }
}

export default function PSOSimulator() {
  const canvasRef = useRef(null)
  const map = useMemo(() => buildFingerprintMap(7), [])
  const simRef = useRef(null)
  const seedRef = useRef(1)
  const [running, setRunning] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [showMap, setShowMap] = useState(true)
  const [metrics, setMetrics] = useState(null)

  // Fingerprint heatmap (|anomaly| µT) — neutral one-hue ramp so the coloured paths stay readable.
  // One pixel per 0.5 m cell, smoothly upscaled when drawn.
  const heatmap = useMemo(() => {
    const c = document.createElement('canvas')
    c.width = WORLD.w / CELL; c.height = WORLD.h / CELL
    const ctx = c.getContext('2d')
    for (let j = 0; j < c.height; j++) {
      for (let i = 0; i < c.width; i++) {
        const a = Math.min(1, map.anomaly((i + 0.5) * CELL, (j + 0.5) * CELL) / 35)
        ctx.fillStyle = `rgba(226,232,240,${0.02 + 0.2 * a})`
        ctx.fillRect(i, j, 1, 1)
      }
    }
    return c
  }, [map])

  if (!simRef.current) simRef.current = new PSOSimulation(map, { seed: seedRef.current })

  const restart = () => {
    seedRef.current += 1
    simRef.current = new PSOSimulation(map, { seed: seedRef.current })
    setRunning(true)
  }

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d')
    let raf, last = performance.now(), acc = 0, lastMetrics = 0, doneAt = null

    const draw = () => {
      const sim = simRef.current
      ctx.fillStyle = '#0f1729'
      ctx.fillRect(0, 0, W, H)
      if (showMap) { ctx.imageSmoothingEnabled = true; ctx.drawImage(heatmap, 0, 0, W, H) }

      ctx.fillStyle = 'rgba(255,255,255,0.05)'
      for (const l of LANES) ctx.fillRect(l.x * PX, l.y * PX, l.w * PX, l.h * PX)
      ctx.strokeStyle = 'rgba(255,255,255,0.12)'
      ctx.lineWidth = 1
      for (const row of BAY_ROWS) {
        for (let x = row.x0; x < row.x1; x += 2.5) ctx.strokeRect(x * PX, row.y * PX, 2.5 * PX, 4 * PX)
      }
      // Entry gate (QR) — the known start position
      ctx.fillStyle = '#12b886'
      ctx.fillRect(ENTRY.x * PX - 9, (ENTRY.y + 1) * PX - 9, 18, 18)
      ctx.fillStyle = '#0f1729'
      ctx.font = 'bold 9px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('QR', ENTRY.x * PX, (ENTRY.y + 1) * PX + 3)

      drawPath(ctx, sim.paths.truth, COLORS.truth, 1.5)
      drawPath(ctx, sim.paths.dr, COLORS.dr, 2)
      drawPath(ctx, sim.paths.pso, COLORS.pso, 2)

      for (const p of sim.particles) dot(ctx, p, 2.2, COLORS.particle)
      // Confidence ring = swarm spread
      ctx.beginPath()
      ctx.arc(sim.estimate.x * PX, sim.estimate.y * PX, Math.max(6, sim.spread * PX), 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(57,135,229,0.5)'
      ctx.lineWidth = 1
      ctx.stroke()

      dot(ctx, sim.dr, 5, COLORS.dr, '#0f1729')
      dot(ctx, sim.truth, 6, 'rgba(255,255,255,0.9)', '#0f1729')
      dot(ctx, sim.estimate, 6, COLORS.pso, '#ffffff')
    }

    const frame = (now) => {
      const sim = simRef.current
      const elapsed = Math.min(250, now - last)
      last = now
      if (running && !sim.done) {
        acc += elapsed * speed
        let n = 0
        while (acc >= DT * 1000 && n < 40) { sim.step(); acc -= DT * 1000; n++ }
      }
      if (sim.done) {
        doneAt ??= now
        if (running && now - doneAt > 3000) {   // loop the demo with a fresh random drift
          seedRef.current += 1
          simRef.current = new PSOSimulation(map, { seed: seedRef.current })
          doneAt = null
        }
      }
      draw()
      if (now - lastMetrics > 200) { setMetrics(simRef.current.metrics()); lastMetrics = now }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [running, speed, showMap, heatmap, map])

  const m = metrics
  const tiles = m && [
    { label: 'Elapsed', value: `${m.t.toFixed(0)} s`, sub: `${Math.round(m.progress * 100)}% of route` },
    { label: 'Dead reckoning error', value: `${m.errDR.toFixed(1)} m`, sub: `RMSE ${m.rmseDR.toFixed(1)} m`, swatch: COLORS.dr },
    { label: 'PSO error', value: `${m.errPSO.toFixed(1)} m`, sub: `RMSE ${m.rmsePSO.toFixed(2)} m`, swatch: COLORS.pso },
    { label: 'Swarm spread', value: `${m.spread.toFixed(1)} m`, sub: `confidence ${Math.round(m.confidence * 100)}%` },
  ]

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="glass-card p-4">
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-lg font-bold text-white">🛰️ PSO magnetic-fingerprint positioning</h3>
            <p className="text-xs text-white/50 mt-0.5 max-w-2xl">
              80 particles @ 10 Hz. Each particle is a position hypothesis, moved by step-and-heading dead reckoning and
              scored by Mahalanobis distance between the last 2 s of magnetometer readings and the fingerprint map.
              Swarm spread is the confidence.
            </p>
          </div>
          <span className="text-[10px] px-2 py-1 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30 whitespace-nowrap">
            Simulation · synthetic fingerprint map
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 mt-4">
          <button onClick={() => setRunning(r => !r)} className="btn-secondary text-xs px-3 py-2">{running ? '⏸ Pause' : '▶ Play'}</button>
          <button onClick={restart} className="btn-secondary text-xs px-3 py-2">↺ New walk</button>
          <button onClick={() => simRef.current.scatter()} className="btn-secondary text-xs px-3 py-2" title="Throw the swarm across the whole floor and watch it relocalise">
            💥 Scatter swarm
          </button>
          <div className="flex rounded-xl overflow-hidden border border-white/15">
            {[1, 2, 4].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                      className={`text-xs px-3 py-2 ${speed === s ? 'bg-navin-600 text-white' : 'bg-white/5 text-white/60 hover:bg-white/10'}`}>
                {s}×
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 text-xs text-white/60 ml-1">
            <input type="checkbox" checked={showMap} onChange={e => setShowMap(e.target.checked)} /> Fingerprint map
          </label>
        </div>
      </div>

      {tiles && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {tiles.map(t => (
            <div key={t.label} className="glass-card p-3">
              <p className="text-[11px] text-white/50 flex items-center gap-1.5">
                {t.swatch && <span className="w-2.5 h-0.5 rounded" style={{ background: t.swatch }} />}{t.label}
              </p>
              <p className="text-xl font-bold text-white mt-0.5">{t.value}</p>
              <p className="text-[10px] text-white/40">{t.sub}</p>
            </div>
          ))}
        </div>
      )}

      <div className="glass-card p-3">
        <div className="flex flex-wrap gap-4 text-[11px] text-white/60 mb-2 px-1">
          <span className="flex items-center gap-1.5"><span className="w-4 h-0.5" style={{ background: COLORS.truth }} />True path</span>
          <span className="flex items-center gap-1.5"><span className="w-4 h-0.5" style={{ background: COLORS.dr }} />Dead reckoning only (drifts)</span>
          <span className="flex items-center gap-1.5"><span className="w-4 h-0.5" style={{ background: COLORS.pso }} />PSO estimate</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full" style={{ background: COLORS.particle }} />Particles</span>
          {showMap && <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: 'rgba(226,232,240,0.22)' }} />Stronger magnetic anomaly</span>}
        </div>
        <canvas ref={canvasRef} width={W} height={H} className="w-full rounded-xl" />
        <p className="text-[10px] text-white/30 mt-2 px-1">
          Real phones need a fingerprint survey of the car park and raw magnetometer access (Generic Sensor API, behind a flag
          in Chrome today) — the live Navigate screen therefore uses dead reckoning; this is the correction layer it is designed for.
        </p>
      </div>
    </div>
  )
}
