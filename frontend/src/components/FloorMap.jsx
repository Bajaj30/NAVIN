import { useEffect, useReducer, useState, useRef } from 'react'
import { getFloorPlanMeta, getFloorPlanUrl } from '../services/api'
import { ACTIVITY } from '../utils/activityClassifier'
import { normalize } from '../utils/heading'
import { FLOOR, ENTRY, STEP_LENGTH_M, bayRect, advance } from '../utils/floorLayout'

const TRAIL_POINTS = 150

/**
 * FloorMap — canvas floor map with pedestrian dead reckoning, drawn in metres (utils/floorLayout).
 * If admin uploaded a floor plan → renders it as background, otherwise a grid.
 *
 * Position = entry gate + one step length per detected step along the phone's heading.
 * The heading is zeroed when the session starts ("up the map" = the way you face walking in),
 * so it works with a true compass and with gyro-only relative headings alike.
 * Dead reckoning drifts; tap the map to correct the dot, or reset it to the entry.
 */
export default function FloorMap({ bays = [], sensors = {}, structureId = 'struct_demo_001', floor = 0, activity }) {
  const canvasRef = useRef(null)
  const [floorPlanImg, setFloorPlanImg] = useState(null)
  const [drawTick, redraw] = useReducer(n => n + 1, 0)
  const posRef = useRef({ ...ENTRY })
  const trailRef = useRef([{ ...ENTRY }])
  const headingZeroRef = useRef(null)      // sensor heading that corresponds to "up the map"
  const lastStepsRef = useRef(0)

  const mapHeading = headingZeroRef.current === null ? 0 : normalize((sensors.heading || 0) - headingZeroRef.current)

  // Check for uploaded floor plan
  useEffect(() => {
    setFloorPlanImg(null)
    getFloorPlanMeta(structureId, floor)
      .then(res => {
        if (res.data.exists) {
          const img = new Image()
          img.crossOrigin = 'anonymous'
          img.src = getFloorPlanUrl(structureId, floor)
          img.onload = () => setFloorPlanImg(img)
        }
      })
      .catch(() => {})
  }, [structureId, floor])

  // Zero the heading on the first orientation reading
  useEffect(() => {
    if (sensors.headingSource && headingZeroRef.current === null) headingZeroRef.current = sensors.heading
  }, [sensors.headingSource, sensors.heading])

  // Dead reckoning: advance one step length per new step
  useEffect(() => {
    const currentSteps = sensors.steps || 0
    const newSteps = currentSteps - lastStepsRef.current
    lastStepsRef.current = currentSteps
    if (newSteps <= 0 || headingZeroRef.current === null) return

    const stepLength = activity === ACTIVITY.STAIRS ? STEP_LENGTH_M.stairs : STEP_LENGTH_M.walking
    const heading = normalize((sensors.heading || 0) - headingZeroRef.current)
    posRef.current = advance(posRef.current, heading, stepLength * newSteps)
    trailRef.current = [...trailRef.current.slice(-TRAIL_POINTS + 1), posRef.current]
  }, [sensors.steps, sensors.heading, activity])

  const moveTo = (pos) => {
    posRef.current = pos
    trailRef.current = [pos]
    redraw()
  }

  const handleTap = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    moveTo({
      x: ((e.clientX - rect.left) / rect.width) * FLOOR.width,
      y: ((e.clientY - rect.top) / rect.height) * FLOOR.height,
    })
  }

  const resetToEntry = () => {
    headingZeroRef.current = sensors.headingSource ? sensors.heading : null
    moveTo({ ...ENTRY })
  }

  // Render canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height
    const S = W / FLOOR.width            // pixels per metre

    // Clear
    ctx.fillStyle = '#0f1729'
    ctx.fillRect(0, 0, W, H)

    // Draw floor plan or grid
    if (floorPlanImg) {
      ctx.globalAlpha = 0.85
      ctx.drawImage(floorPlanImg, 0, 0, W, H)
      ctx.globalAlpha = 1.0
    } else {
      // Grid placeholder
      ctx.strokeStyle = 'rgba(255,255,255,0.05)'
      ctx.lineWidth = 1
      for (let x = 0; x < W; x += S * 2) {   // 2 m grid
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
      }
      for (let y = 0; y < H; y += S * 2) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
      }

      // Label
      ctx.fillStyle = 'rgba(255,255,255,0.15)'
      ctx.font = '12px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(`Floor ${floor + 1} — no floor plan uploaded (Admin → Map)`, W / 2, H * 0.72)
    }

    // Draw bays
    const floorBays = bays.filter(b => b.floor === floor)
    floorBays.forEach((bay, i) => {
      const r = bayRect(i)
      const bx = r.x * S
      const by = r.y * S
      const bw = r.width * S
      const bh = r.height * S

      // Bay rectangle
      const isFree = bay.status === 'free'
      ctx.fillStyle = isFree ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)'
      ctx.strokeStyle = isFree ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.5)'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.roundRect(bx, by, bw, bh, 6)
      ctx.fill()
      ctx.stroke()

      // Bay label
      ctx.fillStyle = isFree ? '#10b981' : '#ef4444'
      ctx.font = 'bold 12px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(bay.bay_number, bx + bw / 2, by + bh / 2)

      // EV indicator
      if (bay.bay_type === 'ev') {
        ctx.fillStyle = '#fbbf24'
        ctx.font = '10px sans-serif'
        ctx.fillText('⚡', bx + bw / 2, by + bh / 2 + 17)
      }

      // Status dot
      ctx.beginPath()
      ctx.arc(bx + bw - 9, by + 9, 3.5, 0, Math.PI * 2)
      ctx.fillStyle = isFree ? '#10b981' : '#ef4444'
      ctx.fill()
    })

    // Entry marker
    ctx.fillStyle = 'rgba(255,255,255,0.35)'
    ctx.font = 'bold 9px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('▲ ENTRY', ENTRY.x * S, ENTRY.y * S + 24)

    // Walked path
    const trail = trailRef.current
    if (trail.length > 1) {
      ctx.beginPath()
      ctx.moveTo(trail[0].x * S, trail[0].y * S)
      trail.slice(1).forEach(p => ctx.lineTo(p.x * S, p.y * S))
      ctx.strokeStyle = 'rgba(129,140,248,0.45)'
      ctx.lineWidth = 2
      ctx.lineJoin = 'round'
      ctx.stroke()
    }

    // Scale bar (5 m)
    ctx.strokeStyle = 'rgba(255,255,255,0.4)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(W - 12 - 5 * S, H - 12)
    ctx.lineTo(W - 12, H - 12)
    ctx.stroke()
    ctx.fillStyle = 'rgba(255,255,255,0.4)'
    ctx.font = '10px Inter, sans-serif'
    ctx.textAlign = 'right'
    ctx.fillText('5 m', W - 12, H - 18)

    // Draw position dot with heading arrow
    const px = posRef.current.x * S
    const py = posRef.current.y * S

    // Glow
    const glow = ctx.createRadialGradient(px, py, 0, px, py, 20)
    glow.addColorStop(0, 'rgba(99,102,241,0.4)')
    glow.addColorStop(1, 'rgba(99,102,241,0)')
    ctx.fillStyle = glow
    ctx.beginPath()
    ctx.arc(px, py, 20, 0, Math.PI * 2)
    ctx.fill()

    // Dot
    ctx.beginPath()
    ctx.arc(px, py, 8, 0, Math.PI * 2)
    ctx.fillStyle = '#6366f1'
    ctx.fill()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.stroke()

    // Heading arrow
    const headingRad = (mapHeading * Math.PI) / 180
    const arrowLen = 18
    const ax = px + Math.sin(headingRad) * arrowLen
    const ay = py - Math.cos(headingRad) * arrowLen
    ctx.beginPath()
    ctx.moveTo(px, py)
    ctx.lineTo(ax, ay)
    ctx.strokeStyle = '#a5b4fc'
    ctx.lineWidth = 2
    ctx.stroke()

    // Arrow tip
    ctx.beginPath()
    ctx.arc(ax, ay, 3, 0, Math.PI * 2)
    ctx.fillStyle = '#a5b4fc'
    ctx.fill()

    // Info text
    ctx.fillStyle = 'rgba(255,255,255,0.4)'
    ctx.font = '10px Inter, sans-serif'
    ctx.textAlign = 'left'
    const source = sensors.headingSource === 'compass' ? 'compass' : sensors.headingSource === 'relative' ? 'gyro, relative' : 'no heading yet'
    ctx.fillText(`Steps ${sensors.steps || 0} · heading ${mapHeading.toFixed(0)}° (${source}) · ${posRef.current.x.toFixed(1)}, ${posRef.current.y.toFixed(1)} m`, 10, 16)

  }, [bays, sensors, floorPlanImg, floor, mapHeading, drawTick])

  return (
    <div className="glass-card overflow-hidden rounded-xl">
      <canvas
        ref={canvasRef}
        width={500}
        height={400}
        onClick={handleTap}
        className="w-full cursor-crosshair"
        style={{ imageRendering: 'crisp-edges' }}
      />
      <div className="flex items-center justify-between gap-3 px-3 py-2 border-t border-white/5">
        <p className="text-[10px] text-white/40">Estimated from steps + heading — tap the map to correct your position</p>
        <button onClick={resetToEntry} className="shrink-0 text-[10px] font-medium text-navin-400 underline">
          Reset to entry
        </button>
      </div>
    </div>
  )
}
