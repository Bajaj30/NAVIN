/**
 * PSO magnetic-fingerprint positioning — simulation.
 *
 * Browsers don't expose raw magnetometer µT on most phones, so this runs on a
 * SYNTHETIC fingerprint map to demonstrate the positioning layer NAVIN is designed
 * around (NavIn_Project_Context.pdf §5.2):
 *   - fingerprint DB: per-cell mean + std of the 3-axis field (0.5 m grid)
 *   - motion model: pedestrian dead reckoning (drifts — gyro bias + step-length error)
 *   - measurement: Mahalanobis distance between recent readings and the map
 *   - optimiser: Particle Swarm Optimisation, 80 particles @ 10 Hz;
 *     swarm spread is the confidence metric
 */

export const WORLD = { w: 50, h: 30 }          // metres
export const CELL = 0.5                        // fingerprint grid resolution (m)
export const DT = 0.1                          // 10 Hz
export const ENTRY = { x: 3, y: 27 }           // QR gate — known start position

// Walking route through the car park lanes (m)
export const ROUTE = [
  { x: 3, y: 27 }, { x: 3, y: 7.5 }, { x: 46, y: 7.5 }, { x: 46, y: 22.5 }, { x: 10, y: 22.5 },
]
// Lanes + bay rows for drawing
export const LANES = [
  { x: 1.5, y: 6, w: 47, h: 3 }, { x: 1.5, y: 21, w: 47, h: 3 },
  { x: 1.5, y: 6, w: 3, h: 23 }, { x: 44.5, y: 6, w: 3, h: 18 },
]
export const BAY_ROWS = [
  { y: 1, x0: 6, x1: 44 }, { y: 10, x0: 6, x1: 44 }, { y: 15.5, x0: 6, x1: 44 }, { y: 25, x0: 6, x1: 44 },
]

// ── Seeded randomness ──────────────────────────────────────────────
export function makeRng(seed) {
  let a = seed >>> 0
  const rand = () => {
    a = (a + 0x6D2B79F5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const gauss = () => {
    const u = 1 - rand(), v = rand()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  return { rand, gauss }
}

// ── Synthetic magnetic field + fingerprint map ─────────────────────
const EARTH = [22, 5, -42]   // µT

export function buildFingerprintMap(seed = 7) {
  const { rand, gauss } = makeRng(seed)
  const sources = []
  // Steel pillars on an 8 m grid — strong, local
  for (let x = 4; x < WORLD.w; x += 8) {
    for (let y = 4; y < WORLD.h; y += 8) {
      sources.push({ x, y, r: 1.4, s: [gauss() * 14, gauss() * 14, gauss() * 18] })
    }
  }
  // Rebar / slab variation — weaker, wider
  for (let i = 0; i < 45; i++) {
    sources.push({ x: rand() * WORLD.w, y: rand() * WORLD.h, r: 1.2 + rand() * 2.5, s: [gauss() * 7, gauss() * 7, gauss() * 9] })
  }

  const field = (x, y) => {
    const b = [...EARTH]
    for (const s of sources) {
      const k = Math.exp(-((x - s.x) ** 2 + (y - s.y) ** 2) / (2 * s.r * s.r))
      b[0] += s.s[0] * k; b[1] += s.s[1] * k; b[2] += s.s[2] * k
    }
    return b
  }

  const nx = Math.round(WORLD.w / CELL) + 1
  const ny = Math.round(WORLD.h / CELL) + 1
  const mean = new Float32Array(nx * ny * 3)
  const std = new Float32Array(nx * ny * 3)
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const b = field(i * CELL, j * CELL)
      const anomaly = Math.hypot(b[0] - EARTH[0], b[1] - EARTH[1], b[2] - EARTH[2])
      for (let a = 0; a < 3; a++) {
        mean[(j * nx + i) * 3 + a] = b[a]
        std[(j * nx + i) * 3 + a] = 0.8 + 0.06 * anomaly   // near steel: more variable across visits
      }
    }
  }

  // Bilinear lookup of mean/std at (x, y) — positions off the map score as "very unlikely"
  const lookup = (x, y, outMean, outStd) => {
    if (x < 0 || y < 0 || x > WORLD.w || y > WORLD.h) return false
    const fx = x / CELL, fy = y / CELL
    const i = Math.min(nx - 2, Math.floor(fx)), j = Math.min(ny - 2, Math.floor(fy))
    const tx = fx - i, ty = fy - j
    for (let a = 0; a < 3; a++) {
      const idx = (jj, ii) => (jj * nx + ii) * 3 + a
      const blend = arr =>
        arr[idx(j, i)] * (1 - tx) * (1 - ty) + arr[idx(j, i + 1)] * tx * (1 - ty) +
        arr[idx(j + 1, i)] * (1 - tx) * ty + arr[idx(j + 1, i + 1)] * tx * ty
      outMean[a] = blend(mean)
      outStd[a] = blend(std)
    }
    return true
  }

  const anomaly = (x, y) => {
    const b = field(x, y)
    return Math.hypot(b[0] - EARTH[0], b[1] - EARTH[1], b[2] - EARTH[2])
  }

  return { field, lookup, anomaly, nx, ny }
}

// ── Route helpers ──────────────────────────────────────────────────
const SEGMENTS = ROUTE.slice(1).map((p, i) => {
  const a = ROUTE[i]
  return { a, b: p, len: Math.hypot(p.x - a.x, p.y - a.y) }
})
export const ROUTE_LENGTH = SEGMENTS.reduce((s, seg) => s + seg.len, 0)

function pointOnRoute(dist) {
  let d = Math.min(dist, ROUTE_LENGTH)
  for (const seg of SEGMENTS) {
    if (d <= seg.len) {
      const t = d / seg.len
      return { x: seg.a.x + (seg.b.x - seg.a.x) * t, y: seg.a.y + (seg.b.y - seg.a.y) * t }
    }
    d -= seg.len
  }
  return { ...ROUTE[ROUTE.length - 1] }
}

// ── Simulation ─────────────────────────────────────────────────────
export const DEFAULTS = {
  particles: 80,
  speed: 1.3,              // m/s walking
  sensorNoise: 1.0,        // µT per axis
  gyroBiasDegPerS: 0.3,    // heading drift
  stepScaleError: 0.06,    // step length mis-estimate
  history: 20,             // readings matched per update (2 s)
  iterations: 2,           // PSO iterations per 10 Hz tick
  inertia: 0.6, cognitive: 1.4, social: 1.4, maxVelocity: 1.5,
}

export class PSOSimulation {
  constructor(map, options = {}) {
    this.map = map
    this.opt = { ...DEFAULTS, ...options }
    this.reset(options.seed ?? 1)
  }

  reset(seed = 1) {
    const { gauss, rand } = makeRng(seed)
    this.gauss = gauss
    this.rand = rand
    this.t = 0
    this.dist = 0
    this.truth = { ...ENTRY }
    this.dr = { ...ENTRY }
    this.headingErr = 0
    this.scaleErr = 1 + this.opt.stepScaleError * (rand() < 0.5 ? -1 : 1)
    this.history = []
    this.paths = { truth: [{ ...ENTRY }], dr: [{ ...ENTRY }], pso: [{ ...ENTRY }] }
    this.sq = { dr: 0, pso: 0, n: 0 }
    this.particles = Array.from({ length: this.opt.particles }, () => this._particleNear(ENTRY, 2))
    this.estimate = { ...ENTRY }
    this.spread = 2
    this.done = false
  }

  _particleNear(c, sigma) {
    const x = c.x + this.gauss() * sigma, y = c.y + this.gauss() * sigma
    return { x, y, vx: 0, vy: 0, bx: x, by: y, bf: Infinity }
  }

  /** Kidnap test: throw the whole swarm uniformly across the floor. */
  scatter() {
    this.particles = this.particles.map(() => {
      const x = this.rand() * WORLD.w, y = this.rand() * WORLD.h
      return { x, y, vx: 0, vy: 0, bx: x, by: y, bf: Infinity }
    })
  }

  // Mean squared Mahalanobis distance of the recent readings if "now" were at (x, y)
  _fitness(x, y) {
    const m = [0, 0, 0], s = [0, 0, 0]
    const noise2 = this.opt.sensorNoise ** 2
    let total = 0
    for (const h of this.history) {
      if (!this.map.lookup(x + h.dx, y + h.dy, m, s)) return 1e6
      for (let a = 0; a < 3; a++) total += (h.obs[a] - m[a]) ** 2 / (s[a] * s[a] + noise2)
    }
    return total / (this.history.length * 3)
  }

  step() {
    if (this.done) return
    const o = this.opt
    const { gauss } = this

    // 1. Truth moves along the route
    this.dist += o.speed * DT
    const prev = this.truth
    this.truth = pointOnRoute(this.dist)
    const tdx = this.truth.x - prev.x, tdy = this.truth.y - prev.y
    if (this.dist >= ROUTE_LENGTH) this.done = true

    // 2. Dead reckoning: true motion seen through a drifting heading + wrong step length
    this.headingErr += (o.gyroBiasDegPerS * DT + gauss() * 0.3 * Math.sqrt(DT)) * Math.PI / 180
    const c = Math.cos(this.headingErr), s = Math.sin(this.headingErr)
    const ddx = (tdx * c - tdy * s) * this.scaleErr, ddy = (tdx * s + tdy * c) * this.scaleErr
    this.dr = { x: this.dr.x + ddx, y: this.dr.y + ddy }

    // 3. Magnetometer reading at the TRUE position
    const b = this.map.field(this.truth.x, this.truth.y)
    const obs = b.map(v => v + gauss() * o.sensorNoise)

    // History holds readings with their position relative to "now" (from DR deltas)
    for (const h of this.history) { h.dx -= ddx; h.dy -= ddy }
    this.history.push({ obs, dx: 0, dy: 0 })
    if (this.history.length > o.history) this.history.shift()

    // 4. PSO: shift swarm by the DR motion, then optimise against the recent readings
    for (const p of this.particles) {
      p.x += ddx + gauss() * 0.05; p.y += ddy + gauss() * 0.05
      p.bx += ddx; p.by += ddy
      p.bf = this._fitness(p.bx, p.by)        // the evidence changed — re-score personal bests
    }
    let best = this.particles.reduce((a, p) => (p.bf < a.bf ? p : a))
    for (let it = 0; it < o.iterations; it++) {
      const gx = best.bx, gy = best.by
      for (const p of this.particles) {
        const r1 = this.rand(), r2 = this.rand()
        p.vx = o.inertia * p.vx + o.cognitive * r1 * (p.bx - p.x) + o.social * r2 * (gx - p.x)
        p.vy = o.inertia * p.vy + o.cognitive * r1 * (p.by - p.y) + o.social * r2 * (gy - p.y)
        const v = Math.hypot(p.vx, p.vy)
        if (v > o.maxVelocity) { p.vx *= o.maxVelocity / v; p.vy *= o.maxVelocity / v }
        p.x += p.vx; p.y += p.vy
        const f = this._fitness(p.x, p.y)
        if (f < p.bf) { p.bf = f; p.bx = p.x; p.by = p.y }
      }
      best = this.particles.reduce((a, q) => (q.bf < a.bf ? q : a))
    }

    // 5. Estimate = mean of the best quarter of personal bests; spread = swarm RMS radius
    const ranked = [...this.particles].sort((a, q) => a.bf - q.bf).slice(0, Math.max(4, o.particles >> 2))
    const rx = ranked.reduce((a, p) => a + p.bx, 0) / ranked.length
    const ry = ranked.reduce((a, p) => a + p.by, 0) / ranked.length
    // Light smoothing of the reported fix (the estimate also rides the DR motion between ticks)
    const ex = 0.4 * rx + 0.6 * (this.estimate.x + ddx)
    const ey = 0.4 * ry + 0.6 * (this.estimate.y + ddy)
    this.estimate = { x: ex, y: ey }
    this.spread = Math.sqrt(this.particles.reduce((a, p) => a + (p.x - ex) ** 2 + (p.y - ey) ** 2, 0) / this.particles.length)
    this.bestFitness = best.bf

    // Poor match everywhere → re-seed the worst quarter around the estimate (recovery)
    if (best.bf > 6) {
      const worst = [...this.particles].sort((a, q) => q.bf - a.bf).slice(0, o.particles >> 2)
      for (const p of worst) Object.assign(p, this._particleNear(this.estimate, 4))
    }

    // 6. Bookkeeping
    this.t += DT
    const eDR = Math.hypot(this.dr.x - this.truth.x, this.dr.y - this.truth.y)
    const ePSO = Math.hypot(ex - this.truth.x, ey - this.truth.y)
    this.sq.dr += eDR ** 2; this.sq.pso += ePSO ** 2; this.sq.n += 1
    this.paths.truth.push({ ...this.truth })
    this.paths.dr.push({ ...this.dr })
    this.paths.pso.push({ ...this.estimate })
  }

  metrics() {
    const n = Math.max(1, this.sq.n)
    return {
      t: this.t,
      errDR: Math.hypot(this.dr.x - this.truth.x, this.dr.y - this.truth.y),
      errPSO: Math.hypot(this.estimate.x - this.truth.x, this.estimate.y - this.truth.y),
      rmseDR: Math.sqrt(this.sq.dr / n),
      rmsePSO: Math.sqrt(this.sq.pso / n),
      spread: this.spread,
      confidence: Math.exp(-this.spread / 3),
      progress: Math.min(1, this.dist / ROUTE_LENGTH),
    }
  }
}
