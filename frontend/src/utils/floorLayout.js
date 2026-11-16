/**
 * Demo floor layout in METRES (origin top-left, x → right, y → down, "up" = into the car park).
 * The canvas, the bays and dead reckoning all share these units, so a step moves the dot a
 * real distance instead of an arbitrary number of pixels.
 */
export const FLOOR = { width: 30, height: 24 }
export const ENTRY = { x: 15, y: 22 }          // entry gate / stairwell door: where a session starts

// Average adult step ≈ 0.7 m on the flat; a stair tread is ≈ 0.3 m of horizontal travel
export const STEP_LENGTH_M = { walking: 0.7, stairs: 0.3 }

const BAY = { width: 3, depth: 5 }
const BAYS_PER_ROW = 5
const ROW_Y = [2, 10]                           // two rows with a 3 m aisle between them
const FIRST_X = 3
const PITCH_X = 5.25

/** Rectangle (metres) of the i-th bay on a floor. */
export function bayRect(index) {
  const col = index % BAYS_PER_ROW
  const row = Math.floor(index / BAYS_PER_ROW)
  return {
    x: FIRST_X + col * PITCH_X,
    y: ROW_Y[row] ?? ROW_Y[ROW_Y.length - 1] + (row - ROW_Y.length + 1) * (BAY.depth + 3),
    width: BAY.width,
    height: BAY.depth,
  }
}

/** One step from `pos` along `headingDeg` (0 = up the map, clockwise), kept inside the floor. */
export function advance(pos, headingDeg, metres) {
  const rad = (headingDeg * Math.PI) / 180
  return {
    x: Math.max(0.5, Math.min(FLOOR.width - 0.5, pos.x + Math.sin(rad) * metres)),
    y: Math.max(0.5, Math.min(FLOOR.height - 0.5, pos.y - Math.cos(rad) * metres)),
  }
}
