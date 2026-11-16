/** Compass maths — headings are degrees clockwise from north, 0 ≤ h < 360. */

export const normalize = (deg) => ((deg % 360) + 360) % 360

/** Signed shortest turn from a to b, in (−180, 180]. */
export const angleDiff = (a, b) => {
  const d = normalize(b - a)
  return d > 180 ? d - 360 : d
}

/** Exponential smoothing that takes the short way round (359° → 1° passes through 0°, not 180°). */
export const smoothHeading = (current, target, alpha = 0.3) =>
  current === null ? normalize(target) : normalize(current + alpha * angleDiff(current, target))

/**
 * Heading from a DeviceOrientationEvent → { heading, absolute } or null.
 *   iOS Safari:  webkitCompassHeading is already clockwise from magnetic north.
 *   Elsewhere:   alpha grows COUNTER-clockwise, so heading = 360 − alpha. It is a real compass
 *                bearing only on `deviceorientationabsolute` events (event.absolute === true);
 *                otherwise it is relative to wherever the phone pointed when the page loaded.
 */
export function headingFromEvent(event) {
  if (typeof event.webkitCompassHeading === 'number' && !Number.isNaN(event.webkitCompassHeading)) {
    return { heading: normalize(event.webkitCompassHeading), absolute: true }
  }
  if (typeof event.alpha !== 'number') return null
  return { heading: normalize(360 - event.alpha), absolute: event.absolute === true }
}
