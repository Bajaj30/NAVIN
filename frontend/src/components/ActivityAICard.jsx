/**
 * ActivityAICard — live 1D-CNN prediction (served by the backend) with heuristic fallback.
 *
 * Props:
 *   ml: { mode, prediction, latency, error } from useMLClassifier
 *   heuristicActivity: string — what the on-device state machine currently says
 */
const CLASS_STYLE = {
  WALKING: { label: 'Walking', icon: '🚶', bar: 'bg-navin-500', text: 'text-navin-400' },
  STAIRS: { label: 'Stairs', icon: '🪜', bar: 'bg-purple-500', text: 'text-purple-400' },
  STATIONARY: { label: 'Stationary', icon: '⏸️', bar: 'bg-amber-400', text: 'text-amber-400' },
}

const MODE_BADGE = {
  ml: { dot: 'bg-emerald-400 animate-pulse', text: 'text-emerald-400', label: 'CNN LIVE' },
  waiting: { dot: 'bg-white/40 animate-pulse', text: 'text-white/50', label: 'BUFFERING' },
  fallback: { dot: 'bg-amber-400', text: 'text-amber-400', label: 'HEURISTIC FALLBACK' },
  'no-sensors': { dot: 'bg-white/20', text: 'text-white/40', label: 'NO MOTION SENSORS' },
  disabled: { dot: 'bg-white/20', text: 'text-white/40', label: 'SENSORS OFF' },
}

export default function ActivityAICard({ ml, heuristicActivity }) {
  const badge = MODE_BADGE[ml.mode] || MODE_BADGE.waiting
  const p = ml.mode === 'ml' ? ml.prediction : null
  const style = p ? CLASS_STYLE[p.activity] : null

  return (
    <div className="glass-card p-4 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-white/80">🤖 Activity AI</h3>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${badge.dot}`} />
          <span className={`text-[10px] font-bold tracking-wide ${badge.text}`}>{badge.label}</span>
          {ml.mode === 'ml' && ml.latency != null && (
            <span className="text-[10px] text-white/30 font-mono">{ml.latency} ms</span>
          )}
        </div>
      </div>

      {p ? (
        <>
          <div className="flex items-baseline justify-center gap-2 mb-3">
            <span className="text-2xl">{style.icon}</span>
            <span className={`text-2xl font-bold ${style.text}`}>{style.label}</span>
            <span className="text-sm text-white/40 font-mono">{Math.round(p.confidence * 100)}%</span>
          </div>
          <div className="space-y-1.5">
            {Object.entries(CLASS_STYLE).map(([cls, s]) => {
              const prob = p.probabilities[cls] ?? 0
              return (
                <div key={cls} className="flex items-center gap-2">
                  <span className="w-20 text-[10px] text-white/50">{s.label}</span>
                  <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div className={`h-full ${s.bar} rounded-full transition-all duration-300`} style={{ width: `${prob * 100}%` }} />
                  </div>
                  <span className="w-9 text-right text-[10px] font-mono text-white/50">{Math.round(prob * 100)}%</span>
                </div>
              )
            })}
          </div>
        </>
      ) : (
        <p className="text-center py-2 text-white/40 text-sm">
          {ml.mode === 'disabled' && 'Tap “Enable Sensors” above to start'}
          {ml.mode === 'no-sensors' && 'Open on a phone to stream motion sensors'}
          {ml.mode === 'waiting' && 'Collecting 2 s of motion…'}
          {ml.mode === 'fallback' && <>Model unreachable — on-device heuristic says <b className="text-white/70">{heuristicActivity}</b></>}
        </p>
      )}

      <p className="mt-3 pt-2 border-t border-white/5 text-[10px] text-white/30 text-center">
        1D-CNN · ONNX Runtime on the server · 2 s windows @ 10 Hz · falls back to on-device heuristic
      </p>
    </div>
  )
}
