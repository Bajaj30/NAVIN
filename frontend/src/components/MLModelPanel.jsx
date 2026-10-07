import { useState, useEffect, useCallback, Fragment } from 'react'
import { getMLModel, getTrainingData, getTrainingDataUrl } from '../services/api'

/**
 * MLModelPanel — admin "model card": honest evaluation, per-class comparison vs the
 * Random Forest baseline, confusion matrix, pipeline, live serving latency, collector uploads.
 */

// Validated (dataviz validator, dark surface #141420): categorical slots 1 & 2, dark steps
const SERIES = {
  cnn: { label: '1D-CNN', color: '#3987e5' },
  rf: { label: 'Random Forest', color: '#d95926' },
}
const CLASS_LABEL = { STAIRS: 'Stairs', STATIONARY: 'Stationary', WALKING: 'Walking' }
const pct = (v, digits = 1) => (v == null ? '—' : `${(v * 100).toFixed(digits)}%`)

function StatTile({ label, value, sub, muted }) {
  return (
    <div className="glass-card p-4">
      <p className="text-[11px] text-white/50">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${muted ? 'text-white/40 line-through decoration-white/30' : 'text-white'}`}>{value}</p>
      {sub && <p className="text-[10px] text-white/40 mt-1">{sub}</p>}
    </div>
  )
}

function PerClassF1({ cnn, rf }) {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h4 className="text-sm font-bold text-white/80">Per-class F1 · held-out sessions</h4>
        <div className="flex gap-3 text-[11px] text-white/60">
          {Object.values(SERIES).map(s => (
            <span key={s.label} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm" style={{ background: s.color }} />{s.label}
            </span>
          ))}
        </div>
      </div>
      <div className="space-y-3">
        {Object.keys(CLASS_LABEL).map(cls => (
          <div key={cls}>
            <p className="text-[11px] text-white/60 mb-1">{CLASS_LABEL[cls]}</p>
            {[['cnn', cnn], ['rf', rf]].map(([key, scores]) => scores && (
              <div key={key} className="flex items-center gap-2 mb-[2px]" title={`${SERIES[key].label} · ${CLASS_LABEL[cls]}: F1 ${pct(scores[cls])}`}>
                <div className="flex-1 h-3 bg-white/[0.04] rounded-r">
                  <div className="h-full rounded-r transition-all duration-500"
                       style={{ width: `${(scores[cls] ?? 0) * 100}%`, background: SERIES[key].color }} />
                </div>
                <span className="w-12 text-right text-[11px] text-white/70" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {pct(scores[cls])}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

function ConfusionMatrix({ matrix, classes }) {
  return (
    <div className="glass-card p-4">
      <h4 className="text-sm font-bold text-white/80 mb-1">Confusion matrix · 1D-CNN</h4>
      <p className="text-[11px] text-white/40 mb-3">Each session predicted by a model that never saw it. Shade = share of the true class.</p>
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `5.5rem repeat(${classes.length}, minmax(0, 1fr))` }}>
        <div className="text-[10px] text-white/40 self-end pb-1">true ↓ / pred →</div>
        {classes.map(c => <div key={c} className="text-[10px] text-white/50 text-center pb-1">{CLASS_LABEL[c]}</div>)}
        {matrix.map((row, i) => {
          const total = row.reduce((a, b) => a + b, 0) || 1
          return [
            <div key={`l${i}`} className="text-[11px] text-white/60 flex items-center">{CLASS_LABEL[classes[i]]}</div>,
            ...row.map((count, j) => {
              const share = count / total
              return (
                <div key={`${i}-${j}`}
                     title={`True ${CLASS_LABEL[classes[i]]} → predicted ${CLASS_LABEL[classes[j]]}: ${count} windows (${pct(share)})`}
                     className="h-14 rounded flex flex-col items-center justify-center"
                     style={{ background: `rgba(57,135,229,${0.06 + 0.8 * share})` }}>
                  <span className={`text-sm font-bold ${share > 0.45 ? 'text-white' : 'text-white/70'}`}>{count}</span>
                  <span className={`text-[10px] ${share > 0.45 ? 'text-white/80' : 'text-white/40'}`}>{pct(share, 0)}</span>
                </div>
              )
            }),
          ]
        })}
      </div>
    </div>
  )
}

export default function MLModelPanel() {
  const [info, setInfo] = useState(null)
  const [uploads, setUploads] = useState([])
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const [m, t] = await Promise.all([getMLModel(), getTrainingData()])
      setInfo(m.data)
      setUploads(t.data.files || [])
      setError(null)
    } catch (err) {
      setError(err.message)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 10000)
    return () => clearInterval(id)
  }, [refresh])

  if (error && !info) return <div className="glass-card p-6 text-red-400 text-sm">Could not load model info: {error}</div>
  if (!info) return <div className="glass-card p-6 text-white/40 text-sm">Loading model card…</div>
  if (!info.loaded) return <div className="glass-card p-6 text-amber-400 text-sm">Model not loaded on the server — the app is running on the heuristic fallback.</div>

  const { card, serving_latency: lat } = info
  const ev = card.evaluation
  const data = card.data

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="glass-card p-4 flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-lg font-bold text-white">🤖 {card.name}</h3>
          <p className="text-xs text-white/50">{card.framework} · {card.params.toLocaleString()} parameters · v{card.version}</p>
        </div>
        <span className="flex items-center gap-1.5 text-xs text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400" /> Serving
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatTile label="1D-CNN weighted F1" value={pct(ev.cnn.weighted_f1_mean)}
                  sub={`± ${(ev.cnn.weighted_f1_std * 100).toFixed(1)} over ${ev.seeds} seeds · leave-one-session-out`} />
        <StatTile label="Random Forest baseline" value={pct(ev.random_forest?.weighted_f1)}
                  sub="Handcrafted features · same evaluation" />
        <StatTile label="Random window split" value={pct(ev.random_split_weighted_f1)} muted
                  sub="Not reported — overlapping windows leak into test" />
        <StatTile label="Server inference (p50 / p95)"
                  value={lat.p50_ms == null ? '—' : `${lat.p50_ms} ms`}
                  sub={lat.count ? `p95 ${lat.p95_ms} ms · last ${lat.count} predictions` : 'No predictions yet'} />
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <PerClassF1 cnn={ev.cnn.per_class_f1} rf={ev.random_forest?.per_class_f1} />
        <ConfusionMatrix matrix={ev.cnn.confusion_matrix} classes={card.classes} />
      </div>

      <div className="glass-card p-4">
        <h4 className="text-sm font-bold text-white/80 mb-3">Pipeline</h4>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          {[
            `Phone: ${card.input.window_samples} samples @ ${card.input.rate_hz} Hz × 6 (accel + rotation rate)`,
            'Posture-invariant: |a|, vertical / horizontal accel, turn & tilt rate',
            'z-score',
            'Conv1D 32 → BN → ReLU → Dropout',
            'Conv1D 64 → BN → ReLU → Dropout',
            'Global avg pool → Linear → softmax',
          ].map((step, i, all) => (
            <span key={step} className="flex items-center gap-2">
              <span className="px-2.5 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white/70">{step}</span>
              {i < all.length - 1 && <span className="text-white/30">→</span>}
            </span>
          ))}
        </div>
        <p className="text-[11px] text-white/40 mt-3">
          The whole chain — features, normalisation and CNN — is one ONNX graph, so the server runs exactly what was trained.
          Gravity is estimated per window, which makes the model indifferent to whether the phone is in a hand, a pocket or held in front.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="glass-card p-4">
          <h4 className="text-sm font-bold text-white/80 mb-3">Training data</h4>
          <dl className="grid grid-cols-2 gap-y-1.5 text-xs">
            <dt className="text-white/50">Recording sessions</dt><dd className="text-white/80 text-right">{data.sessions.length}</dd>
            <dt className="text-white/50">Unique minutes @ 10 Hz</dt><dd className="text-white/80 text-right">{data.unique_minutes}</dd>
            <dt className="text-white/50">2 s windows</dt><dd className="text-white/80 text-right">{data.windows}</dd>
            {Object.entries(data.class_windows).map(([c, n]) => (
              <Fragment key={c}>
                <dt className="text-white/50 pl-3">· {CLASS_LABEL[c]}</dt><dd className="text-white/80 text-right">{n}</dd>
              </Fragment>
            ))}
            <dt className="text-white/50">Duplicate rows removed</dt><dd className="text-white/80 text-right">{data.duplicate_rows_dropped}</dd>
          </dl>
        </div>

        <div className="glass-card p-4">
          <h4 className="text-sm font-bold text-white/80 mb-1">Collector uploads</h4>
          <p className="text-[11px] text-white/40 mb-3">
            From <a href="/collector.html" target="_blank" className="text-navin-400 underline">/collector.html</a> · stored on the server until it restarts
          </p>
          {uploads.length === 0 ? (
            <p className="text-xs text-white/30">No uploads yet</p>
          ) : (
            <ul className="space-y-1.5 text-xs max-h-40 overflow-y-auto">
              {uploads.map(f => (
                <li key={f.filename} className="flex justify-between gap-2">
                  <a href={getTrainingDataUrl(f.filename)} className="text-navin-400 underline truncate">{f.filename}</a>
                  <span className="text-white/40 shrink-0">{f.rows.toLocaleString()} rows</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
