import { useEffect, useRef, useState } from 'react'
import './Predictor.css'

const SECTORS = [
  'Finance and Insurance',
  'Accommodation and Food Services',
  'Health Care',
  'Manufacturing',
  'Retail Trade',
  'Information',
]

const FLAGS = [
  { field: 'giants', icon: '🏰', title: 'Giants', subtitle: 'big incumbent' },
  { field: 'no_budget', icon: '💸', title: 'No Budget', subtitle: 'out of runway' },
  { field: 'competition', icon: '⚔️', title: 'Competition', subtitle: 'crowded market' },
  { field: 'poor_market_fit', icon: '🎯', title: 'Poor Market Fit', subtitle: 'nobody wants it' },
  { field: 'acquisition_stagnation', icon: '🧊', title: 'Acq. Stagnation', subtitle: 'went nowhere' },
  { field: 'monetization_failure', icon: '🪙', title: 'Monetization', subtitle: "can't charge" },
  { field: 'niche_limits', icon: '🕳', title: 'Niche Limits', subtitle: 'market too small' },
  { field: 'execution_flaws', icon: '🔧', title: 'Execution', subtitle: 'flawed delivery' },
  { field: 'trend_shifts', icon: '🌊', title: 'Trend Shifts', subtitle: 'market moved on' },
]

// Real class distribution from the training data (feature_pipeline.build_dataset(),
// 409 rows: early=152, typical=165, long_run=92) -- shown as the baseline segmented
// band before any classification prediction has come back.
const BASELINE_DISTRIBUTION = { early: 37, typical: 40, long_run: 23 }

const BUCKET_LABEL = { early: 'EARLY <5y', typical: 'TYPICAL 5–10y', long_run: 'LONG >10y' }

const BUCKET_SENTENCE = {
  early: 'An early exit. Most startups like this fold before year 5.',
  typical: 'A typical run. Most startups like this fold between years 5 and 10.',
  long_run: 'A long run. Most startups like this survive past year 10.',
}

function bucketFor(years) {
  if (years <= 5) return 'early'
  if (years <= 10) return 'typical'
  return 'long_run'
}

function initialFormState() {
  const state = { raised_musd: 10, sector: SECTORS[0], start_year: 2015 }
  for (const { field } of FLAGS) state[field] = false
  return state
}

function LivePredictionPanel({ track, result, error, loading, onRerun }) {
  const regressionYears = track === 'regression' && result ? result.predicted_duration_years : null
  const bucket =
    track === 'regression' && regressionYears != null
      ? bucketFor(regressionYears)
      : track === 'classification' && result
        ? result.predicted_class
        : null

  const segments =
    track === 'classification' && result
      ? {
          early: Math.round(result.probabilities.early * 100),
          typical: Math.round(result.probabilities.typical * 100),
          long_run: Math.round(result.probabilities.long_run * 100),
        }
      : BASELINE_DISTRIBUTION

  return (
    <div className="live-panel" style={{ opacity: loading ? 0.6 : 1 }}>
      <div className="eyebrow panel-eyebrow">Live prediction</div>

      {track === 'regression' ? (
        <div className="big-number">
          {regressionYears != null ? regressionYears.toFixed(1) : '—'}
          <span className="big-number-unit"> yrs</span>
        </div>
      ) : (
        <div className="big-number big-number-class">{result ? result.predicted_class : '—'}</div>
      )}

      <div className="panel-sentence">
        {error
          ? "Couldn't reach the model — try again"
          : bucket
            ? BUCKET_SENTENCE[bucket]
            : 'Fill in the form to see a prediction.'}
      </div>

      <div className="panel-divider" />

      <div className="eyebrow panel-eyebrow">Where that falls</div>
      <div className="segmented-band">
        {['early', 'typical', 'long_run'].map((key) => (
          <div
            key={key}
            className={`segment ${bucket === key ? 'active' : ''}`}
            style={{ width: `${segments[key]}%` }}
          >
            {BUCKET_LABEL[key]}
          </div>
        ))}
      </div>

      <div className="panel-footnote">
        Updates as you type — no submit step. Decision Tree, trained on 400 real startup post-mortems (test R² 0.38).
      </div>

      <button type="button" className="rerun-button" onClick={onRerun}>
        Re-run prediction →
      </button>
    </div>
  )
}

export default function Predictor() {
  const [form, setForm] = useState(initialFormState())
  const [track, setTrack] = useState('regression')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef(null)

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  function toggleFlag(field) {
    setForm((prev) => ({ ...prev, [field]: !prev[field] }))
  }

  function handleTrackChange(newTrack) {
    setTrack(newTrack)
    setResult(null)
    setError(null)
  }

  async function runPrediction() {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/predict/${track}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail ? JSON.stringify(body.detail) : `Request failed (${response.status})`)
      }
      setResult(await response.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(runPrediction, 400)
    return () => clearTimeout(debounceRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form, track])

  const flagCount = FLAGS.filter(({ field }) => form[field]).length

  return (
    <div>
      <h1 className="predictor-h1">
        Describe a startup.
        <br />
        <span className="accent">Watch the prediction move.</span>
      </h1>
      <div className="predictor-grid">
        <div className="card form-card">
          <div>
            <div className="eyebrow section-label">What to predict</div>
            <div className="predict-type-grid">
              <button
                type="button"
                className={`tile predict-type-tile ${track === 'regression' ? 'selected' : ''}`}
                onClick={() => handleTrackChange('regression')}
              >
                <div className="tile-title">⏱ Years it survives</div>
                <div className="tile-subtitle">regression</div>
              </button>
              <button
                type="button"
                className={`tile predict-type-tile ${track === 'classification' ? 'selected' : ''}`}
                onClick={() => handleTrackChange('classification')}
              >
                <div className="tile-title">Early vs. late failure</div>
                <div className="tile-subtitle">early / typical / long-run</div>
              </button>
            </div>
          </div>

          <div>
            <div className="eyebrow section-label">The basics</div>
            <div className="basics-grid">
              <label className="basic-field">
                <span className="basic-label">Raised</span>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={form.raised_musd}
                  onChange={(e) => updateField('raised_musd', parseFloat(e.target.value))}
                  required
                />
              </label>
              <label className="basic-field">
                <span className="basic-label">Sector</span>
                <select value={form.sector} onChange={(e) => updateField('sector', e.target.value)}>
                  {SECTORS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="basic-field">
                <span className="basic-label">Founded</span>
                <input
                  type="number"
                  min="1900"
                  max="2029"
                  value={form.start_year}
                  onChange={(e) => updateField('start_year', parseInt(e.target.value, 10))}
                  required
                />
              </label>
            </div>
          </div>

          <div>
            <div className="flags-header">
              <span className="eyebrow">What's going wrong?</span>
              <span className="flags-count">{flagCount} of 9 selected</span>
            </div>
            <div className="flags-grid">
              {FLAGS.map(({ field, icon, title, subtitle }) => (
                <button
                  type="button"
                  key={field}
                  className={`chip ${form[field] ? 'selected' : ''}`}
                  onClick={() => toggleFlag(field)}
                >
                  <div className="chip-title">
                    {icon} {title}
                  </div>
                  <div className="chip-subtitle">{subtitle}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <LivePredictionPanel
          track={track}
          result={result}
          error={error}
          loading={loading}
          onRerun={runPrediction}
        />
      </div>

      {import.meta.env.DEV && error && (
        <p style={{ fontSize: 12, color: '#b00' }}>Dev error detail: {error}</p>
      )}
    </div>
  )
}
