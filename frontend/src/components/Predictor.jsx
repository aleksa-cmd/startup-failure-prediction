import { useState } from 'react'

const SECTORS = [
  'Finance and Insurance',
  'Accommodation and Food Services',
  'Health Care',
  'Manufacturing',
  'Retail Trade',
  'Information',
]

const FLAGS = [
  ['giants', 'Giants (a big incumbent dominates this market)'],
  ['no_budget', 'No Budget (ran out of runway)'],
  ['competition', 'Competition (crowded market)'],
  ['poor_market_fit', 'Poor Market Fit'],
  ['acquisition_stagnation', 'Acquisition Stagnation (post-acquisition, went nowhere)'],
  ['monetization_failure', 'Monetization Failure'],
  ['niche_limits', 'Niche Limits (market too small)'],
  ['execution_flaws', 'Execution Flaws'],
  ['trend_shifts', 'Trend Shifts (market moved on)'],
]

function initialFormState() {
  const state = {
    raised_musd: 10,
    sector: SECTORS[0],
    start_year: 2015,
  }
  for (const [field] of FLAGS) {
    state[field] = false
  }
  return state
}

export default function Predictor() {
  const [form, setForm] = useState(initialFormState())
  const [track, setTrack] = useState('regression')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  function handleTrackChange(newTrack) {
    setTrack(newTrack)
    setResult(null)
    setError(null)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
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

  return (
    <div>
      <h2>Startup Failure Predictor</h2>
      <p>Enter a hypothetical startup's characteristics to see what the model predicts.</p>

      <fieldset>
        <legend>Prediction type</legend>
        <label>
          <input
            type="radio"
            name="track"
            value="regression"
            checked={track === 'regression'}
            onChange={() => handleTrackChange('regression')}
          />
          Years of operation (regression)
        </label>
        <label>
          <input
            type="radio"
            name="track"
            value="classification"
            checked={track === 'classification'}
            onChange={() => handleTrackChange('classification')}
          />
          Early / typical / long-run failure (classification)
        </label>
      </fieldset>

      <form onSubmit={handleSubmit}>
        <label>
          Amount raised ($M)
          <input
            type="number"
            min="0"
            step="0.1"
            value={form.raised_musd}
            onChange={(e) => updateField('raised_musd', parseFloat(e.target.value))}
            required
          />
        </label>

        <label>
          Sector
          <select value={form.sector} onChange={(e) => updateField('sector', e.target.value)}>
            {SECTORS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>

        <label>
          Founding year
          <input
            type="number"
            min="1900"
            max="2030"
            value={form.start_year}
            onChange={(e) => updateField('start_year', parseInt(e.target.value, 10))}
            required
          />
        </label>

        <fieldset>
          <legend>Failure reasons that apply</legend>
          {FLAGS.map(([field, label]) => (
            <label key={field}>
              <input
                type="checkbox"
                checked={form[field]}
                onChange={(e) => updateField(field, e.target.checked)}
              />
              {label}
            </label>
          ))}
        </fieldset>

        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {error && <p role="alert">Error: {error}</p>}

      {result && track === 'regression' && (
        <p>Predicted years of operation: <strong>{result.predicted_duration_years}</strong></p>
      )}

      {result && track === 'classification' && (
        <div>
          <p>Predicted class: <strong>{result.predicted_class}</strong></p>
          <ul>
            {Object.entries(result.probabilities).map(([cls, prob]) => (
              <li key={cls}>{cls}: {(prob * 100).toFixed(1)}%</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
