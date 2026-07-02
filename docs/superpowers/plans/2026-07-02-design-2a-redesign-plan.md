# Design 2a "Studio" Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recreate the "2a Studio" design handoff (`docs/design_handoff_2a/`) inside the existing React 19 + Vite frontend, replacing the unstyled Predictor and Dashboard views with the exact visual system specified, including a new debounced live-prediction interaction on the Predictor.

**Architecture:** Pure frontend redesign — one new shared Header component, a full rewrite of `Predictor.jsx`/`Predictor.css` (adds debounced auto-predict), and a full rewrite of `Dashboard.jsx`/`Dashboard.css` (drops `recharts` in favor of plain flex/div pill-bars, matching the reference implementation exactly). The FastAPI backend and its JSON contract are untouched.

**Tech Stack:** React 19.2.7, Vite 8.1.1, plain CSS with custom properties (no framework), Google Fonts (Space Grotesk). `recharts` is removed at the end of Task 3 once nothing imports it.

## Global Constraints

- The API contract in `api/predict.py` (`POST /api/predict/regression`, `POST /api/predict/classification`, the `PredictRequest` shape) is **not modified** by any task in this plan — this is a frontend-only redesign.
- No test framework in this project. Verification is `npm run build` (compile check) plus live Playwright MCP checks against a running `vite` dev server (and, for the Predictor's live-prediction behavior, a running `uvicorn` instance of the existing `api/predict.py`).
- Every specific data-derived number shown in the UI (R², F1, feature-importance percentages, stat-card numbers) **must be computed from the real JSON files at render time** — never hardcoded from the reference mockup's illustrative numbers. Confirmed discrepancy to watch for: the mockup shows "Best F1: 0.60" but the real `XGBoost` `f1_macro` (0.5879) rounds to **0.59** — the real, computed 0.59 is correct; do not "fix" it to match the mockup.
- Literal copy strings that are narrative/editorial (the headline "The simple Decision Tree beat XGBoost on regression 🏆", the footnote text, the three bucket sentences, chip titles/subtitles, "Surprise:" note) **are exact and final** — transcribe them verbatim, do not paraphrase.
- Design tokens (colors, spacing, radii, font) are exact, copied verbatim from `docs/design_handoff_2a/README.md`'s "Design Tokens" section and cross-verified against `docs/design_handoff_2a/2a-reference.html`'s actual inline styles (the reference HTML is ground truth where it and the README prose could be read two ways).
- Reference files for this whole plan: `docs/design_handoff_2a/README.md` (prose spec) and `docs/design_handoff_2a/2a-reference.html` (exact working reference — open this in a browser if anything is ambiguous; it renders correctly as-is).

---

### Task 1: Design tokens, global styles, Header component, font loading

**Files:**
- Modify: `frontend/index.html`
- Rewrite: `frontend/src/index.css`
- Rewrite: `frontend/src/App.css`
- Create: `frontend/src/components/Header.jsx`
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- Consumes: nothing from other tasks (this is the foundation).
- Produces: CSS custom properties on `:root` (`--bg`, `--surface`, `--ink`, `--ink-soft`, `--muted`, `--faint`, `--line`, `--track`, `--bar-muted`, `--accent`, `--accent-deep`, `--accent-soft`) that Tasks 2 and 3 reference by name in their own CSS. A `.card` base class (white surface, radius, shadow) that Tasks 2 and 3 compose into their own card variants. `Header.jsx` exported as default, props `{ view: 'predictor' | 'dashboard', onNavigate: (view: string) => void }`.

- [ ] **Step 1: Add Google Fonts link and update the page title**

In `frontend/index.html`, replace the `<head>` contents:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <title>startup/fail</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 2: Rewrite `frontend/src/index.css` with the design tokens**

Replace the entire file content (this deletes Vite's default scaffold styles):

```css
:root {
  --bg: #f6f5f3;
  --surface: #ffffff;
  --ink: #17161a;
  --ink-soft: #44423c;
  --muted: #8b877e;
  --faint: #b9b6ae;
  --line: #e4e1db;
  --track: #f0eeea;
  --bar-muted: #c9c5bd;
  --accent: oklch(0.55 0.2 275);
  --accent-deep: oklch(0.42 0.18 275);
  --accent-soft: oklch(0.55 0.2 275 / .1);

  color-scheme: light;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: 'Space Grotesk', sans-serif;
}

button {
  font-family: inherit;
  cursor: pointer;
}

input,
select {
  font-family: inherit;
}

.card {
  background: var(--surface);
  border-radius: 18px;
  box-shadow: 0 2px 16px rgba(20, 18, 30, .06);
}
```

- [ ] **Step 3: Rewrite `frontend/src/App.css` with the page shell and header styles**

Replace the entire file content (this deletes Vite's default `.counter`/`.hero` scaffold styles):

```css
.page {
  max-width: 980px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 40px;
}

.wordmark {
  font-size: 16px;
  font-weight: 700;
}

.wordmark .fail {
  color: var(--accent);
}

.nav {
  display: flex;
  gap: 8px;
}

.nav-pill {
  padding: 8px 18px;
  border-radius: 999px;
  font-size: 13px;
  border: 1.5px solid var(--line);
  background: transparent;
  color: var(--ink);
  font-weight: 500;
  transition: border-color 150ms ease, background-color 150ms ease;
}

.nav-pill.active {
  background: var(--ink);
  color: #fff;
  font-weight: 600;
  border-color: var(--ink);
}

.nav-pill:hover:not(.active) {
  border-color: #c9c5bd;
}

.view-content {
  padding: 16px 40px 44px;
}

.eyebrow {
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
}

@media (max-width: 700px) {
  .header,
  .view-content {
    padding-left: 20px;
    padding-right: 20px;
  }
}
```

- [ ] **Step 4: Create `frontend/src/components/Header.jsx`**

```jsx
export default function Header({ view, onNavigate }) {
  return (
    <div className="header">
      <div className="wordmark">
        startup<span className="fail">/fail</span>
      </div>
      <div className="nav">
        <button
          type="button"
          className={`nav-pill ${view === 'predictor' ? 'active' : ''}`}
          onClick={() => onNavigate('predictor')}
        >
          Predict
        </button>
        <button
          type="button"
          className={`nav-pill ${view === 'dashboard' ? 'active' : ''}`}
          onClick={() => onNavigate('dashboard')}
        >
          Results
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Wire `Header` into `frontend/src/App.jsx`**

Replace the entire file content:

```jsx
import { useState } from 'react'
import Header from './components/Header'
import Predictor from './components/Predictor'
import Dashboard from './components/Dashboard'

export default function App() {
  const [view, setView] = useState('predictor')

  return (
    <div className="page">
      <Header view={view} onNavigate={setView} />
      <div className="view-content">
        {view === 'predictor' && <Predictor />}
        {view === 'dashboard' && <Dashboard />}
      </div>
    </div>
  )
}
```

- [ ] **Step 6: Build check**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend"
npm run build
```

Expected: completes with no errors (the old `Predictor`/`Dashboard` components still exist unchanged and will still compile against the new `App.jsx`, since this task doesn't touch their internals).

- [ ] **Step 7: Live Playwright check of just the header**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend" && npm run dev &
```

Using the Playwright MCP tools: navigate to `http://localhost:5173`, take a snapshot, and confirm: the wordmark reads "startup/fail" with "/fail" in a visibly different (purple) color; two pill buttons labeled "Predict" and "Results" are visible; "Predict" starts active (dark filled background, white text) since `view` defaults to `'predictor'`; clicking "Results" switches the active pill to "Results" (the content below will still look like the old unstyled Dashboard at this point — that's expected, Task 3 fixes it). Stop the background dev server when done.

- [ ] **Step 8: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add frontend/index.html frontend/src/index.css frontend/src/App.css frontend/src/App.jsx frontend/src/components/Header.jsx
git commit -m "Add design-2a tokens, global styles, and Header component"
git push
```

---

### Task 2: Predictor view — full rewrite with live prediction

**Files:**
- Rewrite: `frontend/src/components/Predictor.jsx`
- Create: `frontend/src/components/Predictor.css`

**Interfaces:**
- Consumes: `.card`, `.eyebrow` base classes and the CSS custom properties from Task 1's `index.css`/`App.css`. Calls the existing, unmodified `POST /api/predict/regression` and `POST /api/predict/classification` endpoints with the same request shape the old `Predictor.jsx` already used (`raised_musd`, `sector`, `start_year`, and the 9 boolean flag fields).
- Produces: nothing consumed by other tasks — `Predictor.jsx`'s default export is already wired into `App.jsx` by Task 1.

- [ ] **Step 1: Rewrite `frontend/src/components/Predictor.jsx`**

```jsx
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
                onClick={() => setTrack('regression')}
              >
                <div className="tile-title">⏱ Years it survives</div>
                <div className="tile-subtitle">regression</div>
              </button>
              <button
                type="button"
                className={`tile predict-type-tile ${track === 'classification' ? 'selected' : ''}`}
                onClick={() => setTrack('classification')}
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
```

Note on the last block: it's a dev-only diagnostic (`import.meta.env.DEV` is `false` in a production build, so this renders nothing in the deployed app) — the design's actual user-facing error UI is the "Couldn't reach the model — try again" sentence inside `LivePredictionPanel`; this extra line just helps you debug in Step 4 below if something doesn't wire up correctly.

- [ ] **Step 2: Create `frontend/src/components/Predictor.css`**

```css
.predictor-h1 {
  font-size: 38px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.05;
  margin: 0 0 28px;
}

.predictor-h1 .accent {
  color: var(--accent);
}

.section-label {
  display: block;
  margin-bottom: 10px;
}

.predictor-grid {
  display: grid;
  grid-template-columns: 1.3fr 1fr;
  gap: 20px;
  align-items: stretch;
}

.form-card {
  padding: 28px 30px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.predict-type-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.tile {
  border-radius: 12px;
  border: 1.5px solid var(--line);
  padding: 12px 14px;
  background: transparent;
  text-align: left;
  transition: border-color 150ms ease, background-color 150ms ease;
}

.tile:hover:not(.selected) {
  border-color: #c9c5bd;
}

.predict-type-tile.selected {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.tile-title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--ink-soft);
}

.predict-type-tile.selected .tile-title {
  font-weight: 700;
  color: var(--accent-deep);
}

.tile-subtitle {
  font-size: 11.5px;
  color: var(--muted);
  margin-top: 2px;
}

.basics-grid {
  display: grid;
  grid-template-columns: 1fr 1.3fr 1fr;
  gap: 10px;
}

.basic-field {
  background: var(--bg);
  border-radius: 12px;
  padding: 12px 14px;
  display: block;
}

.basic-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.basic-field input,
.basic-field select {
  display: block;
  width: 100%;
  border: none;
  background: transparent;
  font-size: 17px;
  font-weight: 700;
  color: var(--ink);
  margin-top: 2px;
  padding: 0;
}

.basic-field select {
  font-size: 15px;
  font-weight: 600;
  margin-top: 4px;
  appearance: none;
  -webkit-appearance: none;
}

.basic-field input:focus,
.basic-field select:focus {
  outline: none;
}

.flags-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}

.flags-count {
  font-size: 12px;
  font-weight: 700;
  color: var(--accent-deep);
}

.flags-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
}

.chip {
  border-radius: 12px;
  border: 1.5px solid var(--line);
  padding: 11px 13px;
  background: transparent;
  text-align: left;
  transition: border-color 150ms ease, background-color 150ms ease;
}

.chip:hover:not(.selected) {
  border-color: #c9c5bd;
}

.chip-title {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--ink-soft);
}

.chip-subtitle {
  font-size: 11px;
  color: var(--muted);
  margin-top: 2px;
  font-weight: 400;
}

.chip.selected {
  background: var(--ink);
  border-color: var(--ink);
}

.chip.selected .chip-title {
  color: #fff;
}

.chip.selected .chip-subtitle {
  color: var(--faint);
}

.live-panel {
  background: var(--accent);
  border-radius: 18px;
  padding: 28px 28px 30px;
  color: #fff;
  display: flex;
  flex-direction: column;
  transition: opacity 150ms ease;
}

.panel-eyebrow {
  color: #fff;
  opacity: .75;
  letter-spacing: 0.1em;
  margin-bottom: 14px;
}

.big-number {
  font-size: 64px;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
}

.big-number-class {
  font-size: 44px;
  text-transform: capitalize;
}

.big-number-unit {
  font-size: 22px;
  font-weight: 500;
  opacity: .8;
}

.panel-sentence {
  font-size: 13.5px;
  line-height: 1.5;
  opacity: .85;
  margin-top: 10px;
}

.panel-divider {
  height: 1px;
  background: rgba(255, 255, 255, .25);
  margin: 22px 0;
}

.segmented-band {
  display: flex;
  height: 36px;
  border-radius: 10px;
  overflow: hidden;
  font-size: 11px;
  font-weight: 600;
}

.segment {
  background: rgba(255, 255, 255, .14);
  opacity: .7;
  display: flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  overflow: hidden;
  transition: background-color 150ms ease;
}

.segment.active {
  background: #fff;
  color: var(--accent-deep);
  opacity: 1;
}

.panel-footnote {
  font-size: 12.5px;
  line-height: 1.6;
  opacity: .75;
  margin-top: 18px;
}

.rerun-button {
  margin-top: auto;
  background: #fff;
  color: var(--accent-deep);
  border: none;
  border-radius: 12px;
  padding: 13px;
  text-align: center;
  font-size: 14px;
  font-weight: 700;
}

.rerun-button:hover {
  opacity: .9;
}

@media (max-width: 900px) {
  .predictor-grid {
    grid-template-columns: 1fr;
  }

  .live-panel {
    order: -1;
  }
}

@media (max-width: 700px) {
  .flags-grid {
    grid-template-columns: 1fr 1fr;
  }
}
```

- [ ] **Step 3: Build check**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend"
npm run build
```

Expected: completes with no errors.

- [ ] **Step 4: Live Playwright verification against a real local API**

Start both servers as background processes:

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\api" && uvicorn predict:app --port 8000 &
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend" && npm run dev &
```

Using the Playwright MCP tools:
1. Navigate to `http://localhost:5173`.
2. Take a snapshot. Confirm: the h1 reads "Describe a startup." / "Watch the prediction move." (second line in accent color); the "What to predict" tiles show "⏱ Years it survives" selected by default; "The basics" shows Raised/Sector/Founded fields with the default values (10, Finance and Insurance, 2015); the 9 flag chips are visible in a 3-column grid; the purple live-prediction panel on the right already shows a real number (not "—") — this proves the "fire one prediction on mount" behavior worked, since you haven't touched the form yet.
3. **Prove the debounce/auto-predict actually works without a submit click:** change the "Raised" field to a different value (e.g. `50`), wait about 1 second (longer than the 400ms debounce), take another snapshot, and confirm the big number in the panel changed to a new value — with no button click in between.
4. Click one of the flag chips (e.g. "Competition") to select it, confirm its visual state flips to the dark "selected" style and the "N of 9 selected" counter updates, and confirm the prediction updates again after ~1s.
5. Click the "Early vs. late failure" tile to switch tracks. Confirm the panel's big number swaps to a class name (e.g. "typical") and the segmented band below now shows three labeled percentages that sum to roughly 100.
6. Click "Re-run prediction →" and confirm it doesn't error.
7. Check for console errors via the Playwright console-messages tool — expect zero.

Stop both background servers when done.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add frontend/src/components/Predictor.jsx frontend/src/components/Predictor.css
git commit -m "Redesign Predictor view: two-pane layout with debounced live prediction"
git push
```

---

### Task 3: Dashboard view — full rewrite, drop recharts

**Files:**
- Rewrite: `frontend/src/components/Dashboard.jsx`
- Create: `frontend/src/components/Dashboard.css`
- Modify: `frontend/package.json` (remove `recharts` dependency)

**Interfaces:**
- Consumes: `.card`, `.eyebrow` base classes and CSS custom properties from Task 1. Reads the existing, unmodified JSON files `frontend/src/data/regression_leaderboard.json`, `frontend/src/data/classification_leaderboard.json`, `frontend/src/data/regression_feature_importance.json`, `frontend/src/data/classification_feature_importance.json`.
- Produces: nothing consumed by other tasks — already wired into `App.jsx` by Task 1.

- [ ] **Step 1: Rewrite `frontend/src/components/Dashboard.jsx`**

```jsx
import { useState } from 'react'
import regressionLeaderboard from '../data/regression_leaderboard.json'
import classificationLeaderboard from '../data/classification_leaderboard.json'
import regressionImportance from '../data/regression_feature_importance.json'
import classificationImportance from '../data/classification_feature_importance.json'
import './Dashboard.css'

const MODEL_DISPLAY_NAMES = {
  DecisionTree: 'Tree',
  XGBoost: 'XGBoost',
  LassoCV: 'Lasso',
  LogisticRegressionCV: 'LogReg',
  Dummy: 'Guessing',
}

const MEDALS = { 1: '🥇', 2: '🥈', 3: '🥉' }

// Curated, human-readable feature lists per view -- NOT simply "top 5 by
// magnitude" (that would surface n_flags/n_flags_sq on the classification
// side, which are engineered meta-features with no intuitive story for a
// non-technical reader). Order matches descending real importance for both
// views (verified against the source JSON).
const REGRESSION_FEATURES = [
  { raw: 'decade_started', label: 'When founded' },
  { raw: 'raised_musd', label: 'Money raised' },
  { raw: 'Sector', label: 'Sector' },
  { raw: 'Giants', label: 'Giants pressure' },
  { raw: 'No Budget', label: 'No Budget pressure' },
]

const CLASSIFICATION_FEATURES = [
  { raw: 'raised_musd', label: 'Money raised' },
  { raw: 'decade_started', label: 'When founded' },
  { raw: 'Sector', label: 'Sector' },
  { raw: 'big_tech_pressure', label: 'Big Tech pressure' },
  { raw: 'No Budget', label: 'No Budget pressure' },
]

// ".38" style: drop the leading zero, keep the sign, collapse near-zero to a
// clean ".00" (the baseline "guessing the average scores 0" story shouldn't
// render as a confusing "-.00").
function formatScore(v) {
  const rounded = v.toFixed(2)
  if (rounded === '0.00' || rounded === '-0.00') return '.00'
  return rounded.startsWith('0.') ? rounded.slice(1) : rounded
}

// "0.38" style for the big stat-card numbers (keeps the leading zero).
function formatStat(v) {
  return v.toFixed(2)
}

function buildLeaderboardRows(data, valueKey) {
  const sorted = [...data].sort((a, b) => b[valueKey] - a[valueKey])
  const maxValue = sorted[0][valueKey]
  return sorted.map((row, i) => {
    const rank = i + 1
    const isTree = row.model === 'DecisionTree'
    const isDummy = row.model === 'Dummy'
    return {
      model: row.model,
      displayName: MODEL_DISPLAY_NAMES[row.model] ?? row.model,
      value: row[valueKey],
      rank,
      isTree,
      isDummy,
      // Proportional to the card's max value; floored so a near-zero/negative
      // baseline still renders as a visible sliver rather than disappearing.
      widthPct: Math.max((row[valueKey] / maxValue) * 100, 1.5),
      // Medals are earned by the Tree model's OWN rank in this round, not by
      // whichever model actually finished 1st/2nd -- confirmed against the
      // reference: Round 1 Tree wins and gets the only medal (🥇); Round 2
      // XGBoost wins with no medal shown, Tree places 2nd and gets 🥈. This
      // is the "Tree" story throughout, not a generic leaderboard.
      medal: isTree ? MEDALS[rank] : null,
    }
  })
}

function buildFeatureRows(features, importanceData) {
  const lookup = Object.fromEntries(importanceData.map((d) => [d.feature, d.importance]))
  const withValues = features.map((f) => ({ ...f, value: lookup[f.raw] ?? 0 }))
  const maxAbs = Math.max(...withValues.map((f) => Math.abs(f.value)))
  return withValues.map((f, i) => ({
    ...f,
    widthPct: (Math.abs(f.value) / maxAbs) * 100,
    rank: i,
  }))
}

function featureBarColor(rank) {
  if (rank === 0) return 'var(--accent)'
  if (rank === 1) return 'oklch(0.55 0.2 275 / .55)'
  return 'var(--bar-muted)'
}

function LeaderboardCard({ title, caption, rows, formatValue }) {
  return (
    <div className="card leaderboard-card">
      <div className="leaderboard-title">{title}</div>
      <div className="leaderboard-caption">{caption}</div>
      <div className="leaderboard-rows">
        {rows.map((row) => (
          <div key={row.model} className="leaderboard-row">
            <span
              className="leaderboard-label"
              style={{
                fontWeight: row.isTree ? 700 : row.isDummy ? 400 : 600,
                color: row.isTree
                  ? 'var(--accent-deep)'
                  : row.isDummy
                    ? 'var(--faint)'
                    : row.rank === 1
                      ? 'var(--ink)'
                      : 'var(--ink-soft)',
              }}
            >
              {row.displayName} {row.medal ?? ''}
            </span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${row.widthPct}%`,
                  background: row.isTree
                    ? 'var(--accent)'
                    : row.isDummy
                      ? 'var(--line)'
                      : row.rank === 1
                        ? 'var(--ink)'
                        : 'var(--bar-muted)',
                }}
              />
            </div>
            <span
              className="leaderboard-value"
              style={{
                fontWeight: row.isTree || row.rank === 1 ? 700 : 400,
                color: row.isDummy ? 'var(--faint)' : row.isTree || row.rank === 1 ? 'var(--ink)' : 'var(--muted)',
              }}
            >
              {formatValue(row.value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function FeatureImportanceCard({ featureView, onToggle }) {
  const rows =
    featureView === 'regression'
      ? buildFeatureRows(REGRESSION_FEATURES, regressionImportance.xgb_permutation)
      : buildFeatureRows(CLASSIFICATION_FEATURES, classificationImportance.xgb_permutation)

  return (
    <div className="card feature-card">
      <div className="feature-card-header">
        <div className="leaderboard-title">What actually matters</div>
        <div className="feature-toggle">
          <button
            type="button"
            className={`toggle-pill ${featureView === 'regression' ? 'active' : ''}`}
            onClick={() => onToggle('regression')}
          >
            Years survived
          </button>
          <button
            type="button"
            className={`toggle-pill ${featureView === 'classification' ? 'active' : ''}`}
            onClick={() => onToggle('classification')}
          >
            Failure timing
          </button>
        </div>
      </div>
      <div className="feature-rows">
        {rows.map((f) => (
          <div key={f.raw} className="feature-row">
            <span className={`feature-label ${f.rank <= 1 ? 'feature-label-strong' : ''}`}>{f.label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${f.widthPct}%`, background: featureBarColor(f.rank) }} />
            </div>
            <span className="feature-annotation">{f.rank === 0 ? 'strongest' : ''}</span>
          </div>
        ))}
      </div>
      <div className="feature-footnote">
        <strong>Surprise:</strong> the era a startup was born in matters more than any reason it failed.
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [featureView, setFeatureView] = useState('regression')

  const regressionRows = buildLeaderboardRows(regressionLeaderboard, 'test_R2')
  const classificationRows = buildLeaderboardRows(classificationLeaderboard, 'f1_macro')

  const bestRegression = regressionRows[0]
  const bestClassification = classificationRows[0]
  const treeClassification = classificationRows.find((r) => r.isTree)

  return (
    <div>
      <div className="stat-row">
        <div className="card headline-card">
          <div className="eyebrow panel-eyebrow-dark">Headline</div>
          <div className="headline-text">The simple Decision Tree beat XGBoost on regression 🏆</div>
        </div>
        <div className="card stat-card">
          <div className="eyebrow section-label">Best R²</div>
          <div className="stat-number">{formatStat(bestRegression.value)}</div>
          <div className="stat-caption">{bestRegression.displayName} · years survived</div>
        </div>
        <div className="card stat-card">
          <div className="eyebrow section-label">Best F1</div>
          <div className="stat-number">{formatStat(bestClassification.value)}</div>
          <div className="stat-caption">
            {bestClassification.displayName} · timing
            {!bestClassification.isTree && ` (Tree: ${formatStat(treeClassification.value)})`}
          </div>
        </div>
      </div>

      <div className="leaderboard-grid">
        <LeaderboardCard
          title="Round 1: predicting years survived"
          caption="R² — guessing the average scores 0"
          rows={regressionRows}
          formatValue={formatScore}
        />
        <LeaderboardCard
          title="Round 2: timing the failure"
          caption="Early / typical / long-run · F1 score"
          rows={classificationRows}
          formatValue={formatScore}
        />
      </div>

      <FeatureImportanceCard featureView={featureView} onToggle={setFeatureView} />
    </div>
  )
}
```

- [ ] **Step 2: Create `frontend/src/components/Dashboard.css`**

```css
.stat-row {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr;
  gap: 12px;
  margin-bottom: 14px;
}

.headline-card {
  background: var(--accent);
  color: #fff;
  padding: 22px 24px;
}

.panel-eyebrow-dark {
  color: #fff;
  opacity: .75;
  margin-bottom: 8px;
}

.headline-text {
  font-size: 19px;
  font-weight: 700;
  line-height: 1.3;
}

.stat-card {
  padding: 22px 24px;
}

.stat-number {
  font-size: 34px;
  font-weight: 700;
}

.stat-caption {
  font-size: 12px;
  color: var(--muted);
}

.leaderboard-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 12px;
}

.leaderboard-card,
.feature-card {
  padding: 24px 26px;
}

.leaderboard-title {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 4px;
}

.leaderboard-caption {
  font-size: 12.5px;
  color: var(--muted);
  margin-bottom: 16px;
}

.leaderboard-rows,
.feature-rows {
  display: flex;
  flex-direction: column;
  gap: 10px;
  font-size: 13px;
}

.leaderboard-row {
  display: grid;
  grid-template-columns: 110px 1fr 38px;
  gap: 10px;
  align-items: center;
}

.feature-row {
  display: grid;
  grid-template-columns: 130px 1fr 80px;
  gap: 12px;
  align-items: center;
}

.bar-track {
  height: 13px;
  background: var(--track);
  border-radius: 999px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 999px;
}

.leaderboard-value {
  text-align: right;
}

.feature-label {
  color: var(--muted);
  font-size: 13px;
}

.feature-label-strong {
  color: var(--ink);
  font-weight: 600;
}

.feature-annotation {
  font-size: 12px;
  color: var(--muted);
}

.feature-card-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 16px;
}

.feature-toggle {
  display: flex;
  gap: 6px;
  font-size: 12px;
}

.toggle-pill {
  padding: 5px 14px;
  border-radius: 999px;
  border: 1.5px solid var(--line);
  background: transparent;
  color: var(--muted);
  transition: border-color 150ms ease, background-color 150ms ease;
}

.toggle-pill.active {
  background: var(--ink);
  color: #fff;
  font-weight: 600;
  border-color: var(--ink);
}

.feature-footnote {
  font-size: 12.5px;
  color: #6d6a63;
  line-height: 1.5;
  margin-top: 16px;
  border-top: 1px solid var(--track);
  padding-top: 12px;
}

.feature-footnote strong {
  color: var(--ink);
}

@media (max-width: 900px) {
  .stat-row,
  .leaderboard-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 3: Remove the now-unused `recharts` dependency**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend"
npm uninstall recharts
```

Expected: `package.json`'s `dependencies` no longer lists `recharts`; `package-lock.json` updates accordingly.

- [ ] **Step 4: Build check**

```bash
npm run build
```

Expected: completes with no errors (confirms no lingering `import ... from 'recharts'` anywhere).

- [ ] **Step 5: Live Playwright verification with real computed numbers**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend" && npm run dev &
```

Using the Playwright MCP tools: navigate to `http://localhost:5173`, click "Results", take a snapshot, and confirm:
1. The headline card reads "The simple Decision Tree beat XGBoost on regression 🏆".
2. "Best R²" shows **0.38** (Tree's real `test_R2` 0.3774 rounds to 0.38) with caption "Tree · years survived".
3. "Best F1" shows **0.59** — NOT 0.60 — with caption "XGBoost · timing (Tree: 0.57)" (verify these are the real rounded values: XGBoost `f1_macro` 0.5879 → 0.59; Tree `f1_macro` 0.5587 → 0.56 — if the snapshot shows 0.56 rather than 0.57 for Tree, that's still correct, the reference mockup's "0.57" was illustrative; the important check is that the number matches `formatStat(0.5587)` exactly, not that it matches the mockup).
4. "Round 1" card shows 4 rows (Tree/XGBoost/Lasso/Guessing) with Tree's row visually distinct (bold, colored) and carrying "🥇"; "Round 2" card shows 4 rows with XGBoost first (no medal) and Tree second carrying "🥈".
5. The "What actually matters" card shows 5 rows for "Years survived" (When founded / Money raised / Sector / Giants pressure / No Budget pressure) with "When founded" annotated "strongest". Click "Failure timing" and confirm the 5 rows swap to the classification set (Money raised / When founded / Sector / Big Tech pressure / No Budget pressure) with "Money raised" now annotated "strongest".
6. Check for console errors via the Playwright console-messages tool — expect zero.

Stop the background dev server when done.

- [ ] **Step 6: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add frontend/src/components/Dashboard.jsx frontend/src/components/Dashboard.css frontend/package.json frontend/package-lock.json
git commit -m "Redesign Dashboard view: stat row, leaderboard cards, feature-importance toggle; drop recharts"
git push
```

---

### Task 4: Final integration — responsive check, lint, verification pass

**Files:** none created or modified — this task only verifies Tasks 1-3's combined output and fixes anything it finds.

**Interfaces:**
- Consumes: the complete app from Tasks 1-3.
- Produces: nothing further downstream — this is the last task in the plan.

- [ ] **Step 1: Lint check**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend"
npm run lint
```

Expected: no errors. If `oxlint` flags something (e.g. an unused import left over from the `Predictor.jsx`/`Dashboard.jsx` rewrites), fix it and re-run until clean.

- [ ] **Step 2: Final build check**

```bash
npm run build
```

Expected: completes with no errors, and check the reported bundle size in the output — it should be smaller than before this plan started (recharts removed), not larger.

- [ ] **Step 3: Full desktop-width Playwright pass (both views)**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\api" && uvicorn predict:app --port 8000 &
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend" && npm run dev &
```

Using the Playwright MCP tools: resize the browser to 1400x1000 (matching the width the reference design was built at), navigate to `http://localhost:5173`, and take a full-page screenshot of the Predictor view, then click "Results" and take a full-page screenshot of the Dashboard view. Compare both against `docs/design_handoff_2a/2a-reference.html` (open that file locally via a simple `python -m http.server` from its directory and navigate Playwright to it, or just re-reference the visual description already confirmed earlier in this project) — confirm the overall layout, spacing, and color treatment match: two-pane predictor with purple live panel on the right, three-part dashboard (stat row, two leaderboard cards, feature-importance card).

- [ ] **Step 4: Narrow-width responsive check**

Using the Playwright MCP tools: resize the browser to 700x1000. Take a snapshot of the Predictor view and confirm the form card and live-prediction panel are now stacked vertically with the **live prediction panel appearing first** (above the form), and the 9 flag chips are in a 2-column grid instead of 3. Switch to the Dashboard view and confirm the stat row and the two leaderboard cards are each stacked into a single column instead of side-by-side.

Stop both background servers when done.

- [ ] **Step 5: Fix anything found in Steps 3-4**

If any visual or behavioral mismatch turns up, fix it directly in the relevant file from Task 1, 2, or 3 (`Header.jsx`/`App.css`, `Predictor.jsx`/`Predictor.css`, or `Dashboard.jsx`/`Dashboard.css`) and re-run the relevant Playwright check from Step 3 or 4 until it passes. Do not skip this — a mismatch here is a real defect, not a nitpick, since the whole point of this plan is pixel-level fidelity to the approved reference.

- [ ] **Step 6: Commit any fixes and push**

If Step 5 required changes:

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add frontend/
git commit -m "Fix design-2a fidelity issues found in final integration pass"
git push
```

If Step 5 required no changes, this task still counts as complete — note in your report that no fix commit was needed.

---

## Self-Review

**Spec coverage:**
- Design tokens (colors, font, spacing/radius) → Task 1 `index.css`. ✓
- Header with wordmark + nav pills → Task 1 `Header.jsx`/`App.css`. ✓
- Predictor: two-pane grid, "What to predict" tiles, "The basics" tile fields, 3×3 flag chip grid → Task 2. ✓
- Predictor: debounced live prediction (400ms, fire-on-mount, dim-while-loading, keep-last-good-on-error, manual re-run) → Task 2 `useEffect`/`runPrediction`/`LivePredictionPanel`. ✓
- Predictor: regression vs. classification panel swap (big number vs. class name, segmented band as baseline distribution vs. real probabilities) → Task 2 `LivePredictionPanel`. ✓
- Dashboard: stat row (headline + 2 stat cards), 2 leaderboard cards with medal/color-tier rules, feature-importance card with toggle → Task 3. ✓
- `recharts` removal → Task 3 Step 3. ✓
- Responsive behavior (900px predictor stack, 700px chip grid, 900px dashboard stack) → Task 2/3 CSS media queries, verified in Task 4. ✓
- Real-computed-numbers-not-mockup-numbers requirement → called out explicitly in Global Constraints and in Task 3 Step 5's verification (explicitly checks for 0.59 not the mockup's 0.60).
- API contract unchanged → no task modifies `api/predict.py`; Task 2 uses the exact existing request shape.

**Placeholder scan:** no TBD/TODO/"add error handling" patterns. The one `import.meta.env.DEV` diagnostic block in Task 2 is real, complete code (not a placeholder) included specifically to aid the implementer's own debugging during Step 4's live verification.

**Type/name consistency:** `Predictor.jsx`'s request body field names (`raised_musd`, `sector`, `start_year`, and the 9 snake_case flag names) match the existing, unmodified `api/predict.py` `PredictRequest` model exactly — verified against the already-read source in this plan's preparation. `Dashboard.jsx`'s JSON field access (`test_R2`, `f1_macro`, `feature`, `importance`, `model`) matches the already-read exact shapes of `frontend/src/data/*.json`. `Header`'s `view`/`onNavigate` prop names match how `App.jsx` (Task 1) calls it.
