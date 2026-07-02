# D2: React + FastAPI Predictor & Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a React app (live predictor + interactive dashboard) backed by a FastAPI service serving the project's actual trained DecisionTree models, in a new public GitHub repo, live on Vercel.

**Architecture:** A FastAPI app (`api/predict.py`) loads two joblib-serialized sklearn pipelines (retrained on the full dataset) and exposes two POST endpoints. A React app (Vite) has a Predictor view (form → `fetch` → prediction) and a Dashboard view (recharts, fed by JSON generated once from the existing `modeling_output/*.csv` files). Vercel serves the React static build and the Python function from one project, one repo.

**Tech Stack:** Python 3.14 (existing), pandas 3.0.3, scikit-learn 1.9.0, numpy 2.5.0, joblib 1.5.3, FastAPI + Pydantic v2 (new), Node v24.17.0 + Vite + React + recharts (new), Vercel CLI 54.15.1 (already authenticated as `aleksa-cmd`), `gh` CLI (already authenticated as `aleksa-cmd`, credential.helper=manager).

## Global Constraints

- Project directory: `C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure` — already a git repo, 10 local commits (`e824531`..`c27ed73`), no remote yet.
- No test framework in this project. Verification is "run it, inspect the output" — for the API, use FastAPI's `TestClient` (already available via installed `httpx` 0.28.1, no pytest needed) as a plain script, not a pytest suite. For the frontend, use `npm run build` for a static check and Playwright (MCP tools available in this session) for real browser verification against locally-running dev servers.
- `RANDOM_STATE = 42` (from `feature_pipeline.py`) used for both deployed models, matching the already-validated ladder scripts.
- Deployed model depths are fixed, already-validated values, not re-derived: `DecisionTreeRegressor(max_depth=2)`, `DecisionTreeClassifier(max_depth=4)`.
- Design doc of record: `docs/superpowers/specs/2026-07-02-react-fastapi-predictor-dashboard-design.md`.
- Repo creation uses `mcp__github__create_repository` (name=`startup-failure-prediction`, private=false, no `autoInit` so the repo starts empty); the actual content push uses a normal `git push` (not the MCP's `push_files`, which would flatten the 10 existing commits into a single new one and lose the SDD review-cycle history) — `git`/`gh` are already authenticated as `aleksa-cmd` on this machine.
- Deployment uses the Vercel MCP's `deploy_to_vercel` tool, which deploys "the current project" (the working directory) — must be invoked with the repo root as the working directory, after `vercel.json` exists.
- Do not move or rename any existing file at the repo root (`feature_pipeline.py`, `regression_ladder.py`, `classification_ladder.py`, `modeling_notes.md`, `data_notes.md`, `eda_output/`, `modeling_output/`, the 7 CSVs, `docs/`) — only `feature_pipeline.py`'s internals change (Task 1), everything else is additive (`frontend/`, `api/`, `train_and_export_models.py`, `vercel.json`).

---

### Task 1: Create the GitHub repo and push existing history

**Files:** none (no repo files change in this task — it only creates the remote and pushes what already exists locally)

**Interfaces:**
- Produces: a GitHub repo at `https://github.com/aleksa-cmd/startup-failure-prediction`, `origin` remote configured on the local repo, `master` branch pushed with all 10 existing commits intact.

- [ ] **Step 1: Create the empty GitHub repo via MCP**

Call `mcp__github__create_repository` with:
```json
{
  "name": "startup-failure-prediction",
  "description": "Startup failure duration/outcome modeling: feature pipeline, regression + classification ladders, and a live predictor.",
  "private": false
}
```
(Omit `autoInit` — the repo must start empty so the local push below is a clean fast-forward, not a merge.)

- [ ] **Step 2: Add the remote and push**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git remote add origin https://github.com/aleksa-cmd/startup-failure-prediction.git
git push -u origin master
```

Expected: push succeeds, output ends with something like `* [new branch] master -> master` and `branch 'master' set up to track 'origin/master'`. No merge conflicts (the remote was empty).

- [ ] **Step 3: Verify on GitHub**

```bash
git ls-remote origin
```

Expected: prints the SHA for `refs/heads/master`, matching `git rev-parse HEAD` from the local repo (currently `c27ed73...`, confirm they match).

---

### Task 2: Extract `derive_engineered_columns()` in `feature_pipeline.py`

**Files:**
- Modify: `feature_pipeline.py` (whole file currently 161 lines; this task changes lines 71-107, the body of `build_dataset()`)

**Interfaces:**
- Consumes: nothing new — `COMMON_FLAGS` already defined in this file (line 20).
- Produces: `derive_engineered_columns(df: pd.DataFrame) -> pd.DataFrame`, a new module-level function. `build_dataset()` keeps its exact existing signature and return value (same columns, same values) — this task only changes internal structure, not behavior. Task 4 (`api/predict.py`) imports `derive_engineered_columns` directly from this module.

- [ ] **Step 1: Add `derive_engineered_columns()` and update `build_dataset()`**

In `feature_pipeline.py`, replace the body of `build_dataset()` (currently lines 71-107) with a version that calls a new, separately-defined function for the engineered-column logic. The new function must be added *before* `build_dataset()` in the file (e.g. right after the `FundingFeatureEngineer` class, before `def build_dataset`):

```python
def derive_engineered_columns(df):
    """Adds n_flags, n_flags_sq, single_cause_failure, big_tech_pressure, and
    decade_started to a DataFrame that already has Sector, raised_musd,
    start_year, and the 9 COMMON_FLAGS columns. Used by both build_dataset()
    (training, whole-CSV path) and the FastAPI predictor (serving, single-row
    path) so there is exactly one place this derivation logic lives.
    """
    df = df.copy()
    df["n_flags"] = df[COMMON_FLAGS].sum(axis=1)
    df["n_flags_sq"] = df["n_flags"] ** 2
    df["single_cause_failure"] = (df["n_flags"] == 1).astype(int)
    df["big_tech_pressure"] = ((df["Giants"] == 1) & (df["Competition"] == 1)).astype(int)
    df["decade_started"] = pd.cut(
        df["start_year"], bins=[0, 1999, 2009, 2029],
        labels=["pre_2000", "2000s", "2010s_plus"],
    ).astype(str)
    return df
```

Then replace `build_dataset()`'s body so it calls this helper instead of computing those columns inline:

```python
def build_dataset(path=DATA_PATH):
    df = pd.read_csv(path)

    # Undo the earlier (leaky) global-median imputation for the 2 residual
    # raised_musd nulls (Driver, GeoCities) so FundingFeatureEngineer can
    # re-impute them using train-only sector medians instead. See design doc
    # "Corrections to prior work".
    df.loc[df["raised_imputed"] == 1, "raised_musd"] = np.nan

    start_year = df["Years of Operation"].astype(str).str.extract(r"(\d{4})-\d{4}")[0].astype(int)
    df["start_year"] = start_year

    df = derive_engineered_columns(df)

    # Classification target: quartile bins of the FULL duration_years
    # distribution. This only uses each row's own already-known target value
    # to define the stratification variable -- it is not leakage the way
    # fitting a scaler/imputer/selector on test rows would be. See design doc
    # "Targets" section.
    q1, q3 = df["duration_years"].quantile([0.25, 0.75])
    df["duration_class"] = pd.cut(
        df["duration_years"], bins=[-1, q1, q3, 100],
        labels=["early", "typical", "long_run"],
    ).astype(str)

    return df
```

This reorders two independent computations (n_flags/flags-derived columns vs. start_year/decade_started) but changes no values — neither block depends on the other's output.

- [ ] **Step 2: Verify identical output to before the refactor**

Run: `python "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\feature_pipeline.py"`

Expected (must match exactly, since this is a pure refactor): `Dataset shape: (409, 38)`; `raised_musd nulls before split (expect 2): 2`; `duration_class distribution (full data)` shows `typical=165, early=152, long_run=92`; `Transformed test matrix shape` and `Any NaNs in transformed test matrix: False` unchanged from the last verified run.

- [ ] **Step 3: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add feature_pipeline.py
git commit -m "Extract derive_engineered_columns() for reuse by the FastAPI predictor"
git push
```

---

### Task 3: Train and export the deployed models

**Files:**
- Create: `train_and_export_models.py` (project root)
- Creates on run: `api/models/regression_model.joblib`, `api/models/classification_model.joblib`

**Interfaces:**
- Consumes: `feature_pipeline.FEATURE_COLUMNS`, `feature_pipeline.RANDOM_STATE`, `feature_pipeline.build_dataset`, `feature_pipeline.build_pipeline` (Task 2, unchanged signatures).
- Produces: two joblib files Task 4 loads directly.

- [ ] **Step 1: Write `train_and_export_models.py`**

```python
"""One-off script: retrains the deployed models (DecisionTree, at their
already-validated depths from regression_ladder.py / classification_ladder.py)
on the FULL dataset and joblib-dumps them for the FastAPI predictor. The
train/test split used during evaluation was for honestly measuring
generalization -- a production predictor should use all available signal.
See docs/superpowers/specs/2026-07-02-react-fastapi-predictor-dashboard-design.md
"""
import os

import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from feature_pipeline import FEATURE_COLUMNS, RANDOM_STATE, build_dataset, build_pipeline

FOLDER = r"C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
OUT = os.path.join(FOLDER, "api", "models")
os.makedirs(OUT, exist_ok=True)

REGRESSION_DEPTH = 2       # validated via regression_ladder.py's find_best_tree_depth
CLASSIFICATION_DEPTH = 4   # validated via classification_ladder.py's find_best_tree_depth


def main():
    df = build_dataset()
    X = df[FEATURE_COLUMNS]

    y_reg = df["duration_years"]
    reg_pipeline = build_pipeline(DecisionTreeRegressor(max_depth=REGRESSION_DEPTH, random_state=RANDOM_STATE))
    reg_pipeline.fit(X, y_reg)
    reg_path = os.path.join(OUT, "regression_model.joblib")
    joblib.dump(reg_pipeline, reg_path)
    print(f"Regression model trained on {len(X)} rows, saved to {reg_path}")

    # DecisionTreeClassifier (unlike XGBClassifier) accepts string class
    # labels directly -- no integer-encoding workaround needed here, since
    # this deployment path only ever trains a DecisionTree.
    y_clf = df["duration_class"]
    clf_pipeline = build_pipeline(DecisionTreeClassifier(max_depth=CLASSIFICATION_DEPTH, random_state=RANDOM_STATE))
    clf_pipeline.fit(X, y_clf)
    clf_path = os.path.join(OUT, "classification_model.joblib")
    joblib.dump(clf_pipeline, clf_path)
    print(f"Classification model trained on {len(X)} rows, saved to {clf_path}")
    print(f"Classification classes_: {list(clf_pipeline.classes_)}")

    sample = X.iloc[[0]]
    print(f"\nSanity check - regression prediction on row 0: {reg_pipeline.predict(sample)[0]:.2f} years")
    print(f"Sanity check - classification prediction on row 0: {clf_pipeline.predict(sample)[0]}")
    print(f"Sanity check - classification probabilities on row 0: {clf_pipeline.predict_proba(sample)[0]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify**

Run: `python "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\train_and_export_models.py"`

Expected: two "saved to ..." lines print with paths ending in `api\models\regression_model.joblib` and `api\models\classification_model.joblib`; `Classification classes_:` prints `['early', 'long_run', 'typical']` (alphabetical, since these are the model's own sorted string classes — not the `["early","typical","long_run"]` order used elsewhere in this project); both sanity-check lines print a plausible-looking prediction (a positive float for duration, one of the three class name strings, and 3 probabilities summing to ~1.0). Then confirm the files exist:

```bash
ls -la "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\api\models"
```

Expected: both `.joblib` files listed with nonzero size.

- [ ] **Step 3: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add train_and_export_models.py api/models/
git commit -m "Add train_and_export_models.py; export deployed DecisionTree pipelines"
git push
```

---

### Task 4: FastAPI predictor service

**Files:**
- Create: `api/predict.py`
- Create: `api/requirements.txt`
- Create: `api/verify_predict.py` (throwaway-but-committed verification script using `TestClient`, not a pytest test)

**Interfaces:**
- Consumes: `api/models/regression_model.joblib`, `api/models/classification_model.joblib` (Task 3); `feature_pipeline.FEATURE_COLUMNS`, `feature_pipeline.derive_engineered_columns` (Task 2).
- Produces: `app` (FastAPI instance, module-level in `api/predict.py`) — Vercel's Python runtime auto-detects this as the ASGI app to serve. Task 7's `vercel.json` routes `/api/*` to this file. Task 5 (React) calls `POST /api/predict/regression` and `POST /api/predict/classification` with the JSON shape defined here.

- [ ] **Step 1: Install FastAPI**

```bash
pip install fastapi
```

Expected: installs successfully (uvicorn, pydantic, and httpx are already present in this environment from earlier verification).

- [ ] **Step 2: Write `api/predict.py`**

```python
"""FastAPI predictor service for the startup-failure DecisionTree models.
Deployed as a Vercel Python serverless function -- this module exports
`app`, which Vercel's Python runtime auto-detects as an ASGI app to serve
under /api/predict/*. See
docs/superpowers/specs/2026-07-02-react-fastapi-predictor-dashboard-design.md
"""
import os
import sys
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline import FEATURE_COLUMNS, derive_engineered_columns  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

_regression_model = joblib.load(os.path.join(MODELS_DIR, "regression_model.joblib"))
_classification_model = joblib.load(os.path.join(MODELS_DIR, "classification_model.joblib"))

Sector = Literal[
    "Finance and Insurance",
    "Accommodation and Food Services",
    "Health Care",
    "Manufacturing",
    "Retail Trade",
    "Information",
]

# Maps the API's snake_case boolean field names to the exact COMMON_FLAGS
# column names feature_pipeline.py expects (which contain spaces/mixed case).
FLAG_FIELD_TO_COLUMN = {
    "giants": "Giants",
    "no_budget": "No Budget",
    "competition": "Competition",
    "poor_market_fit": "Poor Market Fit",
    "acquisition_stagnation": "Acquisition Stagnation",
    "monetization_failure": "Monetization Failure",
    "niche_limits": "Niche Limits",
    "execution_flaws": "Execution Flaws",
    "trend_shifts": "Trend Shifts",
}


class PredictRequest(BaseModel):
    raised_musd: float = Field(..., ge=0, description="Amount raised, in millions USD")
    sector: Sector
    start_year: int = Field(..., ge=1900, le=2030)
    giants: bool
    no_budget: bool
    competition: bool
    poor_market_fit: bool
    acquisition_stagnation: bool
    monetization_failure: bool
    niche_limits: bool
    execution_flaws: bool
    trend_shifts: bool


class RegressionResponse(BaseModel):
    predicted_duration_years: float


class ClassificationResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]


def _request_to_row(request: PredictRequest) -> pd.DataFrame:
    row = {
        "raised_musd": request.raised_musd,
        "Sector": request.sector,
        "start_year": request.start_year,
    }
    for field_name, column_name in FLAG_FIELD_TO_COLUMN.items():
        row[column_name] = int(getattr(request, field_name))
    df = pd.DataFrame([row])
    df = derive_engineered_columns(df)
    return df[FEATURE_COLUMNS]


@app.post("/api/predict/regression", response_model=RegressionResponse)
def predict_regression(request: PredictRequest):
    X = _request_to_row(request)
    prediction = _regression_model.predict(X)[0]
    return RegressionResponse(predicted_duration_years=round(float(prediction), 2))


@app.post("/api/predict/classification", response_model=ClassificationResponse)
def predict_classification(request: PredictRequest):
    X = _request_to_row(request)
    predicted_class = _classification_model.predict(X)[0]
    probabilities = _classification_model.predict_proba(X)[0]
    classes = _classification_model.classes_
    return ClassificationResponse(
        predicted_class=predicted_class,
        probabilities={cls: round(float(p), 4) for cls, p in zip(classes, probabilities)},
    )
```

- [ ] **Step 3: Write `api/requirements.txt`**

```
fastapi
pydantic
joblib==1.5.3
pandas==3.0.3
scikit-learn==1.9.0
numpy==2.5.0
```

- [ ] **Step 4: Write `api/verify_predict.py`**

```python
"""Verification script for api/predict.py -- exercises both endpoints via
FastAPI's TestClient (no server process needed). Run directly; not a pytest
suite (this project has no test framework).
"""
from fastapi.testclient import TestClient

from predict import app

client = TestClient(app)

SAMPLE_REQUEST = {
    "raised_musd": 15.0,
    "sector": "Information",
    "start_year": 2015,
    "giants": True,
    "no_budget": False,
    "competition": True,
    "poor_market_fit": False,
    "acquisition_stagnation": False,
    "monetization_failure": True,
    "niche_limits": False,
    "execution_flaws": False,
    "trend_shifts": False,
}

reg_response = client.post("/api/predict/regression", json=SAMPLE_REQUEST)
print(f"Regression status: {reg_response.status_code}")
print(f"Regression body: {reg_response.json()}")
assert reg_response.status_code == 200
assert "predicted_duration_years" in reg_response.json()
assert isinstance(reg_response.json()["predicted_duration_years"], float)

clf_response = client.post("/api/predict/classification", json=SAMPLE_REQUEST)
print(f"\nClassification status: {clf_response.status_code}")
print(f"Classification body: {clf_response.json()}")
assert clf_response.status_code == 200
body = clf_response.json()
assert body["predicted_class"] in ("early", "typical", "long_run")
assert set(body["probabilities"].keys()) == {"early", "typical", "long_run"}
assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01

bad_sector_request = dict(SAMPLE_REQUEST, sector="Not A Real Sector")
bad_response = client.post("/api/predict/regression", json=bad_sector_request)
print(f"\nInvalid sector status (expect 422): {bad_response.status_code}")
assert bad_response.status_code == 422

print("\nAll checks passed.")
```

- [ ] **Step 5: Run the verification script**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\api"
python verify_predict.py
```

Expected: `Regression status: 200`, a `predicted_duration_years` float printed; `Classification status: 200`, a `predicted_class` of one of the three names plus 3 probabilities printed; `Invalid sector status (expect 422): 422`; final line `All checks passed.` with no assertion errors.

- [ ] **Step 6: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add api/predict.py api/requirements.txt api/verify_predict.py
git commit -m "Add FastAPI predictor service (api/predict.py) with TestClient verification"
git push
```

---

### Task 5: React frontend scaffold + Predictor view

**Files:**
- Create: `frontend/` (Vite-scaffolded React app)
- Create: `frontend/src/components/Predictor.jsx`
- Modify: `frontend/src/App.jsx` (Vite's default, replaced)
- Create: `frontend/vite.config.js` (Vite's default, modified to add a dev proxy)

**Interfaces:**
- Consumes: the JSON contract from `api/predict.py` (Task 4) — `POST /api/predict/regression` and `POST /api/predict/classification`, request/response shapes exactly as defined there.
- Produces: `frontend/dist/` (Vite build output) that Task 7's `vercel.json` serves as the static site. `App.jsx` exports a tab switcher between `Predictor` and `Dashboard` (Task 6 adds the `Dashboard` import).

- [ ] **Step 1: Scaffold the Vite React app**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install recharts
```

Expected: `frontend/` created with a standard Vite React template (`package.json`, `src/App.jsx`, `src/main.jsx`, `index.html`, etc.); `node_modules/` installed; `recharts` added to `package.json` dependencies.

- [ ] **Step 2: Add a dev-server proxy so `/api/*` works identically in dev and production**

Replace the contents of `frontend/vite.config.js` (Vite's default template file) with:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

(In production on Vercel, `/api/*` is same-origin with the Python function, so the same relative fetch paths work there without any proxy config.)

- [ ] **Step 3: Write `frontend/src/components/Predictor.jsx`**

```jsx
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
            onChange={() => setTrack('regression')}
          />
          Years of operation (regression)
        </label>
        <label>
          <input
            type="radio"
            name="track"
            value="classification"
            checked={track === 'classification'}
            onChange={() => setTrack('classification')}
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
```

- [ ] **Step 4: Write `frontend/src/App.jsx`**

Replace Vite's default template content entirely with:

```jsx
import { useState } from 'react'
import Predictor from './components/Predictor'

export default function App() {
  const [view, setView] = useState('predictor')

  return (
    <div>
      <nav>
        <button onClick={() => setView('predictor')} disabled={view === 'predictor'}>
          Predictor
        </button>
        <button onClick={() => setView('dashboard')} disabled={view === 'dashboard'}>
          Dashboard
        </button>
      </nav>
      {view === 'predictor' && <Predictor />}
      {view === 'dashboard' && <p>Dashboard coming soon.</p>}
    </div>
  )
}
```

(Task 6 replaces the `<p>Dashboard coming soon.</p>` placeholder with a real `<Dashboard />` import — this task's own build/browser check below only needs to confirm the Predictor view works.)

- [ ] **Step 5: Build check**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend"
npm run build
```

Expected: completes with no errors, prints a `dist/` output summary (e.g. `dist/index.html`, `dist/assets/*.js`).

- [ ] **Step 6: Live browser verification with Playwright**

Start both dev servers in the background, then drive the actual Predictor form through a browser:

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\api" && uvicorn predict:app --port 8000 &
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend" && npm run dev &
```

Then, using the Playwright MCP tools:
1. Navigate to `http://localhost:5173`.
2. Take a snapshot to confirm the Predictor form (amount raised, sector select, founding year, 9 flag checkboxes, radio buttons, submit button) rendered.
3. Fill in a sample value for "Amount raised ($M)" (e.g. `15`), leave the rest at defaults, check the "Competition" checkbox.
4. Click "Predict".
5. Take a snapshot and confirm either a `Predicted years of operation:` line (regression, the default track) with a numeric value appeared, or an `Error:` message — if an error appears, that's a real bug to fix before this task is done, not something to wave off.
6. Switch the radio button to classification, click "Predict" again, and confirm a `Predicted class:` line with one of `early`/`typical`/`long_run` and 3 percentage lines appears.

Stop both background dev servers once verified.

- [ ] **Step 7: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add frontend/
git commit -m "Scaffold React frontend (Vite) with a live Predictor view"
git push
```

---

### Task 6: Dashboard view (recharts, fed by existing modeling output)

**Files:**
- Create: `generate_dashboard_data.py` (project root — one-off script, run once, output committed)
- Creates on run: `frontend/src/data/regression_leaderboard.json`, `frontend/src/data/classification_leaderboard.json`, `frontend/src/data/regression_feature_importance.json`, `frontend/src/data/classification_feature_importance.json`
- Create: `frontend/src/components/Dashboard.jsx`
- Modify: `frontend/src/App.jsx` (swap the placeholder `<p>Dashboard coming soon.</p>` for `<Dashboard />`)

**Interfaces:**
- Consumes: `modeling_output/regression_leaderboard.csv`, `modeling_output/classification_leaderboard.csv`, `modeling_output/regression_lasso_coefficients.csv`, `modeling_output/regression_xgb_permutation_importance.csv`, `modeling_output/classification_logreg_coefficients.csv`, `modeling_output/classification_xgb_permutation_importance.csv` (all already exist from the prior modeling plan).
- Produces: 4 JSON files under `frontend/src/data/` with fixed shapes (documented in Step 1) that `Dashboard.jsx` imports directly.

- [ ] **Step 1: Write `generate_dashboard_data.py`**

```python
"""One-off script: converts the existing modeling_output/*.csv files into
JSON the React dashboard can import directly (no CSV-parsing library needed
in the browser). Run once; output is committed to frontend/src/data/.
See docs/superpowers/specs/2026-07-02-react-fastapi-predictor-dashboard-design.md
"""
import json
import os

import pandas as pd

FOLDER = r"C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
MODELING_OUTPUT = os.path.join(FOLDER, "modeling_output")
OUT = os.path.join(FOLDER, "frontend", "src", "data")
os.makedirs(OUT, exist_ok=True)


def write_json(obj, filename):
    path = os.path.join(OUT, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote {path}")


def leaderboard_to_records(csv_name, round_cols):
    df = pd.read_csv(os.path.join(MODELING_OUTPUT, csv_name))
    for col in round_cols:
        df[col] = df[col].round(4)
    return df.to_dict(orient="records")


def top_n_importance(csv_name, n=8):
    """Regression-style single-column importance CSV: index col + one value column."""
    df = pd.read_csv(os.path.join(MODELING_OUTPUT, csv_name), index_col=0)
    df.columns = ["importance"]
    df["importance"] = df["importance"].round(4)
    top = df.reindex(df["importance"].abs().sort_values(ascending=False).index).head(n)
    return [{"feature": idx, "importance": row["importance"]} for idx, row in top.iterrows()]


write_json(
    leaderboard_to_records("regression_leaderboard.csv", ["train_R2", "test_R2", "test_MAE", "test_RMSE", "R2_gap"]),
    "regression_leaderboard.json",
)
write_json(
    leaderboard_to_records(
        "classification_leaderboard.csv",
        ["train_accuracy", "test_accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro_ovr"],
    ),
    "classification_leaderboard.json",
)
write_json(
    {
        "lasso": top_n_importance("regression_lasso_coefficients.csv"),
        "xgb_permutation": top_n_importance("regression_xgb_permutation_importance.csv"),
    },
    "regression_feature_importance.json",
)
write_json(
    {
        "xgb_permutation": top_n_importance("classification_xgb_permutation_importance.csv"),
    },
    "classification_feature_importance.json",
)
```

Note: `classification_logreg_coefficients.csv` has one column per class (not a single importance column like the others), so it's intentionally left out of `classification_feature_importance.json` here — the dashboard only needs the single-column XGBoost permutation importance for its bar chart. `regression_lasso_coefficients.csv` is a single-column file (one coefficient per feature), so `top_n_importance` applies directly to it too.

- [ ] **Step 2: Run it and verify the JSON**

```bash
python "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\generate_dashboard_data.py"
cat "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend\src\data\regression_leaderboard.json"
```

Expected: 4 "Wrote ..." lines print; the `cat` output is a JSON array of 4 objects (one per model: DecisionTree, XGBoost, LassoCV, Dummy), each with `model`, `train_R2`, `test_R2`, `test_MAE`, `test_RMSE`, `R2_gap` keys and numeric values matching `modeling_output/regression_leaderboard.csv`.

- [ ] **Step 3: Write `frontend/src/components/Dashboard.jsx`**

```jsx
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import regressionLeaderboard from '../data/regression_leaderboard.json'
import classificationLeaderboard from '../data/classification_leaderboard.json'
import regressionImportance from '../data/regression_feature_importance.json'
import classificationImportance from '../data/classification_feature_importance.json'

const DUMMY_REGRESSION_R2 = regressionLeaderboard.find((row) => row.model === 'Dummy').test_R2
const DUMMY_CLASSIFICATION_F1 = classificationLeaderboard.find((row) => row.model === 'Dummy').f1_macro

export default function Dashboard() {
  return (
    <div>
      <h2>Model Findings</h2>

      <section>
        <h3>Regression: test R2 by model (dashed line = Dummy baseline)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regressionLeaderboard}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis />
            <Tooltip />
            <Legend />
            <ReferenceLine y={DUMMY_REGRESSION_R2} stroke="red" strokeDasharray="4 4" label="Dummy baseline" />
            <Bar dataKey="test_R2" fill="#4c72b0" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Classification: test F1-macro by model (dashed line = Dummy baseline)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classificationLeaderboard}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis />
            <Tooltip />
            <Legend />
            <ReferenceLine y={DUMMY_CLASSIFICATION_F1} stroke="red" strokeDasharray="4 4" label="Dummy baseline" />
            <Bar dataKey="f1_macro" fill="#dd8452" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Regression: top features (Lasso coefficient magnitude)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regressionImportance.lasso} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={200} />
            <Tooltip />
            <Bar dataKey="importance" fill="#55a868" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Regression: top features (XGBoost permutation importance)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regressionImportance.xgb_permutation} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={200} />
            <Tooltip />
            <Bar dataKey="importance" fill="#8172b2" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Classification: top features (XGBoost permutation importance)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classificationImportance.xgb_permutation} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={200} />
            <Tooltip />
            <Bar dataKey="importance" fill="#c44e52" />
          </BarChart>
        </ResponsiveContainer>
      </section>
    </div>
  )
}
```

- [ ] **Step 4: Wire it into `App.jsx`**

In `frontend/src/App.jsx`, add the import and swap the placeholder:

```jsx
import { useState } from 'react'
import Predictor from './components/Predictor'
import Dashboard from './components/Dashboard'

export default function App() {
  const [view, setView] = useState('predictor')

  return (
    <div>
      <nav>
        <button onClick={() => setView('predictor')} disabled={view === 'predictor'}>
          Predictor
        </button>
        <button onClick={() => setView('dashboard')} disabled={view === 'dashboard'}>
          Dashboard
        </button>
      </nav>
      {view === 'predictor' && <Predictor />}
      {view === 'dashboard' && <Dashboard />}
    </div>
  )
}
```

- [ ] **Step 5: Build check**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend"
npm run build
```

Expected: completes with no errors (confirms the JSON imports resolve and recharts compiles cleanly).

- [ ] **Step 6: Live browser verification with Playwright**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\frontend" && npm run dev &
```

Using the Playwright MCP tools:
1. Navigate to `http://localhost:5173`.
2. Click the "Dashboard" button.
3. Take a snapshot and confirm all 5 chart sections rendered with visible bars (not blank/empty charts) — the two leaderboard charts should each show 4 bars (one per model) plus a red dashed reference line, and the three feature-importance charts should each show up to 8 horizontal bars with feature-name labels.

Stop the background dev server once verified.

- [ ] **Step 7: Commit**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add generate_dashboard_data.py frontend/
git commit -m "Add interactive Dashboard view (recharts) fed by modeling_output CSVs"
git push
```

---

### Task 7: Vercel config and deployment

**Files:**
- Create: `vercel.json` (project root)

**Interfaces:**
- Consumes: `frontend/` (Task 5+6, built via `npm run build` into `frontend/dist/`), `api/predict.py` + `api/requirements.txt` (Task 4).
- Produces: a live Vercel deployment URL.

- [ ] **Step 1: Write `vercel.json`**

```json
{
  "buildCommand": "cd frontend && npm install && npm run build",
  "outputDirectory": "frontend/dist"
}
```

(Vercel auto-detects `.py` files under `/api` as Python serverless functions — no explicit `functions` block needed for that part.)

- [ ] **Step 2: Commit before deploying**

```bash
cd "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
git add vercel.json
git commit -m "Add vercel.json routing the frontend build and the Python API"
git push
```

- [ ] **Step 3: Deploy via the Vercel MCP**

Call `mcp__plugin_vercel_vercel__deploy_to_vercel` with the working directory set to `C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure` (the repo root, where `vercel.json` lives).

Expected: the tool returns a deployment URL (a `*.vercel.app` address) and a status. If the Python function build fails (e.g. a dependency in `api/requirements.txt` isn't available for Vercel's default Python runtime), read the build error from the tool's response, adjust `api/requirements.txt` version pins to the closest compatible versions, commit, push, and redeploy — this is expected troubleshooting, not a sign the plan is wrong.

- [ ] **Step 4: Verify the live deployment**

Use `mcp__plugin_vercel_vercel__get_deployment` (or the URL returned in Step 3 directly) to confirm the deployment status is `READY`. Then use the Playwright MCP tools against the live URL (not localhost) exactly as in Task 5 Step 6 and Task 6 Step 6: load the site, submit a prediction on both tracks, and view the dashboard, confirming everything that worked locally also works against the real deployed API. If the deployment is behind Vercel's preview-protection screen (a 401/403 on first load), use `mcp__plugin_vercel_vercel__get_access_to_vercel_url` first to get a working shareable URL, then retry the Playwright checks against that URL.

- [ ] **Step 5: Record the live URL**

Once verified, note the live URL for the next task — the D2 sub-project's README section (a future task, once D1 also exists per the design's "Explicitly out of scope" note) will link it, and the design doc's original ask ("README.md that links the live app") is satisfied when that later task lands, not this one.

---

## Self-Review

**Spec coverage:**
- GitHub repo creation via MCP + history-preserving push → Task 1. ✓
- `derive_engineered_columns()` refactor → Task 2. ✓
- Models retrained on full data, joblib-exported (DecisionTree, depths 2 and 4) → Task 3. ✓
- FastAPI app, two endpoints, Pydantic request/response shapes, derived features server-side → Task 4. ✓
- React frontend, Predictor view calling both endpoints → Task 5. ✓
- Interactive (recharts, not static images) Dashboard fed by existing CSVs → Task 6. ✓
- `vercel.json` + deployment + live verification → Task 7. ✓
- README deferred until D1 also exists, per the design's explicit scope note → captured in Task 7 Step 5, not silently dropped. ✓

**Placeholder scan:** no TBD/TODO; the one `<Dashboard />` "placeholder" in Task 5's `App.jsx` is an intentional, explicitly-flagged interim state that Task 6 Step 4 replaces with real code — not an unresolved gap.

**Type/name consistency:** `FEATURE_COLUMNS`, `derive_engineered_columns`, `build_dataset`, `build_pipeline`, `RANDOM_STATE` (Task 2) match their Task 3/4 usage exactly. The API's JSON field names (Task 4's `PredictRequest`) match `Predictor.jsx`'s form state keys (Task 5) exactly (`raised_musd`, `sector`, `start_year`, and the 9 snake_case flag names). The `/api/predict/regression` and `/api/predict/classification` paths are consistent across Task 4 (defined), Task 5 (called via the `track` state value interpolated into the fetch URL), and Task 7 (routed by Vercel's auto-detection of `api/predict.py`).
