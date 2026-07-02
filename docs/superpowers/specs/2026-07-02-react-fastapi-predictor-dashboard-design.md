# D2: React + FastAPI Predictor & Dashboard — Design

**Goal:** Build and deploy a React web app (predictor + dashboard) backed by a FastAPI service serving the project's actual trained models, all in a new public GitHub repo, deployed live on Vercel.

**Context:** This is the first of five Phase-2 deliverables (D1 report, D2 this app, D3 AI-workflow reflection, D4 slides, D5 executive summary), decomposed via brainstorming because the full Phase-2 ask spans multiple independent subsystems. Builds on the modeling work already committed in this repo: `feature_pipeline.py`, `regression_ladder.py`, `classification_ladder.py`, `modeling_notes.md`, and their outputs in `eda_output/` and `modeling_output/`.

## Repo

A new public GitHub repo, `aleksa-cmd/startup-failure-prediction`, created via the GitHub MCP and seeded with this project's existing local git history (currently un-pushed, 10 commits from `e824531` to `87a04cc`). This repo becomes the single home for all five Phase-2 deliverables — D1/D3/D4/D5 land in it as later, separate sub-projects.

## Layout

Existing analysis files (`feature_pipeline.py`, `regression_ladder.py`, `classification_ladder.py`, `modeling_notes.md`, `data_notes.md`, `eda_output/`, `modeling_output/`, the 7 data CSVs, `docs/`) stay at the repo root, unmoved. Two new top-level directories are added:

```
frontend/            React app (Vite), predictor form + dashboard
api/                 Vercel Python serverless functions
  predict.py         FastAPI app: /api/predict/regression, /api/predict/classification
  models/            joblib-serialized trained pipelines (committed, small files)
  requirements.txt
train_and_export_models.py   one-off script: trains + serializes the deployed models
vercel.json          routes the frontend static build + the /api functions
```

No reorganization of already-reviewed analysis code — a mixed monorepo layout (data-science root + `frontend/` + `api/`) is normal and avoids unnecessary churn on working, committed code.

## Targeted refactor: `feature_pipeline.py`

`build_dataset()` currently computes the engineered columns (`n_flags`, `n_flags_sq`, `single_cause_failure`, `big_tech_pressure`, `decade_started`) inline, operating on the whole CSV. Extract this into:

```python
def derive_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds n_flags, n_flags_sq, single_cause_failure, big_tech_pressure,
    decade_started to a DataFrame that already has Sector, raised_musd,
    start_year, and the 9 COMMON_FLAGS columns. Used by both build_dataset()
    (training, whole-CSV path) and the API (serving, single-row path) so
    there is exactly one place this derivation logic lives.
    """
```

`build_dataset()` calls this after parsing `start_year` from `Years of Operation` and before computing `duration_class`. The API calls it on a single-row DataFrame built from the request body. This is the only change to already-approved code from the prior modeling plan, scoped narrowly to eliminating train/serve skew.

## Deployed models

Both tracks use **DecisionTree**, per the modeling write-up's own conclusion (clear winner for regression; the more defensible choice for classification given business-audience interpretability). Unlike the evaluation ladders, the deployed models are **retrained on the full 409-row dataset** (not the 80/20 split) at their already-validated depths — the split was for honestly measuring generalization during model selection; a production predictor should use all available signal.

`train_and_export_models.py`:
1. Loads the full dataset via `feature_pipeline.build_dataset()`.
2. Fits `build_pipeline(DecisionTreeRegressor(max_depth=2, random_state=42))` on the full `X, y_duration_years`.
3. Fits `build_pipeline(DecisionTreeClassifier(max_depth=4, random_state=42))` on the full `X, y_duration_class`.
4. `joblib.dump()`s both fitted pipelines to `api/models/regression_model.joblib` and `api/models/classification_model.joblib`.

Both depths are the actual validated values already selected by `find_best_tree_depth`'s CV sweep in `regression_ladder.py` (depth 2) and the current, `StratifiedKFold`-corrected `classification_ladder.py` (depth 4, confirmed by re-running `find_best_tree_depth` directly against the current codebase) — not re-guessed or re-derived differently.

## API

`api/predict.py`, a FastAPI app (Vercel auto-detects a module-level `app = FastAPI()` and serves it as a Python serverless function):

- `POST /api/predict/regression`
- `POST /api/predict/classification`

**Request body** (shared shape, Pydantic model): `raised_musd: float`, `sector: str` (one of the 6 known sector values), `start_year: int`, plus 9 booleans, one per `COMMON_FLAGS` entry. This is a materially shorter form than the 16 raw model columns — `n_flags`, `n_flags_sq`, `single_cause_failure`, `big_tech_pressure`, and `decade_started` are all derived server-side via `derive_engineered_columns()`, not entered by the user.

**Response:**
- Regression: `{"predicted_duration_years": float}`
- Classification: `{"predicted_class": str, "probabilities": {"early": float, "typical": float, "long_run": float}}`

**Error handling:** Pydantic validates the request shape (400 on malformed input); `sector` validated against the known 6-value set (422 on unknown sector); no other server-side validation needed since the model pipeline itself tolerates the full numeric range.

## Frontend

React app (Vite), two views:

1. **Predictor** — a form matching the API's request shape (funding raised, sector dropdown, founding year, 9 flag checkboxes), a track toggle (regression/classification), calls the matching `/api/predict/*` endpoint via `fetch`, displays the result.
2. **Dashboard** — interactive charts (via `recharts`), not embedded static images, fed by the existing `modeling_output/*.csv` files copied into `frontend/src/data/` at build time: both leaderboards (bar charts comparing test R²/F1 across the ladder, with the Dummy baseline visually marked), and the top feature-importance rankings (Lasso/LogReg coefficients vs. XGBoost permutation importance) for both tracks. Rebuilding these as real React-rendered charts (rather than `<img>` tags) is what actually earns "React scores higher on subjective marks" — static images wouldn't.

## Deployment

One Vercel project linked to the GitHub repo. `vercel.json` routes `/api/*` to the Python functions and everything else to the `frontend/` static build. Deployed and verified live using the Vercel MCP tools available in this session.

## Explicitly out of scope

D1 (Quarto report), D3 (AI-workflow reflection), D4 (slides), D5 (executive summary) — each is its own brainstorm/plan/implementation cycle after this one ships. The README.md that ties all five deliverables together is written once D1 also exists, not as part of this sub-project (a partial README referencing not-yet-built deliverables would be misleading); this sub-project's own commit(s) may include a minimal "how to run the frontend/backend locally" note, but the full top-level README is deferred.
