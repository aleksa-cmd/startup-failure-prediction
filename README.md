# Startup Failure Prediction

Predicting how long a startup survives before failing — and whether the failure is early, typical, or a long slow decline — using data from ~400 real failed startups.

**🔗 Live app:** <https://startup-failure-prediction.vercel.app> — enter a hypothetical startup's details for a live prediction, or browse the model comparison dashboard.

## The five deliverables

| | What | Where |
|---|---|---|
| **D1** | Reproducible analysis report (Quarto → HTML): the full pipeline, model ladders, and findings, computed live | [`docs/D1-report.qmd`](docs/D1-report.qmd) → [`docs/D1-report.html`](docs/D1-report.html) |
| **D2** | The live web app (React + FastAPI) | [`frontend/`](frontend/) + [`api/`](api/) — deployed at the link above |
| **D3** | AI-workflow reflection | [`docs/D3-ai-workflow-reflection.md`](docs/D3-ai-workflow-reflection.md) |
| **D4** | Slide deck (Quarto reveal.js) | [`docs/D4-presentation.qmd`](docs/D4-presentation.qmd) → [`docs/D4-presentation.html`](docs/D4-presentation.html) |
| **D5** | One-page executive summary | [`docs/D5-executive-summary.md`](docs/D5-executive-summary.md) |

## The headline finding

A simple, interpretable Decision Tree beat a tuned XGBoost model on real, held-out test data — for predicting years-of-survival outright, and at a fraction of XGBoost's overfitting on the harder early/typical/long-run classification task. The full reasoning is in D1; the plain-language version is in D5.

## Reproducing this locally

**Requirements:** Python 3.11+ and Node 18+.

```bash
git clone https://github.com/aleksa-cmd/startup-failure-prediction.git
cd startup-failure-prediction
pip install -r requirements.txt
```

**Re-run the analysis pipeline** (each step is a standalone script; `RANDOM_STATE = 42` everywhere, so results are reproducible):

```bash
python feature_pipeline.py       # sanity-checks the leakage-safe feature pipeline
python regression_ladder.py      # Dummy -> LassoCV -> DecisionTree -> XGBoost, duration_years
python classification_ladder.py  # Dummy -> LogisticRegressionCV -> DecisionTree -> XGBoost, duration_class
```

Outputs (leaderboards, charts, coefficients) land in `modeling_output/`.

**Render the D1 report** (this re-runs the two ladders live, so it takes a few minutes):

```bash
quarto render docs/D1-report.qmd
```

**Run the app locally:**

```bash
# terminal 1 — API
cd api && pip install -r requirements.txt && uvicorn predict:app --port 8000

# terminal 2 — frontend
cd frontend && npm install && npm run dev
```

Then open <http://localhost:5173>.

## Project layout

```
feature_pipeline.py, regression_ladder.py,        # analysis pipeline (D1)
classification_ladder.py, train_and_export_models.py
data_notes.md, modeling_notes.md                   # data quirks + full findings
docs/D1..D5                                        # the five deliverables
frontend/, api/                                    # the live app (D2)
eda_output/, modeling_output/                      # generated charts + leaderboards
```
