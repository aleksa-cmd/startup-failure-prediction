# Project Summary: Startup Failure Predictor & Dashboard

## What this is

A graduate coursework project (Master's Data Analytics) analyzing a dataset of ~400 failed startups (funding raised, sector, years of operation, and 9 tagged failure reasons like "Competition," "No Budget," "Poor Market Fit"). Two models were trained: a regression predicting how many years a startup operated before failing, and a 3-class classifier predicting whether the failure was "early" (<5yr), "typical" (5–10yr), or "long_run" (>10yr).

The deliverable being redesigned is a small full-stack web app with two views:
1. **Predictor** — a form where a user describes a hypothetical startup and gets a live model prediction.
2. **Dashboard** — charts presenting the model evaluation results (which model won, which features mattered).

## Current state: functionally complete, visually unstyled

The app works end-to-end and is deployed live, but has **zero custom visual design** — it's still running on Vite's default scaffold CSS (unused leftover `.counter`/`.hero` styles from `npm create vite`), so every element renders as plain, unstyled browser-default HTML: default `<select>`, default checkboxes, default buttons, no layout system, no color/spacing system, no typography treatment. This is the thing that most needs design work.

- **Live app:** https://startup-failure-prediction.vercel.app
- **Repo:** https://github.com/aleksa-cmd/startup-failure-prediction (public)

## Predictor view — current content/structure

A single form:
- Radio buttons: "Years of operation (regression)" vs. "Early / typical / long-run failure (classification)"
- Number input: amount raised ($M)
- Dropdown: sector (6 options — Finance and Insurance, Accommodation and Food Services, Health Care, Manufacturing, Retail Trade, Information)
- Number input: founding year
- 9 checkboxes: failure reasons that apply (Giants, No Budget, Competition, Poor Market Fit, Acquisition Stagnation, Monetization Failure, Niche Limits, Execution Flaws, Trend Shifts)
- Submit button → calls a FastAPI backend, displays either a predicted number of years, or a predicted class plus a 3-way probability breakdown (early/typical/long_run %)

## Dashboard view — current content/structure

5 bar charts (via `recharts`), stacked vertically:
1. Regression: test R² by model (4 bars: DecisionTree, XGBoost, LassoCV, Dummy baseline), with a dashed reference line marking the baseline
2. Classification: test F1 by model (same 4-model structure), same baseline line treatment
3. Regression: top ~8 features by Lasso coefficient magnitude (horizontal bars)
4. Regression: top ~8 features by XGBoost permutation importance (horizontal bars)
5. Classification: top ~8 features by XGBoost permutation importance (horizontal bars)

Headline finding these charts should communicate: the simple, interpretable DecisionTree model actually beat the fancier XGBoost model on held-out regression performance, and was competitive on classification — "simpler model won" is the actual editorial point of the dashboard, not incidental.

## Tech stack (constraints for the redesign)

- **Frontend:** React 19 + Vite, plain CSS (no Tailwind/MUI/etc. currently installed — free to add one if it helps, but nothing is there yet). `recharts` is already installed and used for all charts.
- **Backend:** FastAPI (Python), two endpoints (`POST /api/predict/regression`, `POST /api/predict/classification`) — the JSON request/response contract should be treated as fixed unless the redesign has a specific reason to change it (that would require backend changes too).
- **Deployment:** Vercel, auto-deploys on push to the GitHub repo's `master` branch.

## Audience / grading context

This is for a coursework rubric that explicitly scores the web app on subjective/visual polish (a plain-HTML version would score lower than a genuinely designed one), and the app needs to be legible to a **non-technical audience** — the dashboard in particular needs to communicate "which model won and why" without requiring ML background.

## What a redesign should focus on

- A real layout/visual system (spacing, color, typography) — there is currently none to preserve or work around.
- Making the Predictor form feel like a considered product surface, not a raw HTML form (9 checkboxes in particular need better treatment than a bare list).
- Making the Dashboard's "simpler model won" narrative visually obvious, not just five undifferentiated bar charts in a row.
- Nothing about the underlying data, model logic, or API contract needs to change for this — this is a pure front-end/visual pass.
