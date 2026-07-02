# Startup Failure Modeling: Feature Pipeline + Model Ladders — Design

**Goal:** Build a leakage-safe feature engineering and modeling pipeline over the pooled startup-failure dataset, train a regression ladder (predicting `duration_years`) and a classification ladder (predicting `duration_class`), and produce an honest, business-interpretable comparison of all models against their baselines.

**Context:** Builds on prior work in this project: `data_notes.md` (raw data quirks), `eda_output/combined_clean.csv` (409-row pooled dataset from the six enriched sector files, with `duration_years`, `raised_musd`, `log_raised_musd`, `n_flags` already derived and missing values already handled — see "Corrections to prior work" below for one fix needed), and `eda_output/model_ready.csv` (superseded by this design's pipeline).

## Corrections to prior work

The earlier EDA step's median-by-sector imputation for the 2 residual `raised_musd` nulls (`Driver`, `GeoCities`) was fit on the **full** 409-row dataset. Under this design's stricter no-leakage rule, that must be redone as a train-only transformer (see Leakage Safeguards). The `model_ready.csv` produced in that step is superseded by this pipeline's output.

## Targets

- **`duration_years`** (regression) — continuous, already derived as `end_year - start_year`.
- **`duration_class`** (classification) — binned from `duration_years` using the full dataset's own quartiles (`early` <5yr, `typical` 5–10yr, `long_run` >10yr; observed counts 152/165/92, i.e. 37/40/22%). Binning the target on its own known values to create a stratification variable is **not** leakage — it doesn't fit anything to test-row *feature* data or peek at any model's error. This is distinct from fitting a scaler/imputer/selector on test rows, which we do avoid. This point is called out explicitly in the write-up since it's a common point of confusion.

## Split & validation

One `train_test_split(test_size=0.2, stratify=duration_class, random_state=42)` → ~327 train / 82 test, reused for both the regression and classification framing of the same rows. Model selection and hyperparameter tuning use 5-fold cross-validation within the training split only (`LassoCV`'s internal CV, `validation_curve`, `RandomizedSearchCV`). No third explicit validation split — at n=409, a 60/20/20 split would starve each partition; CV-within-train is the better-conditioned choice.

## Feature engineering

Base features (unchanged from EDA): 9 common binary flags (Giants, No Budget, Competition, Poor Market Fit, Acquisition Stagnation, Monetization Failure, Niche Limits, Execution Flaws, Trend Shifts), `Sector` (one-hot), `log_raised_musd`, `n_flags`.

New engineered features — each checked to confirm it does not use `duration_years` (the target) in its own construction:

| Feature | Type | Logic |
|---|---|---|
| `relative_funding_ratio` | ratio | `raised_musd / sector_median_raised_musd`, sector median computed from the **training split only**, applied to test via `.transform()` |
| `log_raised_musd_sq` | polynomial | tests nonlinear (diminishing/increasing-returns) funding effect beyond the existing 0.36 linear correlation with duration |
| `n_flags_sq` | polynomial | tests whether stacking failure reasons has an accelerating (vs. linear) association |
| `single_cause_failure` | domain logic | 1 if exactly one flag is set (clean-cut vs. compound failure) |
| `big_tech_pressure` | interaction | `Giants AND Competition` — the two most common flags (308, 291 of 409 rows) — captures "crowded market already owned by a giant" as one signal rather than two additive terms |
| `decade_started` | binning | start year (parsed from `Years of Operation`) bucketed into 3 eras, one-hot encoded — captures effects independent of eventual duration |

Start years in the pooled data range 1979–2021, concentrated in 2000–2019 (only 9 rows pre-2000, only 2 rows post-2019). Bin edges are therefore data-grounded rather than literal decades, to avoid a near-empty one-hot column: `pre_2000` (<2000, n=9), `2000s` (2000–2009, n=112), `2010s_plus` (2010+, n=287, absorbing the 2 post-2019 rows).

## Leakage safeguards

- All engineered features above are either fixed, data-independent logic, or (for `relative_funding_ratio`) a statistic fit on train and applied via `.transform()`.
- The 2 residual `raised_musd` nulls are imputed via a custom `TransformerMixin` (`SectorMedianImputer`) that computes sector medians from the training split only, with a global-train-median fallback for any sector not represented.
- All preprocessing (imputation, engineered ratio, scaling, one-hot encoding) is wired through one `ColumnTransformer` inside a `Pipeline`, fit via `.fit(X_train, y_train)` only. This same preprocessing shape is reused unchanged across every rung of both ladders — only the final estimator changes (`Pipeline([("preprocess", ct), ("model", <estimator>)])`) — so the comparison across models is apples-to-apples.
- Binary flags are never scaled (would destroy their "difference in outcome between flagged/not-flagged" coefficient interpretation for the linear rungs). `OneHotEncoder` keeps all categories (no `drop="first"`) — L1-regularized linear models handle the resulting collinearity fine via the penalty, and tree/XGBoost models are unaffected by it either way.
- `raised_dirty` and `raised_imputed` (the data-quality flags from the EDA step) are **metadata, not model features** — they describe provenance of the funding figure, not a startup characteristic, so they are excluded from `X`. They're carried alongside the model-ready dataframe only so a sensitivity check ("do results change if we drop the 61 dirty-format / 2 imputed rows?") is possible later if needed.
- `random_state=42` is used consistently across the split and every stochastic model/search (`DecisionTreeRegressor`, `DecisionTreeClassifier`, `XGBRegressor`, `XGBClassifier`, `RandomizedSearchCV`) for reproducibility.

## Regression ladder (target: `duration_years`)

1. **`DummyRegressor(strategy="mean")`** — floor to beat.
2. **`LassoCV`** (5-fold CV on train) — linear rung **and** the feature-selection deliverable (embedded L1 selection) in one model.
3. **`DecisionTreeRegressor`** — `max_depth` swept 1–15 via `validation_curve` (5-fold CV), plotting mean train R² vs. mean CV R² (±std band). This is both the "note where it overfits" deliverable and the required learning/validation-curve artifact: underfitting at shallow depth (both scores low, small gap), overfitting at deep trees (train R²→1.0, CV score plateaus/drops, gap widens).
4. **`XGBRegressor`** — `RandomizedSearchCV` (5-fold, ~30 iterations) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`; best CV model refit and evaluated once on test.

**Regression leaderboard:** one table, rows = the 4 models above, columns = `train_R², test_R², test_MAE, test_RMSE, R²_gap (train − test)`. Paired with a **predicted-vs-actual plot**: 2×2 small multiples (one scatter per model, `y_test` vs `y_pred`, diagonal reference line).

## Classification ladder (target: `duration_class`)

1. **`DummyClassifier(strategy="most_frequent")`** — floor to beat (harder target than `"stratified"`).
2. **`LogisticRegressionCV(penalty="l1", solver="saga", cv=5, scoring="f1_macro")`** — mirrors `LassoCV`'s role: linear baseline with embedded, CV-selected regularization strength and feature selection.
3. **`DecisionTreeClassifier`** — same `validation_curve`-over-`max_depth` treatment, scored on macro-F1.
4. **`XGBClassifier`** — same `RandomizedSearchCV` treatment (scoring `f1_macro`).

**Classification leaderboard:** one table, rows = the 4 models above, columns = `train_accuracy, test_accuracy, precision_macro, recall_macro, f1_macro, roc_auc_macro_ovr` (`roc_auc_score(..., multi_class="ovr", average="macro")` since there are 3 classes). Paired with a **2×2 confusion-matrix grid** (one per model), so it's visible *which* classes each model actually gets right — e.g. whether a higher-accuracy model earns it via genuinely better minority-class (`long_run`) recall, or just by leaning harder on the majority class.

## Honest-evaluation mechanics

Built into the artifacts, not just stated as intent:

- Every leaderboard row reports **train and test** scores side by side, same metric set for every row — no switching metrics to flatter a particular model.
- The train−test gap column/annotation is present for every model, not only the tree rung.
- Each model gets one written sentence anchored to its ladder's baseline ("+X pts over Dummy"); if that delta is small, it is reported as small, not spun as a win.
- If XGBoost does not beat the linear or tree rung on test (plausible at n=409), that is reported as the headline finding for that ladder, not buried in a footnote.

## Business interpretation (final section of the write-up)

- **What the results mean:** framed honestly as *explaining variation in observed duration among startups that already failed*, not *forecasting a live startup's remaining runway*. This distinction is load-bearing here: the failure-reason flags are hindsight labels assigned after the outcome was already known (per `data_notes.md`), so a model trained on them cannot be repositioned as a forward-looking survival predictor without misrepresenting what it was trained on.
- **Which features matter:** cross-check `LassoCV`/`LogisticRegression` coefficients (direction + magnitude) against XGBoost **permutation importance** (not raw gain/split-count, which is biased toward high-cardinality one-hot columns like `Sector`). Features that agree in direction across a linear and a nonlinear model are called out as the more trustworthy signal; features where the two disagree are flagged as unstable, not hidden.
- **What not to trust the model for:** (1) predicting an *active, non-failed* company's remaining lifespan — the hindsight-label issue above; (2) sectors/classes thin on data — Food/services has only 26 rows total, and `Sector × duration_class` cells can be very sparse; (3) extrapolating past the observed funding range (max observed raise is $3.5B); (4) any causal claim — "Poor Market Fit *causes* shorter duration" is not supported, only "companies labeled this way tended to have shorter observed duration in this sample."

## Outputs

- `feature_pipeline.py` — shared preprocessing (`ColumnTransformer`), both targets, the shared split, `SectorMedianImputer`.
- `regression_ladder.py` — the 4-model regression ladder, leaderboard table, predicted-vs-actual plot, validation curve.
- `classification_ladder.py` — the 4-model classification ladder, leaderboard table, confusion-matrix grid, validation curve.
- `modeling_notes.md` — feature justifications, leakage safeguards, both leaderboards, honest-evaluation commentary, business interpretation section.

## Explicitly out of scope

Nothing deferred beyond this design — the model ladder (previously flagged as a possible follow-up) is included here in full, for both tracks.
