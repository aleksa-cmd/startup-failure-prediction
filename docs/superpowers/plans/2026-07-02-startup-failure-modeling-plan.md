# Startup Failure Modeling: Feature Pipeline + Model Ladders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe feature engineering pipeline over the pooled startup-failure dataset, train a regression ladder (Dummy → LassoCV → DecisionTree → XGBoost, predicting `duration_years`) and a classification ladder (Dummy → LogisticRegressionCV → DecisionTree → XGBoost, predicting `duration_class`), and produce honest leaderboards with a business-interpretation write-up.

**Architecture:** One shared module (`feature_pipeline.py`) owns dataset construction, the train/test split, and a `build_pipeline(estimator)` factory that wires a custom leak-safe funding transformer + `ColumnTransformer` in front of any estimator. Two ladder scripts (`regression_ladder.py`, `classification_ladder.py`) import that factory and swap in the four estimators for their track. A final markdown doc synthesizes both ladders' saved outputs into a business write-up.

**Tech Stack:** Python 3.14, pandas 3.0.3, scikit-learn 1.9.0, xgboost 3.3.0, matplotlib, seaborn, scipy.

## Global Constraints

- Project directory: `C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure` — **not a git repository**; there is no commit step in this plan, only "save the file" / "run and inspect output."
- No test framework in use. Verification steps run the actual script and check printed output / saved files against expected shapes, ranges, or presence — not `pytest`.
- Source data: `eda_output/combined_clean.csv` (409 rows). It already contains `duration_years`, `raised_musd`, `raised_dirty`, `raised_imputed`, `log_raised_musd`, `n_flags`, `Sector`, `Years of Operation`, and the flag columns (`Giants`, `No Budget`, `Competition`, `Poor Market Fit`, `Acquisition Stagnation`, `Monetization Failure`, `Niche Limits`, `Execution Flaws`, `Trend Shifts`, `Platform Dependency`, `Toxicity/Trust Issues`, `Regulatory Pressure`, `Overhype`, `High Operational Costs`).
- `RANDOM_STATE = 42` used everywhere a seed is accepted (split, models, searches), per the design doc.
- All new scripts live at the project root (same level as the CSVs), matching the existing flat layout used by `data_notes.md` and `eda_output/`.
- Design doc of record: `docs/superpowers/specs/2026-07-02-startup-failure-modeling-design.md`. Every task below implements a specific section of it; re-read that doc's "Leakage safeguards" and "Corrections to prior work" sections if anything here is ambiguous.

---

### Task 1: Shared feature pipeline module

**Files:**
- Create: `feature_pipeline.py` (project root)

**Interfaces:**
- Produces: `RANDOM_STATE: int`, `COMMON_FLAGS: list[str]` (9 names), `FEATURE_COLUMNS: list[str]` (raw input columns any ladder script selects as `X`), `build_dataset(path=DATA_PATH) -> pd.DataFrame`, `make_split(df, test_size=0.2, random_state=RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]` (train_df, test_df), `build_pipeline(estimator) -> sklearn.pipeline.Pipeline`, class `FundingFeatureEngineer(BaseEstimator, TransformerMixin)`.
- Consumes: nothing from other tasks (this is the foundation).

- [ ] **Step 1: Write `feature_pipeline.py`**

```python
"""Shared, leakage-safe feature pipeline for the startup-failure regression
and classification ladders. See
docs/superpowers/specs/2026-07-02-startup-failure-modeling-design.md
"""
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FOLDER = r"C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
DATA_PATH = os.path.join(FOLDER, "eda_output", "combined_clean.csv")

RANDOM_STATE = 42

COMMON_FLAGS = [
    "Giants", "No Budget", "Competition", "Poor Market Fit", "Acquisition Stagnation",
    "Monetization Failure", "Niche Limits", "Execution Flaws", "Trend Shifts",
]
ENGINEERED_BINARY = ["single_cause_failure", "big_tech_pressure"]
BINARY_COLS = COMMON_FLAGS + ENGINEERED_BINARY
NUMERIC_COLS = ["log_raised_musd", "log_raised_musd_sq", "n_flags", "n_flags_sq", "relative_funding_ratio"]
CATEGORICAL_COLS = ["Sector", "decade_started"]

# raw columns a ladder script selects as X before handing it to a pipeline --
# log_raised_musd / log_raised_musd_sq / relative_funding_ratio are NOT here
# because FundingFeatureEngineer creates them at fit/transform time from
# raised_musd, using train-only statistics.
FEATURE_COLUMNS = (
    ["raised_musd", "Sector", "decade_started", "n_flags", "n_flags_sq"]
    + COMMON_FLAGS
    + ENGINEERED_BINARY
)


class FundingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Fits sector-median funding statistics on training data only, then uses
    them to (a) impute the residual missing raised_musd values and (b) derive
    log / squared-log / sector-relative funding features. Fitting only on the
    split passed to .fit() is what keeps this leakage-safe.
    """

    def __init__(self, funding_col="raised_musd", sector_col="Sector"):
        self.funding_col = funding_col
        self.sector_col = sector_col

    def fit(self, X, y=None):
        self.sector_medians_ = X.groupby(self.sector_col)[self.funding_col].median()
        self.global_median_ = X[self.funding_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        sector_lookup = X[self.sector_col].map(self.sector_medians_).fillna(self.global_median_)

        missing = X[self.funding_col].isna()
        X.loc[missing, self.funding_col] = sector_lookup[missing]

        log_funding = np.log10(X[self.funding_col].replace(0, np.nan))
        X["log_raised_musd"] = log_funding.fillna(0.0)
        X["log_raised_musd_sq"] = X["log_raised_musd"] ** 2
        X["relative_funding_ratio"] = X[self.funding_col] / sector_lookup

        return X


def build_dataset(path=DATA_PATH):
    df = pd.read_csv(path)

    # Undo the earlier (leaky) global-median imputation for the 2 residual
    # raised_musd nulls (Driver, GeoCities) so FundingFeatureEngineer can
    # re-impute them using train-only sector medians instead. See design doc
    # "Corrections to prior work".
    df.loc[df["raised_imputed"] == 1, "raised_musd"] = np.nan

    # Recompute n_flags over just the 9 COMMON_FLAGS. The earlier EDA-step
    # n_flags summed whichever flags existed per source sector file, which
    # mixes in the 4 sector-only / 1 food-only flags this pipeline excludes
    # as predictors -- keep n_flags consistent with the flags actually used.
    df["n_flags"] = df[COMMON_FLAGS].sum(axis=1)
    df["n_flags_sq"] = df["n_flags"] ** 2
    df["single_cause_failure"] = (df["n_flags"] == 1).astype(int)
    df["big_tech_pressure"] = ((df["Giants"] == 1) & (df["Competition"] == 1)).astype(int)

    start_year = df["Years of Operation"].astype(str).str.extract(r"(\d{4})-\d{4}")[0].astype(int)
    df["start_year"] = start_year
    df["decade_started"] = pd.cut(
        df["start_year"], bins=[0, 1999, 2009, 2029],
        labels=["pre_2000", "2000s", "2010s_plus"],
    ).astype(str)

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


def make_split(df, test_size=0.2, random_state=RANDOM_STATE):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["duration_class"], random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def build_pipeline(estimator):
    return Pipeline([
        ("funding_features", FundingFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


if __name__ == "__main__":
    from sklearn.dummy import DummyRegressor

    df = build_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"raised_musd nulls before split (expect 2): {df['raised_musd'].isna().sum()}")
    print(f"\nduration_class distribution (full data):\n{df['duration_class'].value_counts()}")

    train_df, test_df = make_split(df)
    print(f"\nTrain shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train duration_class proportions:\n{train_df['duration_class'].value_counts(normalize=True).round(3)}")
    print(f"Test duration_class proportions:\n{test_df['duration_class'].value_counts(normalize=True).round(3)}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["duration_years"]
    X_test = test_df[FEATURE_COLUMNS]
    print(f"\nraised_musd nulls in X_train: {X_train['raised_musd'].isna().sum()}")
    print(f"raised_musd nulls in X_test: {X_test['raised_musd'].isna().sum()}")

    pipeline = build_pipeline(DummyRegressor())
    pipeline.fit(X_train, y_train)
    engineered_test = pipeline.named_steps["funding_features"].transform(X_test)
    transformed_test = pipeline.named_steps["preprocess"].transform(engineered_test)

    print(f"\nTransformed test matrix shape: {transformed_test.shape}")
    print(f"Any NaNs in transformed test matrix: {bool(np.isnan(transformed_test).any())}")
    print(f"\nFeature names out:\n{list(pipeline.named_steps['preprocess'].get_feature_names_out())}")
```

- [ ] **Step 2: Run it and verify leak-safety + shape**

Run: `python "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\feature_pipeline.py"`

Expected:
- `Dataset shape: (409, ...)`.
- `raised_musd nulls before split (expect 2): 2`.
- `duration_class distribution (full data)` shows `typical=165, early=152, long_run=92`.
- Train/test shapes sum to 409 (e.g. `(327, ...)` and `(82, ...)`), and the train/test `duration_class` proportions are both close to `early≈0.37, typical≈0.40, long_run≈0.22` (stratification working).
- `raised_musd nulls in X_train` and `X_test` sum to 2 across the two (whichever split Driver/GeoCities landed in), confirming the 2 nulls survived the split un-imputed.
- `Transformed test matrix shape` has 409-row-compatible column count (9 flags + 2 engineered binary + 5 numeric + one-hot columns for 6 sectors + 3 decades ⇒ roughly 9+2+5+6+3=25 columns, exact count printed).
- `Any NaNs in transformed test matrix: False` — proves `FundingFeatureEngineer`, fit on train only, still successfully imputes the test row(s) that had a null.

---

### Task 2: Regression ladder

**Files:**
- Create: `regression_ladder.py` (project root)
- Creates on run: `modeling_output/regression_tree_validation_curve.png`, `modeling_output/regression_leaderboard.csv`, `modeling_output/regression_predicted_vs_actual.png`, `modeling_output/regression_lasso_coefficients.csv`, `modeling_output/regression_xgb_permutation_importance.csv`

**Interfaces:**
- Consumes: `feature_pipeline.build_dataset`, `feature_pipeline.make_split`, `feature_pipeline.build_pipeline`, `feature_pipeline.FEATURE_COLUMNS`, `feature_pipeline.RANDOM_STATE` (Task 1).
- Produces: nothing consumed by Task 3 directly; Task 4 reads the CSVs/PNGs this task saves to `modeling_output/`.

- [ ] **Step 1: Write `regression_ladder.py`**

```python
"""Regression ladder for duration_years: Dummy -> LassoCV -> DecisionTree ->
XGBoost. See docs/superpowers/specs/2026-07-02-startup-failure-modeling-design.md
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from feature_pipeline import FEATURE_COLUMNS, RANDOM_STATE, build_dataset, build_pipeline, make_split

FOLDER = r"C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
OUT = os.path.join(FOLDER, "modeling_output")
os.makedirs(OUT, exist_ok=True)


def get_xy(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df["duration_years"].copy()
    return X, y


def find_best_tree_depth(X_train, y_train):
    depths = list(range(1, 16))
    train_scores, cv_scores = validation_curve(
        build_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
        X_train, y_train,
        param_name="model__max_depth", param_range=depths,
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2",
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    cv_mean, cv_std = cv_scores.mean(axis=1), cv_scores.std(axis=1)
    best_idx = int(np.argmax(cv_mean))
    best_depth = depths[best_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(depths, train_mean, "o-", color="steelblue", label="Train R2")
    plt.fill_between(depths, train_mean - train_std, train_mean + train_std, alpha=0.15, color="steelblue")
    plt.plot(depths, cv_mean, "o-", color="darkorange", label="CV R2")
    plt.fill_between(depths, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color="darkorange")
    plt.axvline(best_depth, color="gray", linestyle="--", label=f"best depth={best_depth}")
    plt.xlabel("max_depth")
    plt.ylabel("R2")
    plt.title("DecisionTreeRegressor: validation curve (overfitting demo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "regression_tree_validation_curve.png"), dpi=150)
    plt.close()

    print(f"DecisionTreeRegressor best max_depth by CV R2: {best_depth} "
          f"(train R2={train_mean[best_idx]:.3f}, CV R2={cv_mean[best_idx]:.3f})")
    return best_depth


def tune_xgboost(X_train, y_train):
    param_dist = {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(2, 6),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    }
    search = RandomizedSearchCV(
        build_pipeline(XGBRegressor(random_state=RANDOM_STATE)),
        param_distributions=param_dist, n_iter=30, cv=5, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"XGBRegressor best params: {search.best_params_}")
    print(f"XGBRegressor best CV R2: {search.best_score_:.3f}")
    return search.best_estimator_


def evaluate(pipeline, X_train, y_train, X_test, y_test):
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    metrics = {
        "train_R2": r2_score(y_train, train_pred),
        "test_R2": r2_score(y_test, test_pred),
        "test_MAE": mean_absolute_error(y_test, test_pred),
        "test_RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
    }
    return metrics, test_pred


def main():
    df = build_dataset()
    train_df, test_df = make_split(df)
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)

    best_depth = find_best_tree_depth(X_train, y_train)
    best_xgb_pipeline = tune_xgboost(X_train, y_train)

    models = {
        "Dummy": build_pipeline(DummyRegressor(strategy="mean")),
        "LassoCV": build_pipeline(LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000)),
        "DecisionTree": build_pipeline(DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE)),
        "XGBoost": best_xgb_pipeline,
    }

    rows, predictions = [], {}
    for name, pipeline in models.items():
        if name != "XGBoost":  # XGBoost's best_estimator_ is already fit
            pipeline.fit(X_train, y_train)
        metrics, test_pred = evaluate(pipeline, X_train, y_train, X_test, y_test)
        metrics["R2_gap"] = metrics["train_R2"] - metrics["test_R2"]
        metrics["model"] = name
        rows.append(metrics)
        predictions[name] = test_pred

    leaderboard = pd.DataFrame(rows)[["model", "train_R2", "test_R2", "test_MAE", "test_RMSE", "R2_gap"]]
    leaderboard = leaderboard.sort_values("test_R2", ascending=False)
    leaderboard.to_csv(os.path.join(OUT, "regression_leaderboard.csv"), index=False)
    print("\n=== Regression Leaderboard ===")
    print(leaderboard.to_string(index=False))

    baseline_r2 = leaderboard.loc[leaderboard["model"] == "Dummy", "test_R2"].iloc[0]
    for _, row in leaderboard.iterrows():
        if row["model"] == "Dummy":
            continue
        delta = row["test_R2"] - baseline_r2
        note = "small, marginal" if abs(delta) < 0.05 else "notable"
        print(f"{row['model']}: {delta:+.3f} test R2 over Dummy baseline ({note})")

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (name, pred) in zip(axes.flatten(), predictions.items()):
        ax.scatter(y_test, pred, alpha=0.6, color="steelblue")
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        ax.plot(lims, lims, "r--", label="y = x")
        ax.set_xlabel("Actual duration_years")
        ax.set_ylabel("Predicted duration_years")
        ax.set_title(name)
        ax.legend()
    plt.suptitle("Predicted vs Actual (test set)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "regression_predicted_vs_actual.png"), dpi=150)
    plt.close()

    lasso_pipeline = models["LassoCV"]
    feature_names = lasso_pipeline.named_steps["preprocess"].get_feature_names_out()
    lasso_coefs = pd.Series(
        lasso_pipeline.named_steps["model"].coef_, index=feature_names
    ).sort_values(key=abs, ascending=False)
    lasso_coefs.to_csv(os.path.join(OUT, "regression_lasso_coefficients.csv"))
    print("\nTop LassoCV coefficients (regression):")
    print(lasso_coefs.head(10))

    perm = permutation_importance(
        models["XGBoost"], X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="r2"
    )
    perm_importance = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    perm_importance.to_csv(os.path.join(OUT, "regression_xgb_permutation_importance.csv"))
    print("\nTop XGBoost permutation importances (regression, raw input columns):")
    print(perm_importance.head(10))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify the leaderboard is sane**

Run: `python "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\regression_ladder.py"`

Expected:
- `DecisionTreeRegressor best max_depth by CV R2:` prints some value between 1 and 15 (commonly 2–5 for a 327-row training set).
- `XGBRegressor best params:` and `best CV R2:` print without error.
- `=== Regression Leaderboard ===` prints a 4-row table with columns `model, train_R2, test_R2, test_MAE, test_RMSE, R2_gap`; the `Dummy` row's `test_R2` is at or near `0.0` (a mean-predicting baseline has R2≈0 on its own held-out set by construction) and its `R2_gap` is at or near `0.0`.
- For every non-Dummy model, `test_R2` should differ from `train_R2` (some positive `R2_gap`), and the printed `+X.XXX test R2 over Dummy baseline` line appears once per non-Dummy model.
- `modeling_output/regression_tree_validation_curve.png`, `regression_leaderboard.csv`, `regression_predicted_vs_actual.png`, `regression_lasso_coefficients.csv`, `regression_xgb_permutation_importance.csv` all exist after the run (`ls modeling_output`).
- Open `regression_predicted_vs_actual.png` and `regression_tree_validation_curve.png` (e.g. via the Read tool) and confirm each panel/curve renders with labeled axes and a legend, no blank plots.

---

### Task 3: Classification ladder

**Files:**
- Create: `classification_ladder.py` (project root)
- Creates on run: `modeling_output/classification_tree_validation_curve.png`, `modeling_output/classification_leaderboard.csv`, `modeling_output/classification_confusion_matrices.png`, `modeling_output/classification_logreg_coefficients.csv`, `modeling_output/classification_xgb_permutation_importance.csv`

**Interfaces:**
- Consumes: `feature_pipeline.build_dataset`, `feature_pipeline.make_split`, `feature_pipeline.build_pipeline`, `feature_pipeline.FEATURE_COLUMNS`, `feature_pipeline.RANDOM_STATE` (Task 1).
- Produces: nothing consumed by Task 2; Task 4 reads the CSVs/PNGs this task saves to `modeling_output/`.

- [ ] **Step 1: Write `classification_ladder.py`**

```python
"""Classification ladder for duration_class: Dummy -> LogisticRegressionCV ->
DecisionTree -> XGBoost. See
docs/superpowers/specs/2026-07-02-startup-failure-modeling-design.md
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from feature_pipeline import FEATURE_COLUMNS, RANDOM_STATE, build_dataset, build_pipeline, make_split

FOLDER = r"C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure"
OUT = os.path.join(FOLDER, "modeling_output")
os.makedirs(OUT, exist_ok=True)

CLASSES = ["early", "typical", "long_run"]


def get_xy(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df["duration_class"].copy()
    return X, y


def find_best_tree_depth(X_train, y_train):
    depths = list(range(1, 16))
    train_scores, cv_scores = validation_curve(
        build_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        X_train, y_train,
        param_name="model__max_depth", param_range=depths,
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="f1_macro",
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    cv_mean, cv_std = cv_scores.mean(axis=1), cv_scores.std(axis=1)
    best_idx = int(np.argmax(cv_mean))
    best_depth = depths[best_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(depths, train_mean, "o-", color="steelblue", label="Train F1-macro")
    plt.fill_between(depths, train_mean - train_std, train_mean + train_std, alpha=0.15, color="steelblue")
    plt.plot(depths, cv_mean, "o-", color="darkorange", label="CV F1-macro")
    plt.fill_between(depths, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color="darkorange")
    plt.axvline(best_depth, color="gray", linestyle="--", label=f"best depth={best_depth}")
    plt.xlabel("max_depth")
    plt.ylabel("F1-macro")
    plt.title("DecisionTreeClassifier: validation curve (overfitting demo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "classification_tree_validation_curve.png"), dpi=150)
    plt.close()

    print(f"DecisionTreeClassifier best max_depth by CV F1-macro: {best_depth} "
          f"(train={train_mean[best_idx]:.3f}, CV={cv_mean[best_idx]:.3f})")
    return best_depth


def tune_xgboost(X_train, y_train):
    param_dist = {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(2, 6),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    }
    search = RandomizedSearchCV(
        build_pipeline(XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss")),
        param_distributions=param_dist, n_iter=30, cv=5, scoring="f1_macro",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"XGBClassifier best params: {search.best_params_}")
    print(f"XGBClassifier best CV F1-macro: {search.best_score_:.3f}")
    return search.best_estimator_


def evaluate(pipeline, X_train, y_train, X_test, y_test):
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision_macro": precision_score(y_test, test_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, test_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "roc_auc_macro_ovr": roc_auc_score(y_test, test_proba, multi_class="ovr", average="macro", labels=classes),
    }
    return metrics, test_pred


def main():
    df = build_dataset()
    train_df, test_df = make_split(df)
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)

    best_depth = find_best_tree_depth(X_train, y_train)
    best_xgb_pipeline = tune_xgboost(X_train, y_train)

    models = {
        "Dummy": build_pipeline(DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        "LogisticRegressionCV": build_pipeline(
            LogisticRegressionCV(
                penalty="l1", solver="saga", cv=5, scoring="f1_macro",
                max_iter=5000, random_state=RANDOM_STATE,
            )
        ),
        "DecisionTree": build_pipeline(DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)),
        "XGBoost": best_xgb_pipeline,
    }

    rows, predictions = [], {}
    for name, pipeline in models.items():
        if name != "XGBoost":
            pipeline.fit(X_train, y_train)
        metrics, test_pred = evaluate(pipeline, X_train, y_train, X_test, y_test)
        metrics["model"] = name
        rows.append(metrics)
        predictions[name] = test_pred

    leaderboard = pd.DataFrame(rows)[
        ["model", "train_accuracy", "test_accuracy", "precision_macro",
         "recall_macro", "f1_macro", "roc_auc_macro_ovr"]
    ].sort_values("f1_macro", ascending=False)
    leaderboard.to_csv(os.path.join(OUT, "classification_leaderboard.csv"), index=False)
    print("\n=== Classification Leaderboard ===")
    print(leaderboard.to_string(index=False))

    baseline_f1 = leaderboard.loc[leaderboard["model"] == "Dummy", "f1_macro"].iloc[0]
    for _, row in leaderboard.iterrows():
        if row["model"] == "Dummy":
            continue
        delta = row["f1_macro"] - baseline_f1
        note = "small, marginal" if abs(delta) < 0.05 else "notable"
        print(f"{row['model']}: {delta:+.3f} F1-macro over Dummy baseline ({note})")

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (name, pred) in zip(axes.flatten(), predictions.items()):
        cm = confusion_matrix(y_test, pred, labels=CLASSES)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    plt.suptitle("Confusion Matrices (test set)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "classification_confusion_matrices.png"), dpi=150)
    plt.close()

    logreg_pipeline = models["LogisticRegressionCV"]
    feature_names = logreg_pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = logreg_pipeline.named_steps["model"].coef_  # shape (n_classes, n_features)
    coef_df = pd.DataFrame(
        coefs.T, index=feature_names, columns=logreg_pipeline.named_steps["model"].classes_
    )
    coef_df.to_csv(os.path.join(OUT, "classification_logreg_coefficients.csv"))
    print("\nLogisticRegressionCV coefficients saved to classification_logreg_coefficients.csv")

    perm = permutation_importance(
        models["XGBoost"], X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="f1_macro"
    )
    perm_importance = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    perm_importance.to_csv(os.path.join(OUT, "classification_xgb_permutation_importance.csv"))
    print("\nTop XGBoost permutation importances (classification, raw input columns):")
    print(perm_importance.head(10))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify the leaderboard is sane**

Run: `python "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\classification_ladder.py"`

Expected:
- `DecisionTreeClassifier best max_depth by CV F1-macro:` prints a value between 1 and 15.
- `XGBClassifier best params:` and `best CV F1-macro:` print without error.
- `=== Classification Leaderboard ===` prints a 4-row table with columns `model, train_accuracy, test_accuracy, precision_macro, recall_macro, f1_macro, roc_auc_macro_ovr`. The `Dummy` row's `roc_auc_macro_ovr` should be close to `0.5` (chance level for a majority-class-only predictor).
- The `+X.XXX F1-macro over Dummy baseline` line prints once per non-Dummy model.
- `modeling_output/classification_tree_validation_curve.png`, `classification_leaderboard.csv`, `classification_confusion_matrices.png`, `classification_logreg_coefficients.csv`, `classification_xgb_permutation_importance.csv` all exist after the run.
- Open `classification_confusion_matrices.png` (via the Read tool) and confirm all 4 panels render with the 3×3 grid and `early/typical/long_run` axis labels — check in particular whether the Dummy panel shows all predictions landing in a single column (expected for a most-frequent-class baseline).

---

### Task 4: Modeling write-up

**Files:**
- Create: `modeling_notes.md` (project root)

**Interfaces:**
- Consumes: `modeling_output/regression_leaderboard.csv`, `modeling_output/regression_lasso_coefficients.csv`, `modeling_output/regression_xgb_permutation_importance.csv`, `modeling_output/classification_leaderboard.csv`, `modeling_output/classification_logreg_coefficients.csv`, `modeling_output/classification_xgb_permutation_importance.csv` (Tasks 2 & 3 outputs — read these files' actual contents before writing, do not fabricate numbers).
- Produces: nothing consumed elsewhere — this is the terminal deliverable.

- [ ] **Step 1: Read every CSV produced by Tasks 2 and 3**

Run these and keep the actual printed values at hand for Step 2 — every number in the write-up must come from here, not be invented:

```bash
cat "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\modeling_output\regression_leaderboard.csv"
cat "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\modeling_output\regression_lasso_coefficients.csv"
cat "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\modeling_output\regression_xgb_permutation_importance.csv"
cat "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\modeling_output\classification_leaderboard.csv"
cat "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\modeling_output\classification_logreg_coefficients.csv"
cat "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\modeling_output\classification_xgb_permutation_importance.csv"
```

Expected: six non-empty CSVs print to the terminal.

- [ ] **Step 2: Write `modeling_notes.md`**

Populate this structure using the actual values read in Step 1 (every `<...>` placeholder below must be replaced with a real number/name from the CSVs — this file must contain zero placeholder text when done):

```markdown
# Modeling Notes: Startup Failure Duration

Design doc: `docs/superpowers/specs/2026-07-02-startup-failure-modeling-design.md`

## Regression leaderboard (target: duration_years)

<paste the actual regression_leaderboard.csv table as markdown>

Baseline: DummyRegressor(strategy="mean"), test R2 ≈ 0. <For each non-Dummy
model, one sentence: "<Model> reaches test R2=<value>, a <small/notable>
improvement over baseline (+<value>). Train-test R2 gap of <value> indicates
<low/moderate/high> overfitting.">

See `modeling_output/regression_predicted_vs_actual.png` and
`modeling_output/regression_tree_validation_curve.png`.

## Classification leaderboard (target: duration_class)

<paste the actual classification_leaderboard.csv table as markdown>

Baseline: DummyClassifier(strategy="most_frequent"). <Same honest per-model
commentary as above, anchored to f1_macro and roc_auc_macro_ovr.>

See `modeling_output/classification_confusion_matrices.png` and
`modeling_output/classification_tree_validation_curve.png`.

## Which features matter

<Cross-reference regression_lasso_coefficients.csv against
regression_xgb_permutation_importance.csv (top 5-10 each): list features that
agree in direction/importance across both, and any that disagree. Repeat for
classification_logreg_coefficients.csv vs classification_xgb_permutation_importance.csv.>

## Honest evaluation summary

<One paragraph: does XGBoost actually beat the simpler models here, given the
n=409 sample size? If not, say so plainly. Which model would you actually
recommend and why, given the train-test gaps observed.>

## Business interpretation

**What the results mean:** These models explain variation in *observed*
duration among startups that already failed. They are not a forecast of how
long a currently-operating company will last -- the failure-reason flags used
as predictors are hindsight labels, assigned only after each company's
outcome was already known (see `data_notes.md`). Repositioning this as a
forward-looking survival predictor would misrepresent what it was trained on.

**Which features matter for the business question:** <2-3 sentences drawing
on the feature-importance cross-reference above, in plain business language.>

**What we would not trust this model to do:**
1. Predict the remaining runway of an active, non-failed company (hindsight-label issue above).
2. Generalize to sectors thin on data -- Food/services has only 26 rows total in this dataset, and Sector x duration_class cells can be very sparse.
3. Extrapolate past the observed funding range (max observed raise in this dataset is $3.5B).
4. Support causal claims -- e.g. "Poor Market Fit causes shorter duration" is not established; only "companies labeled this way tended to have shorter observed duration in this sample."
```

- [ ] **Step 3: Confirm the file has no unfilled placeholders**

Run: read back `modeling_notes.md` and grep it for the literal string `<` — any remaining `<...>` marker means a placeholder was left unfilled and must be replaced with the real value before this task is done.

```bash
grep -n "<" "C:\Users\aleks\Desktop\Master - IMB\3. semestar\MA DA\startup failure\modeling_notes.md"
```

Expected: no output (no matches) — if any line prints, go back and fill in that placeholder with the real number/interpretation from the Step 1 data.

---

## Self-Review

**Spec coverage:**
- `SectorMedianImputer` fix → Task 1 (`FundingFeatureEngineer` + `build_dataset`'s `raised_imputed` reset). ✓
- 6 engineered features → Task 1 (`FundingFeatureEngineer` for 3 funding-derived ones; `build_dataset` for `n_flags_sq`, `single_cause_failure`, `big_tech_pressure`, `decade_started`). ✓
- Shared `ColumnTransformer`/`Pipeline` architecture → Task 1 (`build_preprocessor`, `build_pipeline`), reused unmodified by Tasks 2 & 3. ✓
- Two-target single stratified split → Task 1 (`build_dataset`'s `duration_class`, `make_split`). ✓
- Regression ladder (Dummy/LassoCV/DecisionTree+validation_curve/XGBoost+RandomizedSearchCV) → Task 2. ✓
- Classification ladder (Dummy/LogisticRegressionCV/DecisionTree+validation_curve/XGBoost+RandomizedSearchCV) → Task 3. ✓
- Regression leaderboard + predicted-vs-actual → Task 2. ✓
- Classification leaderboard + confusion-matrix grid → Task 3. ✓
- Honest-evaluation mechanics (train+test reported, gap column, baseline-anchored deltas, no cherry-picking) → built into both ladder scripts' leaderboard construction and print statements. ✓
- Business interpretation write-up → Task 4. ✓

**Placeholder scan:** the only `<...>` markers in this plan are inside Task 4's `modeling_notes.md` *template*, which Step 2/3 of that task explicitly requires filling with real values and Step 3 explicitly verifies are gone — this is a documented fill-in-the-template step, not an unresolved plan gap.

**Type/name consistency:** `FEATURE_COLUMNS`, `COMMON_FLAGS`, `build_dataset`, `make_split`, `build_pipeline`, `RANDOM_STATE` are defined once in Task 1 and imported by name, unchanged, in Tasks 2 and 3 — verified matching signatures throughout.
