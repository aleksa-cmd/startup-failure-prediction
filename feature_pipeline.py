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
