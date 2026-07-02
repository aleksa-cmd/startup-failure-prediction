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
