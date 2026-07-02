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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, validation_curve
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from feature_pipeline import FEATURE_COLUMNS, RANDOM_STATE, build_dataset, build_pipeline, make_split

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FOLDER, "modeling_output")
os.makedirs(OUT, exist_ok=True)

CLASSES = ["early", "typical", "long_run"]
# Map string class labels to numeric for XGBoost compatibility
CLASS_TO_CODE = {cls: i for i, cls in enumerate(CLASSES)}
CODE_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}


def get_xy(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df["duration_class"].map(CLASS_TO_CODE).copy()
    return X, y


def find_best_tree_depth(X_train, y_train):
    depths = list(range(1, 16))
    train_scores, cv_scores = validation_curve(
        build_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        X_train, y_train,
        param_name="model__max_depth", param_range=depths,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="f1_macro",
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
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
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
        coefs.T, index=feature_names,
        columns=[CODE_TO_CLASS[c] for c in logreg_pipeline.named_steps["model"].classes_],
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
