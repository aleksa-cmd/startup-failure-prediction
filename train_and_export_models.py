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
