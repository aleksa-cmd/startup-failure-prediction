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
