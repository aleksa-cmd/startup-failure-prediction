"""FastAPI predictor service for the startup-failure DecisionTree models.
Deployed as a Vercel Python serverless function -- this module exports
`app`, which Vercel's Python runtime auto-detects as an ASGI app to serve
under /api/predict/*. See
docs/superpowers/specs/2026-07-02-react-fastapi-predictor-dashboard-design.md
"""
import os
import sys
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline import FEATURE_COLUMNS, derive_engineered_columns  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

_regression_model = joblib.load(os.path.join(MODELS_DIR, "regression_model.joblib"))
_classification_model = joblib.load(os.path.join(MODELS_DIR, "classification_model.joblib"))

Sector = Literal[
    "Finance and Insurance",
    "Accommodation and Food Services",
    "Health Care",
    "Manufacturing",
    "Retail Trade",
    "Information",
]

# Maps the API's snake_case boolean field names to the exact COMMON_FLAGS
# column names feature_pipeline.py expects (which contain spaces/mixed case).
FLAG_FIELD_TO_COLUMN = {
    "giants": "Giants",
    "no_budget": "No Budget",
    "competition": "Competition",
    "poor_market_fit": "Poor Market Fit",
    "acquisition_stagnation": "Acquisition Stagnation",
    "monetization_failure": "Monetization Failure",
    "niche_limits": "Niche Limits",
    "execution_flaws": "Execution Flaws",
    "trend_shifts": "Trend Shifts",
}


class PredictRequest(BaseModel):
    raised_musd: float = Field(..., ge=0, description="Amount raised, in millions USD")
    sector: Sector
    start_year: int = Field(..., ge=1900, le=2029)
    giants: bool
    no_budget: bool
    competition: bool
    poor_market_fit: bool
    acquisition_stagnation: bool
    monetization_failure: bool
    niche_limits: bool
    execution_flaws: bool
    trend_shifts: bool


class RegressionResponse(BaseModel):
    predicted_duration_years: float


class ClassificationResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]


def _request_to_row(request: PredictRequest) -> pd.DataFrame:
    row = {
        "raised_musd": request.raised_musd,
        "Sector": request.sector,
        "start_year": request.start_year,
    }
    for field_name, column_name in FLAG_FIELD_TO_COLUMN.items():
        row[column_name] = int(getattr(request, field_name))
    df = pd.DataFrame([row])
    df = derive_engineered_columns(df)
    return df[FEATURE_COLUMNS]


@app.post("/api/predict/regression", response_model=RegressionResponse)
def predict_regression(request: PredictRequest):
    X = _request_to_row(request)
    prediction = _regression_model.predict(X)[0]
    return RegressionResponse(predicted_duration_years=round(float(prediction), 2))


@app.post("/api/predict/classification", response_model=ClassificationResponse)
def predict_classification(request: PredictRequest):
    X = _request_to_row(request)
    predicted_class = _classification_model.predict(X)[0]
    probabilities = _classification_model.predict_proba(X)[0]
    classes = _classification_model.classes_
    return ClassificationResponse(
        predicted_class=predicted_class,
        probabilities={cls: round(float(p), 4) for cls, p in zip(classes, probabilities)},
    )
