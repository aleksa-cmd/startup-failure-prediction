"""Verification script for api/predict.py -- exercises both endpoints via
FastAPI's TestClient (no server process needed). Run directly; not a pytest
suite (this project has no test framework).
"""
from fastapi.testclient import TestClient

from predict import app

client = TestClient(app)

SAMPLE_REQUEST = {
    "raised_musd": 15.0,
    "sector": "Information",
    "start_year": 2015,
    "giants": True,
    "no_budget": False,
    "competition": True,
    "poor_market_fit": False,
    "acquisition_stagnation": False,
    "monetization_failure": True,
    "niche_limits": False,
    "execution_flaws": False,
    "trend_shifts": False,
}

reg_response = client.post("/api/predict/regression", json=SAMPLE_REQUEST)
print(f"Regression status: {reg_response.status_code}")
print(f"Regression body: {reg_response.json()}")
assert reg_response.status_code == 200
assert "predicted_duration_years" in reg_response.json()
assert isinstance(reg_response.json()["predicted_duration_years"], float)

clf_response = client.post("/api/predict/classification", json=SAMPLE_REQUEST)
print(f"\nClassification status: {clf_response.status_code}")
print(f"Classification body: {clf_response.json()}")
assert clf_response.status_code == 200
body = clf_response.json()
assert body["predicted_class"] in ("early", "typical", "long_run")
assert set(body["probabilities"].keys()) == {"early", "typical", "long_run"}
assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01

bad_sector_request = dict(SAMPLE_REQUEST, sector="Not A Real Sector")
bad_response = client.post("/api/predict/regression", json=bad_sector_request)
print(f"\nInvalid sector status (expect 422): {bad_response.status_code}")
assert bad_response.status_code == 422

print("\nAll checks passed.")
