"""
FastAPI service exposing the credit risk model to other systems
(loan origination software, internal risk tools, etc.). Run with:

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explainability.shap_explainer import ShapExplainer
from src.prediction.predict import CreditRiskPredictor
from src.utils.helper import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Risk Intelligence Platform API",
    description="Serves default-risk predictions, batch scoring, and model metadata.",
    version="0.1.0",
)

_predictor: Optional[CreditRiskPredictor] = None
_explainer: Optional[ShapExplainer] = None


class ApplicantRecord(BaseModel):
    applicant_id: str
    annual_income: float
    loan_amount: float
    credit_score: float = Field(ge=300, le=850)
    employment_years: float
    age: int
    existing_debt: float
    credit_limit: float
    num_open_accounts: int
    num_credit_inquiries_6m: int
    num_previous_defaults: int
    delinquencies_2y: int
    loan_purpose: str
    home_ownership: str
    application_date: Optional[str] = "2026-01-01"
    account_open_date: Optional[str] = "2020-01-01"


class BatchRequest(BaseModel):
    records: List[ApplicantRecord]


@app.on_event("startup")
def load_model_artifacts():
    global _predictor, _explainer
    try:
        _predictor = CreditRiskPredictor()
        logger.info("Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        logger.warning(f"Model artifacts not found at startup: {e}. "
                        f"Run the training pipeline before calling /predict.")
        _predictor = None


def _require_predictor() -> CreditRiskPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline (python main.py --stage all) first.",
        )
    return _predictor


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/model-info")
def model_info():
    predictor = _require_predictor()
    return {
        "model_type": predictor.artifacts.model.__class__.__name__,
        "n_features": len(predictor.artifacts.feature_names),
        "features": predictor.artifacts.feature_names,
    }


@app.post("/predict")
def predict_single(applicant: ApplicantRecord):
    predictor = _require_predictor()
    result = predictor.predict_single(applicant.dict())
    return result


@app.post("/predict-batch")
def predict_batch(request: BatchRequest):
    predictor = _require_predictor()
    df = pd.DataFrame([r.dict() for r in request.records])
    result = predictor.predict_batch(df)
    return result.to_dict(orient="records")


@app.get("/feature-importance")
def feature_importance(top_n: int = 15):
    predictor = _require_predictor()
    test_path = resolve_path(predictor.artifacts.config["data"]["test_path"])
    if not test_path.exists():
        raise HTTPException(status_code=503, detail="Test data not found — run the training pipeline first.")

    x_test = pd.read_csv(test_path).drop(columns=[predictor.artifacts.config["data"]["target_column"]])
    global _explainer
    if _explainer is None:
        _explainer = ShapExplainer(predictor.artifacts.model, x_test.sample(min(200, len(x_test)), random_state=42),
                                    predictor.artifacts.config)
    importance = _explainer.global_feature_importance(x_test.sample(min(500, len(x_test)), random_state=42), top_n=top_n)
    return importance.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
