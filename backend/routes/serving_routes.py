"""Online prediction and drift monitoring for saved models."""
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from database import User
from services import drift, serving
from services.auth import get_current_user

router = APIRouter()


class PredictRequest(BaseModel):
    # One JSON object per row: {"column name": value, ...}
    records: List[Dict[str, Any]]


def _predict(model_id: str, user: User, body: PredictRequest) -> dict:
    try:
        return serving.predict_records(model_id, user.id, body.records)
    except ValueError as exc:
        # "not found" covers both a missing model and one owned by someone else
        status = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status, detail=str(exc))


@router.post("/models/{model_id}/predict")
def predict_with_model(model_id: str, body: PredictRequest, user: User = Depends(get_current_user)):
    return _predict(model_id, user, body)


@router.post("/predict")
def predict_with_production_model(target: str, body: PredictRequest, user: User = Depends(get_current_user)):
    """Score records with whichever model is in production for `target`."""
    try:
        model_id = serving.production_model_id(user.id, target)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _predict(model_id, user, body)


@router.get("/models/{model_id}/serving")
def serving_info(model_id: str, user: User = Depends(get_current_user)):
    try:
        return serving.serving_info(model_id, user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/models/{model_id}/drift")
def model_drift(model_id: str, window: str = "latest", user: User = Depends(get_current_user)):
    try:
        return drift.check_drift(model_id, user.id, window)
    except ValueError as exc:
        status = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status, detail=str(exc))


@router.post("/models/{model_id}/drift/report")
def model_drift_report(model_id: str, window: str = "latest", user: User = Depends(get_current_user)):
    # Download via GET /api/artifacts/{artifact_id}/download-url
    try:
        artifact = drift.drift_report_artifact(model_id, user.id, window)
    except ValueError as exc:
        status = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status, detail=str(exc))
    return {"artifact_id": artifact["id"], "filename": artifact["filename"]}
