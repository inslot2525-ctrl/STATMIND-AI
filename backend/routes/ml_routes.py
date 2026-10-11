from fastapi import APIRouter, Depends, Form, HTTPException

from database import User
from routes.deps import resolve_user_dataset
from services.auth import get_current_user
from services.jobs import enqueue
from services.ml_engine import TASK_TYPES

router = APIRouter()


def _check_test_size(test_size: float):
    if not 0 < test_size < 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1. Example: 0.2")


def _check_task_type(task_type: str | None) -> str | None:
    """Empty or 'auto' means: detect classification vs regression from the target."""
    task_type = (task_type or "").lower().strip()
    if task_type in ("", "auto"):
        return None
    if task_type not in TASK_TYPES:
        raise HTTPException(status_code=400, detail=f"task_type must be auto or one of {list(TASK_TYPES)}.")
    return task_type


# Training runs as a background job: these endpoints return 202 with a job id
# immediately, and the client polls GET /api/jobs/{job_id} for progress and result.
@router.post("/train-model", status_code=202)
def train_model(
    filename: str = Form(None),
    dataset_id: str = Form(None),
    target_column: str = Form(...),
    algorithm: str = Form(...),
    test_size: float = Form(0.2),
    task_type: str = Form(None),
    user: User = Depends(get_current_user),
):
    _check_test_size(test_size)
    task_type = _check_task_type(task_type)
    dataset = resolve_user_dataset(user.id, dataset_id, filename)
    return enqueue(
        "train_model",
        {
            "dataset_id": dataset["id"],
            "target_column": target_column,
            "algorithm": algorithm,
            "test_size": test_size,
            "task_type": task_type,
        },
        owner_id=user.id,
    )


@router.post("/compare-models", status_code=202)
def compare_models(
    filename: str = Form(None),
    dataset_id: str = Form(None),
    target_column: str = Form(...),
    test_size: float = Form(0.2),
    task_type: str = Form(None),
    user: User = Depends(get_current_user),
):
    _check_test_size(test_size)
    task_type = _check_task_type(task_type)
    dataset = resolve_user_dataset(user.id, dataset_id, filename)
    return enqueue(
        "compare_models",
        {
            "dataset_id": dataset["id"],
            "target_column": target_column,
            "test_size": test_size,
            "task_type": task_type,
        },
        owner_id=user.id,
    )


# EXPERIMENT TRACKING ENDPOINTS
from services.experiment_tracker import (  # noqa: E402
    list_experiments, get_experiment, get_experiments_for_model
)

experiment_router = APIRouter(prefix="/experiments", tags=["Experiments"])


@experiment_router.get("/")
def get_experiments(dataset_id: str = None, limit: int = 50, user: User = Depends(get_current_user)):
    return {"experiments": list_experiments(user.id, dataset_id=dataset_id, limit=limit)}


@experiment_router.get("/{experiment_id}")
def get_experiment_detail(experiment_id: str, user: User = Depends(get_current_user)):
    experiment = get_experiment(experiment_id, user.id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@experiment_router.get("/model/{model_id}")
def get_model_experiments(model_id: str, user: User = Depends(get_current_user)):
    return {"experiments": get_experiments_for_model(model_id, user.id)}


# MODEL REGISTRY ENDPOINTS
from services.model_registry import (  # noqa: E402
    VALID_STAGES, get_model, list_models, transition_model_stage, get_production_models
)

model_router = APIRouter(prefix="/models", tags=["Model Registry"])


@model_router.get("/")
def list_models_endpoint(
    algorithm: str = None,
    target_column: str = None,
    stage: str = None,
    limit: int = 50,
    user: User = Depends(get_current_user),
):
    filter_criteria = {}
    if algorithm:
        filter_criteria["algorithm"] = algorithm
    if target_column:
        filter_criteria["target_column"] = target_column
    if stage:
        filter_criteria["stage"] = stage

    models = list_models(user.id, filter_criteria=filter_criteria)
    return {"models": models[:limit]}


@model_router.get("/production/list")
def get_production_models_endpoint(user: User = Depends(get_current_user)):
    return {"production_models": get_production_models(user.id)}


@model_router.get("/{model_id}")
def get_model_details(model_id: str, user: User = Depends(get_current_user)):
    model = get_model(model_id, user.id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@model_router.post("/{model_id}/transition")
def transition_model_stage_endpoint(
    model_id: str,
    stage: str,
    version: str = None,
    user: User = Depends(get_current_user),
):
    if stage not in VALID_STAGES:
        raise HTTPException(status_code=400, detail=f"Invalid stage. Must be one of: {VALID_STAGES}")

    try:
        return transition_model_stage(model_id, stage, user.id, version)
    except LookupError:
        raise HTTPException(status_code=404, detail="Model not found")
