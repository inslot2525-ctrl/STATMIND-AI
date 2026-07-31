import os
from fastapi import APIRouter, Form, HTTPException

from services.ml_engine import train_model_from_file, compare_models_from_file
from services.dataset_registry import get_dataset

router = APIRouter()

UPLOAD_DIR = "uploads"


@router.post("/train-model")
async def train_model(
    filename: str = Form(None),  # Make filename optional
    dataset_id: str = Form(None),  # NEW: dataset_id parameter
    target_column: str = Form(...),
    algorithm: str = Form(...),
    test_size: float = Form(0.2),
):
    try:
        # Determine which file to use
        if dataset_id:
            # Look up dataset by ID
            dataset_info = get_dataset(dataset_id)
            if not dataset_info:
                raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} not found")
            file_path = dataset_info["file_path"]
            # Use original filename for display/logging
            filename_for_display = dataset_info["original_filename"]
        elif filename:
            # Fallback to original behavior (backward compatibility)
            file_path = os.path.join(UPLOAD_DIR, filename)
            filename_for_display = filename
        else:
            raise ValueError("Either filename or dataset_id must be provided")

        result = train_model_from_file(
            file_path=file_path,
            target_column=target_column,
            algorithm=algorithm,
            test_size=test_size,
            dataset_id=dataset_id,  # PASS DATASET_ID TO TRAINING FUNCTION
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-models")
async def compare_models(
    filename: str = Form(None),  # Make filename optional
    dataset_id: str = Form(None),  # NEW: dataset_id parameter
    target_column: str = Form(...),
    test_size: float = Form(0.2),
):
    try:
        # Determine which file to use
        if dataset_id:
            # Look up dataset by ID
            dataset_info = get_dataset(dataset_id)
            if not dataset_info:
                raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} not found")
            file_path = dataset_info["file_path"]
        elif filename:
            # Fallback to original behavior (backward compatibility)
            file_path = os.path.join(UPLOAD_DIR, filename)
        else:
            raise ValueError("Either filename or dataset_id must be provided")

        result = compare_models_from_file(
            file_path=file_path,
            target_column=target_column,
            test_size=test_size,
            dataset_id=dataset_id,  # PASS DATASET_ID TO COMPARISON FUNCTION
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# EXPERIMENT TRACKING ENDPOINTS
from services.experiment_tracker import (
    list_experiments, get_experiment, get_experiments_for_model
)

experiment_router = APIRouter(prefix="/experiments", tags=["Experiments"])

@experiment_router.get("/")
async def get_experiments(dataset_id: str = None, limit: int = 50):
    experiments = list_experiments(dataset_id=dataset_id, limit=limit)
    return {"experiments": experiments}

@experiment_router.get("/{experiment_id}")
async def get_experiment_detail(experiment_id: str):
    experiment = get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment

@experiment_router.get("/model/{model_id}")
async def get_model_experiments(model_id: str):
    experiments = get_experiments_for_model(model_id)
    return {"experiments": experiments}


# MODEL REGISTRY ENDPOINTS
from services.model_registry import (
    register_model, get_model, list_models,
    transition_model_stage, get_production_models
)

model_router = APIRouter(prefix="/models", tags=["Model Registry"])

@model_router.post("/register/{model_id}")
async def register_existing_model(model_id: str):
    """Register an existing model file"""
    try:
        model_info = register_model(model_id)
        return model_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@model_router.get("/")
async def list_models_endpoint(
    algorithm: str = None,
    target_column: str = None,
    stage: str = None,
    limit: int = 50
):
    filter_criteria = {}
    if algorithm:
        filter_criteria["algorithm"] = algorithm
    if target_column:
        filter_criteria["target_column"] = target_column
    if stage:
        filter_criteria["stage"] = stage

    models = list_models(filter_criteria=filter_criteria)
    return {"models": models[:limit]}

@model_router.get("/{model_id}")
async def get_model_details(model_id: str):
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@model_router.post("/{model_id}/transition")
async def transition_model_stage_endpoint(
    model_id: str,
    stage: str,
    version: str = None
):
    valid_stages = ["development", "staging", "production", "archived"]
    if stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Must be one of: {valid_stages}"
        )

    try:
        model_info = transition_model_stage(model_id, stage, version)
        return model_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@model_router.get("/production/list")
async def get_production_models_endpoint():
    models = get_production_models()
    return {"production_models": models}