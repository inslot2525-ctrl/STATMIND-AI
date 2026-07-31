import os
import json
import uuid
from datetime import datetime
from services.dataset_registry import get_dataset

EXPERIMENT_TRACKING_PATH = "experiment_tracking.json"

def _load_tracking():
    if not os.path.exists(EXPERIMENT_TRACKING_PATH):
        return {"experiments": {}, "next_id": 1}
    with open(EXPERIMENT_TRACKING_PATH, 'r') as f:
        return json.load(f)

def _save_tracking(data):
    with open(EXPERIMENT_TRACKING_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def log_experiment(
    dataset_id,
    algorithm,
    hyperparameters=None,
    metrics=None,
    model_id=None,
    tags=None,
    notes=None
):
    """
    Log an ML experiment.
    Returns experiment_id.
    """
    # Validate dataset exists
    dataset_info = get_dataset(dataset_id)
    if not dataset_info:
        raise ValueError(f"Dataset {dataset_id} not found in registry")

    tracking_data = _load_tracking()
    experiment_id = str(tracking_data["next_id"])
    tracking_data["next_id"] += 1

    experiment = {
        "id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "dataset_info": {
            "filename": dataset_info["original_filename"],
            "rows": dataset_info.get("rows"),
            "columns": dataset_info.get("columns")
        },
        "algorithm": algorithm,
        "hyperparameters": hyperparameters or {},
        "metrics": metrics or {},
        "model_id": model_id,
        "tags": tags or [],
        "notes": notes or ""
    }

    tracking_data["experiments"][experiment_id] = experiment
    _save_tracking(tracking_data)
    return experiment_id

def get_experiment(experiment_id):
    tracking_data = _load_tracking()
    return tracking_data["experiments"].get(experiment_id)

def list_experiments(dataset_id=None, limit=50):
    tracking_data = _load_tracking()
    experiments = list(tracking_data["experiments"].values())

    if dataset_id:
        experiments = [exp for exp in experiments if exp["dataset_id"] == dataset_id]

    # Sort by timestamp descending
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    return experiments[:limit]

def get_experiments_for_model(model_id):
    tracking_data = _load_tracking()
    return [
        exp for exp in tracking_data["experiments"].values()
        if exp.get("model_id") == model_id
    ]