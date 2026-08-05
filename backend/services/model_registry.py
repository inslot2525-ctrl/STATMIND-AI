import os
import json
import joblib
from datetime import datetime
from pathlib import Path

MODEL_DIR = "models"
MODEL_REGISTRY_PATH = "registry/model_registry.json"

def _load_registry():
    if not os.path.exists(MODEL_REGISTRY_PATH):
        return {"models": {}, "next_version": 1}
    with open(MODEL_REGISTRY_PATH, 'r') as f:
        return json.load(f)

def _save_registry(data):
    with open(MODEL_REGISTRY_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def register_model(model_id, metadata=None):
    """
    Register a model in the registry.
    If metadata not provided, attempts to load from model file.
    """
    # Load model to get metadata if not provided
    if metadata is None:
        model_path = os.path.join(MODEL_DIR, model_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            artifact = joblib.load(model_path)
            metadata = artifact.get("metadata", {})
        except Exception as e:
            # Fallback to basic info
            metadata = {
                "model_id": model_id,
                "error": f"Failed to load model: {str(e)}"
            }

    registry = _load_registry()

    # Ensure model_id is in metadata
    if "model_id" not in metadata:
        metadata["model_id"] = model_id

    # Check if model already registered
    if model_id in registry["models"]:
        # Update existing entry
        model_entry = registry["models"][model_id]
        model_entry["updated_at"] = datetime.now().isoformat()
        # Merge metadata, preferring new values for conflicts
        model_entry["metadata"] = {**model_entry["metadata"], **metadata}
    else:
        # Create new entry
        model_entry = {
            "model_id": model_id,
            "registered_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata,
            "versions": [{
                "version": "1.0.0",
                "stage": "development",
                "registered_at": datetime.now().isoformat()
            }]
        }
        registry["models"][model_id] = model_entry

    _save_registry(registry)
    return model_entry

def get_model(model_id):
    """Get model metadata by ID"""
    registry = _load_registry()
    return registry["models"].get(model_id)

def list_models(filter_criteria=None):
    """
    List models with optional filtering.
    filter_criteria: dict of metadata fields to match
    """
    registry = _load_registry()
    models = list(registry["models"].values())

    if filter_criteria:
        filtered_models = []
        for model in models:
            match = True
            meta = model.get("metadata", {})
            for key, value in filter_criteria.items():
                if meta.get(key) != value:
                    match = False
                    break
            if match:
                filtered_models.append(model)
        models = filtered_models

    # Sort by registered_at descending (newest first)
    models.sort(key=lambda x: x["registered_at"], reverse=True)
    return models

def transition_model_stage(model_id, stage, version=None):
    """
    Transition a model version to a stage (development, staging, production, archived).
    If version not specified, affects the latest version.
    """
    valid_stages = ["development", "staging", "production", "archived"]
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")

    registry = _load_registry()
    if model_id not in registry["models"]:
        raise ValueError(f"Model {model_id} not found in registry")

    model_entry = registry["models"][model_id]

    # Determine which version to transition
    target_version = None
    if version is None:
        # Use highest version number
        versions = model_entry.get("versions", [])
        if versions:
            # Simple version sorting - assumes semantic versioning
            def version_key(v):
                try:
                    return [int(x) for x in v["version"].split(".")]
                except:
                    return [0, 0, 0]  # fallback for non-standard versions

            versions.sort(key=version_key, reverse=True)
            target_version = versions[0]["version"]
        else:
            target_version = "1.0.0"
    else:
        target_version = version

    # Find and update the version
    version_found = False
    for v_entry in model_entry["versions"]:
        if v_entry["version"] == target_version:
            v_entry["stage"] = stage
            v_entry["transitioned_at"] = datetime.now().isoformat()
            version_found = True
            break

    if not version_found:
        # Add new version entry
        model_entry["versions"].append({
            "version": target_version,
            "stage": stage,
            "transitioned_at": datetime.now().isoformat()
        })

    model_entry["updated_at"] = datetime.now().isoformat()
    # Also update metadata stage for quick access
    model_entry["metadata"]["stage"] = stage

    _save_registry(registry)
    return model_entry

def get_production_models():
    """Get all models currently in production stage"""
    registry = _load_registry()
    production_models = []

    for model_id, model_data in registry["models"].items():
        # Check if any version is in production
        for version in model_data.get("versions", []):
            if version.get("stage") == "production":
                production_models.append({
                    "model_id": model_id,
                    "version": version["version"],
                    **model_data["metadata"]
                })
                break  # Only need one production version per model

    return production_models

def get_model_lineage(model_id):
    """
    Get the lineage of a model - what experiments/datasets led to it.
    This would require integration with experiment tracker.
    For now, returns what we have in metadata.
    """
    model = get_model(model_id)
    if not model:
        return None

    metadata = model.get("metadata", {})
    lineage = {
        "model_id": model_id,
        "dataset_id": metadata.get("dataset_id"),
        "experiment_ids": metadata.get("experiment_ids", []),  # Would come from experiment tracker
        "training_timestamp": metadata.get("created_at"),
    }
    return lineage