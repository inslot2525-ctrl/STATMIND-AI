import os
import json
import joblib
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "registry", "model_registry.json")

def _load_registry():
    os.makedirs(os.path.dirname(MODEL_REGISTRY_PATH), exist_ok=True)
    if not os.path.exists(MODEL_REGISTRY_PATH):
        return {"models": {}, "next_version": 1}
    with open(MODEL_REGISTRY_PATH, "r") as f:
        return json.load(f)

def _save_registry(data):
    os.makedirs(os.path.dirname(MODEL_REGISTRY_PATH), exist_ok=True)
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)

def register_model(model_id, metadata=None):
    if metadata is None:
        model_path = os.path.join(MODEL_DIR, model_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            artifact = joblib.load(model_path)
            metadata = artifact.get("metadata", {})
        except Exception as e:
            metadata = {"model_id": model_id, "error": str(e)}
    registry = _load_registry()
    if "model_id" not in metadata:
        metadata["model_id"] = model_id
    if model_id in registry["models"]:
        model_entry = registry["models"][model_id]
        model_entry["updated_at"] = datetime.now().isoformat()
        model_entry["metadata"] = {**model_entry["metadata"], **metadata}
    else:
        model_entry = {
            "model_id": model_id,
            "registered_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata,
            "versions": [{"version": "1.0.0", "stage": "development", "registered_at": datetime.now().isoformat()}],
        }
        registry["models"][model_id] = model_entry
    _save_registry(registry)
    return model_entry

def get_model(model_id):
    registry = _load_registry()
    return registry["models"].get(model_id)

def list_models(filter_criteria=None):
    registry = _load_registry()
    models = list(registry["models"].values())
    if filter_criteria:
        filtered = []
        for m in models:
            meta = m.get("metadata", {})
            if all(meta.get(k) == v for k, v in filter_criteria.items()):
                filtered.append(m)
        models = filtered
    models.sort(key=lambda x: x["registered_at"], reverse=True)
    return models

def transition_model_stage(model_id, stage, version=None):
    valid_stages = ["development", "staging", "production", "archived"]
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")
    registry = _load_registry()
    if model_id not in registry["models"]:
        raise ValueError(f"Model {model_id} not found")
    model_entry = registry["models"][model_id]
    versions = model_entry.get("versions", [])
    target_version = version or (versions[0]["version"] if versions else "1.0.0")
    found = False
    for v in versions:
        if v["version"] == target_version:
            v["stage"] = stage
            v["transitioned_at"] = datetime.now().isoformat()
            found = True
            break
    if not found:
        versions.append({"version": target_version, "stage": stage, "transitioned_at": datetime.now().isoformat()})
    model_entry["updated_at"] = datetime.now().isoformat()
    model_entry["metadata"]["stage"] = stage
    _save_registry(registry)
    return model_entry

def get_production_models():
    registry = _load_registry()
    result = []
    for model_id, model_data in registry["models"].items():
        for v in model_data.get("versions", []):
            if v.get("stage") == "production":
                result.append({"model_id": model_id, "version": v["version"], **model_data["metadata"]})
                break
    return result
