import os
import json
import joblib
from datetime import datetime
from database import SessionLocal, ModelRegistry, init_db

init_db()

def _to_dict(obj):
    if not obj:
        return None
    return {
        "model_id": obj.model_id,
        "registered_at": obj.registered_at,
        "updated_at": obj.updated_at,
        "metadata": json.loads(obj.metadata_json) if obj.metadata_json else {},
        "versions": json.loads(obj.versions_json) if obj.versions_json else [],
    }

def register_model(model_id, metadata=None):
    """
    Register a model in the registry.
    If metadata not provided, attempts to load from model file.
    """
    from database import SessionLocal
    # Load model to get metadata if not provided
    if metadata is None:
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_path = os.path.join(model_dir, model_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            artifact = joblib.load(model_path)
            metadata = artifact.get("metadata", {})
        except Exception as e:
            metadata = {
                "model_id": model_id,
                "error": f"Failed to load model: {str(e)}"
            }

    if "model_id" not in metadata:
        metadata["model_id"] = model_id

    db = SessionLocal()
    try:
        existing = db.query(ModelRegistry).filter_by(model_id=model_id).first()
        if existing:
            existing.updated_at = datetime.now().isoformat()
            # Merge metadata
            existing_meta = json.loads(existing.metadata_json) if existing.metadata_json else {}
            merged = {**existing_meta, **metadata}
            existing.metadata_json = json.dumps(merged, default=str)
            db.commit()
            return _to_dict(existing)
        else:
            entry = ModelRegistry(
                model_id=model_id,
                registered_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                metadata_json=json.dumps(metadata, default=str),
                versions_json=json.dumps([{
                    "version": "1.0.0",
                    "stage": "development",
                    "registered_at": datetime.now().isoformat()
                }])
            )
            db.add(entry)
            db.commit()
            return _to_dict(entry)
    finally:
        db.close()

def get_model(model_id):
    """Get model metadata by ID"""
    db = SessionLocal()
    try:
        obj = db.query(ModelRegistry).filter_by(model_id=model_id).first()
        return _to_dict(obj)
    finally:
        db.close()

def list_models(filter_criteria=None):
    """
    List models with optional filtering.
    filter_criteria: dict of metadata fields to match
    """
    db = SessionLocal()
    try:
        objs = db.query(ModelRegistry).order_by(ModelRegistry.registered_at.desc()).all()
        models = [_to_dict(o) for o in objs]
        if filter_criteria:
            filtered = []
            for m in models:
                meta = m.get("metadata", {})
                if all(meta.get(k) == v for k, v in filter_criteria.items()):
                    filtered.append(m)
            models = filtered
        return models
    finally:
        db.close()

def transition_model_stage(model_id, stage, version=None):
    """
    Transition a model version to a stage (development, staging, production, archived).
    If version not specified, affects the latest version.
    """
    valid_stages = ["development", "staging", "production", "archived"]
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")

    db = SessionLocal()
    try:
        obj = db.query(ModelRegistry).filter_by(model_id=model_id).first()
        if not obj:
            raise ValueError(f"Model {model_id} not found in registry")

        versions = json.loads(obj.versions_json) if obj.versions_json else []

        # Determine target version
        if version is None:
            if versions:
                def version_key(v):
                    try:
                        return [int(x) for x in v["version"].split(".")]
                    except:
                        return [0, 0, 0]
                versions.sort(key=version_key, reverse=True)
                target_version = versions[0]["version"]
            else:
                target_version = "1.0.0"
        else:
            target_version = version

        version_found = False
        for v in versions:
            if v["version"] == target_version:
                v["stage"] = stage
                v["transitioned_at"] = datetime.now().isoformat()
                version_found = True
                break

        if not version_found:
            versions.append({
                "version": target_version,
                "stage": stage,
                "transitioned_at": datetime.now().isoformat()
            })

        obj.versions_json = json.dumps(versions)
        # Also update metadata stage
        meta = json.loads(obj.metadata_json) if obj.metadata_json else {}
        meta["stage"] = stage
        obj.metadata_json = json.dumps(meta, default=str)
        obj.updated_at = datetime.now().isoformat()
        db.commit()
        return _to_dict(obj)
    finally:
        db.close()

def get_production_models():
    """Get all models currently in production stage"""
    db = SessionLocal()
    try:
        models = [_to_dict(o) for o in db.query(ModelRegistry).all()]
        prod = []
        for m in models:
            for v in m.get("versions", []):
                if v.get("stage") == "production":
                    prod.append({
                        "model_id": m["model_id"],
                        "version": v["version"],
                        **m["metadata"]
                    })
                    break
        return prod
    finally:
        db.close()

def get_model_lineage(model_id):
    """
    Get the lineage of a model - what experiments/datasets led to it.
    """
    m = get_model(model_id)
    if not m:
        return None
    meta = m.get("metadata", {})
    return {
        "model_id": model_id,
        "dataset_id": meta.get("dataset_id"),
        "experiment_ids": meta.get("experiment_ids", []),
        "training_timestamp": meta.get("created_at"),
    }
