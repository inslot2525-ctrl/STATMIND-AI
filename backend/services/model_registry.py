import json
from datetime import datetime

from database import SessionLocal, ModelRegistry
from services import mlflow_tracking

VALID_STAGES = ["development", "staging", "production", "archived"]


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


def register_model(model_id, metadata, owner_id, storage_key):
    """Register a newly trained model, or merge metadata into an existing entry."""
    metadata = {**metadata, "model_id": model_id}

    db = SessionLocal()
    try:
        existing = db.query(ModelRegistry).filter_by(model_id=model_id, owner_id=owner_id).first()
        if existing:
            existing.updated_at = datetime.now().isoformat()
            existing_meta = json.loads(existing.metadata_json) if existing.metadata_json else {}
            existing.metadata_json = json.dumps({**existing_meta, **metadata}, default=str)
            db.commit()
            return _to_dict(existing)

        entry = ModelRegistry(
            model_id=model_id,
            owner_id=owner_id,
            storage_key=storage_key,
            registered_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata_json=json.dumps(metadata, default=str),
            versions_json=json.dumps([{
                "version": "1.0.0",
                "stage": "development",
                "registered_at": datetime.now().isoformat(),
            }]),
        )
        db.add(entry)
        db.commit()
        return _to_dict(entry)
    finally:
        db.close()


def get_model_record(model_id, owner_id):
    """Return the ORM row for a model the user owns, or None."""
    db = SessionLocal()
    try:
        return db.query(ModelRegistry).filter_by(model_id=model_id, owner_id=owner_id).first()
    finally:
        db.close()


def get_model(model_id, owner_id):
    return _to_dict(get_model_record(model_id, owner_id))


def list_models(owner_id, filter_criteria=None):
    """List the user's models, optionally filtered by metadata fields."""
    db = SessionLocal()
    try:
        objs = (
            db.query(ModelRegistry)
            .filter_by(owner_id=owner_id)
            .order_by(ModelRegistry.registered_at.desc())
            .all()
        )
        models = [_to_dict(o) for o in objs]
        if filter_criteria:
            models = [
                m for m in models
                if all(m["metadata"].get(k) == v for k, v in filter_criteria.items())
            ]
        return models
    finally:
        db.close()


def transition_model_stage(model_id, stage, owner_id, version=None):
    """
    Transition a model version to a stage (development, staging, production, archived).
    If version not specified, affects the latest version.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage. Must be one of: {VALID_STAGES}")

    db = SessionLocal()
    try:
        obj = db.query(ModelRegistry).filter_by(model_id=model_id, owner_id=owner_id).first()
        if not obj:
            raise LookupError(f"Model {model_id} not found")

        versions = json.loads(obj.versions_json) if obj.versions_json else []

        if version is None:
            if versions:
                def version_key(v):
                    try:
                        return [int(x) for x in v["version"].split(".")]
                    except (ValueError, KeyError, AttributeError):
                        return [0, 0, 0]
                versions.sort(key=version_key, reverse=True)
                target_version = versions[0]["version"]
            else:
                target_version = "1.0.0"
        else:
            target_version = version

        for v in versions:
            if v["version"] == target_version:
                v["stage"] = stage
                v["transitioned_at"] = datetime.now().isoformat()
                break
        else:
            versions.append({
                "version": target_version,
                "stage": stage,
                "transitioned_at": datetime.now().isoformat(),
            })

        obj.versions_json = json.dumps(versions)
        meta = json.loads(obj.metadata_json) if obj.metadata_json else {}
        meta["stage"] = stage
        obj.metadata_json = json.dumps(meta, default=str)
        obj.updated_at = datetime.now().isoformat()
        db.commit()

        # Keep the MLflow model version's stage and aliases in step, if it has one
        mlflow_tracking.sync_stage(meta.get("mlflow"), stage)

        return _to_dict(obj)
    finally:
        db.close()


def get_production_models(owner_id):
    """Get the user's models currently in production stage"""
    prod = []
    for m in list_models(owner_id):
        for v in m.get("versions", []):
            if v.get("stage") == "production":
                prod.append({"model_id": m["model_id"], "version": v["version"], **m["metadata"]})
                break
    return prod
