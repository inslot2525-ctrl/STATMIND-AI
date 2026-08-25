import os
import json
from datetime import datetime
from services.dataset_registry import get_dataset
from database import SessionLocal, Experiment, init_db

init_db()

def _next_id(db):
    ids = [int(r.id) for r in db.query(Experiment.id).all() if r.id.isdigit()]
    return str(max(ids) + 1) if ids else "1"

def _to_dict(obj):
    if not obj:
        return None
    return {
        "id": obj.id,
        "timestamp": obj.timestamp,
        "dataset_id": obj.dataset_id,
        "dataset_info": json.loads(obj.dataset_info) if obj.dataset_info else {},
        "algorithm": obj.algorithm,
        "hyperparameters": json.loads(obj.hyperparameters) if obj.hyperparameters else {},
        "metrics": json.loads(obj.metrics) if obj.metrics else {},
        "model_id": obj.model_id,
        "tags": json.loads(obj.tags) if obj.tags else [],
        "notes": obj.notes or "",
    }

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
    dataset_info = get_dataset(dataset_id)
    if not dataset_info:
        raise ValueError(f"Dataset {dataset_id} not found in registry")

    db = SessionLocal()
    try:
        experiment_id = _next_id(db)
        exp = Experiment(
            id=experiment_id,
            timestamp=datetime.now().isoformat(),
            dataset_id=str(dataset_id),
            dataset_info=json.dumps({
                "filename": dataset_info["original_filename"],
                "rows": dataset_info.get("rows"),
                "columns": dataset_info.get("columns")
            }),
            algorithm=algorithm,
            hyperparameters=json.dumps(hyperparameters or {}),
            metrics=json.dumps(metrics or {}),
            model_id=model_id,
            tags=json.dumps(tags or []),
            notes=notes or ""
        )
        db.add(exp)
        db.commit()
        return experiment_id
    finally:
        db.close()

def get_experiment(experiment_id):
    db = SessionLocal()
    try:
        obj = db.query(Experiment).filter_by(id=str(experiment_id)).first()
        return _to_dict(obj)
    finally:
        db.close()

def list_experiments(dataset_id=None, limit=50):
    db = SessionLocal()
    try:
        q = db.query(Experiment)
        if dataset_id:
            q = q.filter_by(dataset_id=str(dataset_id))
        # SQLite string timestamp ISO sorts lexicographically same as chronological
        objs = q.order_by(Experiment.timestamp.desc()).limit(limit).all()
        return [_to_dict(o) for o in objs]
    finally:
        db.close()

def get_experiments_for_model(model_id):
    db = SessionLocal()
    try:
        objs = db.query(Experiment).filter_by(model_id=model_id).all()
        return [_to_dict(o) for o in objs]
    finally:
        db.close()
