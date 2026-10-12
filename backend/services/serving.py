"""
Online prediction: score JSON records with a saved model.

This is what gives the registry's "production" stage a meaning: a caller can
ask for "the production model for <target>" without knowing a model id, and
promoting a different model changes what that call uses.
"""
import json
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

from services import prediction_log
from services.dataset_registry import dataset_local_path, get_dataset
from services.file_reader import read_uploaded_file
from services.ml_engine import decode_classification_predictions, load_model_artifact
from services.model_registry import get_model, list_models

MAX_RECORDS_PER_REQUEST = 1000

# Unpickling a model on every request would dominate the latency. Saved models
# never change, so a few stay in memory, keyed by owner as well as id.
_CACHE_SIZE = 8
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def _load(model_id: str, owner_id: str) -> dict:
    key = (owner_id, model_id)

    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    artifact = load_model_artifact(model_id, owner_id)  # checks ownership

    with _cache_lock:
        _cache[key] = artifact
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)

    return artifact


def production_model_id(owner_id: str, target_column: str) -> str:
    """The model currently in production for a target column."""
    target_column = target_column.strip()

    for model in list_models(owner_id):
        stage = (model.get("versions") or [{}])[0].get("stage")
        if stage == "production" and model["metadata"].get("target_column") == target_column:
            return model["model_id"]

    raise LookupError(
        f"No model is in production for target '{target_column}'. "
        "Promote one in MLOps Studio first."
    )


def _records_to_frame(records: list, metadata: dict) -> pd.DataFrame:
    if not records:
        raise ValueError("Send at least one record.")
    if len(records) > MAX_RECORDS_PER_REQUEST:
        raise ValueError(f"Send at most {MAX_RECORDS_PER_REQUEST} records per request.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Each record must be a JSON object of column name to value.")

    frame = pd.DataFrame(records)
    frame.columns = [str(column).strip() for column in frame.columns]

    features = metadata["features_used"]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Records are missing required columns: {missing}")

    frame = frame[features].copy()  # extra columns are ignored

    # JSON often carries numbers as strings; a numeric feature must be numeric
    for column in metadata.get("numeric_features", []):
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = frame[column].notna() & converted.isna()
        if bad.any():
            raise ValueError(
                f"Column '{column}' must be numeric; got {frame.loc[bad, column].iloc[0]!r}."
            )
        frame[column] = converted

    return frame


def predict_records(model_id: str, owner_id: str, records: list, source: str = "api") -> dict:
    artifact = _load(model_id, owner_id)
    pipeline, metadata = artifact["pipeline"], artifact["metadata"]

    frame = _records_to_frame(records, metadata)
    raw = pipeline.predict(frame)

    result = {
        "model_id": model_id,
        "task_type": metadata["task_type"],
        "target_column": metadata["target_column"],
        "rows": int(len(frame)),
    }

    if metadata["task_type"] == "classification":
        label_mapping = metadata.get("label_mapping") or {}
        predictions = decode_classification_predictions(raw, label_mapping)
        result["predictions"] = predictions

        model = pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            probabilities = pipeline.predict_proba(frame)
            labels = [label_mapping.get(str(int(class_id)), str(class_id)) for class_id in model.classes_]
            result["probabilities"] = [
                {label: round(float(p), 4) for label, p in zip(labels, row)}
                for row in probabilities
            ]
    else:
        predictions = [round(float(value), 6) for value in np.asarray(raw, dtype=float)]
        result["predictions"] = predictions

    prediction_log.log_predictions(owner_id, model_id, source, frame, predictions)

    return result


def serving_info(model_id: str, owner_id: str) -> dict:
    """What a caller needs to use a model: its inputs, and a real example record."""
    metadata = _load(model_id, owner_id)["metadata"]
    features = metadata["features_used"]

    # A real row from the training data makes a better example than placeholders
    example = {column: None for column in features}
    frame = training_frame(model_id, owner_id)

    if frame is not None and len(frame) and all(column in frame.columns for column in features):
        complete = frame[features].dropna().head(1)
        example = _first_record(complete if len(complete) else frame[features].head(1))

    return {
        "model_id": model_id,
        "task_type": metadata["task_type"],
        "target_column": metadata["target_column"],
        "features": [
            {
                "name": column,
                "type": (
                    "number" if column in metadata.get("numeric_features", [])
                    else "date" if column in metadata.get("datetime_features", [])
                    else "category"
                ),
            }
            for column in features
        ],
        "example_record": example,
        "max_records_per_request": MAX_RECORDS_PER_REQUEST,
        "predictions_logged": prediction_log.summary(owner_id, model_id),
    }


def _first_record(row: pd.DataFrame) -> dict:
    """One row as plain JSON values (no numpy types, NaN as None)."""
    return json.loads(row.to_json(orient="records", date_format="iso"))[0]


def training_frame(model_id: str, owner_id: str) -> pd.DataFrame | None:
    """The dataset a model was trained on, or None if it is no longer available."""
    model = get_model(model_id, owner_id)
    dataset_id = (model or {}).get("metadata", {}).get("dataset_id")
    dataset = get_dataset(dataset_id, owner_id) if dataset_id else None

    if not dataset:
        return None

    try:
        frame = read_uploaded_file(dataset_local_path(dataset))
    except ValueError:
        return None

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame
