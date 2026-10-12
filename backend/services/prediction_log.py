"""
A record of what each model was asked to predict, and what it answered.

Every prediction, from the API or from an uploaded file, is logged as one
batch. Drift monitoring compares these inputs with the data the model was
trained on. Large batches are sampled, so one big file cannot fill the table.
"""
import json
import uuid

import numpy as np
import pandas as pd

from database import PredictionLog, SessionLocal, utcnow

MAX_LOGGED_ROWS_PER_BATCH = 2000


def log_predictions(owner_id: str, model_id: str, source: str, features: pd.DataFrame, predictions) -> None:
    """Store one batch. owner_id=None (offline use, no user) is not logged."""
    if owner_id is None or len(features) == 0:
        return

    frame = features.copy()
    frame["__prediction__"] = list(predictions)
    total_rows = len(frame)

    if total_rows > MAX_LOGGED_ROWS_PER_BATCH:
        frame = frame.sample(n=MAX_LOGGED_ROWS_PER_BATCH, random_state=42)

    predicted = frame.pop("__prediction__")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    db = SessionLocal()
    try:
        db.add(
            PredictionLog(
                id=uuid.uuid4().hex,
                owner_id=owner_id,
                model_id=model_id,
                source=source,
                rows=total_rows,
                logged_rows=len(frame),
                # to_json writes NaN as null and dates as ISO strings
                features=frame.to_json(orient="records", date_format="iso"),
                predictions=json.dumps(predicted.tolist(), default=str),
                created_at=utcnow(),
            )
        )
        db.commit()
    finally:
        db.close()


def summary(owner_id: str, model_id: str) -> dict:
    """How much has been logged for a model."""
    db = SessionLocal()
    try:
        logs = (
            db.query(PredictionLog.rows, PredictionLog.source, PredictionLog.created_at)
            .filter_by(owner_id=owner_id, model_id=model_id)
            .order_by(PredictionLog.created_at.desc())
            .all()
        )
    finally:
        db.close()

    return {
        "batches": len(logs),
        "rows": int(sum(log.rows for log in logs)),
        "api_rows": int(sum(log.rows for log in logs if log.source == "api")),
        "file_rows": int(sum(log.rows for log in logs if log.source == "file")),
        "last_prediction_at": logs[0].created_at.isoformat() if logs else None,
    }


def load_logged(owner_id: str, model_id: str, window: str = "latest", min_rows: int = 1, max_rows: int = 5000):
    """
    Logged inputs and predictions for a model, newest first.

    window="latest" returns the most recent batch, reaching back to earlier
    batches only while it has fewer than min_rows (one API call is a batch of
    one row, too few to compare). "all" returns everything up to max_rows.
    Returns (features DataFrame, predictions list, batches used).
    """
    db = SessionLocal()
    try:
        query = (
            db.query(PredictionLog)
            .filter_by(owner_id=owner_id, model_id=model_id)
            .order_by(PredictionLog.created_at.desc())
        )
        logs = query.limit(200).all()
    finally:
        db.close()

    frames, predictions, batches = [], [], 0

    for log in logs:
        rows = json.loads(log.features)
        if not rows:
            continue
        frames.append(pd.DataFrame(rows))
        predictions.extend(json.loads(log.predictions))
        batches += 1
        collected = sum(len(frame) for frame in frames)
        if collected >= max_rows or (window == "latest" and collected >= min_rows):
            break

    if not frames:
        return pd.DataFrame(), [], 0

    features = pd.concat(frames, ignore_index=True).head(max_rows)
    return features, predictions[: len(features)], batches
