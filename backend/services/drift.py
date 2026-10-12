"""
Drift monitoring: has the data a model is being asked about moved away from
the data it was trained on?

Reference = the model's training dataset. Current = the inputs logged from its
recent predictions (see prediction_log.py). Evidently runs a statistical test
per column and this module turns the result into a short verdict, plus a full
HTML report on request. The model's own outputs are compared too: predicted
values now against the target values it was trained on.

Drift is a warning, not a measurement of accuracy. A column can drift without
hurting the model, and a model can degrade with no input drift at all. Real
accuracy needs the true outcomes, which arrive later if at all.
"""
import logging

import pandas as pd

from services import prediction_log
from services.artifacts import create_artifact
from services.ml_engine import load_model_artifact
from services.serving import training_frame

logger = logging.getLogger(__name__)

MIN_CURRENT_ROWS = 30
MAX_REFERENCE_ROWS = 5000
PREDICTION_COLUMN = "prediction"
WINDOWS = ("latest", "all")


def _frames(model_id: str, owner_id: str, window: str):
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {list(WINDOWS)}.")

    metadata = load_model_artifact(model_id, owner_id)["metadata"]
    numeric = list(metadata.get("numeric_features", []))
    categorical = list(metadata.get("categorical_features", []))
    target = metadata["target_column"]

    reference_source = training_frame(model_id, owner_id)
    if reference_source is None:
        raise ValueError("The dataset this model was trained on is no longer available, so there is nothing to compare against.")

    current_features, predictions, batches = prediction_log.load_logged(
        owner_id, model_id, window, min_rows=MIN_CURRENT_ROWS
    )
    if len(current_features) < MIN_CURRENT_ROWS:
        raise ValueError(
            f"Only {len(current_features)} prediction rows have been logged for this model; "
            f"at least {MIN_CURRENT_ROWS} are needed for a meaningful comparison. "
            "Make predictions with this model first (Prediction Studio or the API)."
        )

    columns = [column for column in numeric + categorical if column in reference_source.columns]

    reference = reference_source[columns].copy()
    current = current_features.reindex(columns=columns).copy()

    # The model's output: what it was trained to produce vs what it produces now
    reference[PREDICTION_COLUMN] = reference_source[target].values
    current[PREDICTION_COLUMN] = predictions

    if metadata["task_type"] == "regression":
        numeric = numeric + [PREDICTION_COLUMN]
    else:
        categorical = categorical + [PREDICTION_COLUMN]

    for frame in (reference, current):
        for column in numeric:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in categorical:
            if column in frame.columns:
                frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value))

    reference = reference.dropna(subset=[PREDICTION_COLUMN])
    if len(reference) > MAX_REFERENCE_ROWS:
        reference = reference.sample(n=MAX_REFERENCE_ROWS, random_state=42)

    numeric = [column for column in numeric if column in reference.columns]
    categorical = [column for column in categorical if column in reference.columns]

    return reference, current, numeric, categorical, batches, metadata


def _run(reference, current, numeric, categorical):
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)

    return Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )


def _column_results(snapshot) -> list:
    """
    Evidently's per-column tests as plain rows.

    Each column gets its own test at the 0.05 level, so with many columns some
    will "drift" by chance alone: 20 unchanged columns give a false alarm about
    two times in three. The p-value threshold is therefore divided by the number
    of columns tested (a Bonferroni correction). Evidently's own HTML report
    shows the uncorrected tests.
    """
    tests = [
        metric for metric in snapshot.dict()["metrics"]
        if str(metric.get("config", {}).get("type", "")).endswith("ValueDrift")
    ]
    columns = []

    for metric in tests:
        config = metric["config"]
        method, threshold, score = config["method"], float(config["threshold"]), float(metric["value"])
        # A p-value signals drift when it is small; a distance when it is large
        is_p_value = "p_value" in method
        if is_p_value:
            threshold = threshold / len(tests)
        drifted = score < threshold if is_p_value else score >= threshold

        columns.append(
            {
                "column": config["column"],
                "method": method,
                "score": round(score, 6),
                "threshold": round(threshold, 6),
                "drifted": bool(drifted),
                "score_kind": "p-value" if is_p_value else "distance",
            }
        )

    return columns


def check_drift(model_id: str, owner_id: str, window: str = "latest") -> dict:
    reference, current, numeric, categorical, batches, metadata = _frames(model_id, owner_id, window)
    columns = _column_results(_run(reference, current, numeric, categorical))

    prediction = next((row for row in columns if row["column"] == PREDICTION_COLUMN), None)
    features = [row for row in columns if row["column"] != PREDICTION_COLUMN]
    drifted = [row["column"] for row in features if row["drifted"]]
    share = len(drifted) / len(features) if features else 0.0

    if not drifted and not (prediction and prediction["drifted"]):
        status = "stable"
        verdict = "No drift detected: recent inputs look like the training data."
    elif share >= 0.5:
        status = "drift"
        verdict = (
            f"{len(drifted)} of {len(features)} input columns have drifted. "
            "The model is being used on data unlike its training data; consider retraining."
        )
    else:
        status = "warning"
        parts = []
        if drifted:
            parts.append(f"{len(drifted)} of {len(features)} input columns have drifted ({', '.join(drifted[:5])})")
        if prediction and prediction["drifted"]:
            parts.append("the mix of predictions differs from the training target")
        verdict = "; ".join(parts).capitalize() + ". Check whether these changes are expected."

    return {
        "model_id": model_id,
        "target_column": metadata["target_column"],
        "window": window,
        "status": status,
        "verdict": verdict,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "batches_compared": batches,
        "drifted_columns": drifted,
        "drifted_share": round(share, 4),
        "prediction_drift": prediction,
        # Most suspicious first
        "columns": sorted(features, key=lambda row: (not row["drifted"], row["column"])),
        "skipped_columns": list(metadata.get("datetime_features", [])),
    }


def drift_report_artifact(model_id: str, owner_id: str, window: str = "latest") -> dict:
    """The full Evidently report (distributions per column) as a downloadable HTML file."""
    reference, current, numeric, categorical, _batches, _metadata = _frames(model_id, owner_id, window)
    html = _run(reference, current, numeric, categorical).get_html_str(as_iframe=False)

    return create_artifact(
        owner_id=owner_id,
        kind="report",
        data=html.encode("utf-8"),
        extension=".html",
        download_name=f"drift_report_{model_id.rsplit('.', 1)[0]}.html",
    )
