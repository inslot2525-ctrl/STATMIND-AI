import itertools
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from services.file_reader import read_uploaded_file
from services.ml_engine import detect_datetime_columns, detect_task_type


BAD_FEATURE_KEYWORDS = [
    "id",
    "name",
    "email",
    "phone",
    "mobile",
    "address",
    "url",
    "username",
    "password",
    "hash",
    "uuid",
]


LEAKAGE_KEYWORDS = [
    "final",
    "result",
    "outcome",
    "status",
    "approved",
    "rejected",
    "label",
    "target",
    "class",
    "prediction",
]


# Single-feature check: a feature that alone predicts the target this well is suspicious
SOLO_SCORE_THRESHOLD = 0.97
SOLO_MAX_ROWS = 5000
SOLO_MAX_FEATURES = 60

# Arithmetic check: only the columns most correlated with the target are paired up
ARITHMETIC_MAX_COLUMNS = 9

SEVERITY_POINTS = {"Critical": 40, "High": 25, "Medium": 10}


def _name_tokens(column_name: str) -> set:
    """Words in a column name: 'customerID' and 'customer_id' both give {'customer', 'id'}."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(column_name))
    return {token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token}


def _safe_corr(a: pd.Series, b: pd.Series):
    try:
        a_numeric = pd.to_numeric(a, errors="coerce")
        b_numeric = pd.to_numeric(b, errors="coerce")

        valid = ~(a_numeric.isna() | b_numeric.isna())

        if valid.sum() < 5:
            return None

        corr = a_numeric[valid].corr(b_numeric[valid])

        if pd.isna(corr):
            return None

        return round(float(corr), 4)

    except Exception:
        return None


def _is_identifier_like(column_name: str, series: pd.Series):
    # Whole-word match: 'customer_id' is an identifier, 'paid' and 'width' are not
    if _name_tokens(column_name) & set(BAD_FEATURE_KEYWORDS):
        return True

    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

    if unique_ratio > 0.95 and not is_numeric_dtype(series):
        return True

    return False


def _detect_arithmetic_relationship(df: pd.DataFrame, target_column: str):
    """
    Detect simple target leakage patterns such as:
    Profit = Revenue - Expense
    Total = Price * Quantity
    Balance = Credit - Debit

    This is heuristic, not symbolic proof.
    """

    findings = []

    numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).copy()

    if target_column not in numeric_df.columns:
        return findings

    numeric_columns = [col for col in numeric_df.columns if col != target_column]
    target = pd.to_numeric(numeric_df[target_column], errors="coerce")

    if len(numeric_columns) < 2:
        return findings

    # Checking every pair is quadratic, so pair up the columns most related to the target
    numeric_columns = sorted(
        numeric_columns,
        key=lambda col: abs(_safe_corr(numeric_df[col], target) or 0),
        reverse=True,
    )[:ARITHMETIC_MAX_COLUMNS]

    for col_a, col_b in itertools.combinations(numeric_columns, 2):
        a = pd.to_numeric(numeric_df[col_a], errors="coerce")
        b = pd.to_numeric(numeric_df[col_b], errors="coerce")

        candidates = {
            f"{col_a} + {col_b}": a + b,
            f"{col_a} - {col_b}": a - b,
            f"{col_b} - {col_a}": b - a,
            f"{col_a} * {col_b}": a * b,
        }

        for formula, candidate in candidates.items():
            valid = ~(target.isna() | candidate.isna() | np.isinf(candidate))

            if valid.sum() < 10:
                continue

            diff = np.abs(target[valid] - candidate[valid])
            target_scale = np.maximum(np.abs(target[valid]), 1)

            relative_error = diff / target_scale
            match_ratio = float((relative_error < 0.01).mean())

            if match_ratio >= 0.85:
                findings.append(
                    {
                        "type": "arithmetic_leakage",
                        "severity": "High",
                        "feature": formula,
                        "message": f"Target '{target_column}' appears mathematically derived from {formula}.",
                        "evidence": f"{round(match_ratio * 100, 2)}% of rows match this relationship within 1% relative error.",
                        "recommendation": "Do not treat this as a genuine predictive model unless this relationship is expected and acceptable.",
                    }
                )

    return findings


def _solo_feature_scores(df: pd.DataFrame, target_column: str, feature_columns: list) -> dict:
    """
    How well each feature predicts the target on its own: a shallow decision
    tree per feature, scored by 3-fold cross-validation (balanced accuracy for
    classification, R² for regression). Unlike correlation this also covers
    categorical features and non-linear relationships.
    """
    data = df.dropna(subset=[target_column])

    if len(data) > SOLO_MAX_ROWS:
        data = data.sample(n=SOLO_MAX_ROWS, random_state=42)

    target = data[target_column]
    task_type = detect_task_type(target)

    if task_type == "classification":
        y = target.astype(str)
        class_counts = y.value_counts()

        if len(class_counts) < 2 or class_counts.min() < 3:
            return {}

        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        scoring = "balanced_accuracy"
    else:
        y = pd.to_numeric(target, errors="coerce")

        if y.isna().any() or len(y) < 30:
            return {}

        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        scoring = "r2"

    scores = {}

    for feature in feature_columns[:SOLO_MAX_FEATURES]:
        series = data[feature]

        if is_numeric_dtype(series) and not is_bool_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
            values = values.fillna(values.median())
        else:
            values = pd.Series(pd.factorize(series.astype(str))[0], index=series.index)

        if values.isna().all() or values.nunique() < 2:
            continue

        try:
            fold_scores = cross_val_score(
                model, values.to_numpy(dtype=float).reshape(-1, 1), y, cv=3, scoring=scoring
            )
        except Exception:
            continue

        if not np.isnan(fold_scores).any():
            scores[feature] = round(float(fold_scores.mean()), 4)

    return {"metric": scoring, "scores": scores}


def detect_leakage_from_file(file_path: str, target_column: str):
    """
    Main leakage detector.

    Detects:
    - feature columns that look like identifiers
    - target-like feature names
    - features with extremely high correlation to target
    - features that alone predict the target almost perfectly (cross-validated)
    - simple arithmetic leakage
    - duplicated target columns
    """

    df = read_uploaded_file(file_path)
    df.columns = [str(col).strip() for col in df.columns]

    target_column = str(target_column).strip()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}"
        )

    findings = []
    warnings = []

    if len(df) < 20:
        warnings.append(
            "Dataset is small. Leakage detection may be unreliable because there are too few rows."
        )

    target = df[target_column]

    feature_columns = [col for col in df.columns if col != target_column]
    date_columns = set(detect_datetime_columns(df[feature_columns]))

    # Pearson correlation with integer class codes means nothing beyond two classes
    correlation_applies = (
        detect_task_type(target.dropna()) == "regression" or target.nunique(dropna=True) == 2
    )

    for feature in feature_columns:
        series = df[feature]

        if feature not in date_columns and _is_identifier_like(feature, series):
            findings.append(
                {
                    "type": "identifier_feature",
                    "severity": "Medium",
                    "feature": feature,
                    "message": f"Feature '{feature}' looks like an ID, name, contact field, or high-cardinality identifier.",
                    "evidence": f"Unique values: {series.nunique(dropna=True)} out of {len(series)} rows.",
                    "recommendation": "Remove this feature unless it has real predictive meaning.",
                }
            )

        if _name_tokens(feature) & set(LEAKAGE_KEYWORDS):
            findings.append(
                {
                    "type": "target_like_feature_name",
                    "severity": "High",
                    "feature": feature,
                    "message": f"Feature '{feature}' has a name that may indicate future outcome information.",
                    "evidence": "Column name contains words like result, final, outcome, status, approved, label, or target.",
                    "recommendation": "Check whether this column is known before prediction time. If not, remove it.",
                }
            )

        if series.equals(target):
            findings.append(
                {
                    "type": "duplicate_target",
                    "severity": "Critical",
                    "feature": feature,
                    "message": f"Feature '{feature}' is identical to target '{target_column}'.",
                    "evidence": "The feature values exactly match the target values.",
                    "recommendation": "Remove this feature immediately. It causes direct target leakage.",
                }
            )

        corr = _safe_corr(series, target) if correlation_applies else None

        if corr is not None and abs(corr) >= 0.95:
            findings.append(
                {
                    "type": "high_correlation",
                    "severity": "High",
                    "feature": feature,
                    "message": f"Feature '{feature}' has extremely high correlation with target '{target_column}'.",
                    "evidence": f"Correlation = {corr}.",
                    "recommendation": "Investigate whether this feature is derived from the target or unavailable at prediction time.",
                }
            )

    solo = _solo_feature_scores(df, target_column, feature_columns)

    for feature, score in solo.get("scores", {}).items():
        if score >= SOLO_SCORE_THRESHOLD:
            findings.append(
                {
                    "type": "single_feature_predicts_target",
                    "severity": "Critical" if score >= 0.999 else "High",
                    "feature": feature,
                    "message": f"Feature '{feature}' predicts target '{target_column}' almost perfectly on its own.",
                    "evidence": f"A shallow decision tree using only this feature scores cross-validated {solo['metric'].replace('_', ' ')} = {score}.",
                    "recommendation": "Confirm this value is known before the outcome happens. If it is recorded afterwards or derived from the target, remove it.",
                }
            )

    findings.extend(_detect_arithmetic_relationship(df, target_column))

    # One feature can trip several checks; it counts once, at its worst severity
    worst_by_feature = {}

    for item in findings:
        points = SEVERITY_POINTS.get(item.get("severity"), 5)
        worst_by_feature[item["feature"]] = max(worst_by_feature.get(item["feature"], 0), points)

    leakage_risk_score = min(100, sum(worst_by_feature.values()))

    if leakage_risk_score >= 70:
        risk_level = "Critical"
    elif leakage_risk_score >= 40:
        risk_level = "High"
    elif leakage_risk_score >= 20:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    summary = [
        f"Leakage risk level: {risk_level}.",
        f"Leakage risk score: {leakage_risk_score}/100.",
        f"Total findings: {len(findings)}.",
    ]

    if risk_level in ["Critical", "High"]:
        summary.append(
            "High model accuracy may be misleading because one or more features may expose target information."
        )
    elif risk_level == "Medium":
        summary.append(
            "Some suspicious features were found. Review them before trusting model metrics."
        )
    else:
        summary.append(
            "No major leakage pattern was detected by the current heuristic checks."
        )

    return {
        "target_column": target_column,
        "rows_checked": int(len(df)),
        "columns_checked": int(len(df.columns)),
        "leakage_risk_score": leakage_risk_score,
        "risk_level": risk_level,
        "summary": summary,
        "findings": findings,
        "single_feature_scores": {
            "metric": solo.get("metric"),
            "top": sorted(solo.get("scores", {}).items(), key=lambda item: item[1], reverse=True)[:10],
        },
        "warnings": warnings,
    }