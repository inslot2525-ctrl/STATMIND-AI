import itertools
import pandas as pd
import numpy as np

from services.file_reader import read_uploaded_file


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
    col_lower = str(column_name).lower().strip()

    if any(keyword in col_lower for keyword in BAD_FEATURE_KEYWORDS):
        return True

    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

    if unique_ratio > 0.95 and series.dtype == "object":
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

    max_pairs = 40
    pairs_checked = 0

    for col_a, col_b in itertools.combinations(numeric_columns, 2):
        if pairs_checked >= max_pairs:
            break

        pairs_checked += 1

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


def detect_leakage_from_file(file_path: str, target_column: str):
    """
    Main leakage detector.

    Detects:
    - feature columns that look like identifiers
    - target-like feature names
    - features with extremely high correlation to target
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

    for feature in feature_columns:
        series = df[feature]

        if _is_identifier_like(feature, series):
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

        feature_lower = str(feature).lower()

        if any(keyword in feature_lower for keyword in LEAKAGE_KEYWORDS):
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

        corr = _safe_corr(series, target)

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

    findings.extend(_detect_arithmetic_relationship(df, target_column))

    severity_score = 0

    for item in findings:
        severity = item.get("severity")

        if severity == "Critical":
            severity_score += 40
        elif severity == "High":
            severity_score += 25
        elif severity == "Medium":
            severity_score += 10
        else:
            severity_score += 5

    leakage_risk_score = min(100, severity_score)

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
        "warnings": warnings,
    }