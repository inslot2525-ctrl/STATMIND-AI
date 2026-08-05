import os
import pandas as pd

from services.file_reader import read_uploaded_file

BAD_TARGET_KEYWORDS = [
    "id",
    "name",
    "email",
    "phone",
    "mobile",
    "address",
    "url",
    "password",
    "uuid",
]

GOOD_BUSINESS = [
    "profit",
    "revenue",
    "sales",
    "churn",
    "price",
    "rating",
    "score",
    "conversion",
    "cost",
    "income",
]

GOOD_FINANCE = [
    "return",
    "risk",
    "profit",
    "loss",
    "price",
    "close",
    "revenue",
    "expense",
    "default",
    "income",
]

GOOD_RESEARCH = [
    "result",
    "score",
    "outcome",
    "class",
    "label",
    "target",
    "grade",
    "measurement",
]


def _detect_task(series: pd.Series) -> str:
    if (
        series.dtype == "object"
        or series.dtype.name == "category"
        or series.dtype == "bool"
    ):
        return "classification"

    if series.nunique(dropna=True) <= 5:
        return "classification"

    return "regression"


def recommend_target_from_file(file_path: str, domain: str = "business") -> dict:
    if not os.path.exists(file_path):
        raise ValueError("Uploaded dataset not found. Run analysis first.")

    df = read_uploaded_file(file_path)
    df.columns = [str(col).strip() for col in df.columns]

    domain = domain.lower().strip()

    good_keywords = GOOD_BUSINESS

    if domain == "finance":
        good_keywords = GOOD_FINANCE
    elif domain == "research":
        good_keywords = GOOD_RESEARCH

    recommended = []
    rejected = []

    for col in df.columns:
        series = df[col]
        lower = col.lower()

        reasons = []
        warnings = []

        score = 50

        if any(bad in lower for bad in BAD_TARGET_KEYWORDS):
            rejected.append(
                {
                    "column": col,
                    "reason": "Identifier/contact-like column is not a useful prediction target.",
                }
            )
            continue

        missing_ratio = series.isna().mean()
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(len(series), 1)

        if unique_count <= 1:
            rejected.append(
                {
                    "column": col,
                    "reason": "Column has only one unique value.",
                }
            )
            continue

        if missing_ratio > 0.45:
            rejected.append(
                {
                    "column": col,
                    "reason": "Too many missing values for a reliable target.",
                }
            )
            continue

        if any(keyword in lower for keyword in good_keywords):
            score += 25
            reasons.append("Column name matches domain-relevant outcome keywords.")

        if 2 <= unique_count <= 20:
            score += 10
            reasons.append("Target has a manageable number of unique values.")

        if unique_ratio > 0.95 and series.dtype == "object":
            score -= 25
            warnings.append("High-cardinality text target may be hard to model.")

        if pd.api.types.is_numeric_dtype(series):
            score += 8
            reasons.append("Numeric target supports regression or numeric classification.")

        if missing_ratio < 0.05:
            score += 7
            reasons.append("Target has very few missing values.")
        elif missing_ratio > 0:
            warnings.append("Target has missing values; rows with missing target will be removed.")

        task_type = _detect_task(series)

        recommended.append(
            {
                "column": col,
                "score": max(0, min(100, int(score))),
                "task_type": task_type,
                "priority": "High"
                if score >= 75
                else "Medium"
                if score >= 55
                else "Low",
                "reasons": reasons or ["Column is usable as a target."],
                "warnings": warnings,
            }
        )

    recommended = sorted(
        recommended,
        key=lambda x: x["score"],
        reverse=True,
    )

    best = recommended[0] if recommended else None

    summary = []

    if best:
        summary.append(
            f"Best target is '{best['column']}' with score {best['score']}/100."
        )
        summary.append(
            f"Recommended task type is {best['task_type']}."
        )
        summary.append(
            "Review rejected columns before final training."
        )
    else:
        summary.append(
            "No strong target was found. Choose a meaningful outcome column manually."
        )

    return {
        "domain": domain,
        "best_target": best,
        "recommended_targets": recommended,
        "rejected_columns": rejected,
        "advisor_summary": summary,
    }