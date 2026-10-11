"""Leakage detector: what it must flag, and what it must leave alone."""
import os

import numpy as np
import pandas as pd

from services.leakage_detector import _name_tokens, detect_leakage_from_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _detect(tmp_path, df, target):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return detect_leakage_from_file(str(path), target)


def _flagged(result, finding_type=None):
    return {
        finding["feature"] for finding in result["findings"]
        if finding_type is None or finding["type"] == finding_type
    }


def _loans(rows=300):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "paid_amount": rng.normal(1000, 200, size=rows),
        "width": rng.normal(10, 2, size=rows),
        "valid_days": rng.randint(1, 400, size=rows),
        "region": rng.choice(["north", "south", "east", "west"], size=rows),
        "signup_date": pd.date_range("2021-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
    })
    df["defaulted"] = np.where(df["paid_amount"] + rng.normal(0, 250, size=rows) < 950, "yes", "no")
    return df


def test_name_matching_is_by_whole_word():
    assert "id" in _name_tokens("customer_id")
    assert "id" in _name_tokens("customerID")
    assert "id" in _name_tokens("Customer Id")
    for harmless in ("paid_amount", "width", "valid_days", "humidity", "video_length"):
        assert "id" not in _name_tokens(harmless)


def test_ordinary_columns_are_not_flagged(tmp_path):
    result = _detect(tmp_path, _loans(), "defaulted")
    assert result["findings"] == []
    assert result["risk_level"] == "Low"


def test_benchmark_datasets_are_clean():
    for name in ("iris.csv", "breast_cancer.csv", "diabetes.csv"):
        result = detect_leakage_from_file(os.path.join(DATA_DIR, name), "target")
        assert result["findings"] == [], name


def test_identifier_column_is_flagged(tmp_path):
    df = _loans()
    df["customer_id"] = [f"CUST{i:05d}" for i in range(len(df))]
    result = _detect(tmp_path, df, "defaulted")
    assert _flagged(result, "identifier_feature") == {"customer_id"}


def test_categorical_copy_of_the_target_is_flagged(tmp_path):
    """Correlation cannot see this one: both columns are text."""
    df = _loans()
    df["collections_flag"] = df["defaulted"].map({"yes": "sent to collections", "no": "current"})
    result = _detect(tmp_path, df, "defaulted")

    assert "collections_flag" in _flagged(result, "single_feature_predicts_target")
    assert result["risk_level"] in ("High", "Critical")
    assert result["single_feature_scores"]["top"][0][0] == "collections_flag"


def test_numeric_proxy_for_the_target_is_flagged(tmp_path):
    df = _loans()
    df["days_overdue"] = np.where(df["defaulted"] == "yes", 90, 0) + np.arange(len(df)) % 5
    result = _detect(tmp_path, df, "defaulted")
    assert "days_overdue" in _flagged(result, "single_feature_predicts_target")


def test_arithmetic_leakage_is_found_among_many_columns(tmp_path):
    """The two columns that make up the target come last, after 12 unrelated ones."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({f"noise_{i}": rng.normal(size=200) for i in range(12)})
    df["revenue"] = rng.normal(5000, 800, size=200)
    df["expense"] = rng.normal(3000, 500, size=200)
    df["profit"] = df["revenue"] - df["expense"]

    result = _detect(tmp_path, df, "profit")
    assert "revenue - expense" in _flagged(result, "arithmetic_leakage")


def test_one_feature_counts_once_towards_the_score(tmp_path):
    df = _loans()
    df["defaulted_copy"] = df["defaulted"]
    result = _detect(tmp_path, df, "defaulted")

    # Trips both the duplicate check and the single-feature check
    assert len([f for f in result["findings"] if f["feature"] == "defaulted_copy"]) == 2
    assert result["leakage_risk_score"] == 40
