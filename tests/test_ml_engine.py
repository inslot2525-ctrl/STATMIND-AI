import os

import pytest
from services.ml_engine import get_available_models, train_model_from_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def test_iris_classification():
    """Iris: 3-class, should get accuracy > 0.85 with random_forest."""
    result = train_model_from_file(
        os.path.join(DATA_DIR, "iris.csv"),
        target_column="target",
        algorithm="random_forest",
        test_size=0.2
    )
    assert "metrics" in result
    # For classification, check accuracy or f1
    m = result["metrics"]
    score = m.get("accuracy") or m.get("f1_score") or 0
    print(f"\n[iris] metrics: {m}")
    assert score > 0.85, f"iris score too low: {score}"

def test_breast_cancer_classification():
    """Breast cancer: binary, should get accuracy > 0.90."""
    result = train_model_from_file(
        os.path.join(DATA_DIR, "breast_cancer.csv"),
        target_column="target",
        algorithm="random_forest",
        test_size=0.2
    )
    m = result["metrics"]
    score = m.get("accuracy") or m.get("f1_score") or 0
    print(f"\n[breast_cancer] metrics: {m}")
    assert score > 0.90, f"breast_cancer score too low: {score}"

def test_diabetes_regression():
    """Diabetes: regression, should get R2 > 0.30."""
    result = train_model_from_file(
        os.path.join(DATA_DIR, "diabetes.csv"),
        target_column="target",
        algorithm="random_forest",
        test_size=0.2
    )
    m = result["metrics"]
    r2 = m.get("r2_score") or m.get("r2") or 0
    print(f"\n[diabetes] metrics: {m}")
    assert r2 > 0.30, f"diabetes R2 too low: {r2}"

@pytest.mark.parametrize("algo", list(get_available_models("classification")))
def test_classification_algorithms_smoke(algo):
    """Every classification algorithm must train on iris. A failure is a real failure, not a skip."""
    r = train_model_from_file(os.path.join(DATA_DIR, "iris.csv"), "target", algorithm=algo, test_size=0.2)
    assert r["model_id"].endswith(".pkl")
    assert 0 <= r["metrics"]["accuracy"] <= 1


@pytest.mark.parametrize("algo", list(get_available_models("regression")))
def test_regression_algorithms_smoke(algo):
    """Every regression algorithm must train on diabetes."""
    r = train_model_from_file(os.path.join(DATA_DIR, "diabetes.csv"), "target", algorithm=algo, test_size=0.2)
    assert r["model_id"].endswith(".pkl")
    assert "r2_score" in r["metrics"]


def test_model_id_is_filesystem_safe():
    """A target column with path characters must not leak into the model file name."""
    import pandas as pd
    import tempfile

    df = pd.read_csv(os.path.join(DATA_DIR, "iris.csv")).rename(columns={"target": "../evil name"})
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as tmp:
        df.to_csv(tmp, index=False)
    try:
        r = train_model_from_file(tmp.name, "../evil name", algorithm="decision_tree")
    finally:
        os.remove(tmp.name)
    assert "/" not in r["model_id"] and "\\" not in r["model_id"] and ".." not in r["model_id"]
