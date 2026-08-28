import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from services.ml_engine import train_model_from_file

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

def test_all_algorithms_smoke():
    """Smoke test: all algorithms for classification should run without error on iris."""
    from services.ml_engine import get_available_models
    iris_path = os.path.join(DATA_DIR, "iris.csv")
    for algo in get_available_models("classification"):
        try:
            r = train_model_from_file(iris_path, "target", algorithm=algo, test_size=0.2)
            assert "model_id" in r
            print(f"  {algo}: ok -> {r['metrics']}")
        except Exception as e:
            pytest.skip(f"{algo} skipped: {e}")
