"""AutoML, task detection, preprocessing and explainability."""
import os

import numpy as np
import pandas as pd
import pytest

from services.ml_engine import (
    BASELINE_ALGORITHM,
    build_preprocessor,
    compare_models_from_file,
    detect_task_type,
    drop_high_cardinality_columns,
    load_model_artifact,
    train_model_from_file,
)
from tests.conftest import upload

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def iris_automl():
    return compare_models_from_file(os.path.join(DATA_DIR, "iris.csv"), "target")


def _write_csv(tmp_path, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


# ---- AutoML ---------------------------------------------------------------

def test_leaderboard_is_cross_validated(iris_automl):
    assert iris_automl["ranking_metric"] == "f1_macro"
    assert iris_automl["cv_folds"] == 5

    scored = [row for row in iris_automl["leaderboard"] if row["score"] is not None]
    assert len(scored) >= 6
    for row in scored:
        assert 0 <= row["cv_mean"] <= 1 and row["cv_std"] >= 0

    # Sorted best first
    assert [row["score"] for row in scored] == sorted((row["score"] for row in scored), reverse=True)


def test_leaderboard_has_a_baseline_that_the_winner_beats(iris_automl):
    baseline = next(row for row in iris_automl["leaderboard"] if row["is_baseline"])
    winner = next(row for row in iris_automl["leaderboard"] if row["algorithm"] == iris_automl["best_model"])

    assert baseline["algorithm"] == BASELINE_ALGORITHM
    assert iris_automl["best_model"] != BASELINE_ALGORITHM
    assert winner["cv_mean"] > baseline["cv_mean"] + 0.3
    assert iris_automl["test_metrics"]["accuracy"] > iris_automl["baseline_metrics"]["accuracy"]


def test_top_models_are_tuned(iris_automl):
    tuned_rows = [row for row in iris_automl["leaderboard"] if "tuning" in row]
    assert len(tuned_rows) == 2
    for row in tuned_rows:
        # Tuning may only ever keep or improve the default configuration's score
        assert row["cv_mean"] >= row["tuning"]["default_cv_mean"]
        assert row["tuned"] == bool(row["best_params"])


def test_winner_is_scored_on_the_held_out_split(iris_automl):
    winner = iris_automl["winner"]
    assert winner["algorithm"] == iris_automl["best_model"]
    assert winner["train_rows"] + winner["test_rows"] == winner["rows_used"] == 150
    assert winner["cross_validation"]["folds"] == 5
    assert iris_automl["test_metrics"]["accuracy"] > 0.85


def test_automl_job_saves_a_model_usable_for_prediction(client, alice):
    body = upload(client, alice, "breast_cancer.csv")

    r = client.post(
        "/api/compare-models",
        headers=alice,
        data={"dataset_id": body["dataset_id"], "target_column": "target"},
    )
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "succeeded", job
    model_id = job["result"]["model_id"]

    # Registered to the user, with an experiment
    assert client.get(f"/api/models/{model_id}", headers=alice).status_code == 200
    experiments = client.get(f"/api/experiments/model/{model_id}", headers=alice).json()["experiments"]
    assert len(experiments) == 1
    assert experiments[0]["tags"] == ["automl"]
    assert "cross_validation" in experiments[0]["metrics"]

    with open(os.path.join(DATA_DIR, "breast_cancer.csv"), "rb") as f:
        r = client.post(
            "/api/predict-test-data",
            headers=alice,
            data={"model_id": model_id},
            files={"file": ("breast_cancer.csv", f.read(), "text/csv")},
        )
    assert r.status_code == 200, r.text
    assert r.json()["rows_predicted"] == 569


def test_winner_model_can_be_loaded(iris_automl):
    artifact = load_model_artifact(iris_automl["model_id"], owner_id=None)
    assert artifact["metadata"]["algorithm"] == iris_automl["best_model"]


def test_regression_automl_ranks_by_r2():
    result = compare_models_from_file(os.path.join(DATA_DIR, "diabetes.csv"), "target")
    assert result["task_type"] == "regression"
    assert result["ranking_metric"] == "r2"
    assert result["test_metrics"]["r2_score"] > 0.3
    assert result["test_metrics"]["rmse"] < result["baseline_metrics"]["rmse"]
    assert result["winner"]["diagnostics"]["predicted_vs_actual"]


# ---- task type ------------------------------------------------------------

def test_digit_labels_are_classification():
    digits = pd.Series(np.tile(np.arange(10), 100))
    assert detect_task_type(digits) == "classification"


def test_many_valued_integers_are_regression():
    ages = pd.Series(np.random.RandomState(0).randint(18, 90, size=300))
    assert detect_task_type(ages) == "regression"


def test_text_labels_are_classification_however_many_classes():
    labels = pd.Series([f"class_{i % 12}" for i in range(240)])
    assert detect_task_type(labels) == "classification"


def test_task_type_override(tmp_path):
    result = train_model_from_file(
        os.path.join(DATA_DIR, "iris.csv"), "target", algorithm="ridge", task_type="regression"
    )
    assert result["task_type"] == "regression"
    assert "r2_score" in result["metrics"]

    with pytest.raises(ValueError):
        detect_task_type(pd.Series([1, 2, 3]), override="clustering")


def test_task_type_is_validated_by_the_api(client, alice):
    body = upload(client, alice)
    r = client.post(
        "/api/train-model",
        headers=alice,
        data={"dataset_id": body["dataset_id"], "target_column": "target",
              "algorithm": "ridge", "task_type": "clustering"},
    )
    assert r.status_code == 400

    r = client.post(
        "/api/train-model",
        headers=alice,
        data={"dataset_id": body["dataset_id"], "target_column": "target",
              "algorithm": "ridge", "task_type": "regression"},
    )
    assert r.status_code == 202 and r.json()["result"]["task_type"] == "regression"


# ---- preprocessing --------------------------------------------------------

def _mixed_frame(rows=120):
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "customer_id": [f"CUST{i:05d}" for i in range(rows)],
        "signup_date": pd.date_range("2022-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
        "region": rng.choice(["north", "south", "east"], size=rows),
        "income": rng.normal(50000, 9000, size=rows),
        "churned": rng.choice(["yes", "no"], size=rows),
    })


def test_dates_become_numeric_parts_not_one_hot_columns():
    X = _mixed_frame().drop(columns=["churned", "customer_id"])
    preprocessor, numeric, categorical, dates = build_preprocessor(X)

    assert dates == ["signup_date"]
    assert numeric == ["income"]
    assert categorical == ["region"]

    transformed = preprocessor.fit_transform(X)
    # income + 3 regions + year/month/dayofweek/days; not one column per date
    assert transformed.shape == (len(X), 8)


def test_id_columns_are_dropped_but_dates_are_kept():
    X, dropped = drop_high_cardinality_columns(_mixed_frame().drop(columns=["churned"]))
    assert dropped == ["customer_id"]
    assert "signup_date" in X.columns


def test_training_with_text_dates_and_missing_values(tmp_path):
    df = _mixed_frame()
    df.loc[::7, "region"] = None
    df.loc[::9, "income"] = None
    df.loc[::11, "signup_date"] = None

    result = train_model_from_file(_write_csv(tmp_path, df), "churned", algorithm="gradient_boosting")

    assert result["task_type"] == "classification"
    assert result["datetime_features"] == ["signup_date"]
    assert result["dropped_high_cardinality_columns"] == ["customer_id"]


# ---- metrics and explainability -------------------------------------------

def test_classification_report_contents():
    result = train_model_from_file(
        os.path.join(DATA_DIR, "breast_cancer.csv"), "target", algorithm="logistic_regression"
    )

    for metric in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc"):
        assert 0 <= result["metrics"][metric] <= 1
    assert result["metrics"]["roc_auc"] > 0.95

    diagnostics = result["diagnostics"]
    assert [row["label"] for row in diagnostics["per_class"]] == ["0", "1"]
    assert sum(row["support"] for row in diagnostics["per_class"]) == result["test_rows"]
    assert np.sum(diagnostics["confusion_matrix"]["matrix"]) == result["test_rows"]

    curve = diagnostics["roc_curves"][0]["points"]
    assert curve[0] == {"fpr": 0.0, "tpr": 0.0} and curve[-1] == {"fpr": 1.0, "tpr": 1.0}

    # The baseline never separates the classes
    assert result["baseline_metrics"]["balanced_accuracy"] == 0.5


@pytest.mark.parametrize("algorithm", ["svm", "knn", "logistic_regression", "random_forest"])
def test_permutation_importance_works_for_every_kind_of_model(algorithm):
    result = train_model_from_file(os.path.join(DATA_DIR, "iris.csv"), "target", algorithm=algorithm)

    importance = result["feature_importance"]
    assert result["feature_importance_method"] == "permutation"
    # Reported per input column, not per encoded column
    assert {item["feature"] for item in importance} == set(result["features_used"])
    assert importance[0]["feature"].startswith("petal")
    assert all("std" in item for item in importance)


def test_imbalanced_target_is_flagged(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x1": rng.normal(size=300), "x2": rng.normal(size=300)})
    df["fraud"] = (df["x1"] > 1.2).astype(int)  # roughly 1 row in 9

    result = train_model_from_file(_write_csv(tmp_path, df), "fraud", algorithm="random_forest")

    assert result["class_balance"]["imbalanced"] is True
    assert result["hyperparameters"]["class_weight"] == "balanced"
    assert any("imbalanced" in warning for warning in result["warnings"])
