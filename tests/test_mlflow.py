"""MLflow integration, against a throwaway local MLflow store (no server needed)."""
import os

import pytest

import config
from services import mlflow_tracking
from services.ml_engine import compare_models_from_file, train_model_from_file
from services.model_registry import get_model, transition_model_stage

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IRIS = os.path.join(DATA_DIR, "iris.csv")


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Point StatMind at a fresh SQLite-backed MLflow store with artifacts under tmp_path."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT", "statmind-test")
    monkeypatch.setattr(config, "MLFLOW_UI_URL", "http://mlflow.test")

    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    client.create_experiment("statmind-test", artifact_location=(tmp_path / "artifacts").as_uri())
    return client


def test_nothing_is_logged_when_mlflow_is_not_configured():
    assert not mlflow_tracking.enabled()
    result = train_model_from_file(IRIS, "target", algorithm="decision_tree")
    assert result["mlflow"] is None


def test_training_logs_a_run_and_registers_a_version(mlflow_store):
    result = train_model_from_file(IRIS, "target", algorithm="random_forest")
    info = result["mlflow"]

    assert info["registered_model"] == "statmind-target"
    assert info["run_url"] == f"http://mlflow.test/#/experiments/{info['experiment_id']}/runs/{info['run_id']}"

    run = mlflow_store.get_run(info["run_id"])
    assert run.data.params["algorithm"] == "random_forest"
    assert run.data.params["model.n_estimators"] == "120"
    assert run.data.metrics["accuracy"] == result["metrics"]["accuracy"]
    assert run.data.metrics["baseline_accuracy"] == result["baseline_metrics"]["accuracy"]
    assert run.data.tags["statmind.model_id"] == result["model_id"]
    assert run.data.tags["statmind.source"] == "single_model"

    artifacts = {item.path for item in mlflow_store.list_artifacts(info["run_id"])}
    assert {"feature_importance.json", "diagnostics.json", "schema.json"} <= artifacts

    version = mlflow_store.get_model_version("statmind-target", info["version"])
    assert version.tags["statmind.model_id"] == result["model_id"]
    assert version.tags["statmind.stage"] == "development"

    # The StatMind registry remembers where the model lives in MLflow
    assert get_model(result["model_id"], owner_id=None)["metadata"]["mlflow"]["run_id"] == info["run_id"]


def test_logged_model_can_be_loaded_back_and_predicts(mlflow_store):
    import mlflow
    import pandas as pd

    result = train_model_from_file(IRIS, "target", algorithm="logistic_regression")
    info = result["mlflow"]

    model = mlflow.sklearn.load_model(f"models:/{info['registered_model']}/{info['version']}")
    features = pd.read_csv(IRIS).drop(columns=["target"])
    assert len(model.predict(features)) == 150


def test_each_training_run_adds_a_version(mlflow_store):
    first = train_model_from_file(IRIS, "target", algorithm="decision_tree")["mlflow"]
    second = train_model_from_file(IRIS, "target", algorithm="knn")["mlflow"]
    assert first["registered_model"] == second["registered_model"]
    assert int(second["version"]) == int(first["version"]) + 1


def test_automl_logs_every_candidate_as_a_child_run(mlflow_store):
    result = compare_models_from_file(IRIS, "target")
    info = result["winner"]["mlflow"]

    parent = mlflow_store.get_run(info["run_id"])
    assert parent.data.tags["statmind.source"] == "automl"
    assert "cv_f1_macro_mean" in parent.data.metrics

    children = mlflow_store.search_runs(
        [info["experiment_id"]], filter_string=f"tags.`mlflow.parentRunId` = '{info['run_id']}'"
    )
    assert {run.data.params["algorithm"] for run in children} == {row["algorithm"] for row in result["leaderboard"]}

    roles = {run.data.params["algorithm"]: run.data.tags["statmind.role"] for run in children}
    assert roles["baseline"] == "baseline"
    assert all("cv_f1_macro_mean" in run.data.metrics for run in children)


def test_stage_changes_move_the_mlflow_aliases(mlflow_store):
    first = train_model_from_file(IRIS, "target", algorithm="decision_tree")
    second = train_model_from_file(IRIS, "target", algorithm="knn")
    name = first["mlflow"]["registered_model"]

    transition_model_stage(first["model_id"], "production", owner_id=None)
    assert str(mlflow_store.get_model_version_by_alias(name, "production").version) == first["mlflow"]["version"]
    assert mlflow_store.get_model_version(name, first["mlflow"]["version"]).tags["statmind.stage"] == "production"

    # Promoting a newer model takes the alias over
    transition_model_stage(second["model_id"], "production", owner_id=None)
    assert str(mlflow_store.get_model_version_by_alias(name, "production").version) == second["mlflow"]["version"]

    # Archiving the production model releases the alias
    transition_model_stage(second["model_id"], "archived", owner_id=None)
    with pytest.raises(Exception):
        mlflow_store.get_model_version_by_alias(name, "production")


def test_an_unreachable_mlflow_server_does_not_break_training(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")

    result = train_model_from_file(IRIS, "target", algorithm="decision_tree")

    assert result["mlflow"] is None
    assert result["model_id"].endswith(".pkl")
    assert get_model(result["model_id"], owner_id=None) is not None
