"""
Optional MLflow integration, switched on by MLFLOW_TRACKING_URI.

StatMind's own tables stay the source of truth for the app. When MLflow is
configured, every trained model is also logged there as a run (parameters,
metrics, baseline, cross-validation scores, the fitted pipeline) and
registered as a model version, so runs can be compared in the MLflow UI and
models pulled from its registry. An AutoML run logs each candidate as a child
run of the winner.

Nothing here may break training: every public function catches its own
errors, logs them and returns None.
"""
import logging
import os
import re

import config

logger = logging.getLogger(__name__)

# An unreachable server must fail fast instead of stalling a training job on retries
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# MLflow prints a "view run" line with an emoji when a run ends, which raises
# UnicodeEncodeError on a Windows console that is not UTF-8
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")

STAGE_ALIASES = ("staging", "production")


def enabled() -> bool:
    return bool(config.MLFLOW_TRACKING_URI)


def _mlflow():
    """Import lazily: the app must start, and tests must run, without MLflow configured."""
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow


def registered_model_name(target_column: str) -> str:
    """One registered model per prediction target; each training run adds a version."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(target_column)).strip("-") or "target"
    return f"statmind-{slug}"


def _run_url(experiment_id: str, run_id: str) -> str | None:
    if not config.MLFLOW_UI_URL:
        return None
    return f"{config.MLFLOW_UI_URL.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


def _scalar_metrics(metrics: dict, prefix: str = "") -> dict:
    return {
        f"{prefix}{name}": float(value)
        for name, value in (metrics or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _log_candidates(mlflow, leaderboard: list, ranking_metric: str, common_tags: dict) -> None:
    """One child run per AutoML candidate, so the whole leaderboard can be compared in MLflow."""
    for row in leaderboard:
        with mlflow.start_run(run_name=row["algorithm"], nested=True):
            mlflow.set_tags(
                {
                    **common_tags,
                    "statmind.role": "baseline" if row.get("is_baseline") else "candidate",
                    "statmind.tuned": str(bool(row.get("tuned"))).lower(),
                }
            )
            mlflow.log_param("algorithm", row["algorithm"])

            if row.get("best_params"):
                mlflow.log_params({name: str(value) for name, value in row["best_params"].items()})

            if row.get("score") is None:
                mlflow.set_tag("statmind.error", str(row.get("error", ""))[:500])
                continue

            mlflow.log_metrics(
                {
                    f"cv_{ranking_metric}_mean": float(row["cv_mean"]),
                    f"cv_{ranking_metric}_std": float(row["cv_std"]),
                }
            )


def log_training_run(
    pipeline,
    result: dict,
    model_id: str,
    dataset_id: str = None,
    owner_id: str = None,
    leaderboard: list = None,
) -> dict | None:
    """
    Log one trained model. `result` is the dict built by ml_engine for the API;
    `leaderboard` is passed for an AutoML winner. Returns the MLflow identifiers
    to store with the model, or None if MLflow is off or the logging failed.
    """
    if not enabled():
        return None

    try:
        mlflow = _mlflow()
        experiment = mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        cross_validation = result.get("cross_validation") or {}
        name = registered_model_name(result["target_column"])

        tags = {
            "statmind.model_id": model_id,
            "statmind.task_type": result["task_type"],
            "statmind.target": result["target_column"],
            "statmind.source": "automl" if leaderboard else "single_model",
        }
        if dataset_id:
            tags["statmind.dataset_id"] = str(dataset_id)
        if owner_id:
            tags["statmind.owner_id"] = str(owner_id)

        # With lakeFS on: the exact data and model commits behind this run
        data_version = result.get("data_version") or {}
        if data_version.get("dataset_commit"):
            tags["statmind.data_commit"] = data_version["dataset_commit"]
        if data_version.get("model_commit"):
            tags["statmind.model_commit"] = data_version["model_commit"]

        with mlflow.start_run(run_name=f"{result['algorithm']} / {result['target_column']}") as run:
            mlflow.set_tags(tags)

            mlflow.log_params(
                {
                    "algorithm": result["algorithm"],
                    "task_type": result["task_type"],
                    "target_column": result["target_column"],
                    "test_size": result["test_size"],
                    "train_rows": result["train_rows"],
                    "test_rows": result["test_rows"],
                    "n_features": len(result["features_used"]),
                    **{f"model.{key}": str(value) for key, value in result["hyperparameters"].items()},
                }
            )

            mlflow.log_metrics(
                {
                    **_scalar_metrics(result["metrics"]),
                    **_scalar_metrics(result["baseline_metrics"], prefix="baseline_"),
                }
            )

            if cross_validation:
                mlflow.log_metrics(
                    {
                        f"cv_{cross_validation['metric']}_mean": float(cross_validation["mean"]),
                        f"cv_{cross_validation['metric']}_std": float(cross_validation["std"]),
                    }
                )

            mlflow.log_dict(
                {
                    "method": result["feature_importance_method"],
                    "importance": result["feature_importance"],
                },
                "feature_importance.json",
            )
            mlflow.log_dict(result["diagnostics"], "diagnostics.json")
            mlflow.log_dict(
                {
                    "features_used": result["features_used"],
                    "numeric_features": result["numeric_features"],
                    "categorical_features": result["categorical_features"],
                    "datetime_features": result["datetime_features"],
                    "dropped_columns": result["dropped_high_cardinality_columns"],
                    "label_mapping": result["label_mapping"],
                    "class_balance": result["class_balance"],
                },
                "schema.json",
            )

            if leaderboard:
                mlflow.log_dict({"leaderboard": leaderboard}, "automl_leaderboard.json")
                _log_candidates(mlflow, leaderboard, cross_validation.get("metric", "score"), tags)

            # The pipeline holds StatMind's own preprocessing classes, so it is
            # stored with cloudpickle and needs this codebase to load
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=name,
            )

        version = getattr(model_info, "registered_model_version", None)

        if version is not None:
            client = mlflow.MlflowClient()
            client.set_model_version_tag(name, str(version), "statmind.model_id", model_id)
            client.set_model_version_tag(name, str(version), "statmind.stage", "development")

        return {
            "run_id": run.info.run_id,
            "experiment_id": experiment.experiment_id,
            "run_url": _run_url(experiment.experiment_id, run.info.run_id),
            "registered_model": name,
            "version": str(version) if version is not None else None,
        }

    except Exception:
        logger.exception("MLflow logging failed for model %s", model_id)
        return None


def sync_stage(mlflow_info: dict, stage: str) -> bool:
    """
    Mirror a StatMind stage change onto the MLflow model version. 'staging' and
    'production' become registry aliases, so `models:/<name>@production` always
    resolves to the version StatMind currently has in production.
    """
    if not enabled() or not mlflow_info or not mlflow_info.get("version"):
        return False

    try:
        client = _mlflow().MlflowClient()
        name, version = mlflow_info["registered_model"], str(mlflow_info["version"])

        client.set_model_version_tag(name, version, "statmind.stage", stage)

        # A version that leaves staging/production gives up that alias
        for alias in STAGE_ALIASES:
            if alias == stage:
                client.set_registered_model_alias(name, alias, version)
                continue
            try:
                current = client.get_model_version_by_alias(name, alias)
            except Exception:
                continue
            if str(current.version) == version:
                client.delete_registered_model_alias(name, alias)

        return True

    except Exception:
        logger.exception("MLflow stage sync failed for %s", mlflow_info)
        return False
