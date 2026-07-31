import os
import uuid
import joblib
import pandas as pd
import numpy as np

from services.file_reader import read_uploaded_file
from services.experiment_tracker import log_experiment
from services.model_registry import register_model

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

MODEL_DIR = "saved_models"
PREDICTION_DIR = "predictions"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)


def detect_task_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.dtype.name == "category" or y.dtype == "bool":
        return "classification"

    unique_values = y.nunique(dropna=True)

    if unique_values <= 5:
        return "classification"

    return "regression"


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "datetime64[ns]"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


def get_available_models(task_type: str) -> dict:
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
            ),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "svm": SVC(kernel="rbf", probability=True),
        }

    if task_type == "regression":
        return {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.01, max_iter=5000),
            "random_forest": RandomForestRegressor(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
            ),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "knn": KNeighborsRegressor(n_neighbors=5),
            "svm": SVR(kernel="rbf"),
        }

    raise ValueError("Invalid task type.")


def get_model(task_type: str, algorithm: str):
    models = get_available_models(task_type)
    algorithm = algorithm.lower().strip()

    if algorithm not in models:
        raise ValueError(
            f"Invalid algorithm '{algorithm}' for {task_type}. "
            f"Choose from {list(models.keys())}."
        )

    return models[algorithm]


def drop_high_cardinality_columns(X: pd.DataFrame):
    high_cardinality_cols = []

    for col in X.columns:
        unique_ratio = X[col].nunique(dropna=True) / max(len(X), 1)

        if unique_ratio > 0.95 and X[col].dtype == "object":
            high_cardinality_cols.append(col)

    if high_cardinality_cols:
        X = X.drop(columns=high_cardinality_cols)

    return X, high_cardinality_cols


def prepare_target(y: pd.Series, task_type: str):
    label_mapping = None

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))

        label_mapping = {
            str(index): str(label)
            for index, label in enumerate(label_encoder.classes_)
        }

        return y_encoded, label_mapping

    y_numeric = pd.to_numeric(y, errors="coerce")
    return y_numeric, label_mapping


def decode_classification_predictions(predictions, label_mapping):
    if not label_mapping:
        return predictions.tolist()

    decoded = []

    for pred in predictions:
        decoded.append(label_mapping.get(str(int(pred)), str(pred)))

    return decoded


def prepare_ml_data(file_path: str, target_column: str, test_size: float):
    if not os.path.exists(file_path):
        raise ValueError("Uploaded file not found. Please analyze/upload the dataset first.")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1. Example: 0.2")

    df = read_uploaded_file(file_path)

    df.columns = [str(col).strip() for col in df.columns]
    target_column = target_column.strip()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns are: {df.columns.tolist()}"
        )

    if df.shape[0] < 10:
        raise ValueError("Dataset is too small for reliable ML training. Minimum 10 rows required.")

    df = df.dropna(subset=[target_column])

    y = df[target_column]
    X = df.drop(columns=[target_column])

    X, high_cardinality_cols = drop_high_cardinality_columns(X)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found after preprocessing.")

    task_type = detect_task_type(y)
    y_final, label_mapping = prepare_target(y, task_type)

    if task_type == "regression":
        valid_rows = ~pd.isnull(y_final)
        X = X[valid_rows]
        y_final = y_final[valid_rows]

    if len(X) < 10:
        raise ValueError("Not enough valid rows after preparing target column.")

    if task_type == "classification":
        unique_classes, class_counts = np.unique(y_final, return_counts=True)

        if len(unique_classes) < 2:
            raise ValueError("Classification requires at least 2 target classes.")

        stratify_value = y_final if class_counts.min() >= 2 else None
    else:
        stratify_value = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_final,
        test_size=test_size,
        random_state=42,
        stratify=stratify_value,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    return {
        "df": df,
        "X": X,
        "y": y_final,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": task_type,
        "label_mapping": label_mapping,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "high_cardinality_cols": high_cardinality_cols,
    }


def evaluate_model(task_type: str, y_test, predictions) -> dict:
    if task_type == "classification":
        return {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_weighted": round(
                float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                4,
            ),
            "recall_weighted": round(
                float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                4,
            ),
            "f1_weighted": round(
                float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                4,
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }

    rmse = mean_squared_error(y_test, predictions) ** 0.5

    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
    }


def evaluate_external_predictions(task_type, actual_values, predicted_values, label_mapping=None):
    if task_type == "classification":
        actual_clean = actual_values.astype(str)

        decoded_predictions = decode_classification_predictions(predicted_values, label_mapping)
        predicted_clean = pd.Series(decoded_predictions).astype(str)

        return {
            "accuracy": round(float(accuracy_score(actual_clean, predicted_clean)), 4),
            "precision_weighted": round(
                float(precision_score(actual_clean, predicted_clean, average="weighted", zero_division=0)),
                4,
            ),
            "recall_weighted": round(
                float(recall_score(actual_clean, predicted_clean, average="weighted", zero_division=0)),
                4,
            ),
            "f1_weighted": round(
                float(f1_score(actual_clean, predicted_clean, average="weighted", zero_division=0)),
                4,
            ),
            "confusion_matrix": confusion_matrix(actual_clean, predicted_clean).tolist(),
        }

    actual_numeric = pd.to_numeric(actual_values, errors="coerce")
    predicted_numeric = pd.to_numeric(predicted_values, errors="coerce")

    valid_mask = ~(actual_numeric.isnull() | pd.isnull(predicted_numeric))

    actual_numeric = actual_numeric[valid_mask]
    predicted_numeric = predicted_numeric[valid_mask]

    if len(actual_numeric) == 0:
        return {}

    rmse = mean_squared_error(actual_numeric, predicted_numeric) ** 0.5

    return {
        "mae": round(float(mean_absolute_error(actual_numeric, predicted_numeric)), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2_score(actual_numeric, predicted_numeric)), 4),
    }


def generate_model_recommendation(task_type, rows, numeric_features, categorical_features):
    reasons = []

    if task_type == "classification":
        if rows < 200:
            recommended = "logistic_regression"
            reasons.append("Dataset is small, so an interpretable baseline model is safer.")
        else:
            recommended = "random_forest"
            reasons.append("Random Forest is robust for tabular classification datasets.")
    else:
        if rows < 200:
            recommended = "ridge"
            reasons.append("Dataset is small, so regularized regression is safer.")
        else:
            recommended = "random_forest"
            reasons.append("Random Forest is robust for non-linear tabular regression.")

    if categorical_features:
        reasons.append("Categorical variables are handled using one-hot encoding.")

    if numeric_features:
        reasons.append("Numeric variables are imputed and scaled before model training.")

    return {
        "recommended_algorithm": recommended,
        "reasons": reasons,
    }


def generate_model_explanation(task_type: str, algorithm: str, metrics: dict) -> list:
    explanation = [f"The selected algorithm was {algorithm}."]

    if task_type == "classification":
        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1_weighted")

        explanation.append(
            f"The model achieved accuracy = {accuracy} and weighted F1-score = {f1}."
        )

        if f1 is not None:
            if f1 >= 0.85:
                explanation.append("The model performance is strong for this test split.")
            elif f1 >= 0.65:
                explanation.append("The model performance is moderate and may improve with more data or tuning.")
            else:
                explanation.append("The model performance is weak. Data quality, class imbalance, or feature quality may be issues.")

    else:
        r2 = metrics.get("r2_score")
        rmse = metrics.get("rmse")

        explanation.append(f"The model achieved R² = {r2} with RMSE = {rmse}.")

        if r2 is not None:
            if r2 >= 0.8:
                explanation.append("The model explains a high amount of target variance.")
            elif r2 >= 0.5:
                explanation.append("The model explains a moderate amount of target variance.")
            else:
                explanation.append("The model has limited predictive strength on this test split.")

    return explanation


def extract_feature_importance(pipeline):
    trained_model = pipeline.named_steps["model"]

    if not hasattr(trained_model, "feature_importances_"):
        return []

    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = trained_model.feature_importances_

        feature_importance = sorted(
            [
                {
                    "feature": str(feature_names[i]),
                    "importance": round(float(importances[i]), 4),
                }
                for i in range(len(importances))
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:12]

        return feature_importance

    except Exception:
        return []


def save_trained_model(pipeline, metadata):
    model_id = f"{metadata['algorithm']}_{metadata['target_column']}_{uuid.uuid4().hex[:8]}.pkl"
    safe_model_id = model_id.replace(" ", "_").replace("/", "_").replace("\\", "_")

    model_path = os.path.join(MODEL_DIR, safe_model_id)

    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }

    joblib.dump(artifact, model_path)

    return safe_model_id


def train_model_from_file(
    file_path: str,
    target_column: str,
    algorithm: str,
    test_size: float = 0.2,
    dataset_id: str = None,
) -> dict:
    data = prepare_ml_data(file_path, target_column, test_size)

    task_type = data["task_type"]
    model = get_model(task_type, algorithm)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", data["preprocessor"]),
            ("model", model),
        ]
    )

    pipeline.fit(data["X_train"], data["y_train"])
    predictions = pipeline.predict(data["X_test"])

    metrics = evaluate_model(task_type, data["y_test"], predictions)

    metadata = {
        "algorithm": algorithm,
        "task_type": task_type,
        "target_column": target_column,
        "features_used": data["X"].columns.tolist(),
        "numeric_features": data["numeric_features"],
        "categorical_features": data["categorical_features"],
        "label_mapping": data["label_mapping"],
        "test_size": float(test_size),
    }

    model_id = save_trained_model(pipeline, metadata)

    # ── MLOps: register model and log experiment ──────────────────
    try:
        full_metadata = {**metadata, "model_id": model_id, "metrics": metrics}
        if dataset_id:
            full_metadata["dataset_id"] = dataset_id
        register_model(model_id, full_metadata)
    except Exception:
        pass  # don't fail training if registry write fails

    experiment_id = None
    if dataset_id:
        try:
            experiment_id = log_experiment(
                dataset_id=dataset_id,
                algorithm=algorithm,
                hyperparameters={"test_size": float(test_size)},
                metrics=metrics,
                model_id=model_id,
            )
        except Exception:
            pass  # don't fail training if tracking write fails
    # ─────────────────────────────────────────────────────────────

    result = {
        "model_id": model_id,
        "task_type": task_type,
        "algorithm": algorithm,
        "target_column": target_column,
        "rows_used": int(len(data["X"])),
        "train_rows": int(len(data["X_train"])),
        "test_rows": int(len(data["X_test"])),
        "features_used": data["X"].columns.tolist(),
        "numeric_features": data["numeric_features"],
        "categorical_features": data["categorical_features"],
        "dropped_high_cardinality_columns": data["high_cardinality_cols"],
        "test_size": float(test_size),
        "metrics": metrics,
        "label_mapping": data["label_mapping"],
        "warnings": [],
        "model_recommendation": generate_model_recommendation(
            task_type,
            len(data["X"]),
            data["numeric_features"],
            data["categorical_features"],
        ),
        "model_explanation": generate_model_explanation(task_type, algorithm, metrics),
        "feature_importance": extract_feature_importance(pipeline),
    }

    if len(data["X"]) < 100:
        result["warnings"].append(
            "Dataset is small. Model metrics may not be reliable and overfitting risk is high."
        )

    result["warnings"].append(
        "Trained model has been saved and can be used in Prediction Studio."
    )

    return result


def compare_models_from_file(
    file_path: str,
    target_column: str,
    test_size: float = 0.2,
    dataset_id: str = None,
) -> dict:
    data = prepare_ml_data(file_path, target_column, test_size)

    task_type = data["task_type"]

    max_compare_rows = 800

    X_full = data["X"].copy()
    y_full = pd.Series(data["y"], index=X_full.index)

    if len(X_full) > max_compare_rows:
        sampled_indices = X_full.sample(n=max_compare_rows, random_state=42).index
        X_used = X_full.loc[sampled_indices]
        y_used = y_full.loc[sampled_indices]
    else:
        X_used = X_full
        y_used = y_full

    stratify_value = None

    if task_type == "classification":
        unique_classes, class_counts = np.unique(y_used, return_counts=True)

        if len(unique_classes) > 1 and class_counts.min() >= 2:
            stratify_value = y_used

    X_train, X_test, y_train, y_test = train_test_split(
        X_used,
        y_used,
        test_size=test_size,
        random_state=42,
        stratify=stratify_value,
    )

    if task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42),
            "random_forest_fast": RandomForestClassifier(
                n_estimators=40,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
            "extra_trees_fast": ExtraTreesClassifier(
                n_estimators=40,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
        }
    else:
        models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=42),
            "random_forest_fast": RandomForestRegressor(
                n_estimators=40,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
            "extra_trees_fast": ExtraTreesRegressor(
                n_estimators=40,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
        }

    leaderboard = []

    for algorithm, model in models.items():
        try:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", data["preprocessor"]),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            metrics = evaluate_model(task_type, y_test, predictions)

            if task_type == "classification":
                score = metrics["f1_weighted"]
            else:
                score = metrics["r2_score"]

            leaderboard.append(
                {
                    "algorithm": algorithm,
                    "score": score,
                    "metrics": metrics,
                }
            )

        except Exception as e:
            leaderboard.append(
                {
                    "algorithm": algorithm,
                    "score": None,
                    "error": str(e),
                    "metrics": {},
                }
            )

    valid_results = [item for item in leaderboard if item["score"] is not None]
    failed_results = [item for item in leaderboard if item["score"] is None]

    valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)
    final_leaderboard = valid_results + failed_results

    best_model = valid_results[0]["algorithm"] if valid_results else None

    warnings = [
        "AutoML Compare uses lightweight demo models for fast execution.",
        "Heavy models like SVM and KNN are excluded from AutoML Compare to prevent timeout.",
    ]

    if len(X_full) > max_compare_rows:
        warnings.append(
            f"Dataset has {len(X_full)} rows. AutoML used a {max_compare_rows}-row sample for speed."
        )

    return {
        "task_type": task_type,
        "target_column": target_column,
        "test_size": float(test_size),
        "best_model": best_model,
        "ranking_metric": "f1_weighted" if task_type == "classification" else "r2_score",
        "leaderboard": final_leaderboard,
        "rows_used": int(len(X_full)),
        "rows_used_for_compare": int(len(X_used)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "warnings": warnings,
        "model_recommendation": generate_model_recommendation(
            task_type,
            len(X_full),
            data["numeric_features"],
            data["categorical_features"],
        ),
    }


def predict_with_saved_model(model_id: str, test_file_path: str) -> dict:
    model_path = os.path.join(MODEL_DIR, model_id)

    if not os.path.exists(model_path):
        raise ValueError("Saved model not found. Train a model first.")

    if not os.path.exists(test_file_path):
        raise ValueError("Test file not found.")

    artifact = joblib.load(model_path)

    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]

    target_column = metadata["target_column"]
    task_type = metadata["task_type"]
    features_used = metadata["features_used"]
    label_mapping = metadata.get("label_mapping")

    test_df = read_uploaded_file(test_file_path)
    test_df.columns = [str(col).strip() for col in test_df.columns]

    missing_features = [col for col in features_used if col not in test_df.columns]

    if missing_features:
        raise ValueError(
            f"Test data is missing required feature columns: {missing_features}"
        )

    X_new = test_df[features_used].copy()

    raw_predictions = pipeline.predict(X_new)

    output_df = test_df.copy()

    prediction_column = f"predicted_{target_column}"

    if task_type == "classification":
        final_predictions = decode_classification_predictions(raw_predictions, label_mapping)
        output_df[prediction_column] = final_predictions
    else:
        output_df[prediction_column] = raw_predictions

    external_metrics = None

    if target_column in test_df.columns:
        actual_values = test_df[target_column]

        external_metrics = evaluate_external_predictions(
            task_type=task_type,
            actual_values=actual_values,
            predicted_values=raw_predictions,
            label_mapping=label_mapping,
        )

    prediction_filename = f"predictions_{uuid.uuid4().hex[:8]}.csv"
    prediction_path = os.path.join(PREDICTION_DIR, prediction_filename)

    output_df.to_csv(prediction_path, index=False)

    preview = (
        output_df.head(10)
        .replace([np.inf, -np.inf], np.nan)
        .fillna("")
        .to_dict(orient="records")
    )

    return {
        "model_id": model_id,
        "task_type": task_type,
        "target_column": target_column,
        "prediction_column": prediction_column,
        "prediction_filename": prediction_filename,
        "rows_predicted": int(len(output_df)),
        "preview": preview,
        "external_test_metrics": external_metrics,
        "message": "Predictions generated successfully.",
    }

