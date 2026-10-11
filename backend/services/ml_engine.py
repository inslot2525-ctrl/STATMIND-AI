import logging
import os
import tempfile
import time
import uuid
import warnings
from typing import Callable, Optional

import joblib
import pandas as pd
import numpy as np
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from services.file_reader import read_uploaded_file
from services.experiment_tracker import log_experiment
from services import lakefs_versioning, mlflow_tracking
from services.dataset_registry import get_dataset
from services.model_registry import get_model_record, register_model
from services.artifacts import create_artifact
from services.storage import TMP_DIR, get_storage, make_key, safe_component

logger = logging.getLogger(__name__)

# progress(percent, message) - lets background jobs report where training is
ProgressFn = Optional[Callable[[int, str], None]]


def _report(progress: ProgressFn, percent: int, message: str):
    if progress:
        progress(percent, message)

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

TASK_TYPES = ("classification", "regression")

BASELINE_ALGORITHM = "baseline"

# AutoML settings
CV_FOLDS = 5
MAX_AUTOML_ROWS = 20000  # training rows used for the cross-validation and tuning rounds
SLOW_MODEL_ROW_LIMIT = 5000  # SVM and KNN are skipped by AutoML above this many rows
SLOW_MODELS = ("svm", "knn")
TUNED_CANDIDATES = 2
TUNING_ITERATIONS = 10
IMBALANCE_RATIO = 0.5  # smallest class / largest class below this counts as imbalanced

MAX_IMPORTANCE_ROWS = 2000
MAX_CURVE_POINTS = 60
MAX_SCATTER_POINTS = 300

# Metric used to rank models in cross-validation
RANKING_METRIC = {"classification": "f1_macro", "regression": "r2"}

_FOREST_SPACE = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [None, 6, 12, 20],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features": ["sqrt", "log2", 0.5],
}

# Hyperparameter search spaces for the AutoML tuning round
SEARCH_SPACES = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "ridge": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
    "lasso": {"model__alpha": [0.001, 0.01, 0.1, 1]},
    "random_forest": _FOREST_SPACE,
    "extra_trees": _FOREST_SPACE,
    "gradient_boosting": {
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_iter": [100, 200, 400],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0, 0.1, 1],
    },
    "decision_tree": {
        "model__max_depth": [None, 3, 5, 8, 12],
        "model__min_samples_leaf": [1, 2, 5, 10],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 7, 11, 15],
        "model__weights": ["uniform", "distance"],
    },
    "svm": {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]},
}


def detect_task_type(y: pd.Series, override: str = None) -> str:
    if override:
        override = override.lower().strip()
        if override not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {list(TASK_TYPES)}.")
        return override

    if is_bool_dtype(y) or not is_numeric_dtype(y):
        return "classification"

    values = y.dropna()
    unique_values = values.nunique()

    if unique_values <= 5:
        return "classification"

    # Integer codes with few distinct values relative to the row count (digits
    # 0-9, star ratings, ...) are labels, not quantities
    integer_valued = bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0))
    if integer_valued and unique_values <= 20 and unique_values / len(values) <= 0.05:
        return "classification"

    return "regression"


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse a column as dates; values that cannot be parsed become NaT."""
    if not is_datetime64_any_dtype(series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                series = pd.to_datetime(series, errors="coerce", format="mixed")
            except Exception:
                series = pd.Series(pd.NaT, index=series.index)

    if not is_datetime64_any_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if series.dt.tz is not None:
        series = series.dt.tz_localize(None)

    return series


def detect_datetime_columns(X: pd.DataFrame) -> list:
    """Datetime columns, plus text columns whose values look like and parse as dates."""
    datetime_columns = []

    for col in X.columns:
        series = X[col]

        if is_datetime64_any_dtype(series):
            datetime_columns.append(col)
            continue

        if is_numeric_dtype(series) or is_bool_dtype(series):
            continue

        sample = series.dropna().astype(str).head(200)

        if sample.empty:
            continue

        looks_like_date = sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True)

        if looks_like_date.mean() < 0.9:
            continue

        if _to_datetime(sample).notna().mean() >= 0.9:
            datetime_columns.append(col)

    return datetime_columns


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Turn date columns into year, month, day of week and days since the earliest training date."""

    PARTS = ("year", "month", "dayofweek", "days")

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = [str(col) for col in X.columns]
        self.origins_ = [_to_datetime(X[col]).min() for col in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        features = {}

        for name, origin, col in zip(self.columns_, self.origins_, X.columns):
            parsed = _to_datetime(X[col])
            features[f"{name}_year"] = parsed.dt.year
            features[f"{name}_month"] = parsed.dt.month
            features[f"{name}_dayofweek"] = parsed.dt.dayofweek
            features[f"{name}_days"] = (parsed - origin).dt.days

        return pd.DataFrame(features, index=X.index).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [f"{name}_{part}" for name in self.columns_ for part in self.PARTS],
            dtype=object,
        )


def _as_text(X):
    """Cast category values to strings (missing stays missing) so one column never mixes types."""
    X = pd.DataFrame(X)
    return X.apply(
        lambda s: s.astype(object).where(s.notna(), np.nan).map(str, na_action="ignore")
    ).astype(object)


def build_preprocessor(X: pd.DataFrame):
    datetime_features = detect_datetime_columns(X)

    numeric_features = [
        col for col in X.columns
        if col not in datetime_features and is_numeric_dtype(X[col]) and not is_bool_dtype(X[col])
    ]

    categorical_features = [
        col for col in X.columns
        if col not in datetime_features and col not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("text", FunctionTransformer(_as_text, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                # Rare categories beyond the 30 most common share one column
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=30,
                    sparse_output=False,
                ),
            ),
        ]
    )

    datetime_pipeline = Pipeline(
        steps=[
            ("parts", DatetimeFeatures()),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("datetime", datetime_pipeline, datetime_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return preprocessor, numeric_features, categorical_features, datetime_features


def get_available_models(task_type: str, balanced: bool = False) -> dict:
    """All trainable algorithms. balanced=True reweights classes for imbalanced targets."""
    if task_type == "classification":
        class_weight = "balanced" if balanced else None

        return {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight=class_weight),
            "random_forest": RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight,
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                random_state=42,
                class_weight=class_weight,
            ),
            "decision_tree": DecisionTreeClassifier(random_state=42, class_weight=class_weight),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "svm": SVC(kernel="rbf", class_weight=class_weight),
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
            "gradient_boosting": HistGradientBoostingRegressor(random_state=42),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "knn": KNeighborsRegressor(n_neighbors=5),
            "svm": SVR(kernel="rbf"),
        }

    raise ValueError("Invalid task type.")


def get_baseline_model(task_type: str):
    """A model that ignores the features: every real model should beat it."""
    if task_type == "classification":
        return DummyClassifier(strategy="prior")

    return DummyRegressor(strategy="mean")


def get_model(task_type: str, algorithm: str, balanced: bool = False):
    models = get_available_models(task_type, balanced)
    algorithm = algorithm.lower().strip()

    if algorithm not in models:
        raise ValueError(
            f"Invalid algorithm '{algorithm}' for {task_type}. "
            f"Choose from {list(models.keys())}."
        )

    return models[algorithm]


def drop_high_cardinality_columns(X: pd.DataFrame):
    high_cardinality_cols = []
    datetime_columns = detect_datetime_columns(X)

    for col in X.columns:
        if col in datetime_columns or is_numeric_dtype(X[col]) or is_bool_dtype(X[col]):
            continue

        unique_ratio = X[col].nunique(dropna=True) / max(len(X), 1)

        if unique_ratio > 0.95:
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


def prepare_ml_data(file_path: str, target_column: str, test_size: float, task_type: str = None):
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

    task_type = detect_task_type(y, override=task_type)
    y_final, label_mapping = prepare_target(y, task_type)

    if task_type == "regression":
        valid_rows = ~pd.isnull(y_final)
        X = X[valid_rows]
        y_final = y_final[valid_rows].to_numpy(dtype=float)

    if len(X) < 10:
        if task_type == "regression":
            raise ValueError(
                "Not enough numeric values in the target column for regression. "
                "Choose a numeric target or train it as classification."
            )
        raise ValueError("Not enough valid rows after preparing target column.")

    class_balance = None

    if task_type == "classification":
        unique_classes, class_counts = np.unique(y_final, return_counts=True)

        if len(unique_classes) < 2:
            raise ValueError("Classification requires at least 2 target classes.")

        stratify_value = y_final if class_counts.min() >= 2 else None

        class_balance = {
            "classes": int(len(unique_classes)),
            "smallest_class_share": round(float(class_counts.min() / class_counts.sum()), 4),
            "largest_class_share": round(float(class_counts.max() / class_counts.sum()), 4),
            "imbalanced": bool(class_counts.min() / class_counts.max() < IMBALANCE_RATIO),
        }
    else:
        stratify_value = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_final,
        test_size=test_size,
        random_state=42,
        stratify=stratify_value,
    )

    preprocessor, numeric_features, categorical_features, datetime_features = build_preprocessor(X)

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
        "class_balance": class_balance,
        "balanced": bool(class_balance and class_balance["imbalanced"]),
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "datetime_features": datetime_features,
        "high_cardinality_cols": high_cardinality_cols,
    }


def evaluate_model(task_type: str, y_test, predictions, roc_auc: float = None) -> dict:
    if task_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
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
            "f1_macro": round(
                float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                4,
            ),
        }

        if roc_auc is not None:
            metrics["roc_auc"] = round(float(roc_auc), 4)

        metrics["confusion_matrix"] = confusion_matrix(y_test, predictions).tolist()

        return metrics

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


def generate_model_explanation(
    task_type: str,
    algorithm: str,
    metrics: dict,
    baseline_metrics: dict = None,
) -> list:
    explanation = [f"The selected algorithm was {algorithm}."]
    baseline_metrics = baseline_metrics or {}

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

        baseline_accuracy = baseline_metrics.get("accuracy")

        if accuracy is not None and baseline_accuracy is not None:
            explanation.append(
                f"Always predicting the most common class would score accuracy = {baseline_accuracy}; "
                f"the model is {round(accuracy - baseline_accuracy, 4):+} against that baseline."
            )

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

        baseline_rmse = baseline_metrics.get("rmse")

        if rmse is not None and baseline_rmse:
            explanation.append(
                f"Always predicting the average target would give RMSE = {baseline_rmse}; "
                f"the model's error is {round(100 * (1 - rmse / baseline_rmse), 1)}% lower."
            )

    return explanation


def compute_baseline_metrics(task_type: str, y_train, y_test) -> dict:
    """Test-split metrics of the feature-blind baseline model."""
    baseline = get_baseline_model(task_type)
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    predictions = baseline.predict(np.zeros((len(y_test), 1)))

    metrics = evaluate_model(task_type, y_test, predictions)
    metrics.pop("confusion_matrix", None)

    return metrics


def _sample_rows(X: pd.DataFrame, y, max_rows: int):
    y = np.asarray(y)

    if len(X) <= max_rows:
        return X, y

    chosen = np.random.RandomState(42).choice(len(X), size=max_rows, replace=False)
    return X.iloc[chosen], y[chosen]


def compute_feature_importance(pipeline, X_test: pd.DataFrame, y_test, task_type: str) -> list:
    """
    Permutation importance on the test split: how much the score drops when one
    input column is shuffled. Works for every algorithm and is reported per
    original column, not per encoded column.
    """
    try:
        X_sample, y_sample = _sample_rows(X_test, y_test, MAX_IMPORTANCE_ROWS)

        result = permutation_importance(
            pipeline,
            X_sample,
            y_sample,
            scoring=RANKING_METRIC[task_type],
            n_repeats=5 if X_sample.shape[1] <= 50 else 3,
            random_state=42,
        )

        feature_importance = sorted(
            [
                {
                    "feature": str(feature),
                    "importance": round(float(result.importances_mean[i]), 4),
                    "std": round(float(result.importances_std[i]), 4),
                }
                for i, feature in enumerate(X_sample.columns)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:15]

        return feature_importance

    except Exception:
        logger.exception("Permutation importance failed")
        return []


def _class_scores(pipeline, X):
    """Per-class scores for ROC analysis: probabilities, or decision values for models without them."""
    model = pipeline.named_steps["model"]

    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)

    if hasattr(model, "decision_function"):
        return pipeline.decision_function(X)

    return None


def _curve_points(fpr, tpr) -> list:
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), MAX_CURVE_POINTS)).astype(int))

    return [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in keep
    ]


def classification_diagnostics(pipeline, X_test, y_test, predictions, label_mapping) -> dict:
    """Per-class scores, a labelled confusion matrix and one-vs-rest ROC curves."""
    y_test = np.asarray(y_test)
    class_ids = list(range(len(label_mapping)))
    labels = [label_mapping[str(class_id)] for class_id in class_ids]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=class_ids, zero_division=0
    )

    diagnostics = {
        "per_class": [
            {
                "label": labels[i],
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in class_ids
        ],
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, predictions, labels=class_ids).tolist(),
        },
        "roc_curves": [],
        "roc_auc": None,
    }

    try:
        scores = _class_scores(pipeline, X_test)

        if scores is None:
            return diagnostics

        scores = np.asarray(scores)
        model_classes = list(pipeline.named_steps["model"].classes_)

        if scores.ndim == 1:
            # Binary decision values score the second class
            columns = {model_classes[1]: scores}
        elif len(model_classes) == 2:
            columns = {model_classes[1]: scores[:, 1]}
        else:
            columns = {class_id: scores[:, i] for i, class_id in enumerate(model_classes)}

        aucs = []

        for class_id, class_scores in columns.items():
            is_class = (y_test == class_id).astype(int)

            if is_class.min() == is_class.max():
                continue  # class absent from (or the only one in) the test split

            auc = float(roc_auc_score(is_class, class_scores))
            aucs.append(auc)

            if len(diagnostics["roc_curves"]) < 8:
                fpr, tpr, _ = roc_curve(is_class, class_scores)
                diagnostics["roc_curves"].append(
                    {
                        "label": labels[int(class_id)],
                        "auc": round(auc, 4),
                        "points": _curve_points(fpr, tpr),
                    }
                )

        if aucs:
            # Binary: AUC of the positive class. Multiclass: one-vs-rest macro average.
            diagnostics["roc_auc"] = round(float(np.mean(aucs)), 4)

    except Exception:
        logger.exception("ROC analysis failed")

    return diagnostics


def regression_diagnostics(y_test, predictions) -> dict:
    """A sample of actual vs predicted values for residual plots."""
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(predictions, dtype=float)

    if len(actual) > MAX_SCATTER_POINTS:
        chosen = np.random.RandomState(42).choice(len(actual), size=MAX_SCATTER_POINTS, replace=False)
        actual, predicted = actual[chosen], predicted[chosen]

    return {
        "predicted_vs_actual": [
            {
                "actual": round(float(a), 4),
                "predicted": round(float(p), 4),
                "residual": round(float(a - p), 4),
            }
            for a, p in zip(actual, predicted)
        ],
    }


def save_trained_model(pipeline, metadata):
    safe_model_id = (
        f"{safe_component(metadata['algorithm'])}_{safe_component(metadata['target_column'])}"
        f"_{uuid.uuid4().hex[:8]}.pkl"
    )

    key = make_key("models", safe_model_id)

    fd, tmp_path = tempfile.mkstemp(dir=TMP_DIR, suffix=".pkl")
    os.close(fd)
    try:
        joblib.dump({"pipeline": pipeline, "metadata": metadata}, tmp_path)
        get_storage().put_file(key, tmp_path, move=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return safe_model_id, key


def _model_hyperparameters(pipeline) -> dict:
    """The fitted estimator's settings, in a form that can be stored as JSON."""
    params = pipeline.named_steps["model"].get_params(deep=False)

    return {
        name: value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
        for name, value in params.items()
    }


def _finalize_model(
    pipeline,
    data: dict,
    algorithm: str,
    target_column: str,
    test_size: float,
    dataset_id: str = None,
    owner_id: str = None,
    progress: ProgressFn = None,
    cross_validation: dict = None,
    leaderboard: list = None,
) -> dict:
    """
    Score a pipeline already fitted on the training split against the test
    split, explain it, save it and register it. Shared by single-model training
    and by the AutoML winner. cross_validation and leaderboard carry the AutoML
    scores, if any.
    """
    task_type = data["task_type"]

    _report(progress, 75, "Evaluating on the test split")
    predictions = pipeline.predict(data["X_test"])

    if task_type == "classification":
        diagnostics = classification_diagnostics(
            pipeline, data["X_test"], data["y_test"], predictions, data["label_mapping"]
        )
        metrics = evaluate_model(
            task_type, data["y_test"], predictions, roc_auc=diagnostics.pop("roc_auc")
        )
    else:
        diagnostics = regression_diagnostics(data["y_test"], predictions)
        metrics = evaluate_model(task_type, data["y_test"], predictions)

    baseline_metrics = compute_baseline_metrics(task_type, data["y_train"], data["y_test"])

    _report(progress, 82, "Measuring feature importance")
    feature_importance = compute_feature_importance(
        pipeline, data["X_test"], data["y_test"], task_type
    )

    hyperparameters = _model_hyperparameters(pipeline)

    metadata = {
        "algorithm": algorithm,
        "task_type": task_type,
        "target_column": target_column,
        "features_used": data["X"].columns.tolist(),
        "numeric_features": data["numeric_features"],
        "categorical_features": data["categorical_features"],
        "datetime_features": data["datetime_features"],
        "label_mapping": data["label_mapping"],
        "test_size": float(test_size),
    }

    _report(progress, 92, "Saving and registering the model")
    model_id, model_key = save_trained_model(pipeline, metadata)

    # ── MLOps: data lineage. The model file becomes a lakeFS commit, and the
    # model records both that commit and the one holding its training data.
    data_version = None
    if lakefs_versioning.enabled():
        dataset = get_dataset(dataset_id, owner_id) if dataset_id else None
        model_commit = lakefs_versioning.commit(
            f"Add model {model_id}",
            {
                "statmind.kind": "model",
                "statmind.model_id": model_id,
                "statmind.dataset_id": dataset_id,
                "statmind.owner_id": owner_id,
                "statmind.algorithm": algorithm,
                "statmind.target": target_column,
            },
        )
        data_version = lakefs_versioning.describe(
            dataset_commit=(dataset or {}).get("data_commit"),
            model_commit=model_commit,
        )

    result = {
        "model_id": model_id,
        "experiment_id": None,
        "mlflow": None,
        "data_version": data_version,
        "task_type": task_type,
        "algorithm": algorithm,
        "target_column": target_column,
        "rows_used": int(len(data["X"])),
        "train_rows": int(len(data["X_train"])),
        "test_rows": int(len(data["X_test"])),
        "features_used": data["X"].columns.tolist(),
        "numeric_features": data["numeric_features"],
        "categorical_features": data["categorical_features"],
        "datetime_features": data["datetime_features"],
        "dropped_high_cardinality_columns": data["high_cardinality_cols"],
        "test_size": float(test_size),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "hyperparameters": hyperparameters,
        "cross_validation": cross_validation,
        "class_balance": data["class_balance"],
        "diagnostics": diagnostics,
        "label_mapping": data["label_mapping"],
        "warnings": [],
        "model_recommendation": generate_model_recommendation(
            task_type,
            len(data["X"]),
            data["numeric_features"],
            data["categorical_features"],
        ),
        "model_explanation": generate_model_explanation(
            task_type, algorithm, metrics, baseline_metrics
        ),
        "feature_importance": feature_importance,
        "feature_importance_method": "permutation",
    }

    if len(data["X"]) < 100:
        result["warnings"].append(
            "Dataset is small. Model metrics may not be reliable and overfitting risk is high."
        )

    if data["balanced"]:
        result["warnings"].append(
            "Target classes are imbalanced "
            f"(smallest class is {round(100 * data['class_balance']['smallest_class_share'], 1)}% of rows). "
            "Classes were reweighted where the algorithm supports it; "
            "judge the model by balanced accuracy and macro F1 rather than accuracy."
        )

    result["warnings"].append(
        "Trained model has been saved and can be used in Prediction Studio."
    )

    # ── MLOps: MLflow (if configured), model registry, experiment log ──
    if mlflow_tracking.enabled():
        _report(progress, 95, "Logging the run to MLflow")
        result["mlflow"] = mlflow_tracking.log_training_run(
            pipeline,
            result,
            model_id=model_id,
            dataset_id=dataset_id,
            owner_id=owner_id,
            leaderboard=leaderboard,
        )

    # Registration is required: models are only reachable through the registry.
    full_metadata = {
        **metadata,
        "model_id": model_id,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "hyperparameters": hyperparameters,
    }
    if cross_validation:
        full_metadata["cross_validation"] = cross_validation
    if dataset_id:
        full_metadata["dataset_id"] = dataset_id
    if result["mlflow"]:
        full_metadata["mlflow"] = result["mlflow"]
    if data_version:
        full_metadata["data_version"] = data_version
    register_model(model_id, full_metadata, owner_id=owner_id, storage_key=model_key)

    if dataset_id and owner_id:
        try:
            result["experiment_id"] = log_experiment(
                owner_id=owner_id,
                dataset_id=dataset_id,
                algorithm=algorithm,
                hyperparameters={"test_size": float(test_size), **hyperparameters},
                metrics={**metrics, **({"cross_validation": cross_validation} if cross_validation else {})},
                model_id=model_id,
                tags=["automl"] if cross_validation else [],
                notes=f"MLflow run {result['mlflow']['run_id']}" if result["mlflow"] else None,
            )
        except Exception:
            logger.exception("Experiment logging failed for model %s", model_id)
    # ─────────────────────────────────────────────────────────────

    return result


def train_model_from_file(
    file_path: str,
    target_column: str,
    algorithm: str,
    test_size: float = 0.2,
    dataset_id: str = None,
    owner_id: str = None,
    task_type: str = None,
    progress: ProgressFn = None,
) -> dict:
    """
    Train one model and register it. owner_id=None registers an unowned model
    (used by offline benchmarks); API calls always pass the current user.
    task_type=None detects classification vs regression from the target.
    """
    _report(progress, 5, "Loading and preparing data")
    data = prepare_ml_data(file_path, target_column, test_size, task_type)

    model = get_model(data["task_type"], algorithm, data["balanced"])

    pipeline = Pipeline(
        steps=[
            ("preprocessor", data["preprocessor"]),
            ("model", model),
        ]
    )

    _report(progress, 20, f"Training {algorithm}")
    pipeline.fit(data["X_train"], data["y_train"])

    return _finalize_model(
        pipeline,
        data,
        algorithm=algorithm,
        target_column=target_column,
        test_size=test_size,
        dataset_id=dataset_id,
        owner_id=owner_id,
        progress=progress,
    )


def _cv_splitter(task_type: str, y):
    if task_type == "classification":
        smallest_class = int(np.unique(y, return_counts=True)[1].min())

        if smallest_class >= 2:
            return StratifiedKFold(
                n_splits=min(CV_FOLDS, smallest_class), shuffle=True, random_state=42
            )

    return KFold(n_splits=min(CV_FOLDS, max(2, len(y) // 5)), shuffle=True, random_state=42)


def _candidate_pipeline(data: dict, model) -> Pipeline:
    return Pipeline(
        steps=[
            # Fresh copy per model so candidates never share fitted state
            ("preprocessor", clone(data["preprocessor"])),
            ("model", model),
        ]
    )


def _cross_validate_candidate(algorithm: str, model, data: dict, X_cv, y_cv, cv) -> dict:
    row = {
        "algorithm": algorithm,
        "is_baseline": algorithm == BASELINE_ALGORITHM,
        "tuned": False,
        "best_params": {},
    }

    started = time.perf_counter()

    try:
        scores = cross_validate(
            _candidate_pipeline(data, model),
            X_cv,
            y_cv,
            cv=cv,
            scoring=RANKING_METRIC[data["task_type"]],
            error_score="raise",
        )["test_score"]

        if np.isnan(scores).any():
            raise ValueError("The score is undefined on at least one fold.")

        row.update(
            {
                "score": round(float(scores.mean()), 4),
                "cv_mean": round(float(scores.mean()), 4),
                "cv_std": round(float(scores.std()), 4),
                "cv_scores": [round(float(score), 4) for score in scores],
                "seconds": round(time.perf_counter() - started, 2),
            }
        )

    except Exception as e:
        row.update({"score": None, "error": str(e)})

    return row


def _tune_candidate(row: dict, model, data: dict, X_cv, y_cv, cv) -> None:
    """Random search over the algorithm's search space; keeps the defaults if they score better."""
    space = SEARCH_SPACES.get(row["algorithm"])

    if not space:
        return

    combinations = int(np.prod([len(values) for values in space.values()]))
    iterations = min(TUNING_ITERATIONS, combinations)

    try:
        search = RandomizedSearchCV(
            _candidate_pipeline(data, model),
            space,
            n_iter=iterations,
            scoring=RANKING_METRIC[data["task_type"]],
            cv=cv,
            random_state=42,
            error_score=np.nan,
            refit=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_cv, y_cv)

        best_score = float(search.best_score_)

    except Exception:
        logger.exception("Tuning failed for %s", row["algorithm"])
        return

    improved = not np.isnan(best_score) and best_score > row["cv_mean"]

    row["tuning"] = {
        "configurations_tried": iterations,
        "default_cv_mean": row["cv_mean"],
        "improved": bool(improved),
    }

    if improved:
        row.update(
            {
                "tuned": True,
                "best_params": {
                    name.replace("model__", "", 1): value
                    for name, value in search.best_params_.items()
                },
                "score": round(best_score, 4),
                "cv_mean": round(best_score, 4),
                "cv_std": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
            }
        )
        row.pop("cv_scores", None)


def _rank(leaderboard: list) -> list:
    valid_results = [item for item in leaderboard if item["score"] is not None]
    failed_results = [item for item in leaderboard if item["score"] is None]

    return sorted(valid_results, key=lambda x: x["score"], reverse=True) + failed_results


def compare_models_from_file(
    file_path: str,
    target_column: str,
    test_size: float = 0.2,
    dataset_id: str = None,
    owner_id: str = None,
    task_type: str = None,
    progress: ProgressFn = None,
) -> dict:
    """
    AutoML: the test split is set aside first, every algorithm is ranked by
    cross-validation on the training split, the best ones are tuned, and the
    winner is refit on the whole training split and scored once on the test
    split. The winner is saved and registered like any trained model.
    """
    _report(progress, 5, "Loading and preparing data")
    data = prepare_ml_data(file_path, target_column, test_size, task_type)

    task_type = data["task_type"]
    ranking_metric = RANKING_METRIC[task_type]

    X_cv, y_cv = _sample_rows(data["X_train"], data["y_train"], MAX_AUTOML_ROWS)
    cv = _cv_splitter(task_type, y_cv)

    models = get_available_models(task_type, data["balanced"])
    skipped = []

    if len(X_cv) > SLOW_MODEL_ROW_LIMIT:
        skipped = [name for name in SLOW_MODELS if name in models]
        models = {name: model for name, model in models.items() if name not in skipped}

    candidates = {BASELINE_ALGORITHM: get_baseline_model(task_type), **models}
    leaderboard = []

    for index, (algorithm, model) in enumerate(candidates.items()):
        _report(progress, 8 + int(42 * index / len(candidates)), f"Cross-validating {algorithm}")
        leaderboard.append(_cross_validate_candidate(algorithm, clone(model), data, X_cv, y_cv, cv))

    contenders = [
        row for row in _rank(leaderboard)
        if row["score"] is not None and not row["is_baseline"]
    ]

    if not contenders:
        raise ValueError("None of the candidate models could be trained on this dataset.")

    for index, row in enumerate(contenders[:TUNED_CANDIDATES]):
        _report(progress, 50 + int(20 * index / TUNED_CANDIDATES), f"Tuning {row['algorithm']}")
        _tune_candidate(row, clone(models[row["algorithm"]]), data, X_cv, y_cv, cv)

    final_leaderboard = _rank(leaderboard)
    winner = next(row for row in final_leaderboard if row["score"] is not None and not row["is_baseline"])
    baseline = next((row for row in final_leaderboard if row["is_baseline"]), None)

    _report(progress, 70, f"Refitting {winner['algorithm']} on the full training split")
    best_model = clone(models[winner["algorithm"]]).set_params(**winner["best_params"])
    pipeline = _candidate_pipeline(data, best_model)
    pipeline.fit(data["X_train"], data["y_train"])

    winner_result = _finalize_model(
        pipeline,
        data,
        algorithm=winner["algorithm"],
        target_column=target_column,
        test_size=test_size,
        dataset_id=dataset_id,
        owner_id=owner_id,
        progress=progress,
        cross_validation={
            "metric": ranking_metric,
            "folds": cv.get_n_splits(),
            "mean": winner["cv_mean"],
            "std": winner["cv_std"],
            "tuned": winner["tuned"],
        },
        leaderboard=final_leaderboard,
    )

    warnings_list = [
        f"Models were ranked by {cv.get_n_splits()}-fold cross-validated {ranking_metric} on the training split. "
        "The test split was used once, to score the winner.",
        f"The top {min(TUNED_CANDIDATES, len(contenders))} models were tuned with a random search "
        f"of up to {TUNING_ITERATIONS} configurations each.",
    ]

    if baseline and baseline["score"] is not None:
        warnings_list.append(
            f"The '{BASELINE_ALGORITHM}' row ignores the features. "
            "A model that does not clearly beat it has learned nothing useful."
        )

    if skipped:
        warnings_list.append(
            f"{', '.join(skipped)} skipped: too slow for more than {SLOW_MODEL_ROW_LIMIT} training rows."
        )

    if len(data["X_train"]) > len(X_cv):
        warnings_list.append(
            f"Cross-validation used a {len(X_cv)}-row sample of the {len(data['X_train'])} training rows. "
            "The winner was refit on all of them."
        )

    warnings_list.extend(
        warning for warning in winner_result["warnings"] if "Prediction Studio" not in warning
    )
    warnings_list.append(
        f"The winning model was saved as {winner_result['model_id']} and can be used in Prediction Studio."
    )

    return {
        "task_type": task_type,
        "target_column": target_column,
        "test_size": float(test_size),
        "best_model": winner["algorithm"],
        "model_id": winner_result["model_id"],
        "ranking_metric": ranking_metric,
        "cv_folds": cv.get_n_splits(),
        "leaderboard": final_leaderboard,
        "test_metrics": winner_result["metrics"],
        "baseline_metrics": winner_result["baseline_metrics"],
        "skipped_algorithms": skipped,
        "rows_used": int(len(data["X"])),
        "rows_used_for_compare": int(len(X_cv)),
        "train_rows": int(len(data["X_train"])),
        "test_rows": int(len(data["X_test"])),
        "warnings": warnings_list,
        "model_recommendation": winner_result["model_recommendation"],
        "winner": winner_result,
    }


def load_model_artifact(model_id: str, owner_id: str) -> dict:
    """
    Load a model the user owns.

    Models are pickles, and unpickling runs code, so a model is only loaded if
    it is registered to this user; its storage key was written by the server.
    """
    record = get_model_record(model_id, owner_id) if model_id else None

    if record is None or not record.storage_key:
        raise ValueError("Saved model not found. Train a model first.")

    return joblib.load(get_storage().local_path(record.storage_key))


def predict_with_saved_model(model_id: str, test_file_path: str, owner_id: str) -> dict:
    artifact = load_model_artifact(model_id, owner_id)

    if not os.path.exists(test_file_path):
        raise ValueError("Test file not found.")

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

    prediction_filename = f"predictions_{safe_component(target_column)}_{uuid.uuid4().hex[:8]}.csv"
    artifact_record = create_artifact(
        owner_id=owner_id,
        kind="prediction",
        data=output_df.to_csv(index=False).encode("utf-8"),
        extension=".csv",
        download_name=prediction_filename,
    )

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
        "artifact_id": artifact_record["id"],
        "rows_predicted": int(len(output_df)),
        "preview": preview,
        "external_test_metrics": external_metrics,
        "message": "Predictions generated successfully.",
    }

