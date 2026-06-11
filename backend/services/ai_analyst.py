def _clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def _extract_context_summary(context: dict):
    dataset = context.get("dataset") or {}
    statistics = context.get("statistics") or {}
    schema = context.get("schema") or {}
    target_advice = context.get("targetAdvice") or context.get("target_advice") or {}
    ml_result = context.get("mlResult") or context.get("ml_result") or {}
    compare_result = context.get("compareResult") or context.get("compare_result") or {}
    prediction_result = context.get("predictionResult") or context.get("prediction_result") or {}
    leakage_result = context.get("leakageResult") or context.get("leakage_result") or {}

    summary = {
        "rows": dataset.get("rows"),
        "columns": dataset.get("columns"),
        "domain": dataset.get("domain"),
        "numeric_columns": schema.get("numeric_columns", []),
        "categorical_columns": schema.get("categorical_columns", []),
        "datetime_columns": schema.get("datetime_columns", []),
        "text_columns": schema.get("text_columns", []),
        "data_quality": statistics.get("data_quality", {}),
        "best_target": target_advice.get("best_target"),
        "ml_task_type": ml_result.get("task_type"),
        "ml_algorithm": ml_result.get("algorithm"),
        "ml_metrics": ml_result.get("metrics"),
        "automl_best_model": compare_result.get("best_model"),
        "automl_metric": compare_result.get("ranking_metric"),
        "prediction_rows": prediction_result.get("rows_predicted"),
        "leakage_risk": leakage_result.get("risk_level"),
        "leakage_score": leakage_result.get("leakage_risk_score"),
    }

    return summary


def _answer_target_question(summary):
    best_target = summary.get("best_target")

    if not best_target:
        return [
            "I do not have target-advisor output yet.",
            "Run Target Advisor first. A good target column is usually a meaningful outcome such as Profit, Revenue, Risk, Churn, Score, Result, Price, or Rating.",
        ]

    reasons = best_target.get("reasons", [])
    warnings = best_target.get("warnings", [])

    answer = [
        f"The best recommended target is '{best_target.get('column')}'.",
        f"Recommended task type: {best_target.get('task_type')}.",
        f"Suitability score: {best_target.get('score')}/100.",
    ]

    if reasons:
        answer.append("Reasons: " + " ".join(reasons))

    if warnings:
        answer.append("Warnings: " + " ".join(warnings))

    return answer


def _answer_metrics_question(summary):
    metrics = summary.get("ml_metrics")

    if not metrics:
        return [
            "No ML metrics are available yet.",
            "Train a model first in ML Studio, then ask about accuracy, R², RMSE, F1-score, or model quality.",
        ]

    answer = ["The current model metrics are:"]

    for key, value in metrics.items():
        if isinstance(value, list):
            continue

        answer.append(f"- {key.replace('_', ' ')}: {value}")

    if "r2_score" in metrics:
        r2 = metrics.get("r2_score")

        if r2 >= 0.8:
            answer.append("The R² score is strong, but you should still check leakage and external test performance.")
        elif r2 >= 0.5:
            answer.append("The R² score is moderate. Feature engineering or better data may improve the model.")
        else:
            answer.append("The R² score is weak. The dataset may not contain enough predictive signal.")

    if "f1_weighted" in metrics:
        f1 = metrics.get("f1_weighted")

        if f1 >= 0.85:
            answer.append("The weighted F1-score is strong, but class balance and leakage should still be checked.")
        elif f1 >= 0.65:
            answer.append("The weighted F1-score is moderate.")
        else:
            answer.append("The weighted F1-score is weak. Check class imbalance, noisy labels, or poor features.")

    return answer


def _answer_leakage_question(summary):
    risk = summary.get("leakage_risk")
    score = summary.get("leakage_score")

    if risk is None:
        return [
            "Leakage Detector has not been run yet.",
            "Run Leakage Detector after selecting a target column. It checks high-correlation features, duplicate target columns, ID-like columns, and arithmetic target leakage.",
        ]

    answer = [
        f"Leakage risk level: {risk}.",
        f"Leakage risk score: {score}/100.",
    ]

    if risk in ["Critical", "High"]:
        answer.append(
            "You should not fully trust high accuracy until suspicious features are removed and the model is tested on external data."
        )
    elif risk == "Medium":
        answer.append(
            "Some columns may be suspicious. Review leakage findings before finalizing the model."
        )
    else:
        answer.append(
            "No major leakage pattern was detected by current checks."
        )

    return answer


def _answer_data_quality_question(summary):
    quality = summary.get("data_quality") or {}

    if not quality:
        return [
            "Data quality information is not available yet.",
            "Run Statistics Studio first to calculate data quality score, missing values, and schema information.",
        ]

    score = quality.get("data_quality_score")
    grade = quality.get("grade")

    answer = [
        f"Data quality score: {score}/100.",
        f"Grade: {grade}.",
    ]

    if score is not None:
        if score >= 85:
            answer.append("The dataset quality looks good for initial analysis.")
        elif score >= 65:
            answer.append("The dataset is usable, but missing values, outliers, or inconsistent columns may affect reliability.")
        else:
            answer.append("The dataset quality is weak. Clean the data before trusting model results.")

    return answer


def _answer_prediction_question(summary):
    rows = summary.get("prediction_rows")

    if rows is None:
        return [
            "Prediction Studio has not generated predictions yet.",
            "Train a model, go to Prediction Studio, upload an unseen test dataset, and generate predictions.",
        ]

    return [
        f"Prediction Studio generated predictions for {rows} rows.",
        "If the test file contains the actual target column, external test metrics are also calculated.",
        "External test metrics are more useful than internal split metrics because they test the model on separate data.",
    ]


def _answer_general_question(summary):
    answer = [
        "Here is the current project context I can see:",
        f"- Rows: {summary.get('rows')}",
        f"- Columns: {summary.get('columns')}",
        f"- Domain: {summary.get('domain')}",
        f"- Numeric columns: {len(summary.get('numeric_columns') or [])}",
        f"- Categorical columns: {len(summary.get('categorical_columns') or [])}",
    ]

    if summary.get("best_target"):
        answer.append(f"- Recommended target: {summary['best_target'].get('column')}")

    if summary.get("ml_algorithm"):
        answer.append(f"- Current model: {summary.get('ml_algorithm')}")

    if summary.get("automl_best_model"):
        answer.append(f"- AutoML best model: {summary.get('automl_best_model')}")

    if summary.get("leakage_risk"):
        answer.append(f"- Leakage risk: {summary.get('leakage_risk')}")

    answer.append(
        "Recommended next step: run Leakage Detector, then generate the final intelligence report."
    )

    return answer


def answer_data_scientist_question(question: str, context: dict):
    """
    Rule-based AI analyst.

    This does not require Gemini/OpenAI API.
    Later, this can be upgraded to use an LLM by passing the same context.
    """

    question_clean = _clean_text(question).lower()

    if not question_clean:
        raise ValueError("Question cannot be empty.")

    summary = _extract_context_summary(context)

    if any(word in question_clean for word in ["target", "predict", "column", "label"]):
        answer = _answer_target_question(summary)

    elif any(word in question_clean for word in ["accuracy", "metric", "r2", "rmse", "mae", "f1", "precision", "recall", "performance"]):
        answer = _answer_metrics_question(summary)

    elif any(word in question_clean for word in ["leakage", "trust", "reliable", "misleading"]):
        answer = _answer_leakage_question(summary)

    elif any(word in question_clean for word in ["quality", "missing", "clean", "null", "data issue"]):
        answer = _answer_data_quality_question(summary)

    elif any(word in question_clean for word in ["prediction", "test data", "unseen", "external"]):
        answer = _answer_prediction_question(summary)

    else:
        answer = _answer_general_question(summary)

    return {
        "question": question,
        "answer": answer,
        "analyst_mode": "rule_based_contextual",
        "note": "This analyst uses StatMind's generated context. It can later be upgraded with an LLM API.",
    }