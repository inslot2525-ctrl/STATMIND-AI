import numpy as np
import pandas as pd

from services.file_reader import read_uploaded_file


def safe_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if np.isinf(value):
            return None
        return round(float(value), 4)

    return str(value)


def detect_schema(df: pd.DataFrame) -> dict:
    numeric_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime_columns = []
    text_columns = []

    for col in list(categorical_columns):
        sample = df[col].dropna().astype(str).head(30)

        if sample.empty:
            continue

        avg_length = sample.str.len().mean()
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

        if avg_length > 40 and unique_ratio > 0.5:
            text_columns.append(col)
            categorical_columns.remove(col)

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "text_columns": text_columns,
        "missing_values": {
            col: int(df[col].isna().sum()) for col in df.columns
        },
        "dtypes": {
            col: str(df[col].dtype) for col in df.columns
        },
    }


def calculate_data_quality(df: pd.DataFrame) -> dict:
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_ratio = missing_cells / total_cells
    duplicate_ratio = duplicate_rows / max(len(df), 1)

    score = 100
    score -= missing_ratio * 45
    score -= duplicate_ratio * 25

    score = max(0, round(score, 2))

    if score >= 85:
        grade = "Excellent"
    elif score >= 70:
        grade = "Good"
    elif score >= 55:
        grade = "Moderate"
    else:
        grade = "Weak"

    return {
        "data_quality_score": score,
        "grade": grade,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
    }


def numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    output = {}

    for col in numeric_columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if series.empty:
            continue

        output[col] = {
            "count": int(series.count()),
            "mean": safe_value(series.mean()),
            "median": safe_value(series.median()),
            "std": safe_value(series.std()),
            "min": safe_value(series.min()),
            "max": safe_value(series.max()),
            "q1": safe_value(series.quantile(0.25)),
            "q3": safe_value(series.quantile(0.75)),
        }

    return output


def categorical_summary(df: pd.DataFrame, categorical_columns: list[str]) -> dict:
    output = {}

    for col in categorical_columns:
        value_counts = df[col].astype(str).value_counts().head(8)

        output[col] = {
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_value": str(value_counts.index[0]) if len(value_counts) else None,
            "top_frequency": int(value_counts.iloc[0]) if len(value_counts) else 0,
            "top_categories": [
                {
                    "category": str(category),
                    "count": int(count),
                }
                for category, count in value_counts.items()
            ],
        }

    return output


def generate_charts(df: pd.DataFrame, schema: dict) -> dict:
    numeric = schema["numeric_columns"]
    categorical = schema["categorical_columns"]

    charts = {
        "bar_charts": [],
        "pie_charts": [],
        "line_charts": [],
        "area_charts": [],
        "histograms": [],
        "scatter_plots": [],
        "top_n_charts": [],
        "missing_value_chart": [],
        "correlation_heatmap": None,
    }

    for col in df.columns:
        missing_count = int(df[col].isna().sum())

        if missing_count > 0:
            charts["missing_value_chart"].append(
                {
                    "column": col,
                    "missing_count": missing_count,
                }
            )

    for col in categorical[:3]:
        value_counts = df[col].astype(str).value_counts().head(8)

        chart_data = [
            {
                "category": str(category),
                "count": int(count),
            }
            for category, count in value_counts.items()
        ]

        charts["bar_charts"].append(
            {
                "title": f"Distribution of {col}",
                "x_axis": "category",
                "y_axis": "count",
                "data": chart_data,
            }
        )

        charts["pie_charts"].append(
            {
                "title": f"Share of {col}",
                "name_key": "category",
                "value_key": "count",
                "data": chart_data,
            }
        )

    for col in numeric[:4]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) < 2:
            continue

        line_data = [
            {
                "index": int(index),
                col: safe_value(value),
            }
            for index, value in enumerate(series.head(50))
        ]

        charts["line_charts"].append(
            {
                "title": f"Trend of {col}",
                "x_axis": "index",
                "y_axis": col,
                "data": line_data,
            }
        )

        charts["area_charts"].append(
            {
                "title": f"Area Trend of {col}",
                "x_axis": "index",
                "y_axis": col,
                "data": line_data,
            }
        )

        counts, bin_edges = np.histogram(series, bins=min(8, max(3, len(series) // 5)))

        histogram_data = []

        for i in range(len(counts)):
            histogram_data.append(
                {
                    "range": f"{round(bin_edges[i], 2)} - {round(bin_edges[i + 1], 2)}",
                    "count": int(counts[i]),
                }
            )

        charts["histograms"].append(
            {
                "title": f"Histogram of {col}",
                "data": histogram_data,
            }
        )

    if len(numeric) >= 2:
        x_col = numeric[0]
        y_col = numeric[1]

        scatter_df = df[[x_col, y_col]].dropna().head(100)

        charts["scatter_plots"].append(
            {
                "title": f"{x_col} vs {y_col}",
                "x_axis": x_col,
                "y_axis": y_col,
                "data": scatter_df.to_dict(orient="records"),
            }
        )

    if categorical and numeric:
        cat_col = categorical[0]
        num_col = numeric[0]

        grouped = (
            df.groupby(cat_col)[num_col]
            .mean(numeric_only=True)
            .sort_values(ascending=False)
            .head(10)
        )

        charts["top_n_charts"].append(
            {
                "title": f"Top {cat_col} by average {num_col}",
                "x_axis": cat_col,
                "y_axis": num_col,
                "data": [
                    {
                        cat_col: str(category),
                        num_col: safe_value(value),
                    }
                    for category, value in grouped.items()
                ],
            }
        )

    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True).fillna(0)

        heatmap_data = []

        for row_col in corr.index:
            for column_col in corr.columns:
                heatmap_data.append(
                    {
                        "x": str(column_col),
                        "y": str(row_col),
                        "value": round(float(corr.loc[row_col, column_col]), 4),
                    }
                )

        charts["correlation_heatmap"] = {
            "title": "Correlation Heatmap",
            "data": heatmap_data,
        }

    return charts


def generate_domain_insights(df: pd.DataFrame, domain: str, schema: dict) -> list[str]:
    insights = []

    numeric = schema["numeric_columns"]
    categorical = schema["categorical_columns"]

    if domain == "business":
        insights.append(
            "Business mode activated: analyzing revenue, profit, category performance, and business anomalies."
        )
    elif domain == "finance":
        insights.append(
            "Finance mode activated: analyzing volatility, numeric trends, financial distribution, and risk indicators."
        )
    elif domain == "research":
        insights.append(
            "Research mode activated: analyzing variable distributions, data quality, and statistical structure."
        )
    else:
        insights.append("General analytics mode activated.")

    if numeric:
        highest_mean_col = max(
            numeric,
            key=lambda col: pd.to_numeric(df[col], errors="coerce").mean(skipna=True),
        )

        insights.append(
            f"'{highest_mean_col}' has the highest average value among numeric columns."
        )

    if categorical:
        most_diverse_col = max(
            categorical,
            key=lambda col: df[col].nunique(dropna=True),
        )

        insights.append(
            f"'{most_diverse_col}' has the highest categorical diversity."
        )

    missing_total = int(df.isna().sum().sum())

    if missing_total > 0:
        insights.append(
            f"The dataset contains {missing_total} missing values. Cleaning is recommended before model training."
        )
    else:
        insights.append("No missing values were detected.")

    return insights


def generate_dataset_intelligence(df: pd.DataFrame, schema: dict, quality: dict) -> list[str]:
    intelligence = [
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.",
        f"Detected {len(schema['numeric_columns'])} numeric columns.",
        f"Detected {len(schema['categorical_columns'])} categorical columns.",
        f"Data quality score is {quality['data_quality_score']}/100 with grade '{quality['grade']}'.",
    ]

    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows > 0:
        intelligence.append(f"{duplicate_rows} duplicate rows were detected.")
    else:
        intelligence.append("No duplicate rows were detected.")

    return intelligence


def analyze_file(file_path: str, domain: str) -> dict:
    df = read_uploaded_file(file_path)

    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    schema = detect_schema(df)
    quality = calculate_data_quality(df)

    statistics = {
        "data_quality": quality,
        "data_quality_breakdown": {
            "missing_cells": quality["missing_cells"],
            "duplicate_rows": quality["duplicate_rows"],
        },
        "dataset_intelligence": generate_dataset_intelligence(df, schema, quality),
        "numeric_summary": numeric_summary(df, schema["numeric_columns"]),
        "categorical_summary": categorical_summary(df, schema["categorical_columns"]),
    }

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "domain": domain,
        "schema": schema,
        "statistics": statistics,
        "domain_insights": generate_domain_insights(df, domain, schema),
        "charts": generate_charts(df, schema),
        "large_dataset_mode": False,
    }