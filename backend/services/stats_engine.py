import pandas as pd
import numpy as np


def generate_statistics(df: pd.DataFrame, schema: dict) -> dict:
    """
    Generates descriptive statistics for numeric and categorical columns.
    """

    results = {
        "numeric_summary": {},
        "categorical_summary": {},
        "data_quality": {}
    }

    numeric_columns = schema.get("numeric_columns", [])
    categorical_columns = schema.get("categorical_columns", [])

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        results["numeric_summary"][column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "std_dev": float(series.std()) if len(series) > 1 else 0.0,
            "variance": float(series.var()) if len(series) > 1 else 0.0,
            "missing_values": int(df[column].isnull().sum())
        }

    for column in categorical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        value_counts = series.value_counts().head(10)

        results["categorical_summary"][column] = {
            "unique_values": int(series.nunique()),
            "top_values": value_counts.to_dict(),
            "missing_values": int(df[column].isnull().sum())
        }

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())

    data_quality_score = 100

    if total_cells > 0:
        missing_percentage = (missing_cells / total_cells) * 100
        data_quality_score -= min(missing_percentage, 40)

    duplicate_rows = int(df.duplicated().sum())

    if df.shape[0] > 0:
        duplicate_percentage = (duplicate_rows / df.shape[0]) * 100
        data_quality_score -= min(duplicate_percentage, 20)

    results["data_quality"] = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "total_missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "data_quality_score": round(max(data_quality_score, 0), 2)
    }

    return results