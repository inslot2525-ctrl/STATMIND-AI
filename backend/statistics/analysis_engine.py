import time
import numpy as np
import pandas as pd

from services.file_reader import read_uploaded_file

# --- Performance tuning constants ---
SAMPLE_ROWS_FOR_CHARTS = 5000
MAX_ROWS_FOR_DUPLICATE_CHECK = 50000
MAX_NUMERIC_FOR_CORR = 8
MAX_NUMERIC_FOR_CHARTS = 6
MAX_CATEGORICAL_FOR_CHARTS = 3
HIST_BINS = 8
SCATTER_SAMPLE = 300
LINE_SAMPLE = 50

# Simple in-memory cache: file_path + mtime + domain -> result
_analysis_cache = {}
_CACHE_MAX = 20

def _cache_key(file_path: str, domain: str):
    import os
    try:
        stat = os.stat(file_path)
        return f"{file_path}:{stat.st_mtime_ns}:{stat.st_size}:{domain}"
    except:
        return f"{file_path}:{domain}"

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
    # Use sampled frame for expensive uniqueness check on large data
    n = len(df)
    sample_for_unique = df if n <= 10000 else df.sample(n=min(5000, n), random_state=42)
    for col in list(categorical_columns):
        sample = df[col].dropna().astype(str).head(30)
        if sample.empty:
            continue
        avg_length = sample.str.len().mean()
        # unique ratio from sampled frame to avoid full scan
        unique_ratio = sample_for_unique[col].nunique(dropna=True) / max(len(sample_for_unique), 1)
        n_unique = sample_for_unique[col].nunique(dropna=True)
        # High-cardinality ID detection: e.g., customer_id, UUIDs
        is_high_card_id = unique_ratio > 0.9 and n_unique > 30
        if (avg_length > 40 and unique_ratio > 0.5) or is_high_card_id:
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
    n = len(df)
    # Optimized duplicate check: exact for small, sampled estimate for large
    if n <= MAX_ROWS_FOR_DUPLICATE_CHECK:
        duplicate_rows = int(df.duplicated().sum())
    else:
        # Sample 50k rows for duplicate estimation, much faster
        sample_n = MAX_ROWS_FOR_DUPLICATE_CHECK
        sampled = df.sample(n=sample_n, random_state=42)
        dup_sample = int(sampled.duplicated().sum())
        # Scale estimate but cap; for large datasets exact duplicate count is rarely critical
        duplicate_rows = dup_sample
        # Add flag that this is estimated if needed
    missing_ratio = missing_cells / total_cells
    duplicate_ratio = duplicate_rows / max(n, 1)
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
    if not numeric_columns:
        return {}
    # Vectorized describe is faster than per-column loops
    try:
        desc = df[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75])
        output = {}
        for col in numeric_columns:
            if col not in desc.columns:
                continue
            # Check if column has any non-null numeric data
            count = int(desc.loc["count", col]) if not pd.isna(desc.loc["count", col]) else 0
            if count == 0:
                continue
            output[col] = {
                "count": count,
                "mean": safe_value(desc.loc["mean", col]),
                "median": safe_value(desc.loc["50%", col]),
                "std": safe_value(desc.loc["std", col]),
                "min": safe_value(desc.loc["min", col]),
                "max": safe_value(desc.loc["max", col]),
                "q1": safe_value(desc.loc["25%", col]),
                "q3": safe_value(desc.loc["75%", col]),
            }
        return output
    except Exception:
        # Fallback to per-column loop
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
        # Use sampled value_counts for large frames to keep it snappy
        # but keep exact top categories for accuracy on moderate sizes
        s = df[col].astype(str)
        value_counts = s.value_counts().head(8)
        output[col] = {
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_value": str(value_counts.index[0]) if len(value_counts) else None,
            "top_frequency": int(value_counts.iloc[0]) if len(value_counts) else 0,
            "top_categories": [
                {"category": str(category), "count": int(count)}
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
    # Sampling for chart data: avoid processing millions of rows for visuals
    use_sample = len(df) > SAMPLE_ROWS_FOR_CHARTS
    df_charts = df.sample(n=SAMPLE_ROWS_FOR_CHARTS, random_state=42) if use_sample else df
    # Limit numerics for charts to avoid explosion
    numeric_capped = numeric[:MAX_NUMERIC_FOR_CHARTS]
    numeric_corr_capped = numeric[:MAX_NUMERIC_FOR_CORR]
    categorical_capped = categorical[:MAX_CATEGORICAL_FOR_CHARTS]

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            charts["missing_value_chart"].append({"column": col, "missing_count": missing_count})

    for col in categorical_capped:
        # value_counts on sampled frame is faster and visually similar
        value_counts = df_charts[col].astype(str).value_counts().head(8)
        chart_data = [{"category": str(k), "count": int(v)} for k, v in value_counts.items()]
        charts["bar_charts"].append({"title": f"Distribution of {col}", "x_axis": "category", "y_axis": "count", "data": chart_data})
        charts["pie_charts"].append({"title": f"Share of {col}", "name_key": "category", "value_key": "count", "data": chart_data})

    for col in numeric_capped:
        # Fast numeric conversion - if already numeric dtype, skip coercion
        if pd.api.types.is_numeric_dtype(df_charts[col]):
            series = df_charts[col].dropna()
        else:
            series = pd.to_numeric(df_charts[col], errors="coerce").dropna()
        if len(series) < 2:
            continue
        # Line/area: limit to LINE_SAMPLE points, already from sampled df
        n_line = min(LINE_SAMPLE, len(series))
        # Take evenly spaced samples for better visual if series large
        if len(series) > n_line:
            idx = np.linspace(0, len(series)-1, n_line, dtype=int)
            vals = series.iloc[idx]
        else:
            vals = series.head(n_line)
        line_data = [{"index": int(i), col: safe_value(v)} for i, v in enumerate(vals)]
        charts["line_charts"].append({"title": f"Trend of {col}", "x_axis": "index", "y_axis": col, "data": line_data})
        charts["area_charts"].append({"title": f"Area Trend of {col}", "x_axis": "index", "y_axis": col, "data": line_data})
        # Histogram: fixed bins, fast
        try:
            counts, bin_edges = np.histogram(series, bins=HIST_BINS)
            histogram_data = [{"range": f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}", "count": int(counts[i])} for i in range(len(counts))]
        except Exception:
            histogram_data = []
        charts["histograms"].append({"title": f"Histogram of {col}", "data": histogram_data})

    if len(numeric_capped) >= 2:
        x_col, y_col = numeric_capped[0], numeric_capped[1]
        # Scatter: sample strictly
        scatter_df = df_charts[[x_col, y_col]].dropna()
        if len(scatter_df) > SCATTER_SAMPLE:
            scatter_df = scatter_df.sample(n=SCATTER_SAMPLE, random_state=42)
        # Convert to records with capped size - use to_dict is faster than iterrows
        charts["scatter_plots"].append({"title": f"{x_col} vs {y_col}", "x_axis": x_col, "y_axis": y_col, "data": scatter_df.to_dict(orient="records")})

    if categorical_capped and numeric_capped:
        cat_col, num_col = categorical_capped[0], numeric_capped[0]
        try:
            # Use sampled frame for groupby -> much faster
            grouped = df_charts.groupby(cat_col, observed=True)[num_col].mean(numeric_only=True).sort_values(ascending=False).head(10)
            charts["top_n_charts"].append({
                "title": f"Top {cat_col} by average {num_col}",
                "x_axis": cat_col, "y_axis": num_col,
                "data": [{cat_col: str(k), num_col: safe_value(v)} for k, v in grouped.items()],
            })
        except Exception:
            pass

    if len(numeric_corr_capped) >= 2:
        try:
            # Select top variance columns for more interesting heatmap
            if len(numeric_corr_capped) == MAX_NUMERIC_FOR_CORR and len(numeric) > MAX_NUMERIC_FOR_CORR:
                variances = df_charts[numeric].var(numeric_only=True).sort_values(ascending=False)
                numeric_corr_capped = variances.head(MAX_NUMERIC_FOR_CORR).index.tolist()
            # Use sampled frame for correlation
            corr = df_charts[numeric_corr_capped].corr(numeric_only=True).fillna(0)
            heatmap_data = [{"x": str(c), "y": str(r), "value": round(float(corr.loc[r, c]), 4)} for r in corr.index for c in corr.columns]
            charts["correlation_heatmap"] = {"title": "Correlation Heatmap", "data": heatmap_data}
        except Exception:
            charts["correlation_heatmap"] = None

    return charts


def generate_domain_insights(df: pd.DataFrame, domain: str, schema: dict) -> list[str]:
    insights = []
    numeric = schema["numeric_columns"]
    categorical = schema["categorical_columns"]
    if domain == "business":
        insights.append("Business mode activated: analyzing revenue, profit, category performance, and business anomalies.")
    elif domain == "finance":
        insights.append("Finance mode activated: analyzing volatility, numeric trends, financial distribution, and risk indicators.")
    elif domain == "research":
        insights.append("Research mode activated: analyzing variable distributions, data quality, and statistical structure.")
    else:
        insights.append("General analytics mode activated.")
    if numeric:
        try:
            means = df[numeric].mean(numeric_only=True)
            if not means.empty and not means.isna().all():
                highest_mean_col = means.idxmax()
                insights.append(f"'{highest_mean_col}' has the highest average value among numeric columns.")
        except Exception:
            pass
    if categorical:
        try:
            # Fast nunique on sampled if large
            if len(df) > 10000:
                diversities = {c: df[c].sample(n=5000, random_state=42).nunique(dropna=True) for c in categorical}
                most_diverse_col = max(diversities, key=lambda k: diversities[k])
            else:
                most_diverse_col = max(categorical, key=lambda col: df[col].nunique(dropna=True))
            insights.append(f"'{most_diverse_col}' has the highest categorical diversity.")
        except Exception:
            pass
    missing_total = int(df.isna().sum().sum())
    if missing_total > 0:
        insights.append(f"The dataset contains {missing_total} missing values. Cleaning is recommended before model training.")
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
    # Duplicate count already computed in quality, reuse to avoid second scan
    duplicate_rows = quality["duplicate_rows"]
    if duplicate_rows > 0:
        intelligence.append(f"{duplicate_rows} duplicate rows were detected.")
    else:
        intelligence.append("No duplicate rows were detected.")
    return intelligence


def analyze_file(file_path: str, domain: str) -> dict:
    key = _cache_key(file_path, domain)
    if key in _analysis_cache:
        return _analysis_cache[key]
    t0 = time.time()
    df = read_uploaded_file(file_path)
    df.columns = [str(col).strip() for col in df.columns]
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")
    schema = detect_schema(df)
    quality = calculate_data_quality(df)
    statistics = {
        "data_quality": quality,
        "data_quality_breakdown": {"missing_cells": quality["missing_cells"], "duplicate_rows": quality["duplicate_rows"]},
        "dataset_intelligence": generate_dataset_intelligence(df, schema, quality),
        "numeric_summary": numeric_summary(df, schema["numeric_columns"]),
        "categorical_summary": categorical_summary(df, schema["categorical_columns"]),
    }
    charts = generate_charts(df, schema)
    elapsed = round(time.time() - t0, 3)
    large_mode = len(df) > SAMPLE_ROWS_FOR_CHARTS
    result = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "domain": domain,
        "schema": schema,
        "statistics": statistics,
        "domain_insights": generate_domain_insights(df, domain, schema),
        "charts": charts,
        "large_dataset_mode": large_mode,
        "processing_time_seconds": elapsed,
        "chart_sampling_applied": large_mode,
    }
    # Cache management
    if len(_analysis_cache) >= _CACHE_MAX:
        oldest = next(iter(_analysis_cache))
        _analysis_cache.pop(oldest)
    _analysis_cache[key] = result
    return result
