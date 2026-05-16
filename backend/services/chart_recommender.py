def recommend_charts(df, schema: dict) -> list:
    """
    Recommends chart types based on detected column types.
    """

    recommendations = []

    numeric_columns = schema.get("numeric_columns", [])
    categorical_columns = schema.get("categorical_columns", [])
    datetime_columns = schema.get("datetime_columns", [])

    if categorical_columns and numeric_columns:
        recommendations.append({
            "chart_type": "bar_chart",
            "x_axis": categorical_columns[0],
            "y_axis": numeric_columns[0],
            "reason": "Best for comparing numeric values across categories."
        })

    if datetime_columns and numeric_columns:
        recommendations.append({
            "chart_type": "line_chart",
            "x_axis": datetime_columns[0],
            "y_axis": numeric_columns[0],
            "reason": "Best for showing trends over time."
        })

    if numeric_columns:
        recommendations.append({
            "chart_type": "histogram",
            "x_axis": numeric_columns[0],
            "y_axis": None,
            "reason": "Best for understanding the distribution of a numeric variable."
        })

    if len(numeric_columns) >= 2:
        recommendations.append({
            "chart_type": "scatter_plot",
            "x_axis": numeric_columns[0],
            "y_axis": numeric_columns[1],
            "reason": "Best for checking relationships between two numeric variables."
        })

        recommendations.append({
            "chart_type": "correlation_heatmap",
            "x_axis": numeric_columns,
            "y_axis": numeric_columns,
            "reason": "Best for identifying correlations between numeric variables."
        })

    return recommendations