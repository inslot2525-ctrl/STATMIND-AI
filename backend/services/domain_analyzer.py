def generate_domain_insights(dataframe, schema: dict, statistics: dict, domain: str) -> list:
    """
    Generates rule-based domain-specific insights.
    """

    if domain == "business":
        return analyze_business(schema, statistics)

    if domain == "research":
        return analyze_research(schema, statistics)

    if domain == "finance":
        return analyze_finance(schema, statistics)

    return []


def analyze_business(schema: dict, statistics: dict) -> list:
    insights = []

    numeric_summary = statistics.get("numeric_summary", {})
    categorical_summary = statistics.get("categorical_summary", {})

    insights.append("Business mode activated: focusing on KPIs, trends, category performance, and anomalies.")

    for column, stats in numeric_summary.items():
        if any(keyword in column.lower() for keyword in ["sales", "revenue", "profit", "income"]):
            insights.append(
                f"{column} has an average value of {round(stats['mean'], 2)} with a maximum of {round(stats['max'], 2)}."
            )

    for column, stats in categorical_summary.items():
        if stats.get("top_values"):
            top_category = list(stats["top_values"].keys())[0]
            insights.append(
                f"The most frequent category in {column} is '{top_category}'."
            )

    return insights


def analyze_research(schema: dict, statistics: dict) -> list:
    insights = []

    numeric_columns = schema.get("numeric_columns", [])

    insights.append("Research mode activated: focusing on descriptive statistics, distributions, and variable relationships.")

    if len(numeric_columns) >= 2:
        insights.append(
            "The dataset has at least two numeric variables, so correlation analysis can be performed."
        )

    for column, stats in statistics.get("numeric_summary", {}).items():
        insights.append(
            f"{column}: mean = {round(stats['mean'], 2)}, standard deviation = {round(stats['std_dev'], 2)}."
        )

    return insights


def analyze_finance(schema: dict, statistics: dict) -> list:
    insights = []

    numeric_summary = statistics.get("numeric_summary", {})

    insights.append("Finance mode activated: focusing on income, expenses, profit, risk, and financial trends.")

    for column, stats in numeric_summary.items():
        lower_col = column.lower()

        if any(keyword in lower_col for keyword in ["expense", "cost", "debit"]):
            insights.append(
                f"{column} average spending/cost is {round(stats['mean'], 2)}."
            )

        if any(keyword in lower_col for keyword in ["income", "revenue", "credit", "profit"]):
            insights.append(
                f"{column} average inflow/profit-related value is {round(stats['mean'], 2)}."
            )

    insights.append("This system provides financial analytics only, not investment advice.")

    return insights