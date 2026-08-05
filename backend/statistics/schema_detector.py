import pandas as pd


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Detects column types and basic data quality information.
    """

    schema = {
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "missing_values": {},
        "duplicate_rows": int(df.duplicated().sum())
    }

    for column in df.columns:
        missing_count = int(df[column].isnull().sum())
        schema["missing_values"][column] = missing_count

        if pd.api.types.is_numeric_dtype(df[column]):
            schema["numeric_columns"].append(column)

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            schema["datetime_columns"].append(column)

        else:
            # Try detecting date columns stored as strings
            try:
                converted = pd.to_datetime(df[column], errors="coerce")
                valid_ratio = converted.notnull().mean()

                if valid_ratio > 0.7:
                    schema["datetime_columns"].append(column)
                else:
                    unique_ratio = df[column].nunique() / max(len(df), 1)

                    if unique_ratio < 0.5:
                        schema["categorical_columns"].append(column)
                    else:
                        schema["text_columns"].append(column)

            except Exception:
                schema["text_columns"].append(column)

    return schema