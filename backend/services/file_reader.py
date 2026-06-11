import os
import pandas as pd


def read_uploaded_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise ValueError("File not found.")

    lower_path = file_path.lower()

    if lower_path.endswith(".csv"):
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="latin1")

    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        return pd.read_excel(file_path)

    raise ValueError("Unsupported file type. Upload CSV or Excel only.")