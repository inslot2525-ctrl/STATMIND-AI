import pandas as pd


def read_uploaded_file(file_path: str) -> pd.DataFrame:
    """
    Reads CSV or Excel files and returns a pandas DataFrame.
    """

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)

    else:
        raise ValueError("Unsupported file type. Upload CSV or Excel files only.")

    if df.empty:
        raise ValueError("Uploaded file is empty.")

    return df