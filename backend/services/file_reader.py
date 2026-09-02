import os
import pandas as pd


def read_uploaded_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise ValueError("File not found.")
    lower_path = file_path.lower()
    if lower_path.endswith(".csv"):
        # Optimized: try fast engine, handle large files with low_memory, use pyarrow if available
        for encoding in ("utf-8", "latin1"):
            try:
                # For very large files, read with pyarrow or c engine quickly
                # Use on_bad_lines skip to avoid choke on malformed rows
                return pd.read_csv(
                    file_path,
                    encoding=encoding,
                    engine="c",
                    low_memory=False,
                    on_bad_lines="skip",
                )
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # fallback without on_bad_lines for older pandas
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except Exception:
                    raise e
        return pd.read_csv(file_path, encoding="latin1", engine="c", low_memory=False, on_bad_lines="skip")
    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        # openpyxl is faster for xlsx; limit reading to first sheet
        return pd.read_excel(file_path, engine="openpyxl" if lower_path.endswith(".xlsx") else None)
    raise ValueError("Unsupported file type. Upload CSV or Excel only.")
