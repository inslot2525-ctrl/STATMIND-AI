"""
Central file-storage helpers.

Every path the API reads or writes goes through this module, so that:
  * user-supplied names can never escape their storage directory (path traversal),
  * uploads are stored under a content hash, so two different files that share a
    name never overwrite each other and a dataset_id always points at the exact
    bytes it was registered with,
  * the storage root can be moved (tests, Docker volumes) with STATMIND_DATA_DIR.
"""
import hashlib
import os
import re
import tempfile

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.getenv("STATMIND_DATA_DIR", BACKEND_DIR))

UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
MODEL_DIR = os.path.join(DATA_DIR, "models")
PREDICTION_DIR = os.path.join(DATA_DIR, "predictions")
REPORT_DIR = os.path.join(DATA_DIR, "reports")

for _dir in (UPLOAD_DIR, MODEL_DIR, PREDICTION_DIR, REPORT_DIR):
    os.makedirs(_dir, exist_ok=True)

ALLOWED_DATA_EXTENSIONS = (".csv", ".xlsx", ".xls")
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._()-]*$")


def resolve_in(directory: str, name: str) -> str:
    """
    Return the absolute path of `name` inside `directory`.

    Raises ValueError if the name contains path separators or unusual characters,
    or would resolve to a location outside `directory`.
    """
    if not name or not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError("Invalid file name.")

    base = os.path.realpath(directory)
    path = os.path.realpath(os.path.join(base, name))

    if os.path.commonpath([base, path]) != base:
        raise ValueError("Invalid file name.")

    return path


def safe_component(value: str) -> str:
    """Turn arbitrary text (e.g. a column name) into a safe file-name fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned[:60] or "x"


def data_extension(filename: str) -> str:
    """Return the lower-case extension of an uploaded data file, or raise ValueError."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_DATA_EXTENSIONS:
        raise ValueError("Only CSV and Excel files are supported.")
    return ext


def save_upload(upload_file) -> tuple[str, str]:
    """
    Stream an UploadFile to disk under a content-addressed name.

    Returns (stored_name, absolute_path). The original client filename is never
    used as a path; it is only kept as metadata by the caller.
    """
    if not upload_file.filename:
        raise ValueError("No file uploaded.")

    ext = data_extension(upload_file.filename)
    sha256 = hashlib.sha256()
    size = 0

    fd, tmp_path = tempfile.mkstemp(dir=UPLOAD_DIR, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := upload_file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"File is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                    )
                sha256.update(chunk)
                out.write(chunk)

        if size == 0:
            raise ValueError("Uploaded file is empty.")

        stored_name = f"{sha256.hexdigest()[:32]}{ext}"
        final_path = os.path.join(UPLOAD_DIR, stored_name)
        os.replace(tmp_path, final_path)
        return stored_name, final_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
