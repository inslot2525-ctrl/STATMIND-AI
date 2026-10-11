import hashlib
import uuid
from datetime import datetime

from sqlalchemy.exc import IntegrityError

from database import SessionLocal, Dataset
from services import lakefs_versioning
from services.file_reader import read_uploaded_file
from services.storage import get_storage


def _calculate_file_hash(filepath):
    """Calculate SHA256 hash of file content"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _to_dict(obj):
    if not obj:
        return None
    return {
        "id": obj.id,
        "content_hash": obj.content_hash,
        "original_filename": obj.original_filename,
        "stored_filename": obj.stored_filename,
        "storage_key": obj.storage_key,
        "upload_time": obj.upload_time,
        "rows": obj.rows,
        "columns": obj.columns,
        "missing_cells": obj.missing_cells,
        "duplicate_rows": obj.duplicate_rows,
        "error": obj.error,
        "data_commit": obj.data_commit,
        "data_commit_url": lakefs_versioning.commit_url(obj.data_commit),
    }


def register_dataset(owner_id, stored_name, storage_key, local_path, original_filename):
    """
    Register an uploaded dataset for a user and return its dataset_id.
    Uploading identical content again returns the user's existing dataset_id.
    """
    file_hash = _calculate_file_hash(local_path)
    db = SessionLocal()
    try:
        existing = db.query(Dataset).filter_by(owner_id=owner_id, content_hash=file_hash).first()
        if existing:
            return existing.id

        try:
            df = read_uploaded_file(local_path)
            rows, cols = df.shape
            missing_count = int(df.isna().sum().sum())
            duplicate_rows = int(df.duplicated().sum())
            error = None
        except Exception as e:
            rows = cols = missing_count = duplicate_rows = None
            error = str(e)

        dataset_id = uuid.uuid4().hex

        # With lakeFS on, the upload becomes a commit and the dataset remembers it
        data_commit = lakefs_versioning.commit(
            f"Add dataset {original_filename}",
            {
                "statmind.kind": "dataset",
                "statmind.dataset_id": dataset_id,
                "statmind.owner_id": owner_id,
                "statmind.content_hash": file_hash,
                "statmind.storage_key": storage_key,
            },
        )

        ds = Dataset(
            id=dataset_id,
            owner_id=owner_id,
            content_hash=file_hash,
            original_filename=original_filename,
            stored_filename=stored_name,
            storage_key=storage_key,
            upload_time=datetime.now().isoformat(),
            rows=rows,
            columns=cols,
            missing_cells=missing_count,
            duplicate_rows=duplicate_rows,
            file_path=storage_key,
            error=error,
            data_commit=data_commit,
        )
        db.add(ds)
        try:
            db.commit()
        except IntegrityError:
            # The same user uploaded the same file concurrently; use the winner's row
            db.rollback()
            return db.query(Dataset).filter_by(owner_id=owner_id, content_hash=file_hash).one().id
        return ds.id
    finally:
        db.close()


def get_dataset(dataset_id, owner_id):
    """Get metadata for a dataset the user owns, or None."""
    db = SessionLocal()
    try:
        obj = db.query(Dataset).filter_by(id=str(dataset_id), owner_id=owner_id).first()
        return _to_dict(obj)
    finally:
        db.close()


def get_dataset_by_stored_name(stored_name, owner_id):
    """Look up a user's dataset by the `filename` returned from /api/analyze."""
    db = SessionLocal()
    try:
        obj = (
            db.query(Dataset)
            .filter_by(stored_filename=stored_name, owner_id=owner_id)
            .order_by(Dataset.upload_time.desc())
            .first()
        )
        return _to_dict(obj)
    finally:
        db.close()


def list_datasets(owner_id):
    """List the user's registered datasets, newest first."""
    db = SessionLocal()
    try:
        objs = db.query(Dataset).filter_by(owner_id=owner_id).order_by(Dataset.upload_time.desc()).all()
        return [_to_dict(o) for o in objs]
    finally:
        db.close()


def dataset_local_path(dataset: dict) -> str:
    """A readable local file for a dataset (downloads from object storage if needed)."""
    if not dataset.get("storage_key"):
        raise ValueError("Dataset file is missing. Please upload it again.")
    return get_storage().local_path(dataset["storage_key"])
