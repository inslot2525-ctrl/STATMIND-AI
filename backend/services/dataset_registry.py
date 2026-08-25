import os
import hashlib
import json
from datetime import datetime
from services.file_reader import read_uploaded_file
from database import SessionLocal, Dataset, init_db

# Ensure DB and tables exist
init_db()

def _calculate_file_hash(filepath):
    """Calculate SHA256 hash of file content"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def _next_id(db):
    ids = [int(r.id) for r in db.query(Dataset.id).all() if r.id.isdigit()]
    return str(max(ids) + 1) if ids else "1"

def _to_dict(obj):
    if not obj:
        return None
    return {
        "id": obj.id,
        "content_hash": obj.content_hash,
        "original_filename": obj.original_filename,
        "stored_filename": obj.stored_filename,
        "upload_time": obj.upload_time,
        "rows": obj.rows,
        "columns": obj.columns,
        "missing_cells": obj.missing_cells,
        "duplicate_rows": obj.duplicate_rows,
        "file_path": obj.file_path,
        "error": obj.error,
    }

def register_dataset(file_path, original_filename):
    """
    Register a dataset in the registry, returning dataset_id.
    If identical content exists, returns existing ID.
    """
    file_hash = _calculate_file_hash(file_path)
    db = SessionLocal()
    try:
        # Check dedup
        existing = db.query(Dataset).filter_by(content_hash=file_hash).first()
        if existing:
            return existing.id

        dataset_id = _next_id(db)

        # Read dataset to get metadata
        try:
            df = read_uploaded_file(file_path)
            rows, cols = df.shape
            missing_count = int(df.isna().sum().sum())
            duplicate_rows = int(df.duplicated().sum())
            error = None
        except Exception as e:
            rows = cols = missing_count = duplicate_rows = None
            error = str(e)

        ds = Dataset(
            id=dataset_id,
            content_hash=file_hash,
            original_filename=original_filename,
            stored_filename=os.path.basename(file_path),
            upload_time=datetime.now().isoformat(),
            rows=rows,
            columns=cols,
            missing_cells=missing_count,
            duplicate_rows=duplicate_rows,
            file_path=file_path,
            error=error
        )
        db.add(ds)
        db.commit()
        return dataset_id
    finally:
        db.close()

def get_dataset(dataset_id):
    """Get dataset metadata by ID"""
    db = SessionLocal()
    try:
        obj = db.query(Dataset).filter_by(id=str(dataset_id)).first()
        return _to_dict(obj)
    finally:
        db.close()

def list_datasets():
    """List all registered datasets"""
    db = SessionLocal()
    try:
        return [_to_dict(o) for o in db.query(Dataset).all()]
    finally:
        db.close()

def get_dataset_by_hash(file_hash):
    """Find dataset by content hash"""
    db = SessionLocal()
    try:
        obj = db.query(Dataset).filter_by(content_hash=file_hash).first()
        return _to_dict(obj)
    finally:
        db.close()
