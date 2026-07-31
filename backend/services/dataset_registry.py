import os
import json
import hashlib
from datetime import datetime
from services.file_reader import read_uploaded_file

DATASET_REGISTRY_PATH = "dataset_registry.json"
DATASETS_DIR = "uploads"

def _load_registry():
    if not os.path.exists(DATASET_REGISTRY_PATH):
        return {"datasets": {}, "next_id": 1}
    with open(DATASET_REGISTRY_PATH, 'r') as f:
        return json.load(f)

def _save_registry(data):
    with open(DATASET_REGISTRY_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def _calculate_file_hash(filepath):
    """Calculate SHA256 hash of file content"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def register_dataset(file_path, original_filename):
    """
    Register a dataset in the registry, returning dataset_id.
    If identical content exists, returns existing ID.
    """
    # Calculate content hash
    file_hash = _calculate_file_hash(file_path)

    # Load registry
    registry = _load_registry()

    # Check if we've seen this exact content before
    for data_id, data_info in registry["datasets"].items():
        if data_info.get("content_hash") == file_hash:
            return data_id  # Return existing ID

    # Register new dataset
    dataset_id = str(registry["next_id"])
    registry["next_id"] += 1

    # Read dataset to get metadata
    try:
        df = read_uploaded_file(file_path)
        rows, cols = df.shape

        # Basic stats for quick reference
        missing_count = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        dataset_info = {
            "id": dataset_id,
            "content_hash": file_hash,
            "original_filename": original_filename,
            "stored_filename": os.path.basename(file_path),
            "upload_time": datetime.now().isoformat(),
            "rows": rows,
            "columns": cols,
            "missing_cells": missing_count,
            "duplicate_rows": duplicate_rows,
            "file_path": file_path
        }

        registry["datasets"][dataset_id] = dataset_info
        _save_registry(registry)
        return dataset_id

    except Exception as e:
        # If we can't read the file, still register basic info
        dataset_info = {
            "id": dataset_id,
            "content_hash": file_hash,
            "original_filename": original_filename,
            "stored_filename": os.path.basename(file_path),
            "upload_time": datetime.now().isoformat(),
            "error": str(e),
            "file_path": file_path
        }
        registry["datasets"][dataset_id] = dataset_info
        _save_registry(registry)
        return dataset_id

def get_dataset(dataset_id):
    """Get dataset metadata by ID"""
    registry = _load_registry()
    return registry["datasets"].get(dataset_id)

def list_datasets():
    """List all registered datasets"""
    registry = _load_registry()
    return list(registry["datasets"].values())

def get_dataset_by_hash(file_hash):
    """Find dataset by content hash"""
    registry = _load_registry()
    for data_id, data_info in registry["datasets"].items():
        if data_info.get("content_hash") == file_hash:
            return data_info
    return None