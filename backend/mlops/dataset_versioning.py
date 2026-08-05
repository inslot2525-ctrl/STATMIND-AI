import os
import json
import hashlib
from datetime import datetime
from file_reader import read_uploaded_file

DATASET_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "dataset_registry.json")

def _load_registry():
    os.makedirs(os.path.dirname(DATASET_REGISTRY_PATH), exist_ok=True)
    if not os.path.exists(DATASET_REGISTRY_PATH):
        return {"datasets": {}, "next_id": 1}
    with open(DATASET_REGISTRY_PATH, "r") as f:
        return json.load(f)

def _save_registry(data):
    os.makedirs(os.path.dirname(DATASET_REGISTRY_PATH), exist_ok=True)
    with open(DATASET_REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)

def _calculate_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def register_dataset(file_path, original_filename):
    file_hash = _calculate_file_hash(file_path)
    registry = _load_registry()
    for data_id, data_info in registry["datasets"].items():
        if data_info.get("content_hash") == file_hash:
            return data_id
    dataset_id = str(registry["next_id"])
    registry["next_id"] += 1
    try:
        df = read_uploaded_file(file_path)
        rows, cols = df.shape
        missing_count = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        dataset_info = {
            "id": dataset_id, "content_hash": file_hash,
            "original_filename": original_filename,
            "stored_filename": os.path.basename(file_path),
            "upload_time": datetime.now().isoformat(),
            "rows": rows, "columns": cols,
            "missing_cells": missing_count,
            "duplicate_rows": duplicate_rows,
            "file_path": file_path,
        }
    except Exception as e:
        dataset_info = {
            "id": dataset_id, "content_hash": file_hash,
            "original_filename": original_filename,
            "stored_filename": os.path.basename(file_path),
            "upload_time": datetime.now().isoformat(),
            "error": str(e), "file_path": file_path,
        }
    registry["datasets"][dataset_id] = dataset_info
    _save_registry(registry)
    return dataset_id

def get_dataset(dataset_id):
    registry = _load_registry()
    return registry["datasets"].get(dataset_id)

def list_datasets():
    registry = _load_registry()
    return list(registry["datasets"].values())
