from fastapi import HTTPException

from services.dataset_registry import get_dataset, get_dataset_by_stored_name


def resolve_user_dataset(owner_id: str, dataset_id: str | None = None, filename: str | None = None) -> dict:
    """
    Find one of the user's datasets by id, or by the stored `filename` that
    /api/analyze returned. Another user's dataset is reported as not found.
    """
    if dataset_id:
        dataset = get_dataset(dataset_id, owner_id)
    elif filename:
        dataset = get_dataset_by_stored_name(filename, owner_id)
    else:
        raise HTTPException(status_code=400, detail="Either filename or dataset_id must be provided")

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload and analyze it first.")
    return dataset
