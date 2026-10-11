"""
Background job implementations. Each task receives the JSON params stored on the
job, the owning user's id and a progress callback, and returns a JSON-able result.
Raise ValueError for problems the user can fix; the message is shown to them.
"""
from services.dataset_registry import dataset_local_path, get_dataset
from services.ml_engine import compare_models_from_file, train_model_from_file


def _dataset_path(params: dict, owner_id: str) -> str:
    dataset = get_dataset(params["dataset_id"], owner_id)
    if not dataset:
        raise ValueError(f"Dataset {params['dataset_id']} not found")
    return dataset_local_path(dataset)


def train_model_task(params: dict, owner_id: str, progress) -> dict:
    return train_model_from_file(
        file_path=_dataset_path(params, owner_id),
        target_column=params["target_column"],
        algorithm=params["algorithm"],
        test_size=params["test_size"],
        dataset_id=params["dataset_id"],
        owner_id=owner_id,
        task_type=params.get("task_type"),
        progress=progress,
    )


def compare_models_task(params: dict, owner_id: str, progress) -> dict:
    return compare_models_from_file(
        file_path=_dataset_path(params, owner_id),
        target_column=params["target_column"],
        test_size=params["test_size"],
        dataset_id=params["dataset_id"],
        owner_id=owner_id,
        task_type=params.get("task_type"),
        progress=progress,
    )


TASKS = {
    "train_model": train_model_task,
    "compare_models": compare_models_task,
}
