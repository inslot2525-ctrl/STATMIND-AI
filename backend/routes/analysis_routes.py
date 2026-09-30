import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.analysis_engine import analyze_file
from services.dataset_registry import register_dataset, list_datasets
from services.storage import save_upload

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/datasets")
def get_datasets():
    try:
        datasets = list_datasets()
        return {"datasets": datasets}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so heavy
# pandas work does not block the event loop for other requests.
@router.post("/analyze")
def analyze_dataset(
    file: UploadFile = File(...),
    domain: str = Form("business"),
):
    try:
        # The client filename is metadata only; the file is stored under its content hash.
        stored_name, file_path = save_upload(file)
        logger.info("Analyze request: %s stored as %s (domain=%s)", file.filename, stored_name, domain)

        dataset_id = register_dataset(file_path, file.filename)

        result = analyze_file(file_path, domain)

        result["filename"] = stored_name
        result["original_filename"] = file.filename
        result["dataset_id"] = dataset_id

        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
