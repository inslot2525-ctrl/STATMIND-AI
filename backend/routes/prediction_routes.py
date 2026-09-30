import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from services.ml_engine import predict_with_saved_model
from services.storage import PREDICTION_DIR, resolve_in, save_upload

router = APIRouter()


@router.post("/predict-test-data")
def predict_test_data(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    dataset_id: str = Form(None),  # Optional dataset_id for tracking
):
    try:
        _, test_file_path = save_upload(file)

        result = predict_with_saved_model(
            model_id=model_id,
            test_file_path=test_file_path,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-predictions/{filename}")
def download_predictions(filename: str):
    try:
        file_path = resolve_in(PREDICTION_DIR, filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Prediction file not found.")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv",
    )
