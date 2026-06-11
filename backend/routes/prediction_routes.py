import os
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from services.ml_engine import predict_with_saved_model

router = APIRouter()

UPLOAD_DIR = "uploads"
PREDICTION_DIR = "predictions"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)


@router.post("/predict-test-data")
async def predict_test_data(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        filename = file.filename

        if not filename:
            raise ValueError("No test file uploaded.")

        if not (
            filename.endswith(".csv")
            or filename.endswith(".xlsx")
            or filename.endswith(".xls")
        ):
            raise ValueError("Only CSV and Excel files are supported.")

        test_file_path = os.path.join(UPLOAD_DIR, f"test_{filename}")

        with open(test_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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
async def download_predictions(filename: str):
    try:
        file_path = os.path.join(PREDICTION_DIR, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Prediction file not found.")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/csv",
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))