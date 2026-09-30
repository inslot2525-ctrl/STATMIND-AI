from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from database import User
from services.auth import get_current_user
from services.ml_engine import predict_with_saved_model
from services.storage import save_upload

router = APIRouter()


@router.post("/predict-test-data")
def predict_test_data(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    dataset_id: str = Form(None),  # Optional dataset_id for tracking
    user: User = Depends(get_current_user),
):
    try:
        _, _, test_file_path = save_upload(file)

        # The prediction CSV is stored as an artifact; download it via
        # GET /api/artifacts/{artifact_id}/download-url
        return predict_with_saved_model(
            model_id=model_id,
            test_file_path=test_file_path,
            owner_id=user.id,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
