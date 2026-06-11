import os
from fastapi import APIRouter, Form, HTTPException

from services.ml_engine import train_model_from_file, compare_models_from_file

router = APIRouter()

UPLOAD_DIR = "uploads"


@router.post("/train-model")
async def train_model(
    filename: str = Form(...),
    target_column: str = Form(...),
    algorithm: str = Form(...),
    test_size: float = Form(0.2),
):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)

        result = train_model_from_file(
            file_path=file_path,
            target_column=target_column,
            algorithm=algorithm,
            test_size=test_size,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-models")
async def compare_models(
    filename: str = Form(...),
    target_column: str = Form(...),
    test_size: float = Form(0.2),
):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)

        result = compare_models_from_file(
            file_path=file_path,
            target_column=target_column,
            test_size=test_size,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))