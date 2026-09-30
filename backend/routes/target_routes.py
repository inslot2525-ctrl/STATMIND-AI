from fastapi import APIRouter, Form, HTTPException

from services.target_advisor import recommend_target_from_file
from services.storage import UPLOAD_DIR, resolve_in

router = APIRouter()


@router.post("/recommend-target")
def recommend_target(
    filename: str = Form(...),
    domain: str = Form("business"),
):
    try:
        file_path = resolve_in(UPLOAD_DIR, filename)

        return recommend_target_from_file(file_path, domain)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
