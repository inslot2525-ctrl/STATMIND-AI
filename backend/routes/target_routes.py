from fastapi import APIRouter, Depends, Form, HTTPException

from database import User
from routes.deps import resolve_user_dataset
from services.auth import get_current_user
from services.dataset_registry import dataset_local_path
from services.target_advisor import recommend_target_from_file

router = APIRouter()


@router.post("/recommend-target")
def recommend_target(
    filename: str = Form(None),
    dataset_id: str = Form(None),
    domain: str = Form("business"),
    user: User = Depends(get_current_user),
):
    dataset = resolve_user_dataset(user.id, dataset_id, filename)
    try:
        return recommend_target_from_file(dataset_local_path(dataset), domain)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
