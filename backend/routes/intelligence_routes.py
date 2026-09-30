from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from database import User
from routes.deps import resolve_user_dataset
from services.ai_analyst import answer_data_scientist_question
from services.auth import get_current_user
from services.dataset_registry import dataset_local_path
from services.leakage_detector import detect_leakage_from_file
from services.report_generator import generate_intelligence_report

router = APIRouter()


class AnalystChatRequest(BaseModel):
    question: str
    context: Dict[str, Any]


class ReportRequest(BaseModel):
    context: Dict[str, Any]


class LeakageRequest(BaseModel):
    target_column: str
    filename: str | None = None
    dataset_id: str | None = None


@router.post("/chat-analyst")
def chat_analyst(request: AnalystChatRequest, user: User = Depends(get_current_user)):
    try:
        return answer_data_scientist_question(
            question=request.question,
            context=request.context,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-leakage")
def detect_leakage(request: LeakageRequest, user: User = Depends(get_current_user)):
    dataset = resolve_user_dataset(user.id, request.dataset_id, request.filename)
    try:
        return detect_leakage_from_file(
            file_path=dataset_local_path(dataset),
            target_column=request.target_column,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-report")
def generate_report(request: ReportRequest, user: User = Depends(get_current_user)):
    # The HTML report is stored as an artifact; download it via
    # GET /api/artifacts/{artifact_id}/download-url
    try:
        return generate_intelligence_report(request.context, owner_id=user.id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
