import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict

from services.ai_analyst import answer_data_scientist_question
from services.leakage_detector import detect_leakage_from_file
from services.report_generator import generate_intelligence_report

router = APIRouter()

UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


class AnalystChatRequest(BaseModel):
    question: str
    context: Dict[str, Any]


class ReportRequest(BaseModel):
    context: Dict[str, Any]


@router.post("/chat-analyst")
async def chat_analyst(request: AnalystChatRequest):
    try:
        result = answer_data_scientist_question(
            question=request.question,
            context=request.context,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-leakage")
async def detect_leakage(payload: Dict[str, Any]):
    try:
        filename = payload.get("filename")
        target_column = payload.get("target_column")

        if not filename:
            raise ValueError("filename is required.")

        if not target_column:
            raise ValueError("target_column is required.")

        file_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.exists(file_path):
            raise ValueError("Uploaded dataset file not found. Run analysis first.")

        result = detect_leakage_from_file(
            file_path=file_path,
            target_column=target_column,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-report")
async def generate_report(request: ReportRequest):
    try:
        result = generate_intelligence_report(request.context)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-report/{filename}")
async def download_report(filename: str):
    try:
        file_path = os.path.join(REPORT_DIR, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report file not found.")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/html",
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))