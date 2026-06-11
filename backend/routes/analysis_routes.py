import os
import shutil
import traceback

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.analysis_engine import analyze_file

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    domain: str = Form("business"),
):
    try:
        print("===== ANALYZE REQUEST RECEIVED =====")
        print("Filename:", file.filename)
        print("Domain:", domain)

        if not file.filename:
            raise ValueError("No file uploaded.")

        allowed_extensions = [".csv", ".xlsx", ".xls"]

        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError("Only CSV and Excel files are supported.")

        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("File saved at:", file_path)
        print("Starting analysis...")

        result = analyze_file(file_path, domain)

        print("Analysis completed.")
        print("Rows:", result.get("rows"))
        print("Columns:", result.get("columns"))

        result["filename"] = safe_filename

        return result

    except ValueError as exc:
        print("VALUE ERROR:", str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        print("SERVER ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))