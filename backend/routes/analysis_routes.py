import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.file_reader import read_uploaded_file
from services.schema_detector import detect_schema
from services.stats_engine import generate_statistics
from services.chart_recommender import recommend_charts
from services.domain_analyzer import generate_domain_insights

router = APIRouter()

UPLOAD_DIR = "uploads"


@router.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    domain: str = Form(...)
):
    try:
        if domain.lower() not in ["business", "research", "finance"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid domain. Choose business, research, or finance."
            )

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        dataframe = read_uploaded_file(file_path)

        schema = detect_schema(dataframe)

        statistics = generate_statistics(dataframe, schema)

        chart_recommendations = recommend_charts(dataframe, schema)

        domain_insights = generate_domain_insights(
            dataframe=dataframe,
            schema=schema,
            statistics=statistics,
            domain=domain.lower()
        )

        return {
            "filename": file.filename,
            "domain": domain.lower(),
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "schema": schema,
            "statistics": statistics,
            "chart_recommendations": chart_recommendations,
            "domain_insights": domain_insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))