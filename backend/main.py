from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.analysis_routes import router as analysis_router
from routes.target_routes import router as target_router
from routes.ml_routes import router as ml_router, model_router, experiment_router
from routes.prediction_routes import router as prediction_router
from routes.intelligence_routes import router as intelligence_router

app = FastAPI(
    title="StatMind AI API",
    description="Statistics, ML, AutoML, target recommendation, prediction, leakage detection, AI analyst, and report generation API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api", tags=["Analysis"])
app.include_router(target_router, prefix="/api", tags=["Target Advisor"])
app.include_router(ml_router, prefix="/api", tags=["Machine Learning"])
app.include_router(experiment_router, prefix="/api", tags=["Experiments"])
app.include_router(model_router, prefix="/api", tags=["Model Registry"])
app.include_router(prediction_router, prefix="/api", tags=["Prediction Studio"])
app.include_router(intelligence_router, prefix="/api", tags=["Intelligence Center"])


@app.get("/")
def root():
    return {
        "message": "StatMind AI backend is running",
        "docs": "/docs",
        "features": [
            "Statistics Studio",
            "Target Advisor",
            "ML Studio",
            "AutoML Compare",
            "Prediction Studio",
            "Leakage Detector",
            "AI Data Scientist Chatbot",
            "Document Generator",
            "Dataset Versioning",
            "Experiment Tracking",
            "Model Registry",
        ],
    }


@app.get("/api/health")
def health():
    """Health check with DB connectivity."""
    try:
        from sqlalchemy import text
        from database import SessionLocal

        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
    status = "ok" if db_status == "ok" else "degraded"
    return {"status": status, "database": db_status, "version": "1.0.0"}