import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("statmind")

import config  # noqa: E402
from database import init_db  # noqa: E402
from services.auth import get_current_user  # noqa: E402
from services.storage import get_storage  # noqa: E402
from routes.analysis_routes import router as analysis_router  # noqa: E402
from routes.artifact_routes import router as artifact_router, public_router as artifact_public_router  # noqa: E402
from routes.auth_routes import router as auth_router  # noqa: E402
from routes.intelligence_routes import router as intelligence_router  # noqa: E402
from routes.job_routes import router as job_router  # noqa: E402
from routes.ml_routes import router as ml_router, model_router, experiment_router  # noqa: E402
from routes.prediction_routes import router as prediction_router  # noqa: E402
from routes.target_routes import router as target_router  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    get_storage()  # fail fast on bad storage configuration
    worker = None
    if config.JOB_BACKEND == "thread":
        from services.jobs import start_background_worker

        worker = start_background_worker()
    logger.info(
        "StatMind API ready (env=%s, storage=%s, jobs=%s)",
        config.ENVIRONMENT, config.STORAGE_BACKEND, config.JOB_BACKEND,
    )
    yield
    if worker:
        worker.stop()


app = FastAPI(
    title="StatMind AI API",
    description="Statistics, ML, AutoML, target recommendation, prediction, leakage detection, AI analyst, and report generation API.",
    version="1.1.0",
    lifespan=lifespan,
)

# Allow Vercel frontend + local dev. Set ALLOWED_ORIGINS env as comma-separated list to override.
_allowed = os.getenv("ALLOWED_ORIGINS", "")
if _allowed:
    _origins = [o.strip() for o in _allowed.split(",") if o.strip()]
else:
    _origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Narrow this to your project, e.g. r"https://statmind-.*\.vercel\.app"
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*\.vercel\.app"),
    # Auth uses a Bearer header, not cookies; never combine credentials with a wildcard regex
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public: login/registration and signed-token downloads
app.include_router(auth_router, prefix="/api", tags=["Auth"])
app.include_router(artifact_public_router, prefix="/api", tags=["Artifacts"])

# Everything else requires a logged-in user. The router-level dependency is a
# backstop: a new endpoint can never be accidentally left public.
_protected = [Depends(get_current_user)]
app.include_router(analysis_router, prefix="/api", tags=["Analysis"], dependencies=_protected)
app.include_router(target_router, prefix="/api", tags=["Target Advisor"], dependencies=_protected)
app.include_router(ml_router, prefix="/api", tags=["Machine Learning"], dependencies=_protected)
app.include_router(job_router, prefix="/api", tags=["Jobs"], dependencies=_protected)
app.include_router(experiment_router, prefix="/api", tags=["Experiments"], dependencies=_protected)
app.include_router(model_router, prefix="/api", tags=["Model Registry"], dependencies=_protected)
app.include_router(prediction_router, prefix="/api", tags=["Prediction Studio"], dependencies=_protected)
app.include_router(artifact_router, prefix="/api", tags=["Artifacts"], dependencies=_protected)
app.include_router(intelligence_router, prefix="/api", tags=["Intelligence Center"], dependencies=_protected)


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
            "Background Jobs",
            "User Accounts",
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
    except Exception:
        logger.exception("Health check: database unavailable")
        db_status = "error"
    status = "ok" if db_status == "ok" else "degraded"
    return {
        "status": status,
        "database": db_status,
        "storage": config.STORAGE_BACKEND,
        "jobs": config.JOB_BACKEND,
        "version": app.version,
    }
