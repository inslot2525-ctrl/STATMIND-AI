from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analysis_routes import router as analysis_router

app = FastAPI(
    title="StatMind AI",
    description="Multi-domain AI statistical analyst for business, research, and finance data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "StatMind AI backend is running",
        "docs": "/docs"
    }