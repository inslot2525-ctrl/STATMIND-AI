from fastapi import APIRouter, Depends, HTTPException

from database import User
from services.auth import get_current_user
from services.jobs import get_job, list_jobs

router = APIRouter(prefix="/jobs")


@router.get("/")
def get_jobs(limit: int = 50, user: User = Depends(get_current_user)):
    return {"jobs": list_jobs(user.id, limit=min(limit, 200))}


@router.get("/{job_id}")
def get_job_detail(job_id: str, user: User = Depends(get_current_user)):
    job = get_job(job_id, user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
