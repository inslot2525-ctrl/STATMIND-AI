from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from database import Artifact, SessionLocal, User
from services.artifacts import download_url, get_artifact_row, list_artifacts
from services.auth import create_download_token, get_current_user, read_download_token
from services.storage import content_type_for, get_storage

# Authenticated endpoints (Authorization header)
router = APIRouter(prefix="/artifacts")

# The download endpoint is authorised by the signed token in its URL instead,
# because browsers follow plain links without the Authorization header.
public_router = APIRouter(prefix="/artifacts")


def _api_download_path(row: Artifact) -> str:
    return f"/api/artifacts/download?token={quote(create_download_token(row.owner_id, row.id))}"


@router.get("/")
def get_artifacts(kind: str | None = None, user: User = Depends(get_current_user)):
    return {"artifacts": list_artifacts(user.id, kind=kind)}


@router.get("/{artifact_id}/download-url")
def get_download_url(artifact_id: str, user: User = Depends(get_current_user)):
    row = get_artifact_row(artifact_id, user.id)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return {"url": download_url(row, _api_download_path), "filename": row.filename}


@public_router.get("/download")
def download(token: str):
    owner_id, artifact_id = read_download_token(token)

    db = SessionLocal()
    try:
        row = db.query(Artifact).filter_by(id=artifact_id, owner_id=owner_id).first()
    finally:
        db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        path = get_storage().local_path(row.storage_key)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path, filename=row.filename, media_type=content_type_for(row.filename))
