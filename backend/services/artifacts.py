"""
Downloadable outputs (prediction CSVs, HTML reports) owned by a user.

Browsers download files with a plain link, which cannot carry the Authorization
header, so downloads use short-lived signed URLs: a presigned S3 URL, or an API
URL with a signed token for local storage.
"""
import uuid

from database import Artifact, SessionLocal, utcnow
from services.storage import get_storage, make_key


def create_artifact(owner_id, kind, data: bytes, extension: str, download_name: str) -> dict:
    artifact_id = uuid.uuid4().hex
    folder = {"prediction": "predictions", "report": "reports"}[kind]
    key = make_key(folder, f"{artifact_id}{extension}")
    get_storage().put_bytes(key, data)

    db = SessionLocal()
    try:
        row = Artifact(
            id=artifact_id,
            owner_id=owner_id,
            kind=kind,
            storage_key=key,
            filename=download_name,
            created_at=utcnow(),
        )
        db.add(row)
        db.commit()
        return _to_dict(row)
    finally:
        db.close()


def _to_dict(row):
    return {
        "id": row.id,
        "kind": row.kind,
        "filename": row.filename,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def get_artifact_row(artifact_id, owner_id):
    db = SessionLocal()
    try:
        return db.query(Artifact).filter_by(id=artifact_id, owner_id=owner_id).first()
    finally:
        db.close()


def list_artifacts(owner_id, kind=None, limit=50):
    db = SessionLocal()
    try:
        q = db.query(Artifact).filter_by(owner_id=owner_id)
        if kind:
            q = q.filter_by(kind=kind)
        return [_to_dict(r) for r in q.order_by(Artifact.created_at.desc()).limit(limit).all()]
    finally:
        db.close()


def download_url(row, api_path_builder) -> str:
    """Presigned object-storage URL if available, otherwise a signed API URL."""
    url = get_storage().presigned_url(row.storage_key, row.filename)
    return url or api_path_builder(row)
