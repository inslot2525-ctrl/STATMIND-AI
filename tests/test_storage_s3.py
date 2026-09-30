"""
The S3 storage backend (also Cloudflare R2 / MinIO), against a mocked S3 (moto).
Runs the full upload -> train -> predict -> download flow with STORAGE_BACKEND=s3.
"""
import os
import shutil

import boto3
import pytest
from moto import mock_aws

import config
from services import storage
from tests.conftest import DATA_DIR, train, upload

BUCKET = "statmind-test"


@pytest.fixture
def s3_backend(monkeypatch):
    with mock_aws():
        monkeypatch.setattr(config, "S3_BUCKET", BUCKET)
        monkeypatch.setattr(config, "S3_ENDPOINT_URL", None)
        monkeypatch.setattr(config, "S3_REGION", "us-east-1")
        monkeypatch.setattr(config, "S3_PREFIX", "prod")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)

        backend = storage.S3Storage()
        monkeypatch.setattr(storage, "_storage", backend)
        yield backend
    # Clear the read-through cache so later tests cannot see mocked objects
    shutil.rmtree(storage.CACHE_DIR, ignore_errors=True)
    os.makedirs(storage.CACHE_DIR, exist_ok=True)


def _keys():
    listing = boto3.client("s3", region_name="us-east-1").list_objects_v2(Bucket=BUCKET)
    return sorted(o["Key"] for o in listing.get("Contents", []))


def test_full_flow_on_s3(client, alice, s3_backend):
    body = upload(client, alice)
    model_id = train(client, alice, body["dataset_id"])["result"]["model_id"]

    keys = _keys()
    assert f"prod/uploads/{body['filename']}" in keys
    assert f"prod/models/{model_id}" in keys
    # The model went to the bucket, not the local models folder
    assert model_id not in os.listdir(storage.MODEL_DIR)

    # Wipe the local cache: prediction must fetch the model back from S3
    shutil.rmtree(storage.CACHE_DIR)
    os.makedirs(storage.CACHE_DIR)

    with open(os.path.join(DATA_DIR, "iris.csv"), "rb") as f:
        r = client.post(
            "/api/predict-test-data",
            headers=alice,
            data={"model_id": model_id},
            files={"file": ("iris.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text

    link = client.get(f"/api/artifacts/{r.json()['artifact_id']}/download-url", headers=alice).json()
    # With S3 the browser downloads straight from the bucket via a presigned URL
    assert link["url"].startswith("https://")
    assert "Signature" in link["url"]  # SigV2 "Signature=" or SigV4 "X-Amz-Signature="
    assert "/api/artifacts/download" not in link["url"]


def test_missing_object_is_a_clean_error(s3_backend):
    with pytest.raises(ValueError):
        s3_backend.local_path("uploads/does-not-exist.csv")


def test_s3_requires_bucket(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", None)
    with pytest.raises(RuntimeError):
        storage.S3Storage()
