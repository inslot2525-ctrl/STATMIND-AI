"""
Regression tests for path-traversal and unsafe-deserialisation bugs.
Each test reproduces an attack that worked before the fix.
"""
import io
import os

import joblib
import pytest
from fastapi.testclient import TestClient

from main import app
from services import storage

client = TestClient(app)


def _upload(name, content, mime="text/csv"):
    return client.post("/api/analyze", files={"file": (name, content, mime)})


def test_upload_filename_cannot_escape_upload_dir():
    r = _upload("../../escaped.csv", b"a,b\n1,2\n3,4\n")
    assert r.status_code == 200, r.text
    assert not os.path.exists(os.path.join(storage.DATA_DIR, "escaped.csv"))
    assert not os.path.exists(os.path.join(os.path.dirname(storage.DATA_DIR), "escaped.csv"))
    assert os.path.exists(os.path.join(storage.UPLOAD_DIR, r.json()["filename"]))


def test_same_name_different_content_are_separate_versions():
    first = _upload("sales.csv", b"x,y\n" + b"1,2\n" * 20).json()
    second = _upload("sales.csv", b"x,y\n" + b"9,9\n" * 50).json()

    assert first["dataset_id"] != second["dataset_id"]
    assert first["filename"] != second["filename"]

    datasets = {d["id"]: d for d in client.get("/api/datasets").json()["datasets"]}
    assert datasets[first["dataset_id"]]["rows"] == 20
    assert datasets[second["dataset_id"]]["rows"] == 50
    # The first version's bytes are still intact on disk
    with open(datasets[first["dataset_id"]]["file_path"], "rb") as f:
        assert f.read().count(b"1,2") == 20


def test_identical_content_is_deduplicated():
    content = b"p,q\n" + b"5,6\n" * 15
    assert _upload("a.csv", content).json()["dataset_id"] == _upload("b.csv", content).json()["dataset_id"]


def test_uploaded_pickle_cannot_be_loaded_as_model():
    buf = io.BytesIO()
    joblib.dump({"pipeline": "not-a-model", "metadata": {}}, buf)
    stored = _upload("innocent.csv", buf.getvalue()).json().get("filename", "innocent.csv")

    for model_id in (f"../uploads/{stored}", f"..\\uploads\\{stored}", stored, "/etc/passwd"):
        r = client.post(
            "/api/predict-test-data",
            data={"model_id": model_id},
            files={"file": ("t.csv", b"a\n1\n", "text/csv")},
        )
        assert r.status_code == 400, (model_id, r.text)


def test_register_endpoint_rejects_non_model_files():
    r = client.post("/api/models/register/..%5Cuploads%5Cx.csv")
    assert r.status_code == 400


@pytest.mark.parametrize("path", [
    "/api/download-predictions/..%5C..%5Cstatmind.db",
    "/api/download-report/..%5Cstatmind.db",
    "/api/download-predictions/.hidden",
])
def test_downloads_reject_traversal(path):
    assert client.get(path).status_code in (400, 404)


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/recommend-target", {"filename": "../statmind.db"}),
    ("/api/train-model", {"filename": "../statmind.db", "target_column": "t", "algorithm": "ridge"}),
    ("/api/compare-models", {"filename": "..\\statmind.db", "target_column": "t"}),
])
def test_filename_params_reject_traversal(endpoint, payload):
    assert client.post(endpoint, data=payload).status_code == 400


def test_leakage_rejects_traversal():
    r = client.post("/api/detect-leakage", json={"filename": "../statmind.db", "target_column": "t"})
    assert r.status_code == 400


def test_rejects_non_data_extension():
    assert _upload("script.py", b"print(1)").status_code == 400


def test_rejects_oversized_upload():
    too_big = b"a\n" + b"1\n" * (storage.MAX_UPLOAD_BYTES // 2 + 10)
    r = _upload("big.csv", too_big)
    assert r.status_code == 400
    assert "too large" in r.json()["detail"]
    assert not [f for f in os.listdir(storage.UPLOAD_DIR) if f.endswith(".part")]
