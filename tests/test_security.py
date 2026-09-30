"""
Regression tests for path-traversal and unsafe-deserialisation bugs.
Each test reproduces an attack that worked before the step-1 fixes.
"""
import hashlib
import io
import os

import joblib
import pytest

from services import storage
from tests.conftest import upload


def test_upload_filename_cannot_escape_upload_dir(client, alice):
    body = upload(client, alice, name="../../escaped.csv", content=b"a,b\n1,2\n3,4\n")
    assert not os.path.exists(os.path.join(storage.DATA_DIR, "escaped.csv"))
    assert not os.path.exists(os.path.join(os.path.dirname(storage.DATA_DIR), "escaped.csv"))
    assert os.path.exists(os.path.join(storage.UPLOAD_DIR, body["filename"]))


def test_same_name_different_content_are_separate_versions(client, alice):
    first = upload(client, alice, name="sales.csv", content=b"x,y\n" + b"1,2\n" * 20)
    second = upload(client, alice, name="sales.csv", content=b"x,y\n" + b"9,9\n" * 50)

    assert first["dataset_id"] != second["dataset_id"]
    assert first["filename"] != second["filename"]

    datasets = {d["id"]: d for d in client.get("/api/datasets", headers=alice).json()["datasets"]}
    assert datasets[first["dataset_id"]]["rows"] == 20
    assert datasets[second["dataset_id"]]["rows"] == 50
    with open(os.path.join(storage.UPLOAD_DIR, first["filename"]), "rb") as f:
        assert f.read().count(b"1,2") == 20


def test_identical_content_is_deduplicated_per_user(client, alice):
    content = b"p,q\n" + b"5,6\n" * 15
    assert (upload(client, alice, name="a.csv", content=content)["dataset_id"]
            == upload(client, alice, name="b.csv", content=content)["dataset_id"])


def test_uploaded_pickle_cannot_be_loaded_as_model(client, alice):
    buf = io.BytesIO()
    joblib.dump({"pipeline": "not-a-model", "metadata": {}}, buf)
    payload = buf.getvalue()
    # Analysis rejects it, but the upload itself is stored first - that is the attack surface
    client.post("/api/analyze", headers=alice, files={"file": ("innocent.csv", payload, "text/csv")})
    stored = hashlib.sha256(payload).hexdigest()[:32] + ".csv"
    assert os.path.exists(os.path.join(storage.UPLOAD_DIR, stored))

    for model_id in (f"../uploads/{stored}", f"..\\uploads\\{stored}", stored, "/etc/passwd"):
        r = client.post(
            "/api/predict-test-data",
            headers=alice,
            data={"model_id": model_id},
            files={"file": ("t.csv", b"a\n1\n", "text/csv")},
        )
        assert r.status_code == 400, (model_id, r.text)


def test_model_file_registration_endpoint_is_gone(client, alice):
    # It used to unpickle any file named in the URL; models are now registered only by training
    assert client.post("/api/models/register/x.pkl", headers=alice).status_code in (404, 405)


@pytest.mark.parametrize("token", ["", "garbage", "..%5C..%5Cstatmind.db"])
def test_artifact_download_requires_valid_signed_token(client, token):
    assert client.get(f"/api/artifacts/download?token={token}").status_code == 403


def test_tampered_download_token_rejected(client, alice):
    r = client.post("/api/generate-report", headers=alice, json={"context": {}})
    url = client.get(f"/api/artifacts/{r.json()['artifact_id']}/download-url", headers=alice).json()["url"]
    assert client.get(url[:-2] + ("AA" if not url.endswith("AA") else "BB")).status_code == 403


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/recommend-target", {"filename": "../statmind.db"}),
    ("/api/train-model", {"filename": "../statmind.db", "target_column": "t", "algorithm": "ridge"}),
    ("/api/compare-models", {"filename": "..\\statmind.db", "target_column": "t"}),
])
def test_filename_params_only_match_the_users_datasets(client, alice, endpoint, payload):
    assert client.post(endpoint, headers=alice, data=payload).status_code == 404


def test_leakage_rejects_unknown_file(client, alice):
    r = client.post("/api/detect-leakage", headers=alice, json={"filename": "../statmind.db", "target_column": "t"})
    assert r.status_code == 404


def test_rejects_non_data_extension(client, alice):
    r = client.post("/api/analyze", headers=alice, files={"file": ("script.py", b"print(1)", "text/plain")})
    assert r.status_code == 400


def test_rejects_oversized_upload(client, alice):
    too_big = b"a\n" + b"1\n" * (storage.MAX_UPLOAD_BYTES // 2 + 10)
    r = client.post("/api/analyze", headers=alice, files={"file": ("big.csv", too_big, "text/csv")})
    assert r.status_code == 400
    assert "too large" in r.json()["detail"]
    assert not [f for f in os.listdir(storage.TMP_DIR) if f.endswith(".part")]


@pytest.mark.parametrize("key", ["uploads/../statmind.db", "secrets/x.csv", "uploads/", "models/a/b.pkl"])
def test_storage_keys_are_validated(key):
    with pytest.raises(ValueError):
        storage.get_storage().local_path(key)
