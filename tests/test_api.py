import os

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"


def test_analyze_and_train_flow():
    """Full flow: upload iris.csv -> analyze -> train -> experiment logged -> predict."""
    with open(os.path.join(DATA_DIR, "iris.csv"), "rb") as f:
        r = client.post("/api/analyze", files={"file": ("iris.csv", f, "text/csv")}, data={"domain": "research"})
    assert r.status_code == 200, r.text
    body = r.json()
    dataset_id = body["dataset_id"]
    assert body["original_filename"] == "iris.csv"

    r2 = client.post(
        "/api/train-model",
        data={"dataset_id": dataset_id, "target_column": "target", "algorithm": "random_forest", "test_size": "0.2"},
    )
    assert r2.status_code == 200, r2.text
    trained = r2.json()
    assert trained["metrics"]["accuracy"] > 0.85

    r3 = client.get(f"/api/experiments/?dataset_id={dataset_id}")
    assert r3.status_code == 200
    experiments = r3.json()["experiments"]
    assert any(e["model_id"] == trained["model_id"] for e in experiments)

    with open(os.path.join(DATA_DIR, "iris.csv"), "rb") as f:
        r4 = client.post(
            "/api/predict-test-data",
            data={"model_id": trained["model_id"]},
            files={"file": ("iris.csv", f, "text/csv")},
        )
    assert r4.status_code == 200, r4.text
    prediction_file = r4.json()["prediction_filename"]

    r5 = client.get(f"/api/download-predictions/{prediction_file}")
    assert r5.status_code == 200


def test_legacy_filename_flow():
    """The frontend passes the stored `filename` from /api/analyze to later endpoints."""
    with open(os.path.join(DATA_DIR, "iris.csv"), "rb") as f:
        stored = client.post("/api/analyze", files={"file": ("iris.csv", f, "text/csv")}).json()["filename"]

    r = client.post("/api/recommend-target", data={"filename": stored})
    assert r.status_code == 200, r.text

    r = client.post("/api/detect-leakage", json={"filename": stored, "target_column": "target"})
    assert r.status_code == 200, r.text


def test_unknown_dataset_id_is_client_error():
    r = client.post("/api/train-model", data={"dataset_id": "does-not-exist", "target_column": "t", "algorithm": "ridge"})
    assert r.status_code == 400


def test_list_endpoints():
    for path in ["/api/datasets", "/api/models/", "/api/experiments/"]:
        r = client.get(path)
        assert r.status_code == 200, f"{path} failed {r.status_code}"
