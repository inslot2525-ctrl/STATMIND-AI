"""Multi-tenancy: one user can never see or use another user's data."""
import os

from tests.conftest import DATA_DIR, train, upload


def test_users_see_only_their_own_data(client, alice, bob):
    body = upload(client, alice)
    job = train(client, alice, body["dataset_id"])
    model_id = job["result"]["model_id"]

    assert client.get("/api/datasets", headers=bob).json()["datasets"] == []
    assert client.get("/api/models/", headers=bob).json()["models"] == []
    assert client.get("/api/experiments/", headers=bob).json()["experiments"] == []
    assert client.get("/api/jobs/", headers=bob).json()["jobs"] == []

    assert client.get(f"/api/models/{model_id}", headers=bob).status_code == 404
    assert client.get(f"/api/jobs/{job['id']}", headers=bob).status_code == 404
    experiment_id = job["result"]["experiment_id"]
    assert client.get(f"/api/experiments/{experiment_id}", headers=bob).status_code == 404


def test_cannot_use_another_users_dataset(client, alice, bob):
    body = upload(client, alice)

    r = client.post(
        "/api/train-model",
        headers=bob,
        data={"dataset_id": body["dataset_id"], "target_column": "target", "algorithm": "ridge"},
    )
    assert r.status_code == 404
    r = client.post("/api/recommend-target", headers=bob, data={"filename": body["filename"]})
    assert r.status_code == 404
    r = client.post("/api/detect-leakage", headers=bob, json={"filename": body["filename"], "target_column": "target"})
    assert r.status_code == 404


def test_cannot_predict_with_or_promote_another_users_model(client, alice, bob):
    model_id = train(client, alice, upload(client, alice)["dataset_id"])["result"]["model_id"]

    with open(os.path.join(DATA_DIR, "iris.csv"), "rb") as f:
        r = client.post(
            "/api/predict-test-data",
            headers=bob,
            data={"model_id": model_id},
            files={"file": ("iris.csv", f, "text/csv")},
        )
    assert r.status_code == 400

    assert client.post(f"/api/models/{model_id}/transition?stage=production", headers=bob).status_code == 404


def test_cannot_download_another_users_artifact(client, alice, bob):
    r = client.post("/api/generate-report", headers=alice, json={"context": {}})
    artifact_id = r.json()["artifact_id"]

    assert client.get(f"/api/artifacts/{artifact_id}/download-url", headers=bob).status_code == 404
    assert client.get("/api/artifacts/", headers=bob).json()["artifacts"] == []


def test_same_file_uploaded_by_two_users_gives_separate_datasets(client, alice, bob):
    a = upload(client, alice)
    b = upload(client, bob)
    assert a["dataset_id"] != b["dataset_id"]
    assert len(client.get("/api/datasets", headers=bob).json()["datasets"]) == 1
