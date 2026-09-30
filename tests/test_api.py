import os

from tests.conftest import DATA_DIR, train, upload


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"


def test_analyze_train_predict_download_flow(client, alice):
    """upload -> analyze -> train job -> experiment logged -> predict -> download."""
    body = upload(client, alice)
    assert body["original_filename"] == "iris.csv"

    job = train(client, alice, body["dataset_id"])
    assert job["status"] == "succeeded", job
    assert job["progress"] == 100
    trained = job["result"]
    assert trained["metrics"]["accuracy"] > 0.85

    # The job is visible through the jobs API
    r = client.get(f"/api/jobs/{job['id']}", headers=alice)
    assert r.status_code == 200 and r.json()["status"] == "succeeded"

    experiments = client.get(f"/api/experiments/?dataset_id={body['dataset_id']}", headers=alice).json()["experiments"]
    assert any(e["model_id"] == trained["model_id"] for e in experiments)

    with open(os.path.join(DATA_DIR, "iris.csv"), "rb") as f:
        r = client.post(
            "/api/predict-test-data",
            headers=alice,
            data={"model_id": trained["model_id"]},
            files={"file": ("iris.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    prediction = r.json()
    assert prediction["rows_predicted"] == 150

    link = client.get(f"/api/artifacts/{prediction['artifact_id']}/download-url", headers=alice).json()
    # Downloading needs no Authorization header: the signed token in the URL is enough
    r = client.get(link["url"])
    assert r.status_code == 200
    assert r.content.startswith(b"sepal length (cm)")
    assert "attachment" in r.headers["content-disposition"]


def test_compare_models_job(client, alice):
    body = upload(client, alice)
    r = client.post(
        "/api/compare-models",
        headers=alice,
        data={"dataset_id": body["dataset_id"], "target_column": "target", "test_size": "0.2"},
    )
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "succeeded", job
    assert job["result"]["best_model"]
    assert len(job["result"]["leaderboard"]) >= 3


def test_failed_job_reports_a_useful_error(client, alice):
    body = upload(client, alice)
    job = train(client, alice, body["dataset_id"], target="no_such_column")
    assert job["status"] == "failed"
    assert "no_such_column" in job["error"]


def test_legacy_filename_flow(client, alice):
    """The frontend passes the stored `filename` from /api/analyze to later endpoints."""
    stored = upload(client, alice)["filename"]

    r = client.post("/api/recommend-target", headers=alice, data={"filename": stored})
    assert r.status_code == 200, r.text

    r = client.post("/api/detect-leakage", headers=alice, json={"filename": stored, "target_column": "target"})
    assert r.status_code == 200, r.text

    r = client.post(
        "/api/train-model",
        headers=alice,
        data={"filename": stored, "target_column": "target", "algorithm": "decision_tree"},
    )
    assert r.status_code == 202 and r.json()["status"] == "succeeded"


def test_report_generation_and_download(client, alice):
    r = client.post("/api/generate-report", headers=alice, json={"context": {"dataset": {"rows": 3}}})
    assert r.status_code == 200, r.text
    link = client.get(f"/api/artifacts/{r.json()['artifact_id']}/download-url", headers=alice).json()
    page = client.get(link["url"])
    assert page.status_code == 200 and b"<html" in page.content.lower()


def test_model_stage_transition(client, alice):
    model_id = train(client, alice, upload(client, alice)["dataset_id"])["result"]["model_id"]

    r = client.post(f"/api/models/{model_id}/transition?stage=production", headers=alice)
    assert r.status_code == 200
    production = client.get("/api/models/production/list", headers=alice).json()["production_models"]
    assert [m["model_id"] for m in production] == [model_id]

    assert client.post(f"/api/models/{model_id}/transition?stage=bogus", headers=alice).status_code == 400


def test_unknown_dataset_id_is_not_found(client, alice):
    r = client.post(
        "/api/train-model",
        headers=alice,
        data={"dataset_id": "does-not-exist", "target_column": "t", "algorithm": "ridge"},
    )
    assert r.status_code == 404


def test_invalid_test_size_rejected_before_queueing(client, alice):
    body = upload(client, alice)
    r = client.post(
        "/api/train-model",
        headers=alice,
        data={"dataset_id": body["dataset_id"], "target_column": "target", "algorithm": "ridge", "test_size": "1.5"},
    )
    assert r.status_code == 400
    assert client.get("/api/jobs/", headers=alice).json()["jobs"] == []


def test_list_endpoints(client, alice):
    for path in ["/api/datasets", "/api/models/", "/api/experiments/", "/api/jobs/", "/api/artifacts/"]:
        r = client.get(path, headers=alice)
        assert r.status_code == 200, f"{path} failed {r.status_code}"
