"""
lakeFS integration. Most tests replace the lakeFS REST call with a small fake,
so they need no server. The last one runs the real upload -> train flow through
a real lakeFS and is skipped unless LAKEFS_TEST_ENDPOINT is set, e.g.

    LAKEFS_TEST_ENDPOINT=http://127.0.0.1:8001 \
    LAKEFS_TEST_ACCESS_KEY_ID=... LAKEFS_TEST_SECRET_ACCESS_KEY=... pytest tests/test_lakefs.py
"""
import os
import uuid

import pytest

import config
from services import lakefs_versioning
from services.lakefs_versioning import LakeFSError
from tests.conftest import train, upload

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class FakeLakeFS:
    """Stands in for the lakeFS REST API: one repository, one branch, a list of commits."""

    def __init__(self, repository_exists=True):
        self.repository_exists = repository_exists
        self.commits = []
        self.pending_changes = True
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))

        if method == "GET" and path == "/repositories/statmind":
            if not self.repository_exists:
                raise LakeFSError(404, "repository not found")
            return {"id": "statmind"}

        if method == "POST" and path == "/repositories":
            self.repository_exists = True
            self.created_with = body
            return {"id": body["name"]}

        if method == "GET" and path == "/repositories/statmind/branches/main":
            return {"id": "main", "commit_id": self.commits[-1]["id"] if self.commits else "root"}

        if method == "POST" and path == "/repositories/statmind/branches/main/commits":
            if not self.pending_changes:
                raise LakeFSError(400, "commit: no changes")
            commit = {"id": uuid.uuid4().hex * 2, **body}
            self.commits.append(commit)
            return commit

        raise AssertionError(f"unexpected lakeFS call: {method} {path}")


@pytest.fixture
def lakefs(monkeypatch):
    fake = FakeLakeFS()
    monkeypatch.setattr(config, "LAKEFS_ENDPOINT", "http://lakefs.test")
    monkeypatch.setattr(config, "LAKEFS_UI_URL", "http://lakefs.test")
    monkeypatch.setattr(config, "LAKEFS_ACCESS_KEY_ID", "key")
    monkeypatch.setattr(config, "LAKEFS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(lakefs_versioning, "_api", fake)
    return fake


def _unique_csv():
    """A dataset no other test has uploaded, so it registers as new."""
    with open(os.path.join(DATA_DIR, "iris.csv"), "rb") as f:
        return f.read() + f"5.{uuid.uuid4().int % 10**9},3.5,1.4,0.2,0\n".encode()


def test_off_by_default():
    assert not lakefs_versioning.enabled()
    assert lakefs_versioning.commit("anything") is None
    assert lakefs_versioning.describe("a", "b") is None


def test_commit_returns_the_new_commit_id(lakefs):
    commit_id = lakefs_versioning.commit("Add dataset x.csv", {"statmind.dataset_id": "d1", "skipped": None})

    assert commit_id == lakefs.commits[0]["id"]
    assert lakefs.commits[0]["message"] == "Add dataset x.csv"
    assert lakefs.commits[0]["metadata"] == {"statmind.dataset_id": "d1"}


def test_commit_with_no_changes_returns_the_current_head(lakefs):
    first = lakefs_versioning.commit("Add dataset x.csv")
    lakefs.pending_changes = False

    assert lakefs_versioning.commit("Add dataset x.csv again") == first
    assert len(lakefs.commits) == 1


def test_a_failed_commit_is_reported_as_missing_not_raised(lakefs, monkeypatch):
    def broken(method, path, body=None):
        raise LakeFSError(500, "boom")

    monkeypatch.setattr(lakefs_versioning, "_api", broken)
    assert lakefs_versioning.commit("Add dataset x.csv") is None

    def unreachable(method, path, body=None):
        raise OSError("connection refused")

    monkeypatch.setattr(lakefs_versioning, "_api", unreachable)
    assert lakefs_versioning.commit("Add dataset x.csv") is None


def test_repository_is_created_on_first_start(lakefs):
    lakefs.repository_exists = False

    lakefs_versioning.ensure_repository()

    assert lakefs.created_with == {
        "name": "statmind",
        "storage_namespace": config.LAKEFS_STORAGE_NAMESPACE,
        "default_branch": "main",
    }

    lakefs.calls.clear()
    lakefs_versioning.ensure_repository()
    assert lakefs.calls == [("GET", "/repositories/statmind")]


def test_startup_fails_clearly_when_lakefs_is_unreachable(lakefs, monkeypatch):
    def unreachable(method, path, body=None):
        raise OSError("connection refused")

    monkeypatch.setattr(lakefs_versioning, "_api", unreachable)
    with pytest.raises(RuntimeError, match="Cannot reach lakeFS"):
        lakefs_versioning.ensure_repository()


def test_upload_and_training_record_their_commits(client, alice, lakefs):
    body = upload(client, alice, content=_unique_csv())

    datasets = client.get("/api/datasets", headers=alice).json()["datasets"]
    dataset = next(d for d in datasets if d["id"] == body["dataset_id"])
    dataset_commit = lakefs.commits[0]

    assert dataset["data_commit"] == dataset_commit["id"]
    assert dataset["data_commit_url"] == f"http://lakefs.test/repositories/statmind/commits/{dataset_commit['id']}"
    assert dataset_commit["metadata"]["statmind.kind"] == "dataset"
    assert dataset_commit["metadata"]["statmind.dataset_id"] == body["dataset_id"]

    job = train(client, alice, body["dataset_id"], algorithm="decision_tree")
    assert job["status"] == "succeeded", job
    version = job["result"]["data_version"]
    model_commit = lakefs.commits[1]

    assert version["repository"] == "statmind" and version["branch"] == "main"
    assert version["dataset_commit"] == dataset_commit["id"]
    assert version["model_commit"] == model_commit["id"]
    assert model_commit["metadata"]["statmind.model_id"] == job["result"]["model_id"]
    assert model_commit["metadata"]["statmind.dataset_id"] == body["dataset_id"]

    # The registry keeps the lineage
    model = client.get(f"/api/models/{job['result']['model_id']}", headers=alice).json()
    assert model["metadata"]["data_version"]["dataset_commit"] == dataset_commit["id"]


def test_training_survives_a_failed_commit(client, alice, lakefs, monkeypatch):
    body = upload(client, alice, content=_unique_csv())

    def broken(method, path, body=None):
        raise LakeFSError(500, "boom")

    monkeypatch.setattr(lakefs_versioning, "_api", broken)
    job = train(client, alice, body["dataset_id"], algorithm="decision_tree")

    assert job["status"] == "succeeded", job
    assert job["result"]["data_version"]["model_commit"] is None


@pytest.mark.skipif(not os.getenv("LAKEFS_TEST_ENDPOINT"), reason="needs a running lakeFS (set LAKEFS_TEST_ENDPOINT)")
def test_against_a_real_lakefs_server(client, alice, monkeypatch):
    """Files really go through the S3 gateway, and the recorded commit really contains them."""
    import boto3

    from services import storage

    repository = f"statmind-test-{uuid.uuid4().hex[:8]}"
    endpoint = os.environ["LAKEFS_TEST_ENDPOINT"]
    key, secret = os.environ["LAKEFS_TEST_ACCESS_KEY_ID"], os.environ["LAKEFS_TEST_SECRET_ACCESS_KEY"]

    for name, value in {
        "LAKEFS_ENDPOINT": endpoint, "LAKEFS_UI_URL": endpoint,
        "LAKEFS_ACCESS_KEY_ID": key, "LAKEFS_SECRET_ACCESS_KEY": secret,
        "LAKEFS_REPOSITORY": repository, "LAKEFS_STORAGE_NAMESPACE": f"local://{repository}",
        "STORAGE_BACKEND": "lakefs",
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(storage, "_storage", None)

    try:
        lakefs_versioning.ensure_repository()

        content = _unique_csv()
        body = upload(client, alice, content=content)
        dataset = next(
            d for d in client.get("/api/datasets", headers=alice).json()["datasets"] if d["id"] == body["dataset_id"]
        )
        assert len(dataset["data_commit"]) == 64

        job = train(client, alice, body["dataset_id"], algorithm="decision_tree")
        assert job["status"] == "succeeded", job
        version = job["result"]["data_version"]
        assert version["dataset_commit"] == dataset["data_commit"]
        assert version["model_commit"] not in (None, version["dataset_commit"])

        # Read the dataset back *at the recorded commit*: byte-identical to what was uploaded
        s3 = boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret, region_name="us-east-1"
        )
        stored = s3.get_object(Bucket=repository, Key=f"{version['dataset_commit']}/uploads/{body['filename']}")
        assert stored["Body"].read() == content

        # The model file exists at the model commit but not at the earlier dataset commit
        model_key = f"models/{job['result']['model_id']}"
        s3.head_object(Bucket=repository, Key=f"{version['model_commit']}/{model_key}")
        with pytest.raises(Exception):
            s3.head_object(Bucket=repository, Key=f"{version['dataset_commit']}/{model_key}")

        # Prediction still works with files served from lakeFS
        r = client.post(
            "/api/predict-test-data", headers=alice,
            data={"model_id": job["result"]["model_id"]},
            files={"file": ("iris.csv", open(os.path.join(DATA_DIR, "iris.csv"), "rb").read(), "text/csv")},
        )
        assert r.status_code == 200, r.text
    finally:
        monkeypatch.setattr(storage, "_storage", None)
        try:
            lakefs_versioning._api("DELETE", f"/repositories/{repository}")
        except Exception:
            pass
