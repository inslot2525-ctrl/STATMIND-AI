"""
Test isolation: point the app at a throwaway data directory *before* any backend
module is imported, so tests never write to the real statmind.db, uploads/ or models/.

By default tests use a fresh SQLite database. Set TEST_DATABASE_URL to run the
same suite against PostgreSQL (CI does this).
"""
import os
import shutil
import sys
import tempfile
import uuid

import pytest

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="statmind-test-")
os.environ["STATMIND_DATA_DIR"] = _TEST_DATA_DIR
os.environ["STATMIND_ENV"] = "test"
os.environ["JWT_SECRET"] = "test-secret-" + uuid.uuid4().hex
os.environ["JOB_BACKEND"] = "inline"  # jobs finish inside the request: deterministic tests
os.environ["STORAGE_BACKEND"] = "local"
os.environ["ANALYST_LLM"] = "off"  # never call the Anthropic API from tests
os.environ.pop("MLFLOW_TRACKING_URI", None)  # MLflow tests point it at a temp store themselves
os.environ.pop("LAKEFS_ENDPOINT", None)  # lakeFS tests run against a fake server
os.environ.setdefault("MAX_UPLOAD_MB", "1")
if os.getenv("TEST_DATABASE_URL"):
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
else:
    os.environ.pop("DATABASE_URL", None)

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def pytest_sessionfinish(session, exitstatus):
    from database import engine

    engine.dispose()  # release the SQLite file so it can be deleted on Windows
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _migrated_db():
    from database import init_db

    init_db()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from main import app

    # No `with`: lifespan (migrations + worker thread) is handled by the fixtures above
    return TestClient(app)


@pytest.fixture
def make_user(client):
    """Factory: register a fresh user and return (email, auth headers)."""

    def _make():
        email = f"user-{uuid.uuid4().hex[:10]}@example.com"
        r = client.post("/api/auth/register", json={"email": email, "password": "correct-horse-1"})
        assert r.status_code == 201, r.text
        return email, {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _make


@pytest.fixture
def alice(make_user):
    return make_user()[1]


@pytest.fixture
def bob(make_user):
    return make_user()[1]


def upload(client, headers, name="iris.csv", content=None):
    """Upload a dataset (from tests/data by default) and return the /api/analyze JSON."""
    if content is None:
        with open(os.path.join(DATA_DIR, name), "rb") as f:
            content = f.read()
    r = client.post("/api/analyze", headers=headers, files={"file": (name, content, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def train(client, headers, dataset_id, target="target", algorithm="random_forest"):
    """Start a training job (runs inline in tests) and return the finished job."""
    r = client.post(
        "/api/train-model",
        headers=headers,
        data={"dataset_id": dataset_id, "target_column": target, "algorithm": algorithm, "test_size": "0.2"},
    )
    assert r.status_code == 202, r.text
    return r.json()
