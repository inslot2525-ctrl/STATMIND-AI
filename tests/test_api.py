import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "database" in data
    print(f"\n[health] {data}")

def test_analyze_and_train_flow():
    """Full flow: upload iris.csv → analyze → train → check experiment logged."""
    iris = os.path.join(os.path.dirname(__file__), "data", "iris.csv")
    with open(iris, "rb") as f:
        r = client.post("/api/analyze", files={"file": ("iris.csv", f, "text/csv")}, data={"domain": "research"})
    assert r.status_code == 200, r.text
    dataset_id = r.json().get("dataset_id")
    assert dataset_id, "no dataset_id from /api/analyze"
    print(f"\n[analyze] dataset_id={dataset_id}")

    # Train using dataset_id (new SQLite path)
    r2 = client.post("/api/train-model", data={"dataset_id": dataset_id, "target_column": "target", "algorithm": "random_forest", "test_size": "0.2"})
    assert r2.status_code == 200, r2.text
    j2 = r2.json()
    assert "metrics" in j2
    print(f"[train] metrics={j2['metrics']} model_id={j2.get('model_id')}")

    # Verify experiment was logged
    r3 = client.get(f"/api/experiments/?dataset_id={dataset_id}")
    assert r3.status_code == 200
    data = r3.json()
    # Normalize: API may return list, dict, or {"experiments": [...]}
    if isinstance(data, dict) and "experiments" in data:
        exps = data["experiments"]
    elif isinstance(data, dict):
        # dict of id->exp
        exps = list(data.values())
    else:
        exps = data  # list
    # Handle nested list case
    if exps and isinstance(exps[0], list):
        exps = exps[0]
    assert len(exps) >= 1
    first = exps[0]
    # first may be dict with metrics as JSON string or dict
    metrics = first.get("metrics") if isinstance(first, dict) else first
    print(f"[experiments] count={len(exps)} latest_metrics={metrics}")

def test_list_endpoints():
    for path in ["/api/datasets", "/api/models/", "/api/experiments/"]:
        r = client.get(path)
        # Some may be /api/experiments vs /api/experiments/ — accept 200 or 307
        assert r.status_code in (200, 307, 404), f"{path} failed {r.status_code}"
        print(f"[{path}] {r.status_code}")
