"""Prediction API, the prediction log, and drift monitoring."""
import io
import itertools
import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import train, upload

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _customers(rows=400, seed=0, income_shift=0.0, region_weights=(0.4, 0.3, 0.3)):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "income": rng.normal(50000 + income_shift, 8000, size=rows).round(2),
        "age": rng.randint(18, 70, size=rows),
        "region": rng.choice(["north", "south", "east"], size=rows, p=region_weights),
    })
    df["bought"] = np.where(df["income"] + rng.normal(0, 6000, size=rows) > 50000, "yes", "no")
    return df


# A different training set per test (so each registers a new dataset), the same every run
_seeds = itertools.count(101)


def _csv(df):
    return df.to_csv(index=False).encode()


@pytest.fixture
def model(client, alice):
    """A trained classifier on a small customer dataset; returns (model_id, training frame)."""
    df = _customers(seed=next(_seeds))
    body = upload(client, alice, name="customers.csv", content=_csv(df))
    job = train(client, alice, body["dataset_id"], target="bought", algorithm="logistic_regression")
    assert job["status"] == "succeeded", job
    return job["result"]["model_id"], df


def _records(df):
    return df.drop(columns=["bought"]).to_dict(orient="records")


# ---- prediction API ---------------------------------------------------------

def test_predict_returns_labels_and_probabilities(client, alice, model):
    model_id, df = model
    r = client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(df.head(3))})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 3 and body["task_type"] == "classification"
    assert set(body["predictions"]) <= {"yes", "no"}
    for probabilities in body["probabilities"]:
        assert set(probabilities) == {"yes", "no"}
        assert abs(sum(probabilities.values()) - 1) < 1e-3


def test_predict_accepts_numbers_as_strings_and_ignores_extra_columns(client, alice, model):
    model_id, _ = model
    record = {"income": "81000.5", "age": "40", "region": "north", "note": "ignored"}
    r = client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": [record]})
    assert r.status_code == 200, r.text
    assert r.json()["predictions"] == ["yes"]


def test_predict_handles_a_missing_value(client, alice, model):
    model_id, _ = model
    record = {"income": None, "age": 40, "region": "north"}
    r = client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": [record]})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("records, message", [
    ([], "at least one record"),
    ([{"income": 1, "age": 2}], "missing required columns"),
    ([{"income": "lots", "age": 2, "region": "north"}], "must be numeric"),
    ([{"income": 1, "age": 2, "region": "north"}] * 1001, "at most 1000"),
])
def test_predict_rejects_bad_requests(client, alice, model, records, message):
    model_id, _ = model
    r = client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": records})
    assert r.status_code == 400
    assert message in r.json()["detail"]


def test_another_users_model_cannot_be_used(client, alice, bob, model):
    model_id, df = model
    payload = {"records": _records(df.head(1))}

    assert client.post(f"/api/models/{model_id}/predict", headers=bob, json=payload).status_code == 404
    assert client.get(f"/api/models/{model_id}/serving", headers=bob).status_code == 404
    assert client.get(f"/api/models/{model_id}/drift", headers=bob).status_code == 404


def test_regression_model_returns_numbers(client, alice):
    body = upload(client, alice, name="diabetes.csv")
    job = train(client, alice, body["dataset_id"], algorithm="ridge")
    model_id = job["result"]["model_id"]

    record = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv")).drop(columns=["target"]).iloc[0].to_dict()
    r = client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": [record]})

    assert r.status_code == 200, r.text
    assert isinstance(r.json()["predictions"][0], float)
    assert "probabilities" not in r.json()


def test_production_endpoint_follows_the_registry_stage(client, alice, model):
    model_id, df = model
    payload = {"records": _records(df.head(2))}

    r = client.post("/api/predict?target=bought", headers=alice, json=payload)
    assert r.status_code == 404 and "No model is in production" in r.json()["detail"]

    assert client.post(f"/api/models/{model_id}/transition?stage=production", headers=alice).status_code == 200
    r = client.post("/api/predict?target=bought", headers=alice, json=payload)
    assert r.status_code == 200 and r.json()["model_id"] == model_id

    # Demoting it takes it out of service
    client.post(f"/api/models/{model_id}/transition?stage=archived", headers=alice)
    assert client.post("/api/predict?target=bought", headers=alice, json=payload).status_code == 404


def test_serving_info_describes_inputs_with_a_real_example(client, alice, model):
    model_id, df = model
    info = client.get(f"/api/models/{model_id}/serving", headers=alice).json()

    assert info["features"] == [
        {"name": "income", "type": "number"},
        {"name": "age", "type": "number"},
        {"name": "region", "type": "category"},
    ]
    assert info["example_record"]["region"] in {"north", "south", "east"}

    # The example works as a request as-is
    r = client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": [info["example_record"]]})
    assert r.status_code == 200, r.text


# ---- prediction log ---------------------------------------------------------

def test_api_and_file_predictions_are_logged(client, alice, model):
    model_id, df = model
    assert client.get(f"/api/models/{model_id}/serving", headers=alice).json()["predictions_logged"]["rows"] == 0

    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(df.head(5))})
    r = client.post(
        "/api/predict-test-data", headers=alice, data={"model_id": model_id},
        files={"file": ("new.csv", io.BytesIO(_csv(df.head(40))), "text/csv")},
    )
    assert r.status_code == 200, r.text

    logged = client.get(f"/api/models/{model_id}/serving", headers=alice).json()["predictions_logged"]
    assert logged == {**logged, "batches": 2, "rows": 45, "api_rows": 5, "file_rows": 40}
    assert logged["last_prediction_at"]


# ---- drift ------------------------------------------------------------------

def test_drift_needs_enough_logged_predictions(client, alice, model):
    model_id, df = model
    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(df.head(5))})

    r = client.get(f"/api/models/{model_id}/drift", headers=alice)
    assert r.status_code == 400
    assert "at least 30" in r.json()["detail"]


def test_data_like_the_training_data_is_stable(client, alice, model):
    model_id, _ = model
    fresh = _customers(rows=300, seed=999)
    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(fresh)})

    drift = client.get(f"/api/models/{model_id}/drift", headers=alice).json()

    assert drift["status"] == "stable", drift
    assert drift["drifted_columns"] == []
    assert drift["current_rows"] == 300 and drift["reference_rows"] == 400
    assert {row["column"] for row in drift["columns"]} == {"income", "age", "region"}


def test_shifted_data_is_flagged_column_by_column(client, alice, model):
    model_id, _ = model
    shifted = _customers(rows=300, seed=999, income_shift=25000, region_weights=(0.05, 0.05, 0.9))
    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(shifted)})

    drift = client.get(f"/api/models/{model_id}/drift", headers=alice).json()

    assert drift["status"] == "drift", drift
    assert set(drift["drifted_columns"]) == {"income", "region"}  # age was not changed
    assert drift["columns"][0]["drifted"] is True  # drifted columns are listed first
    # Higher incomes mean far more "yes" predictions than in training
    assert drift["prediction_drift"]["drifted"] is True


def test_latest_window_looks_only_at_the_most_recent_batch(client, alice, model):
    model_id, _ = model
    shifted = _customers(rows=200, seed=5, income_shift=25000)
    normal = _customers(rows=200, seed=6)

    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(shifted)})
    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(normal)})

    latest = client.get(f"/api/models/{model_id}/drift?window=latest", headers=alice).json()
    everything = client.get(f"/api/models/{model_id}/drift?window=all", headers=alice).json()

    assert latest["current_rows"] == 200 and latest["batches_compared"] == 1
    assert "income" not in latest["drifted_columns"]
    assert everything["current_rows"] == 400 and everything["batches_compared"] == 2
    assert "income" in everything["drifted_columns"]

    assert client.get(f"/api/models/{model_id}/drift?window=yesterday", headers=alice).status_code == 400


def test_drift_report_can_be_downloaded(client, alice, model):
    model_id, _ = model
    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(_customers(rows=100, seed=3))})

    r = client.post(f"/api/models/{model_id}/drift/report", headers=alice)
    assert r.status_code == 200, r.text

    link = client.get(f"/api/artifacts/{r.json()['artifact_id']}/download-url", headers=alice).json()
    page = client.get(link["url"])
    assert page.status_code == 200 and b"<html" in page.content.lower()
    assert len(page.content) > 10000


def test_promoting_a_model_retires_the_previous_production_model(client, alice, model):
    first_id, df = model
    body = upload(client, alice, name="customers.csv", content=_csv(df))  # same data, second model
    second_id = train(client, alice, body["dataset_id"], target="bought", algorithm="decision_tree")["result"]["model_id"]
    payload = {"records": _records(df.head(1))}

    client.post(f"/api/models/{first_id}/transition?stage=production", headers=alice)
    assert client.post("/api/predict?target=bought", headers=alice, json=payload).json()["model_id"] == first_id

    client.post(f"/api/models/{second_id}/transition?stage=production", headers=alice)
    assert client.post("/api/predict?target=bought", headers=alice, json=payload).json()["model_id"] == second_id

    stages = {
        m["model_id"]: m["versions"][0]["stage"]
        for m in client.get("/api/models/", headers=alice).json()["models"]
    }
    assert stages[first_id] == "archived" and stages[second_id] == "production"
    assert len(client.get("/api/models/production/list", headers=alice).json()["production_models"]) == 1


def test_latest_window_reaches_back_when_the_last_batch_is_tiny(client, alice, model):
    """One API call is a one-row batch; the check must still have enough rows to compare."""
    model_id, df = model
    shifted = _customers(rows=200, seed=5, income_shift=25000)

    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(shifted)})
    client.post(f"/api/models/{model_id}/predict", headers=alice, json={"records": _records(df.head(1))})

    drift = client.get(f"/api/models/{model_id}/drift", headers=alice).json()
    assert drift["current_rows"] == 201 and drift["batches_compared"] == 2
    assert "income" in drift["drifted_columns"]
