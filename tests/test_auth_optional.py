"""AUTH_REQUIRED=false (the local-development default): no login, one built-in local user."""
import pytest

import config
from tests.conftest import upload


@pytest.fixture
def no_login(monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)


def test_login_is_required_by_default_outside_development(client):
    assert client.get("/api/auth/config").json() == {"auth_required": True}
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/datasets").status_code == 401


def test_requests_without_a_token_act_as_the_local_user(client, no_login):
    assert client.get("/api/auth/config").json() == {"auth_required": False}

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == config.LOCAL_USER_EMAIL

    # The same user every time, and it can use the app end to end
    assert client.get("/api/auth/me").json()["id"] == me.json()["id"]
    body = upload(client, headers={})
    datasets = client.get("/api/datasets").json()["datasets"]
    assert body["dataset_id"] in [d["id"] for d in datasets]


def test_an_expired_or_invalid_token_falls_back_to_the_local_user(client, no_login):
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 200 and r.json()["email"] == config.LOCAL_USER_EMAIL


def test_a_real_account_still_sees_its_own_data(client, no_login, alice):
    body = upload(client, alice)

    mine = client.get("/api/datasets", headers=alice).json()["datasets"]
    local = client.get("/api/datasets").json()["datasets"]

    assert body["dataset_id"] in [d["id"] for d in mine]
    assert body["dataset_id"] not in [d["id"] for d in local]


def test_nobody_can_log_in_as_the_local_user(client, no_login):
    client.get("/api/auth/me")  # make sure the local user exists

    for password in ("!", "anything-at-all"):
        r = client.post("/api/auth/login", data={"username": config.LOCAL_USER_EMAIL, "password": password})
        assert r.status_code == 401
