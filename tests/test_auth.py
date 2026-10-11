"""Accounts, tokens, and a sweep proving no API route is accidentally public."""
import time

import jwt
import pytest

PUBLIC_PATHS = {
    "/", "/api/health",
    "/api/auth/register", "/api/auth/login",
    "/api/auth/config",  # only says whether a login is needed
    "/api/artifacts/download",  # authorised by a signed token in the URL
}


def test_register_login_me(client):
    r = client.post("/api/auth/register", json={"email": "Carol@Example.com ", "password": "s3cure-pass"})
    assert r.status_code == 201
    assert r.json()["user"]["email"] == "carol@example.com"

    r = client.post("/api/auth/login", data={"username": "carol@example.com", "password": "s3cure-pass"})
    assert r.status_code == 200
    token = r.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["email"] == "carol@example.com"


def test_duplicate_email_rejected(client):
    body = {"email": "dave@example.com", "password": "long-enough-1"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json={**body, "email": "DAVE@example.com"}).status_code == 400


@pytest.mark.parametrize("email,password", [
    ("not-an-email", "long-enough-1"),
    ("erin@example.com", "short"),
])
def test_registration_validation(client, email, password):
    assert client.post("/api/auth/register", json={"email": email, "password": password}).status_code == 400


def test_wrong_password_and_unknown_user_look_the_same(client, make_user):
    email, _ = make_user()
    wrong = client.post("/api/auth/login", data={"username": email, "password": "nope-nope-nope"})
    unknown = client.post("/api/auth/login", data={"username": "ghost@example.com", "password": "nope-nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_password_is_hashed(make_user):
    from database import SessionLocal, User

    email, _ = make_user()
    db = SessionLocal()
    stored = db.query(User).filter_by(email=email).one().password_hash
    db.close()
    assert "correct-horse-1" not in stored and stored.startswith("$argon2")


@pytest.mark.parametrize("header", [
    None,
    "Bearer not-a-jwt",
    "Bearer " + jwt.encode(
        {"sub": "x", "purpose": "access", "exp": time.time() + 60}, "wrong-secret-" + "x" * 32, algorithm="HS256"
    ),
])
def test_bad_tokens_rejected(client, header):
    headers = {"Authorization": header} if header else {}
    assert client.get("/api/datasets", headers=headers).status_code == 401


def test_expired_token_rejected(client, make_user):
    import config

    email, _ = make_user()
    from database import SessionLocal, User
    db = SessionLocal()
    user_id = db.query(User).filter_by(email=email).one().id
    db.close()

    expired = jwt.encode(
        {"sub": user_id, "purpose": "access", "exp": time.time() - 10},
        config.JWT_SECRET, algorithm="HS256",
    )
    assert client.get("/api/datasets", headers={"Authorization": f"Bearer {expired}"}).status_code == 401


def test_download_token_cannot_be_used_as_access_token(client, alice):
    from services.auth import create_download_token

    token = create_download_token("someone", "artifact")
    assert client.get("/api/datasets", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_every_non_public_route_requires_auth(client):
    """If someone adds an endpoint and forgets auth, this test fails."""
    from main import app

    # The OpenAPI schema lists every endpoint however routers are nested
    checked = 0
    for route_path, operations in app.openapi()["paths"].items():
        if route_path in PUBLIC_PATHS:
            continue
        path = route_path.replace("{", "").replace("}", "")  # fill path params with dummy text
        for method in operations:
            r = client.request(method.upper(), path)
            assert r.status_code == 401, f"{method.upper()} {route_path} returned {r.status_code} without a token"
            checked += 1
    assert checked >= 15
