"""
Authentication: password hashing (argon2), JWT access tokens, and the
`get_current_user` FastAPI dependency that every protected route uses.
"""
import re
import uuid
from datetime import timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash

import config
from database import SessionLocal, User, utcnow

password_hasher = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8

# Dummy hash so unknown emails take as long to reject as wrong passwords
_DUMMY_HASH = password_hasher.hash("statmind-timing-equaliser")


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL.match(email) or len(email) > 320:
        raise ValueError("Enter a valid email address.")
    return email


def create_user(email: str, password: str, is_admin: bool = False) -> User:
    email = normalize_email(email)
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    db = SessionLocal()
    try:
        if db.query(User).filter_by(email=email).first():
            raise ValueError("An account with this email already exists.")
        user = User(
            id=uuid.uuid4().hex,
            email=email,
            password_hash=password_hasher.hash(password),
            is_admin=is_admin,
            created_at=utcnow(),
        )
        db.add(user)
        db.commit()
        return user
    finally:
        db.close()


def authenticate(email: str, password: str) -> User | None:
    try:
        email = normalize_email(email)
    except ValueError:
        return None

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).first()
    finally:
        db.close()

    if user is None:
        password_hasher.verify(password or "", _DUMMY_HASH)
        return None
    if not password_hasher.verify(password or "", user.password_hash):
        return None
    return user


def _encode(claims: dict, lifetime: timedelta) -> str:
    now = utcnow()
    payload = {**claims, "iat": now, "exp": now + lifetime}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def _decode(token: str, purpose: str) -> dict:
    payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
    if payload.get("purpose") != purpose:
        raise jwt.InvalidTokenError("wrong token purpose")
    return payload


def create_access_token(user: User) -> str:
    return _encode(
        {"sub": user.id, "purpose": "access"},
        timedelta(minutes=config.ACCESS_TOKEN_MINUTES),
    )


def create_download_token(owner_id: str, artifact_id: str) -> str:
    """Short-lived token that lets a plain <a href> download one artifact."""
    return _encode(
        {"sub": owner_id, "artifact": artifact_id, "purpose": "download"},
        timedelta(seconds=config.DOWNLOAD_LINK_SECONDS),
    )


def read_download_token(token: str) -> tuple[str, str]:
    try:
        payload = _decode(token, "download")
    except jwt.PyJWTError:
        raise HTTPException(status_code=403, detail="Download link is invalid or has expired.")
    return payload["sub"], payload["artifact"]


_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated. Please log in again.",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = _decode(token, "access")
    except jwt.PyJWTError:
        raise _CREDENTIALS_ERROR

    db = SessionLocal()
    try:
        user = db.get(User, payload.get("sub"))
    finally:
        db.close()

    if user is None:
        raise _CREDENTIALS_ERROR
    return user
