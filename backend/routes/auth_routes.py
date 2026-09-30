from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

import config
from database import User
from services.auth import authenticate, create_access_token, create_user, get_current_user

router = APIRouter(prefix="/auth")


class RegisterRequest(BaseModel):
    email: str
    password: str


def _user_dict(user: User) -> dict:
    return {"id": user.id, "email": user.email, "is_admin": user.is_admin}


def _token_response(user: User) -> dict:
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "expires_in": config.ACCESS_TOKEN_MINUTES * 60,
        "user": _user_dict(user),
    }


@router.post("/register", status_code=201)
def register(body: RegisterRequest):
    if not config.ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="Registration is disabled on this server.")
    try:
        user = create_user(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _token_response(user)


# OAuth2 password form (fields: username, password) so Swagger's "Authorize" works too
@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_response(user)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return _user_dict(user)
