"""Auth router — 7 endpoints: register, login, logout, refresh, me (GET/PUT), google."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from schemas.auth import (
    RegisterRequest, LoginRequest, LoginResponse, UserOut,
    TokenRefreshRequest, TokenRefreshResponse, UpdateProfileRequest
)
from services import auth_service
from routers.deps import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Auth"])


@router.post("/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = auth_service.register_user(
            db, body.email, body.password, body.displayName, body.role
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "accessToken": auth_service.create_access_token(str(user.id), user.email, user.role),
        "refreshToken": auth_service.create_refresh_token(str(user.id)),
        "user": {
            "id": user.id,
            "email": user.email,
            "displayName": user.display_name,
            "role": user.role,
            "createdAt": user.created_at,
        },
    }


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "accessToken": auth_service.create_access_token(str(user.id), user.email, user.role),
        "refreshToken": auth_service.create_refresh_token(str(user.id)),
        "user": {
            "id": user.id,
            "email": user.email,
            "displayName": user.display_name,
            "role": user.role,
            "createdAt": user.created_at,
        },
    }


@router.post("/logout")
def logout(current_user: dict = Depends(get_current_user)):
    # JWT is stateless — client drops the token. No server-side action needed.
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=TokenRefreshResponse)
def refresh_token(body: TokenRefreshRequest, db: Session = Depends(get_db)):
    payload = auth_service.decode_token(body.refreshToken)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = auth_service.get_user_by_id(db, payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {
        "accessToken": auth_service.create_access_token(str(user.id), user.email, user.role)
    }


@router.get("/me", response_model=UserOut)
def get_me(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    user = auth_service.get_user_by_id(db, current_user["id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "email": user.email,
        "displayName": user.display_name,
        "role": user.role,
        "createdAt": user.created_at,
    }


@router.put("/me", response_model=UserOut)
def update_me(
    body: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = auth_service.update_user_profile(
        db, current_user["id"], body.displayName, body.role
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "email": user.email,
        "displayName": user.display_name,
        "role": user.role,
        "createdAt": user.created_at,
    }


@router.post("/google")
def google_oauth():
    """Placeholder — integrate Firebase or Google OAuth2 later."""
    raise HTTPException(
        status_code=501,
        detail="Google OAuth not yet configured. Use email/password auth.",
    )
