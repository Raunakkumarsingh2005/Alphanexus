"""
Auth dependency — shared FastAPI dependency for protected routes.
Import get_current_user into any router that needs authentication.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database import get_db
from services import auth_service

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> dict:
    """FastAPI dependency — decode JWT and return user payload. 401 if invalid."""
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = auth_service.decode_token(credentials.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user = auth_service.get_user_by_id(db, payload["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return {"id": str(user.id), "email": user.email, "role": user.role, "display_name": user.display_name}


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Optional[dict]:
    """Like get_current_user but returns None instead of raising if unauthenticated."""
    if not credentials:
        return None
    payload = auth_service.decode_token(credentials.credentials)
    if not payload:
        return None
    return {"id": payload.get("sub"), "email": payload.get("email"), "role": payload.get("role")}
