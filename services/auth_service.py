"""
Auth Service — Local JWT Authentication
=========================================
Handles: register, login, token refresh, profile management.
Uses bcrypt for password hashing and python-jose for JWTs.

No PocketBase required. PocketBase integration can be added
as a proxy layer without changing this service.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional
import uuid

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from config import settings
from models.user_models import User

logger = logging.getLogger(__name__)


def _prehash(plain: str) -> bytes:
    """SHA-256 pre-hash so bcrypt never sees > 72 bytes."""
    return hashlib.sha256(plain.encode("utf-8")).hexdigest().encode("utf-8")


def hash_password(plain: str) -> str:
    hashed = bcrypt.hashpw(_prehash(plain), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(plain), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: str, email: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=30)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as e:
        logger.debug(f"Token decode failed: {e}")
        return None


def register_user(db: Session, email: str, password: str, display_name: str, role: str) -> User:
    """Create a new user. Raises ValueError if email already exists."""
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise ValueError("Email already registered")
    user = User(
        email=email,
        display_name=display_name,
        password_hash=hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Verify credentials and return user or None."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return None
    return user


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    try:
        return db.query(User).filter(User.id == uuid.UUID(user_id)).first()
    except Exception:
        return None


def update_user_profile(
    db: Session, user_id: str,
    display_name: Optional[str] = None,
    role: Optional[str] = None,
) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    if display_name is not None:
        user.display_name = display_name
    if role is not None:
        user.role = role
    db.commit()
    db.refresh(user)
    return user
