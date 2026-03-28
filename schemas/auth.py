from pydantic import BaseModel, EmailStr
from typing import Optional, Literal
from datetime import datetime
import uuid


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    displayName: str
    role: Literal["analyst", "trader"] = "trader"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    displayName: Optional[str]
    role: str
    createdAt: datetime

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    accessToken: str
    refreshToken: str
    user: UserOut


class TokenRefreshRequest(BaseModel):
    refreshToken: str


class TokenRefreshResponse(BaseModel):
    accessToken: str


class UpdateProfileRequest(BaseModel):
    displayName: Optional[str] = None
    role: Optional[Literal["analyst", "trader"]] = None
