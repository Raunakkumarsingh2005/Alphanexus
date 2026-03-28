from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/alphanexus"

    # ── Auth ──────────────────────────────────────────────────
    JWT_SECRET: str = "alphanexus-jwt-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ── PocketBase (optional) ─────────────────────────────────
    POCKETBASE_URL: Optional[str] = None

    # ── API Keys ──────────────────────────────────────────────
    FINNHUB_API_KEY: str = ""
    NEWSAPI_KEY: str = ""
    QUIVER_API_KEY: Optional[str] = None

    # ── App ───────────────────────────────────────────────────
    APP_NAME: str = "AlphaNexus"
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
