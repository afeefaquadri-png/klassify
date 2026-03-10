"""
Klassify – Central configuration management.
All environment-driven settings live here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "Klassify"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── API ────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Storage ────────────────────────────────────────────────────────────
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    MODEL_DIR: Path = BASE_DIR / "data" / "models"
    EXPERIMENT_DIR: Path = BASE_DIR / "data" / "experiments"
    MAX_UPLOAD_SIZE_MB: int = 100

    # ── Redis / Celery ─────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # ── ML defaults ────────────────────────────────────────────────────────
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_RANDOM_STATE: int = 42
    MAX_TRAINING_TIME_SECONDS: int = 3600

    # ── Security ───────────────────────────────────────────────────────────
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".parquet", ".json"]
    SECRET_KEY: str = "change-me-in-production"

    # ── Monitoring ─────────────────────────────────────────────────────────
    PROMETHEUS_ENABLED: bool = False
    SENTRY_DSN: Optional[str] = None

    def create_dirs(self) -> None:
        """Ensure all required directories exist."""
        for d in (self.UPLOAD_DIR, self.MODEL_DIR, self.EXPERIMENT_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_dirs()
