from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Report Decommissioning API"
    app_version: str = "0.1.0"

    environment: Literal[
        "local",
        "development",
        "test",
        "production",
    ] = "local"

    log_level: str = "INFO"
    enable_docs: bool = True

    engine_config_path: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="REPORT_API_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
