"""Application settings, loaded from environment variables.

Centralizes configuration so the API, worker, and services read from one typed
source instead of scattered os.environ lookups. Values mirror the env vars the
engine already honors (report_cleanup.security) so security behavior stays
consistent between the CLI, the legacy Streamlit app, and this API.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Service metadata ---------------------------------------------------
    app_name: str = "report-decommissioning-api"
    environment: str = Field(default="development")

    # --- Security (mirrors report_cleanup.security env vars) ----------------
    require_auth: bool = Field(default=False, alias="REPORT_CLEANUP_REQUIRE_AUTH")
    access_code: str = Field(default="", alias="REPORT_CLEANUP_ACCESS_CODE")
    sensitive_mode: bool = Field(default=False, alias="REPORT_CLEANUP_SENSITIVE_MODE")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, alias="REPORT_CLEANUP_MAX_UPLOAD_BYTES")

    # --- Storage / async plumbing (filled in later phases) ------------------
    # OneDrive / SharePoint source for Power BI inputs.
    onedrive_tenant_id: str = Field(default="", alias="ONEDRIVE_TENANT_ID")
    onedrive_client_id: str = Field(default="", alias="ONEDRIVE_CLIENT_ID")
    onedrive_client_secret: str = Field(default="", alias="ONEDRIVE_CLIENT_SECRET")
    # Queue backing asynchronous run processing (e.g. Azure Storage Queue).
    queue_connection_string: str = Field(default="", alias="QUEUE_CONNECTION_STRING")
    queue_name: str = Field(default="analysis-runs", alias="QUEUE_NAME")
    # Where run artifacts (Excel + SQLite) are written.
    output_dir: str = Field(default="output", alias="REPORT_CLEANUP_OUTPUT_DIR")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
