"""Inbound request models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class InputSource(BaseModel):
    """Location of one uploaded/linked input table.

    Either a previously uploaded blob reference or a OneDrive/SharePoint item id
    is supplied; the resolution logic lives in the services layer.
    """

    table1: str = Field(..., description="Reference to the Metadata (Comprehensive) export.")
    table2: str | None = Field(default=None, description="Reference to the Execution export.")
    table3_fields: str | None = Field(default=None, description="Reference to the Fields export.")


class CreateRunRequest(BaseModel):
    """Payload to submit a new analysis run."""

    sources: InputSource
    sensitive_mode: bool = Field(
        default=False,
        description="Mask person-identifying fields in outputs and the database.",
    )
    keep_history: bool = Field(
        default=False,
        description="Retain prior runs in the database instead of resetting.",
    )
