from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


FileRole = Literal[
    "metadata",
    "execution_history",
    "report_fields",
]


class OneDriveFileReference(BaseModel):
    role: FileRole
    item_id: str = Field(min_length=1)
    filename: str | None = None


class CreateAnalysisRunRequest(BaseModel):
    submission_id: str = Field(
        min_length=1,
        max_length=100,
    )

    analysis_as_of_date: date
    source_drive_id: str = Field(min_length=1)
    files: list[OneDriveFileReference]

    @model_validator(mode="after")
    def validate_file_roles(self):
        roles = [file.role for file in self.files]

        required = {
            "metadata",
            "execution_history",
            "report_fields",
        }

        if set(roles) != required:
            raise ValueError(
                "The submission must contain exactly one metadata, "
                "one execution_history, and one report_fields file."
            )

        if len(roles) != len(set(roles)):
            raise ValueError(
                "Each file role can appear only once."
            )

        return self
