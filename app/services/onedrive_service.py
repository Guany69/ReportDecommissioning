"""Fetch input exports from OneDrive / SharePoint for the Power BI workflow.

Resolves item references from a run request into local file paths the engine can
read, using the Microsoft Graph credentials in Settings. Implemented later.
"""
from __future__ import annotations

from app.settings import Settings


class OneDriveService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def download(self, item_ref: str, dest_path: str) -> str:
        """Download a OneDrive/SharePoint item to dest_path and return the path."""
        raise NotImplementedError("OneDriveService.download is implemented in a later phase.")
