"""FastAPI web service for the Report Decommissioning platform.

This package exposes the HTTP API (`app.main:app`) that fronts the existing
`report_cleanup` engine. It replaces the legacy Streamlit UI
(`legacy_streamlit_app.py`) with an API-first workflow suitable for Power BI
and asynchronous processing via the `worker` package.
"""
