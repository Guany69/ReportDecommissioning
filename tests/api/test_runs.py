"""Runs endpoints — contract checks for the migration scaffold.

The handlers are intentionally not implemented yet (async processing phase), so
these assert the wiring is in place and that unimplemented handlers signal 501
rather than 500. Tighten these into behavioral tests as the phase lands.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_create_run_not_implemented_yet():
    resp = client.post("/runs", json={"sources": {"table1": "ref-1"}})
    assert resp.status_code == 501


def test_get_run_not_implemented_yet():
    resp = client.get("/runs/abc123")
    assert resp.status_code == 501


def test_create_run_validates_request_body():
    # Missing required `sources` -> 422 from request validation, not 501.
    resp = client.post("/runs", json={})
    assert resp.status_code == 422
