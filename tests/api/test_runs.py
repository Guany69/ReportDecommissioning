from fastapi.testclient import TestClient

from app.main import app


VALID_REQUEST = {
    "submission_id": "RUN-20260702-001",
    "analysis_as_of_date": "2026-07-02",
    "source_drive_id": "drive-id",
    "files": [
        {
            "role": "metadata",
            "item_id": "metadata-item",
        },
        {
            "role": "execution_history",
            "item_id": "execution-item",
        },
        {
            "role": "report_fields",
            "item_id": "fields-item",
        },
    ],
}


def test_run_endpoint_validates_contract():
    with TestClient(app) as client:
        response = client.post(
            "/v1/analysis-runs",
            json=VALID_REQUEST,
        )

    # Expected until repository and queue are implemented.
    assert response.status_code == 503


def test_run_endpoint_rejects_missing_role():
    payload = {
        **VALID_REQUEST,
        "files": VALID_REQUEST["files"][:2],
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/analysis-runs",
            json=payload,
        )

    assert response.status_code == 422
