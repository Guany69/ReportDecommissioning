#!/usr/bin/env bash
# Azure App Service startup command for the Report Decommissioning API.
# Serves app.main:app with Uvicorn workers behind Gunicorn.
set -euo pipefail

PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-2}"

exec gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers "${WORKERS}" \
  --bind "0.0.0.0:${PORT}" \
  --timeout 600 \
  --access-logfile - \
  --error-logfile -
