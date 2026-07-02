#!/usr/bin/env bash

set -euo pipefail

exec gunicorn \
  --bind 0.0.0.0:8000 \
  --workers "${WEB_CONCURRENCY:-2}" \
  --worker-class uvicorn_worker.UvicornWorker \
  --timeout "${GUNICORN_TIMEOUT:-600}" \
  --access-logfile - \
  --error-logfile - \
  app.main:app
