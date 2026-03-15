#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:/app"

exec python3 -m uvicorn server:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}"
