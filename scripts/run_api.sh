#!/usr/bin/env bash
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Starting EVOLYX REST API on http://127.0.0.1:8000..."
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
