#!/usr/bin/env bash
set -e
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Running EVOLYX test suite..."
python -m pytest tests/ -v
