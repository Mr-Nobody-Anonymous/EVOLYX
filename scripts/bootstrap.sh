#!/usr/bin/env bash
set -e
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=========================================="
echo "  Bootstrapping EVOLYX AI Research Platform"
echo "=========================================="

mkdir -p artifacts/models artifacts/datasets artifacts/evaluations experiments/results

echo "Running initial verification tests..."
python -m pytest tests/

echo "[SUCCESS] EVOLYX Platform verified and ready."
