#!/usr/bin/env bash
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Launching EVOLYX Evolution Experiment..."
python -m apps.cli.main evolve
