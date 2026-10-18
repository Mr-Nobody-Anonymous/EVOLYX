#!/usr/bin/env bash
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m apps.cli.main "$@"
