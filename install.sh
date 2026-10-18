#!/usr/bin/env bash
# EVOLYX One-Click Linux/macOS/Server Installer
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================================="
echo "      EVOLYX Linux / macOS / Server Automated Installer   "
echo "=========================================================="

# 1. Detect Python 3
PYTHON_BIN=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[!] Error: Python 3.10+ is required but was not found."
    echo "    Install Python via your package manager: apt install python3 python3-venv (Ubuntu/Debian) or brew install python (macOS)"
    exit 1
fi

echo "[+] Found Python: $($PYTHON_BIN --version)"

# 2. Virtual Environment Setup
VENV_DIR="$DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment at .venv..."
    $PYTHON_BIN -m venv "$VENV_DIR" || {
        echo "[!] Warning: python3-venv not installed. Proceeding with user environment."
    }
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "[*] Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

# 3. Run Universal Installer
export PYTHONPATH="$DIR"
python3 "$DIR/install.py" "$@"
