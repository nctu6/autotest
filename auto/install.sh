#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "[install] Creating Python virtual environment in $ROOT_DIR/.venv ..."
python3 -m venv .venv

echo "[install] Activating virtual environment..."
source .venv/bin/activate

echo "[install] Installing guidellm in editable mode..."
pip install -e ".[plot]"

echo "[install] Done. Activate with: source $ROOT_DIR/.venv/bin/activate"
