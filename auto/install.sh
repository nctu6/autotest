#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[install] Creating Python virtual environment..."
python3 -m venv .venv

echo "[install] Activating virtual environment..."
source .venv/bin/activate

echo "[install] Installing guidellm in editable mode..."
pip install -e .

echo "[install] Done. Activate with: source .venv/bin/activate"
