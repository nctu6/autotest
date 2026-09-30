#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

LOG_FILE="$ROOT_DIR/install.log"
exec > >(tee "$LOG_FILE") 2>&1
echo "[install] Logging to $LOG_FILE"

echo "[install] Creating Python virtual environment in $ROOT_DIR/.venv ..."
python3 -m venv .venv

echo "[install] Activating virtual environment..."
source .venv/bin/activate

echo "[install] Installing guidellm in editable mode..."
pip install -e ".[plot,audio,vision]"

# Optuna server-arg search (auto/workflow.py --optuna / auto/optuna.sh).
# pyyaml is already a project dependency; optuna is opt-in for auto/.
echo "[install] Installing optuna (for auto/ --optuna)..."
pip install optuna

echo "[install] Done. Activate with: source $ROOT_DIR/.venv/bin/activate"
