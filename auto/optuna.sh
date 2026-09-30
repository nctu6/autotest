#!/bin/bash
# Optuna server-arg optimization wrapper (params only in opt.yml args).
#
# Usage (from repo root; paths in opt.yml are relative to CWD):
#   bash auto/optuna.sh
#   bash auto/optuna.sh ./auto/example-opt-dir/opt.yml
#
# Default opt.yml: ./auto/example-opt-dir/opt.yml (or SCRIPT_DIR fallback).
# Absolute paths in opt.yml work as-is. No hidden auto/ prefixing.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

LOG_FILE="$ROOT_DIR/optuna.log"
exec > >(tee "$LOG_FILE") 2>&1
echo "[optuna] Logging to $LOG_FILE"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.venv/bin/activate"
else
    echo "[warn] venv not found at $ROOT_DIR/.venv" >&2
fi

OPT_YML="${1:-./auto/example-opt-dir/opt.yml}"
if [ "$#" -gt 0 ]; then
    shift
fi

if [[ "$OPT_YML" != /* ]]; then
    if [ -f "$OPT_YML" ]; then
        OPT_YML="$(cd "$(dirname "$OPT_YML")" && pwd)/$(basename "$OPT_YML")"
    elif [ -f "$SCRIPT_DIR/../$OPT_YML" ] && [[ "$OPT_YML" == ./auto/* || "$OPT_YML" == auto/* ]]; then
        # repo-root-style path while CWD is auto/: strip leading auto/
        REL="${OPT_YML#./}"
        REL="${REL#auto/}"
        if [ -f "$SCRIPT_DIR/$REL" ]; then
            OPT_YML="$SCRIPT_DIR/$REL"
        fi
    elif [ -f "$SCRIPT_DIR/$OPT_YML" ]; then
        OPT_YML="$SCRIPT_DIR/$OPT_YML"
    elif [ -f "$SCRIPT_DIR/example-opt-dir/opt.yml" ] && [ "$OPT_YML" = "./auto/example-opt-dir/opt.yml" ]; then
        OPT_YML="$SCRIPT_DIR/example-opt-dir/opt.yml"
    fi
fi

if [ ! -f "$OPT_YML" ]; then
    echo "[error] opt.yml not found: $OPT_YML" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/workflow.py" --optuna "$OPT_YML" "$@"
