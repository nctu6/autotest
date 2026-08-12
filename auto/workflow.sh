#!/bin/bash
# Example: run the autotest workflow
# Adjust --service-dir and --test-dir to your actual paths.
#
# Extra options (pass via "$@"):
#   --nostop          Keep container alive across concurrency levels
#                     (compose up once per test, down after all concurrency done)
#   --output          Output directory for benchmark results (default: ./results)
#   --health-timeout  Health check timeout in seconds (default: 2160)
#   --log-level       DEBUG|INFO|WARNING|ERROR
#
# Example:
#   ./workflow.sh --nostop
#   ./workflow.sh --concurrency 1,32,128 --nostop
#   ./workflow.sh --output ./my-results --nostop
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate venv if available
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
else
    echo "[warn] venv not found at $ROOT_DIR/.venv" >&2
fi

python3 "$SCRIPT_DIR/workflow.py" \
  --service-dir "$SCRIPT_DIR/example-service-dir" \
  --test-dir "$SCRIPT_DIR/example-test-dir" \
  --concurrency 1,16,32,64,128,256,512 \
  --output "$SCRIPT_DIR/results" \
  "$@"
