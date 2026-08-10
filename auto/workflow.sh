#!/bin/bash
# Example: run the autotest workflow
# Adjust --service-dir and --test-dir to your actual paths.
#
# Extra options (pass via "$@"):
#   --nostop          Keep container alive across concurrency levels
#                     (compose up once per test, down after all concurrency done)
#   --health-timeout  Health check timeout in seconds (default: 2160)
#   --log-level       DEBUG|INFO|WARNING|ERROR
#
# Example:
#   ./workflow.sh --nostop
#   ./workflow.sh --concurrency 1,32,128 --nostop
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv if available
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

python3 "$SCRIPT_DIR/workflow.py" \
  --service-dir "$SCRIPT_DIR/example-service-dir" \
  --test-dir "$SCRIPT_DIR/example-test-dir" \
  --concurrency 1,16,32,64,128,256,512 \
  "$@"
