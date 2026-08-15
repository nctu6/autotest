#!/bin/bash
# GuideLLM Autotest Workflow
#
# Usage:
#   bash auto/workflow.sh                         # run with default config below
#   bash auto/workflow.sh --nostop                # keep container alive across concurrency
#   bash auto/workflow.sh --concurrency 1,32,128  # loop concurrency (test uses $CONCURRENCY)
#
# Options (pass via "$@"):
#   --config          Pairs of (service_dir,test_dir), one per line or semicolon-separated
#   --concurrency     Comma-separated concurrency values. If not set, loop once and
#                     let the test script decide its own concurrency.
#   --nostop          Keep container alive across concurrency levels
#   --output          Output directory for benchmark results (default: ./results)
#   --health-timeout  Health check timeout in seconds (default: 2160)
#   --log-level       DEBUG|INFO|WARNING|ERROR
#
#   --target URL      Test a running service (no container management)
#                     Requires: --model, --service-name, --tp, --test-dir
#
# --config format:
#   Each line is: service_dir,test_dir
#   Example:
#     --config "
#       /path/to/rag-services,/path/to/rag-tests;
#       /path/to/mllm-services,/path/to/mllm-tests;
#     "
#
# Legacy (single pair):
#   --service-dir /path/to/services --test-dir /path/to/tests
#
# Output filename convention:
#   ${OUTPUT_PREFIX}.${SERVICE_NAME}.tp${TP}.${TEST_NAME}.${RUN_TAG}.{json,csv,png}
#   - OUTPUT_PREFIX = service_dir_name.test_dir_name (auto from --config pair)
#   - RUN_TAG = c${CONCURRENCY} (when --concurrency is set) or empty (test decides)
#
# Examples:
#   # No --concurrency (test script decides, loop once per test):
#   python3 auto/workflow.py --config "/svc,/tests" --nostop
#
#   # With --concurrency (workflow loops each value):
#   python3 auto/workflow.py --config "/svc,/tests" --concurrency 1,16,32,64
#
#   # Semicolon-separated single-line:
#   python3 auto/workflow.py --config "/svc1,/test1;/svc2,/test2"
#
#   # Multiple pairs (multiline):
#   python3 auto/workflow.py --config "
#     /path/to/rag-service,/path/to/rag-test;
#     /path/to/mllm-service,/path/to/mllm-test;
#   " --concurrency 1,16,32 --nostop
#
#   # Against a running service (--target mode):
#   python3 auto/workflow.py --target http://127.0.0.1:8976 \
#     --model /models/Qwen/Qwen3.5-9B --service-name qwen9b --tp 1 \
#     --test-dir /path/to/tests --concurrency 1,16,32
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate venv if available
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
else
    echo "[warn] venv not found at $ROOT_DIR/.venv" >&2
fi

# Verify guidellm is reachable
if ! command -v guidellm &>/dev/null; then
    echo "[error] guidellm not found in PATH. Did you run install.sh?" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/workflow.py" \
  --config "
    $SCRIPT_DIR/example-service-dir,$SCRIPT_DIR/example-test-dir
  " \
  --output "$SCRIPT_DIR/results" \
  "$@"
