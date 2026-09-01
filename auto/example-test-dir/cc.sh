#!/bin/bash
# Claude Code agentic trace replay benchmark for guidellm autotest
#
# Replays the WEKA-format agentic coding traces from
# semianalysisai/cc-traces-weka-062126-256k against an OpenAI-compatible
# /v1/chat/completions endpoint. The dataset ships per-request timestamps (t),
# input tokens (in), output tokens (out) and prefix-cache hash_ids nested in a
# "requests" column, with a top-level "id" conversation UUID and a 64-token
# KV block size -- exactly the guidellm `weka` trace format defaults.
#
# The guidellm `weka` data kind is a file-based trace loader: it reads a local
# `path` to the trace file and does NOT accept a HuggingFace hub `source`. So we
# resolve the dataset's single traces.jsonl via huggingface_hub (download once +
# cache) and point --data at that local path.
#
# Subagent handling: this "with-subagents" 256k variant interleaves subagent
# GROUP entries (type == "subagent") into each conversation's "requests" list.
# Those group entries carry no "in"/"out"/"hash_ids" fields (they wrap their own
# inner requests instead), and guidellm's weka loader (which still lists subagent
# support as in-development) raises KeyError('in') on them. We therefore
# pre-process the file, dropping subagent group entries so only replayable
# main-agent requests remain. Set KEEP_SUBAGENTS=1 to skip filtering.
#
# Two benchmark modes (MODE env var):
#   throughput (default) - fixed-concurrency load test that sweeps CONCURRENCY,
#                          exactly like rag.sh. CONCURRENCY=1 -> 10 requests,
#                          CONCURRENCY=16 -> 160, etc. Trace timestamps are
#                          ignored; the weka dataset only supplies per-request
#                          prompt/output token lengths. Use this to compare
#                          performance across concurrency levels.
#   replay               - true trace replay driven by the trace timestamps
#                          (--profile kind=replay). CONCURRENCY has no effect;
#                          load is scaled by TIME_SCALE instead (smaller = faster
#                          replay = higher instantaneous concurrency). Preserves
#                          multi-turn prefix-cache reuse, unlike throughput mode.
#
# Usage examples (run from the auto/ directory):
#   # throughput: sweep concurrency (like rag.sh). c1 -> 10 reqs, c16 -> 160, ...
#   python3 workflow.py \
#       --service-dir ./example-service-dir \
#       --test-dir ./example-test-dir \
#       --concurrency 1,16,32,64,128,256
#
#   # replay: sweep TIME_SCALE to vary load. Each value writes a "...cc.ts<val>"
#   # result set. MODE/TIME_SCALE are env vars, forwarded to this script.
#   for ts in 2.0 1.0 0.5 0.25; do
#     MODE=replay TIME_SCALE=$ts python3 workflow.py \
#         --service-dir ./example-service-dir \
#         --test-dir ./example-test-dir
#   done
#
# Environment variables injected by workflow.py:
#   PORT         - server port
#   MODEL        - model path (tokenizer)
#   CONCURRENCY  - current concurrency level (drives load in throughput mode)
#   OUTPUT_DIR   - output directory for results
#   SERVICE_NAME - compose file stem
#   TEST_NAME    - test script stem
#   TP           - tensor parallel size

HOST=127.0.0.1
PORT=${PORT:-8976}
MODEL=${MODEL:-/models/Qwen/Qwen3.5-27B}
CONCURRENCY=${CONCURRENCY:-1}
COUNT=${COUNT:-$((CONCURRENCY * 10))}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
SERVICE_NAME=${SERVICE_NAME:-cc}
TEST_NAME=${TEST_NAME:-cc}
TP=${TP:-1}
SERVED_MODEL=${SERVED_MODEL:-test}
RUN_TAG=${RUN_TAG:-c${CONCURRENCY}}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-}

# Benchmark mode:
#   throughput (default) - fixed-concurrency load test, sweeps CONCURRENCY like
#                          rag.sh. Trace timestamps are ignored; the weka data
#                          only supplies per-request prompt/output token lengths.
#   replay               - true trace replay driven by the trace timestamps.
#                          CONCURRENCY has no effect in this mode.
MODE=${MODE:-throughput}

# Time scale for replay intervals (replay mode only): 1.0 preserves original timing.
# In replay mode this is the knob you sweep to test performance under heavier
# load: 0.5 replays twice as fast (higher instantaneous concurrency), 2.0 slower.
TIME_SCALE=${TIME_SCALE:-1.0}
# Limit how many trace rows (conversations) are loaded/replayed.
SAMPLES=${SAMPLES:-100}

# Input-length filter (throughput mode only). Keep only requests whose `in` token
# count is within [MIN_IN_TOKENS, MAX_IN_TOKENS]. Defaults below target the
# 200K-256K bucket (~20% of the dataset, the longest-context requests). Override
# with env vars, or set either to empty for no bound (e.g. MIN_IN_TOKENS="").
# NOTE: this filters per-request. In replay mode it is ignored, because dropping
# individual turns would break each conversation's timeline and hash_id prefix
# chain (prefix-cache reuse). Filter is applied only when MODE != replay.
MIN_IN_TOKENS=${MIN_IN_TOKENS:-200000}
MAX_IN_TOKENS=${MAX_IN_TOKENS:-256000}

# In replay mode, CONCURRENCY is meaningless (load is driven by time_scale), so
# tag the output files by time_scale instead to avoid collisions when sweeping it.
if [ "${MODE}" = "replay" ]; then
  RUN_TAG="ts${TIME_SCALE}"
fi

# WEKA trace dataset (HuggingFace hub repo) and the single trace file within it.
DATASET_REPO=${DATASET_REPO:-semianalysisai/cc-traces-weka-062126-256k}
DATASET_FILE=${DATASET_FILE:-traces.jsonl}

# guidellm's `weka` loader reads a LOCAL trace file (its `path` field), it does
# not fetch from the hub. Resolve the file via huggingface_hub (which ships with
# guidellm): it downloads once, caches, honors HUGGING_FACE_HUB_TOKEN, and prints
# the local path. Subsequent runs are served from cache.
RAW_TRACE_PATH=$(python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id='${DATASET_REPO}', filename='${DATASET_FILE}', repo_type='dataset'))
") || { echo 'ERROR: failed to download WEKA trace file from the hub' >&2; exit 1; }

if [ ! -s "${RAW_TRACE_PATH}" ]; then
  echo "ERROR: WEKA trace file not found at ${RAW_TRACE_PATH}" >&2
  exit 1
fi

# Preprocess the trace so guidellm's weka loader can parse it. Two things are
# needed for the current loader:
#   1. Drop subagent GROUP entries (type == "subagent"): they lack the required
#      per-request fields and subagent replay is not yet implemented.
#   2. Slim each request dict to ONLY the required columns (t, in, out, hash_ids).
#      The loader casts each conversation to exactly those columns and rejects any
#      extras (model, api_time, type, ttft, think_time, ...) with
#      "columns in features must be identical as the columns in the dataset".
# Cache the processed file next to the raw download. The cache filename carries a
# version tag (PREP_VERSION): bump it whenever the preprocessing logic changes so
# stale cached files from an older version are not silently reused. We also
# regenerate whenever the raw download is newer than the processed file.
PREP_VERSION="v3"

# The input-length filter only applies outside replay mode (see note above).
EFF_MIN_IN="${MIN_IN_TOKENS}"
EFF_MAX_IN="${MAX_IN_TOKENS}"
if [ "${MODE}" = "replay" ]; then
  if [ -n "${MIN_IN_TOKENS}${MAX_IN_TOKENS}" ]; then
    echo "[cc.sh] NOTE: MIN_IN_TOKENS/MAX_IN_TOKENS ignored in replay mode" >&2
  fi
  EFF_MIN_IN=""
  EFF_MAX_IN=""
fi

TRACE_PATH="${RAW_TRACE_PATH}"
if [ "${KEEP_SUBAGENTS:-0}" != "1" ]; then
  # Include the input-length window in the cache filename so different filters
  # don't clobber each other.
  RANGE_TAG="all"
  if [ -n "${EFF_MIN_IN}${EFF_MAX_IN}" ]; then
    RANGE_TAG="in${EFF_MIN_IN:-0}-${EFF_MAX_IN:-max}"
  fi
  FILTERED_TRACE_PATH="${RAW_TRACE_PATH%.jsonl}.main-only.${PREP_VERSION}.${RANGE_TAG}.jsonl"
  if [ ! -s "${FILTERED_TRACE_PATH}" ] || [ "${RAW_TRACE_PATH}" -nt "${FILTERED_TRACE_PATH}" ]; then
    TS_COL="${TIMESTAMP_COLUMN:-t}" IN_COL="${PROMPT_TOKENS_COLUMN:-in}" \
    OUT_COL="${OUTPUT_TOKENS_COLUMN:-out}" HASH_COL="${HASH_IDS_COLUMN:-hash_ids}" \
    MIN_IN="${EFF_MIN_IN}" MAX_IN="${EFF_MAX_IN}" \
    python3 -c "
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
ts, tin, tout, thash = os.environ['TS_COL'], os.environ['IN_COL'], os.environ['OUT_COL'], os.environ['HASH_COL']
keep = (ts, tin, tout, thash)
min_in = int(os.environ['MIN_IN']) if os.environ.get('MIN_IN') else None
max_in = int(os.environ['MAX_IN']) if os.environ.get('MAX_IN') else None
kept_rows = dropped_sub = dropped_range = 0
with open(src) as fin, open(dst, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        reqs = row.get('requests', [])
        slimmed = []
        for r in reqs:
            d = r if isinstance(r, dict) else json.loads(r)
            if d.get('type') == 'subagent':
                dropped_sub += 1
                continue
            n_in = d[tin]
            if (min_in is not None and n_in < min_in) or (max_in is not None and n_in > max_in):
                dropped_range += 1
                continue
            # Keep only the columns the weka loader expects.
            slimmed.append({k: d[k] for k in keep})
        if not slimmed:
            continue
        row['requests'] = slimmed
        fout.write(json.dumps(row) + '\n')
        kept_rows += 1
window = f'[{min_in if min_in is not None else 0}, {max_in if max_in is not None else \"inf\"}]'
print(f'[cc.sh] preprocessed trace: kept {kept_rows} conversations, dropped {dropped_sub} subagent + {dropped_range} out-of-range requests, in-window {window}', file=sys.stderr)
if kept_rows == 0:
    print('[cc.sh] ERROR: no conversations left after filtering (check MIN_IN_TOKENS/MAX_IN_TOKENS)', file=sys.stderr)
    sys.exit(1)
" "${RAW_TRACE_PATH}" "${FILTERED_TRACE_PATH}" || { echo 'ERROR: failed to preprocess trace file' >&2; exit 1; }
  fi
  TRACE_PATH="${FILTERED_TRACE_PATH}"
fi

if [ ! -s "${TRACE_PATH}" ]; then
  echo "ERROR: WEKA trace file not found at ${TRACE_PATH}" >&2
  exit 1
fi

# Build output base name
OUT_BASE="${OUTPUT_PREFIX:+${OUTPUT_PREFIX}.}${SERVICE_NAME}.tp${TP}.${TEST_NAME}${RUN_TAG:+.${RUN_TAG}}"

# Select profile/constraint by MODE.
if [ "${MODE}" = "replay" ]; then
  # Time-driven replay: CONCURRENCY unused; SAMPLES bounds how many
  # conversations are replayed.
  PROFILE_ARGS=(--profile "kind=replay,time_scale=${TIME_SCALE}")
else
  # Fixed-concurrency load test (sweepable like rag.sh): CONCURRENCY drives the
  # concurrent request count and COUNT (=CONCURRENCY*10) bounds total requests.
  PROFILE_ARGS=(
    --profile "kind=throughput,max_concurrency=${CONCURRENCY}"
    --constraint "kind=max_requests,count=${COUNT}"
  )
fi

guidellm run \
  --backend "kind=openai_http,target=http://${HOST}:${PORT},model=${SERVED_MODEL},request_format=/v1/chat/completions" \
  "${PROFILE_ARGS[@]}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${MODEL}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data "{\"kind\":\"weka\",\"path\":\"${TRACE_PATH}\"}" \
  --data-loader "kind=pytorch,samples=${SAMPLES}" \
  --output "kind=json,path=${OUTPUT_DIR}/${OUT_BASE}.json" \
  --output "kind=csv,path=${OUTPUT_DIR}/${OUT_BASE}.csv" \
  --output "kind=plot,path=${OUTPUT_DIR}/${OUT_BASE}.png" \
  --seed "kind=static,value=0"
