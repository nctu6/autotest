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

# Per-request read timeout (seconds) for the backend. The cc traces are long-
# context: a single ~200K-token request can take 6+ minutes to complete. With no
# timeout (guidellm default), slow-but-valid requests that are still in flight
# when the max_requests constraint completes get cancelled and counted as
# ERRORED. A generous timeout lets them finish and be counted as successful.
# Raise further if you push very high concurrency or the 200K-256K bucket.
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}

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

# Input handling (throughput mode only; both ignored in replay mode).
#
# TRUNCATE_IN_TOKENS: if set, CLAMP every request's input to at most this many
#   tokens instead of dropping requests. The weka prompt is synthesized from `in`
#   (token count) + `hash_ids` (64-token blocks), so truncation = cap `in` to N
#   and keep the first ceil(N/64) hash_ids. Every request is still sent, just with
#   a shorter prompt. This is the simplest way to bound prompt size / KV usage.
#   Default 25000 (25K). Set empty to disable truncation.
TRUNCATE_IN_TOKENS=${TRUNCATE_IN_TOKENS:-25000}
#
# MIN_IN_TOKENS / MAX_IN_TOKENS: per-request range FILTER (drops requests outside
#   the window), evaluated on the ORIGINAL `in` before truncation. Filtering runs
#   FIRST, then survivors are truncated to TRUNCATE_IN_TOKENS.
#
#   For a uniform ~25K-prompt throughput test we want every request to end up at
#   ~25K tokens, so we keep only requests whose ORIGINAL `in` is already >= the
#   truncation target (otherwise truncation is a no-op and the prompt stays small,
#   diluting the "~25K" target). Default floor = TRUNCATE_IN_TOKENS, no upper bound.
#   Set both empty to keep every request regardless of size.
MIN_IN_TOKENS=${MIN_IN_TOKENS:-${TRUNCATE_IN_TOKENS}}
MAX_IN_TOKENS=${MAX_IN_TOKENS:-}
#
# FLATTEN_TURNS: in throughput mode we want each REQUEST to be an independent
#   single-turn ~25K prompt so nothing accumulates across turns and load scales
#   purely with CONCURRENCY. The weka loader replays each trace row as a chained
#   MULTI-TURN conversation (turn N sees turns 0..N-1 in its context), which is
#   what pushes late turns past the server context window (the 400 Bad Request
#   errors). With FLATTEN_TURNS=1 (default) every surviving turn is emitted as its
#   own one-turn conversation, giving a flat pool of uniform single-turn requests.
#   Set FLATTEN_TURNS=0 to preserve original multi-turn conversations.
#   Ignored in replay mode (replay needs the real multi-turn timing/structure).
FLATTEN_TURNS=${FLATTEN_TURNS:-1}

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
PREP_VERSION="v6"

# The input filter/truncation/flattening only apply outside replay mode.
EFF_MIN_IN="${MIN_IN_TOKENS}"
EFF_MAX_IN="${MAX_IN_TOKENS}"
EFF_TRUNC="${TRUNCATE_IN_TOKENS}"
EFF_FLATTEN="${FLATTEN_TURNS}"
if [ "${MODE}" = "replay" ]; then
  if [ -n "${MIN_IN_TOKENS}${MAX_IN_TOKENS}${TRUNCATE_IN_TOKENS}" ]; then
    echo "[cc.sh] NOTE: MIN_IN_TOKENS/MAX_IN_TOKENS/TRUNCATE_IN_TOKENS ignored in replay mode" >&2
  fi
  if [ "${FLATTEN_TURNS}" = "1" ]; then
    echo "[cc.sh] NOTE: FLATTEN_TURNS ignored in replay mode (multi-turn structure preserved)" >&2
  fi
  EFF_MIN_IN=""
  EFF_MAX_IN=""
  EFF_TRUNC=""
  EFF_FLATTEN="0"
fi

TRACE_PATH="${RAW_TRACE_PATH}"
if [ "${KEEP_SUBAGENTS:-0}" != "1" ]; then
  # Encode filter window + truncation in the cache filename so different settings
  # don't clobber each other.
  RANGE_TAG="all"
  if [ -n "${EFF_MIN_IN}${EFF_MAX_IN}" ]; then
    RANGE_TAG="in${EFF_MIN_IN:-0}-${EFF_MAX_IN:-max}"
  fi
  if [ -n "${EFF_TRUNC}" ]; then
    RANGE_TAG="${RANGE_TAG}.trunc${EFF_TRUNC}"
  fi
  if [ "${EFF_FLATTEN}" = "1" ]; then
    RANGE_TAG="${RANGE_TAG}.flat"
  fi

  # Cache the processed file in a stable, writable directory (default: next to
  # this script under .cc_cache) instead of inside the read-only-ish HuggingFace
  # snapshot dir. Override with PREP_CACHE_DIR.
  PREP_CACHE_DIR="${PREP_CACHE_DIR:-$(dirname "$0")/.cc_cache}"
  mkdir -p "${PREP_CACHE_DIR}" 2>/dev/null || true

  # Derive a stable identity for the raw trace from its resolved content
  # signature (size + mtime of the real blob), NOT a live mtime comparison.
  # Embedding it in the cache filename means: same raw file + same params ->
  # same cache name -> reuse; a changed raw download -> new name -> regenerate.
  RAW_SIG="$(python3 -c "
import os, sys
p = os.path.realpath(sys.argv[1])
st = os.stat(p)
print(f'{st.st_size}-{int(st.st_mtime)}')
" "${RAW_TRACE_PATH}" 2>/dev/null || echo "nosig")"

  RAW_STEM="$(basename "${RAW_TRACE_PATH%.jsonl}")"
  FILTERED_TRACE_PATH="${PREP_CACHE_DIR}/${RAW_STEM}.main-only.${PREP_VERSION}.${RANGE_TAG}.${RAW_SIG}.jsonl"
  if [ -s "${FILTERED_TRACE_PATH}" ]; then
    echo "[cc.sh] reusing cached preprocessed trace: ${FILTERED_TRACE_PATH}" >&2
  fi
  if [ ! -s "${FILTERED_TRACE_PATH}" ]; then
    TS_COL="${TIMESTAMP_COLUMN:-t}" IN_COL="${PROMPT_TOKENS_COLUMN:-in}" \
    OUT_COL="${OUTPUT_TOKENS_COLUMN:-out}" HASH_COL="${HASH_IDS_COLUMN:-hash_ids}" \
    MIN_IN="${EFF_MIN_IN}" MAX_IN="${EFF_MAX_IN}" TRUNC_IN="${EFF_TRUNC}" \
    FLATTEN="${EFF_FLATTEN}" \
    BLOCK_SIZE="${HASH_ID_BLOCK_SIZE:-64}" \
    python3 -c "
import json, math, os, sys
src, dst = sys.argv[1], sys.argv[2]
ts, tin, tout, thash = os.environ['TS_COL'], os.environ['IN_COL'], os.environ['OUT_COL'], os.environ['HASH_COL']
keep = (ts, tin, tout, thash)
min_in = int(os.environ['MIN_IN']) if os.environ.get('MIN_IN') else None
max_in = int(os.environ['MAX_IN']) if os.environ.get('MAX_IN') else None
trunc = int(os.environ['TRUNC_IN']) if os.environ.get('TRUNC_IN') else None
flatten = os.environ.get('FLATTEN') == '1'
block = int(os.environ['BLOCK_SIZE'])
kept_rows = dropped_sub = dropped_range = truncated = emitted_reqs = 0
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
            rec = {k: d[k] for k in keep}
            # Truncate: clamp the in-token count to trunc tokens and keep the
            # first ceil(trunc/block) hash_ids so the synthesized prompt matches.
            if trunc is not None and rec[tin] > trunc:
                rec[tin] = trunc
                n_blocks = math.ceil(trunc / block)
                rec[thash] = rec[thash][:n_blocks]
                truncated += 1
            slimmed.append(rec)
        if not slimmed:
            continue
        if flatten:
            # Emit each surviving turn as its own single-turn conversation so the
            # weka loader replays it as an independent request (no multi-turn
            # context accumulation). Give each a unique conversation id.
            base_id = row.get('id', f'conv{kept_rows}')
            for i, rec in enumerate(slimmed):
                out_row = dict(row)
                out_row['id'] = f'{base_id}.t{i}'
                out_row['requests'] = [rec]
                fout.write(json.dumps(out_row) + '\n')
                emitted_reqs += 1
            kept_rows += 1
        else:
            row['requests'] = slimmed
            fout.write(json.dumps(row) + '\n')
            emitted_reqs += len(slimmed)
            kept_rows += 1
window = f'[{min_in if min_in is not None else 0}, {max_in if max_in is not None else \"inf\"}]'
mode = 'flattened single-turn' if flatten else 'multi-turn'
print(f'[cc.sh] preprocessed trace ({mode}): kept {kept_rows} conversations / {emitted_reqs} requests, dropped {dropped_sub} subagent + {dropped_range} out-of-range, truncated {truncated} requests to {trunc if trunc is not None else \"-\"} tokens, in-window {window}', file=sys.stderr)
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
  --backend "kind=openai_http,target=http://${HOST}:${PORT},model=${SERVED_MODEL},request_format=/v1/chat/completions,timeout=${REQUEST_TIMEOUT}" \
  "${PROFILE_ARGS[@]}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${MODEL}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data "{\"kind\":\"weka\",\"path\":\"${TRACE_PATH}\",\"validate\":${WEKA_VALIDATE:-false}}" \
  --data-loader "kind=pytorch,samples=${SAMPLES}" \
  --output "kind=json,path=${OUTPUT_DIR}/${OUT_BASE}.json" \
  --output "kind=csv,path=${OUTPUT_DIR}/${OUT_BASE}.csv" \
  --output "kind=plot,path=${OUTPUT_DIR}/${OUT_BASE}.png" \
  --seed "kind=static,value=0"
