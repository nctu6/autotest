#!/bin/bash
# LongBench-v2 long-context multiple-choice QA benchmark for guidellm autotest
#
# Loads zai-org/LongBench-v2 from HuggingFace and sends long-context MCQ prompts
# to an OpenAI-compatible endpoint. Dataset fields (per HF card):
#   context, question, choice_A/B/C/D, answer (plus _id/domain/difficulty/length)
#
# Prompt = context + question + choice_A…D (generative_column_mapper concatenates
# text_column fields with spaces, same as OpenAI text completion join). Labels
# like "(A)" from LongBench's 0-shot template are not injected; no answer scoring
# with the stock CLI.
#
# Rows whose joined prompt tokenizes to >= MAX_PROMPT_TOKENS (default 256K =
# 262144) are dropped before guidellm runs (LongBench contexts can reach ~2M words).
#
# Environment variables injected by workflow.py:
#   PORT          - server port
#   MODEL         - model/tokenizer path (host path after compose volume remap)
#   TOKENIZER     - optional tokenizer override (defaults to MODEL)
#   SERVED_MODEL  - API model name (from --served-model-name / compose env)
#   CONCURRENCY   - current concurrency level (optional, test decides if not set)
#   COUNT         - number of requests (CONCURRENCY * 10, optional)
#   OUTPUT_DIR    - output directory for results
#   OUTPUT_PREFIX - filename prefix from workflow pair dirs
#   SERVICE_NAME  - compose file stem
#   TEST_NAME     - test script stem
#   TP            - tensor parallel size
#   RUN_TAG       - filename suffix (e.g. "c32" or "" if concurrency not set by workflow)
#   HOST          - server host (default 127.0.0.1)

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8976}
MODEL=${MODEL:-/models/Qwen/Qwen3.5-9B}
TOKENIZER=${TOKENIZER:-${MODEL}}
CONCURRENCY=${CONCURRENCY:-1}
COUNT=${COUNT:-$((CONCURRENCY * 10))}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
SERVICE_NAME=${SERVICE_NAME:-unknown}
TEST_NAME=${TEST_NAME:-longbenchv2}
TP=${TP:-1}
SERVED_MODEL=${SERVED_MODEL:-${SERVICE_NAME:-test}}
RUN_TAG=${RUN_TAG:-c${CONCURRENCY}}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-}

# Per-request read timeout (seconds). LongBench-v2 prompts are long-context;
# raise further for very high concurrency or the "long" length bucket.
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-3600}

# Drop rows whose space-joined prompt (context+question+choices) has this many
# tokens or more. Default 256K = 262144. Set empty to disable filtering.
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-262144}

DATASET_SOURCE=${DATASET_SOURCE:-zai-org/LongBench-v2}
DATASET_SPLIT=${DATASET_SPLIT:-train}

# Optional row cap for the loader after filtering. Empty = use all kept rows
# (subject to COUNT). Set SAMPLES=N to clip loaded rows.
SAMPLES=${SAMPLES:-}

# Build output base name
OUT_BASE="${OUTPUT_PREFIX:+${OUTPUT_PREFIX}.}${SERVICE_NAME}.tp${TP}.${TEST_NAME}${RUN_TAG:+.${RUN_TAG}}"

PREP_CACHE_DIR=${PREP_CACHE_DIR:-${OUTPUT_DIR}/.longbenchv2_cache}
mkdir -p "${PREP_CACHE_DIR}"

# Cache key includes tokenizer path + max tokens + dataset id/split.
TOK_TAG=$(printf '%s' "${TOKENIZER}" | shasum -a 256 | awk '{print substr($1,1,12)}')
MAX_TAG=${MAX_PROMPT_TOKENS:-none}
FILTERED_JSONL="${PREP_CACHE_DIR}/longbenchv2.${DATASET_SPLIT}.max${MAX_TAG}.tok${TOK_TAG}.jsonl"

if [ -s "${FILTERED_JSONL}" ]; then
  echo "[longbenchv2.sh] reusing filtered dataset: ${FILTERED_JSONL}" >&2
else
  echo "[longbenchv2.sh] filtering ${DATASET_SOURCE} (${DATASET_SPLIT}) to prompt tokens < ${MAX_PROMPT_TOKENS:-∞} with tokenizer ${TOKENIZER}" >&2
  TOKENIZER_PATH="${TOKENIZER}" DATASET_SOURCE="${DATASET_SOURCE}" DATASET_SPLIT="${DATASET_SPLIT}" \
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" FILTERED_JSONL="${FILTERED_JSONL}" python3 - <<'PY'
import json
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

source = os.environ["DATASET_SOURCE"]
split = os.environ["DATASET_SPLIT"]
tok_path = os.environ["TOKENIZER_PATH"]
out_path = os.environ["FILTERED_JSONL"]
max_raw = os.environ.get("MAX_PROMPT_TOKENS", "").strip()
max_tokens = int(max_raw) if max_raw else None

cols = ["context", "question", "choice_A", "choice_B", "choice_C", "choice_D"]

print(f"[longbenchv2.sh] loading tokenizer: {tok_path}", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(tok_path, use_fast=False, trust_remote_code=True)

print(f"[longbenchv2.sh] loading dataset: {source} split={split}", file=sys.stderr)
ds = load_dataset(source, split=split)

kept = dropped = 0
tmp_path = out_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as out:
    for row in ds:
        parts = [str(row.get(c) or "") for c in cols]
        prompt = " ".join(parts)
        n_tokens = len(tok.encode(prompt, add_special_tokens=False))
        if max_tokens is not None and n_tokens >= max_tokens:
            dropped += 1
            continue
        kept += 1
        payload = {c: row.get(c) for c in cols}
        # keep useful metadata for debugging / later scoring
        for meta in ("_id", "domain", "sub_domain", "difficulty", "length", "answer"):
            if meta in row:
                payload[meta] = row[meta]
        payload["prompt_tokens"] = n_tokens
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")

os.replace(tmp_path, out_path)
print(
    f"[longbenchv2.sh] kept {kept} rows, dropped {dropped} "
    f"(max_prompt_tokens={max_tokens if max_tokens is not None else 'disabled'}) -> {out_path}",
    file=sys.stderr,
)
if kept == 0:
    print("[longbenchv2.sh] ERROR: no rows left after prompt-token filter", file=sys.stderr)
    sys.exit(1)
PY
fi

if [ ! -s "${FILTERED_JSONL}" ]; then
  echo "[longbenchv2.sh] ERROR: filtered dataset missing: ${FILTERED_JSONL}" >&2
  exit 1
fi

DATA_LOADER_ARGS=()
if [ -n "${SAMPLES}" ]; then
  DATA_LOADER_ARGS=(--data-loader "kind=pytorch,samples=${SAMPLES}")
fi

guidellm run \
  --backend "kind=openai_http,target=http://${HOST}:${PORT},model=${SERVED_MODEL},timeout=${REQUEST_TIMEOUT}" \
  --profile "kind=throughput,max_concurrency=${CONCURRENCY}" \
  --constraint "kind=max_requests,count=${COUNT}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${TOKENIZER}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data "{\"kind\":\"json_file\",\"path\":\"${FILTERED_JSONL}\"}" \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":["context","question","choice_A","choice_B","choice_C","choice_D"]}}' \
  "${DATA_LOADER_ARGS[@]}" \
  --output "kind=json,path=${OUTPUT_DIR}/${OUT_BASE}.json" \
  --output "kind=csv,path=${OUTPUT_DIR}/${OUT_BASE}.csv" \
  --output "kind=plot,path=${OUTPUT_DIR}/${OUT_BASE}.png" \
  --seed "kind=static,value=0"
