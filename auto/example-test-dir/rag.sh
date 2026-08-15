#!/bin/bash
# RAG benchmark test config for guidellm autotest
# Environment variables injected by workflow.py:
#   PORT         - server port
#   MODEL        - model path
#   CONCURRENCY  - current concurrency level (optional, test decides if not set)
#   COUNT        - number of requests (CONCURRENCY * 10, optional)
#   OUTPUT_DIR   - output directory for results
#   SERVICE_NAME - compose file stem
#   TEST_NAME    - test script stem
#   TP           - tensor parallel size
#   RUN_TAG      - filename suffix (e.g. "c32" or "" if concurrency not set by workflow)

HOST=127.0.0.1
PORT=${PORT:-8976}
MODEL=${MODEL:-/models/Qwen/Qwen3.5-9B}
CONCURRENCY=${CONCURRENCY:-1}
COUNT=${COUNT:-$((CONCURRENCY * 10))}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
SERVICE_NAME=${SERVICE_NAME:-unknown}
TEST_NAME=${TEST_NAME:-rag}
TP=${TP:-1}
SERVED_MODEL=${SERVED_MODEL:-test}
RUN_TAG=${RUN_TAG:-c${CONCURRENCY}}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-}

# Build output base name
OUT_BASE="${OUTPUT_PREFIX:+${OUTPUT_PREFIX}.}${SERVICE_NAME}.tp${TP}.${TEST_NAME}${RUN_TAG:+.${RUN_TAG}}"

guidellm run \
  --backend "kind=openai_http,target=http://${HOST}:${PORT},model=${SERVED_MODEL}" \
  --profile "kind=throughput,max_concurrency=${CONCURRENCY}" \
  --constraint "kind=max_requests,count=${COUNT}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${MODEL}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data '{"kind":"huggingface","source":"neural-bridge/rag-dataset-12000","load_kwargs":{"split":"train"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":["context","question"]}}' \
  --output "kind=json,path=${OUTPUT_DIR}/${OUT_BASE}.json" \
  --output "kind=csv,path=${OUTPUT_DIR}/${OUT_BASE}.csv" \
  --output "kind=plot,path=${OUTPUT_DIR}/${OUT_BASE}.png" \
  --seed "kind=static,value=0"
