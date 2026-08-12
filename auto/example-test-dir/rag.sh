#!/bin/bash
# RAG benchmark test config for guidellm autotest
# Environment variables injected by workflow.py:
#   PORT         - server port
#   MODEL        - model path
#   CONCURRENCY  - current concurrency level
#   COUNT        - number of requests (CONCURRENCY * 10)
#   OUTPUT_DIR   - output directory for results
#   SERVICE_NAME - compose file stem
#   TEST_NAME    - test script stem
#   TP           - tensor parallel size

HOST=0.0.0.0
PORT=${PORT:-8976}
MODEL=${MODEL:-/models/Qwen/Qwen3.5-9B}
CONCURRENCY=${CONCURRENCY:-1}
COUNT=${COUNT:-$((CONCURRENCY * 10))}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
SERVICE_NAME=${SERVICE_NAME:-unknown}
TEST_NAME=${TEST_NAME:-rag_bench}
TP=${TP:-1}

guidellm run \
  --backend "kind=openai_http,target=http://${HOST}:${PORT}" \
  --profile "kind=throughput,max_concurrency=${CONCURRENCY}" \
  --constraint "kind=max_requests,count=${COUNT}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${MODEL}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data '{"kind":"huggingface","source":"neural-bridge/rag-dataset-12000","load_kwargs":{"split":"train"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"prefix_column":"context","text_column":"question"}}' \
  --output "kind=json,path=${OUTPUT_DIR}/${SERVICE_NAME}.tp${TP}.${TEST_NAME}.c${CONCURRENCY}.json" \
  --output "kind=csv,path=${OUTPUT_DIR}/${SERVICE_NAME}.tp${TP}.${TEST_NAME}.c${CONCURRENCY}.csv" \
  --output "kind=plot,path=${OUTPUT_DIR}/${SERVICE_NAME}.tp${TP}.${TEST_NAME}.c${CONCURRENCY}.png" \
  --seed "kind=static,value=0"
