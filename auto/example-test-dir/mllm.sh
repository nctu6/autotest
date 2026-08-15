#!/bin/bash
# Multimodal LLM (VLM) benchmark for guidellm autotest
#
# Sends synthetic text + synthetic image to /v1/chat/completions endpoint.
# Uses two --data sources: synthetic_text for prompts, synthetic_image for images.
#
# Environment variables injected by workflow.py:
#   PORT         - server port
#   MODEL        - model path
#   CONCURRENCY  - current concurrency level
#   COUNT        - number of requests (CONCURRENCY * 10)
#   OUTPUT_DIR   - output directory for results
#   SERVICE_NAME - compose file stem
#   TEST_NAME    - test script stem
#   TP           - tensor parallel size

HOST=127.0.0.1
PORT=${PORT:-8976}
MODEL=${MODEL:-/models/Qwen/Qwen2.5-VL-7B-Instruct}
CONCURRENCY=${CONCURRENCY:-1}
COUNT=${COUNT:-$((CONCURRENCY * 10))}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
SERVICE_NAME=${SERVICE_NAME:-mllm}
TEST_NAME=${TEST_NAME:-mllm}
TP=${TP:-1}
SERVED_MODEL=${SERVED_MODEL:-test}
RUN_TAG=${RUN_TAG:-c${CONCURRENCY}}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-}

# Build output base name
OUT_BASE="${OUTPUT_PREFIX:+${OUTPUT_PREFIX}.}${SERVICE_NAME}.tp${TP}.${TEST_NAME}${RUN_TAG:+.${RUN_TAG}}"

guidellm run \
  --backend "kind=openai_http,target=http://${HOST}:${PORT},model=${SERVED_MODEL},request_format=/v1/chat/completions" \
  --profile "kind=throughput,max_concurrency=${CONCURRENCY}" \
  --constraint "kind=max_requests,count=${COUNT}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${MODEL}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data '{"kind":"huggingface","source":"lmms-lab/VQAv2","load_kwargs":{"split":"validation"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"question","image_column":"image"}}' \
  --output "kind=json,path=${OUTPUT_DIR}/${OUT_BASE}.json" \
  --output "kind=csv,path=${OUTPUT_DIR}/${OUT_BASE}.csv" \
  --output "kind=plot,path=${OUTPUT_DIR}/${OUT_BASE}.png" \
  --seed "kind=static,value=0"
