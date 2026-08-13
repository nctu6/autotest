#!/bin/bash
# Whisper audio transcription benchmark for guidellm autotest
#
# Uses a HuggingFace audio dataset and sends to /v1/audio/transcriptions endpoint.
# guidellm auto-detects the "audio" column and encodes it for the request.
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

HOST=0.0.0.0
PORT=${PORT:-8976}
MODEL=${MODEL:-/models/openai/whisper-large-v3}
CONCURRENCY=${CONCURRENCY:-1}
COUNT=${COUNT:-$((CONCURRENCY * 10))}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
SERVICE_NAME=${SERVICE_NAME:-whisper}
TEST_NAME=${TEST_NAME:-whisper}
TP=${TP:-1}

guidellm run \
  --backend "kind=openai_http,target=http://${HOST}:${PORT},model=whisper,request_format=/v1/audio/transcriptions" \
  --profile "kind=throughput,max_concurrency=${CONCURRENCY}" \
  --constraint "kind=max_requests,count=${COUNT}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${MODEL}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data '{"kind":"huggingface","source":"hf-internal-testing/librispeech_asr_dummy","load_kwargs":{"name":"clean","split":"validation"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"audio_column":"audio"}}' \
  --output "kind=json,path=${OUTPUT_DIR}/${SERVICE_NAME}.tp${TP}.${TEST_NAME}.c${CONCURRENCY}.json" \
  --output "kind=csv,path=${OUTPUT_DIR}/${SERVICE_NAME}.tp${TP}.${TEST_NAME}.c${CONCURRENCY}.csv" \
  --output "kind=plot,path=${OUTPUT_DIR}/${SERVICE_NAME}.tp${TP}.${TEST_NAME}.c${CONCURRENCY}.png" \
  --seed "kind=static,value=0"
