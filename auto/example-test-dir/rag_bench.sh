#!/bin/bash
# RAG benchmark test config for guidellm autotest
# Environment variables injected by workflow.py:
#   PORT        - server port
#   MODEL       - model path
#   CONCURRENCY - current concurrency level
#   COUNT       - number of requests (CONCURRENCY * 10)

HOST=0.0.0.0
PORT=${PORT:-8976}
MODEL=${MODEL:-/models/Qwen/Qwen3.5-9B}
CONCURRENCY=${CONCURRENCY:-1}
COUNT=${COUNT:-$((CONCURRENCY * 10))}

guidellm run \
  --backend "kind=openai_http,target=http://${HOST}:${PORT}" \
  --profile "kind=throughput" \
  --constraint "kind=max_requests,count=${COUNT},streams=${CONCURRENCY}" \
  --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"${MODEL}\",\"load_kwargs\":{\"use_fast\":false}}" \
  --data '{"kind":"huggingface","source":"neural-bridge/rag-dataset-12000","load_kwargs":{"split":"train"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"prefix_column":"context","text_column":"question"}}' \
  --seed "kind=static,value=0"
