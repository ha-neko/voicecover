#!/usr/bin/env bash
set -euo pipefail

LLAMA_ROOT="${VOICECOVER_LLAMA_ROOT:-$HOME/.local/src/llama.cpp}"
MODEL="${VOICECOVER_LLM_MODEL_PATH:-$HOME/.local/share/voicecover/models/Qwen3.5-4B-Q4_K_M.gguf}"
SERVER="${VOICECOVER_LLAMA_SERVER:-$LLAMA_ROOT/build/bin/llama-server}"

if [[ ! -x "$SERVER" ]]; then
  printf 'llama-server is missing: %s\nrun scripts/setup-llm.sh first.\n' "$SERVER" >&2
  exit 1
fi
if [[ ! -s "$MODEL" ]]; then
  printf 'the translation model is missing: %s\nrun scripts/setup-llm.sh first.\n' "$MODEL" >&2
  exit 1
fi

exec "$SERVER" \
  --model "$MODEL" \
  --host "${VOICECOVER_LLM_HOST:-127.0.0.1}" \
  --port "${VOICECOVER_LLM_PORT:-8081}" \
  --threads "${VOICECOVER_LLM_THREADS:-16}" \
  --ctx-size "${VOICECOVER_LLM_CONTEXT:-16384}" \
  --batch-size 1024 \
  --ubatch-size 256 \
  --parallel 1 \
  --jinja \
  --reasoning off \
  --reasoning-format none
