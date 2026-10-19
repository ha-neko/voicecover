#!/usr/bin/env bash
set -euo pipefail

LLAMA_ROOT="${VOICECOVER_LLAMA_ROOT:-$HOME/.local/src/llama.cpp}"
MODEL_DIR="${VOICECOVER_LLM_MODEL_DIR:-$HOME/.local/share/voicecover/models}"
MODEL="$MODEL_DIR/Qwen3.5-4B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
MODEL_SHA256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"

mkdir -p "$(dirname "$LLAMA_ROOT")" "$MODEL_DIR"
if [[ ! -d "$LLAMA_ROOT/.git" ]]; then
  git clone --depth 1 --branch v0.4.1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_ROOT"
fi

cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" \
  -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_ROOT/build" --config Release \
  --parallel "${VOICECOVER_BUILD_JOBS:-16}" --target llama-server

curl --http1.1 -L --fail --retry 10 --retry-all-errors --continue-at - \
  --output "$MODEL" "$MODEL_URL"
printf '%s  %s\n' "$MODEL_SHA256" "$MODEL" | sha256sum --check --status
printf 'ready. start it with scripts/llm-server.sh\n'
