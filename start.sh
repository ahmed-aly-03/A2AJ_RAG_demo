#!/bin/bash
# Container entrypoint:
#   1. build the FAISS index (data prep + GPU embedding) if it's not there yet
#   2. bring up a local Ollama server (if OLLAMA_HOST points at this container)
#   3. pull the standard model lineup
#   4. start the Gradio chat UI in the background
#   5. stay alive so RunPod/docker exec can attach
set -uo pipefail

MODELS=(
  "llama3.2:3b"
  "llama3.1:8b"
  "phi4:14b"
  "qwen2.5:32b"
)

cd /app

if [[ -f index_store/faiss.index && -f index_store/chunk_meta.jsonl ]]; then
    echo "[start.sh] FAISS index already present, skipping data prep/build."
else
    if [[ ! -f data/corpus.jsonl ]]; then
        echo "[start.sh] Preparing A2AJ corpus (download + filter + chunk) ..."
        python3 data/prepare_data.py
    fi
    echo "[start.sh] Building FAISS index (embedding on GPU if available) ..."
    python3 -c "import torch; print('[start.sh] CUDA available:', torch.cuda.is_available())"
    python3 index_store/build_index.py
    echo "[start.sh] FAISS index ready."
fi

if [[ "$OLLAMA_HOST" == *"localhost"* || "$OLLAMA_HOST" == *"127.0.0.1"* ]]; then
    echo "[start.sh] Starting local Ollama server ($OLLAMA_HOST) ..."
    nohup ollama serve > /var/log/ollama.log 2>&1 &

    echo "[start.sh] Waiting for Ollama server to come up ..."
    until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
        sleep 1
    done
    echo "[start.sh] Ollama server is up."

    for model in "${MODELS[@]}"; do
        echo "[start.sh] Pulling $model ..."
        ollama pull "$model" || echo "[start.sh] WARNING: failed to pull $model, continuing"
    done
    echo "[start.sh] Model pulls done."
else
    echo "[start.sh] OLLAMA_HOST=$OLLAMA_HOST points elsewhere -- skipping local Ollama bootstrap."
fi

echo "[start.sh] Starting Gradio chat UI on :7860 ..."
nohup python3 chatbot_ui.py > /var/log/chatbot_ui.log 2>&1 &

echo "[start.sh] Ready. Container staying alive for exec/SSH access."
exec sleep infinity
