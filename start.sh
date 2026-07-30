#!/bin/bash
# Container entrypoint:
#   1. build the FAISS index (data prep + GPU embedding) if it's not there yet
#   2. bring up a local Ollama server (if OLLAMA_HOST points at this container)
#   3. pull the default model (qwen2.5:32b) and launch the Gradio chat UI with
#      it right away -- don't make the demo wait on the other 3 models
#   4. pull the remaining models in the background for later comparison
#   5. stay alive so RunPod/docker exec can attach
set -uo pipefail

DEFAULT_MODEL="qwen2.5:32b"
OTHER_MODELS=(
  "llama3.2:3b"
  "llama3.1:8b"
  "phi4:14b"
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

    echo "[start.sh] Pulling default model $DEFAULT_MODEL ..."
    ollama pull "$DEFAULT_MODEL" || echo "[start.sh] WARNING: failed to pull $DEFAULT_MODEL"

    echo "[start.sh] Pulling remaining models in the background (llama3.2:3b, llama3.1:8b, phi4:14b) ..."
    (
        for model in "${OTHER_MODELS[@]}"; do
            echo "[start.sh] Pulling $model ..."
            ollama pull "$model" || echo "[start.sh] WARNING: failed to pull $model, continuing"
        done
        echo "[start.sh] Remaining model pulls done."
    ) &
else
    echo "[start.sh] OLLAMA_HOST=$OLLAMA_HOST points elsewhere -- skipping local Ollama bootstrap."
fi

echo "[start.sh] Starting Gradio chat UI on :7860 with $DEFAULT_MODEL ..."
nohup python3 chatbot_ui.py --model "$DEFAULT_MODEL" > /var/log/chatbot_ui.log 2>&1 &

echo "[start.sh] Ready ($DEFAULT_MODEL live now, other models pulling in background if applicable). Container staying alive for exec/SSH access."
exec sleep infinity
