FROM --platform=linux/amd64 nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl zstd pciutils \
    && rm -rf /var/lib/apt/lists/*

# Bake the Ollama binary into the image at build time. Models are NOT
# pulled here (multi-GB each) -- start.sh pulls them at container start.
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
# Pin torch to a CUDA 12.1 build explicitly -- letting pip grab the newest
# default torch wheel bundles a CUDA 13.x runtime that many RunPod hosts'
# NVIDIA drivers are too old for, silently falling back to CPU embedding.
# CUDA 12.1 only needs driver >=525, which is broadly compatible.
RUN pip3 install --upgrade pip \
    && pip3 install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install -r requirements.txt

COPY . .

# Default assumes Ollama runs inside this same container (RunPod single-
# container pods). The docker-compose sidecar setup overrides this to
# http://ollama:11434, in which case start.sh skips its local bootstrap.
ENV OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_MODEL=llama3.2:3b

# Gradio chat UI (chatbot_ui.py). start.sh auto-launches one instance on
# 7860 with the default model; 7861-7863 are free for extra --port instances
# if you want to run multiple models side by side (see README).
EXPOSE 7860-7863

CMD ["bash", "/app/start.sh"]
