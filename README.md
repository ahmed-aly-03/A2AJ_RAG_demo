# A2AJ Immigration RAG Demo

A small, self-contained test bench for comparing a local FAISS+LLM RAG stack
against MCP-based retrieval, using the [A2AJ](https://a2aj.ca/data/) Canadian
case-law dataset filtered to immigration/refugee decisions.

Pipeline: **A2AJ (HuggingFace parquet) → chunk → embed with
[BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) → hybrid
retrieval (FAISS `IndexFlatIP` dense search + BM25 sparse search, fused with
Reciprocal Rank Fusion) → any local Ollama model for generation.**

## Corpus

Subsets pulled from `a2aj/canadian-case-law`: `RAD`, `RPD`, `RLLR`, and `FC`
(Federal Court rows are kept only if they match immigration-related keywords
— see `FC_IMMIGRATION_KEYWORDS` in [config.py](config.py), since raw FC
covers all Federal Court matters, not just immigration). Deduplicated by
citation, then sampled down to `TARGET_CORPUS_SIZE` decisions (default 1000)
and chunked into ~350-word passages with 60-word overlap.

All of this is configurable in [config.py](config.py).

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure Ollama is running locally (`ollama serve`) and you've pulled at
least one model, e.g. `ollama pull llama3.2:3b`.

## Usage

Run these in order:

```bash
# 1. Download + filter + chunk the A2AJ corpus -> data/corpus.jsonl
python data/prepare_data.py

# 2. Embed chunks with bge-large-en and build the FAISS index
python index_store/build_index.py

# 3. Chat with it -- terminal (swap --model for any model in `ollama list`)
python chatbot.py --model llama3.2:3b --k 5

# ...or the browser UI instead, at http://localhost:7860
python chatbot_ui.py

# 4. Auto-generate a retrieval eval set (question, answer, source chunk)
#    using a local Ollama model as the question-writer
python eval/generate_eval_set.py --n 150 --model llama3.2:3b

# 5. Run the evaluation harness
python eval/evaluate.py --model llama3.2:3b --k 5
```

## Metrics reported by `eval/evaluate.py`

Each synthetic eval question is generated from one specific chunk, so that
chunk is the single ground-truth "relevant document" for retrieval purposes:

- **Accuracy** = Recall@1 — top-1 retrieval exact-match rate
- **Recall@k** — fraction of questions where the source chunk appears in the
  top-k retrieved
- **Precision@k** — 1/k when the (single) relevant chunk is retrieved, else 0
- **F1@k** — harmonic mean of Precision@k and Recall@k
- **MRR** — mean reciprocal rank of the source chunk
- **Answer token-F1** — SQuAD-style token overlap between the model's full
  generated answer and the expected answer. This is a coarse proxy for answer
  quality (not an LLM judge) — always reported separately from the retrieval
  numbers, and per-question detail is written to `eval/eval_results.jsonl` for
  manual inspection.

Swap `--model` on `chatbot.py` / `evaluate.py` to compare different local
Ollama models against the same fixed retrieval index and eval set.

## Retrieval: hybrid (FAISS + BM25)

`HybridRetriever` in [rag_utils.py](rag_utils.py) runs both a dense FAISS
search and a BM25 sparse search over the same chunks, then fuses the two
rankings with Reciprocal Rank Fusion (`RRF_K` / `HYBRID_CANDIDATE_K` in
[config.py](config.py)). BM25 catches exact legal terms — citations, section
numbers, party names — that dense embeddings tend to under-weight; RRF
combines the two without having to reconcile their very different score
scales. No separate build step is needed: BM25 is built in-memory from
`chunk_meta.jsonl` each time a script starts (a few seconds for this corpus
size).

## Chatbot conversation behavior

Both `chatbot.py` and `chatbot_ui.py` hold a real multi-turn conversation —
retrieval runs fresh on every turn, and the model also sees the full
conversation so far, so follow-ups like "what about the second prong?" build
on earlier answers rather than starting from a blank slate each time.

## Project layout

```
config.py                    # all tunables (dataset subset, chunk size, model names, paths)
rag_utils.py                 # Embedder, HybridRetriever (FAISS+BM25), OllamaClient, prompt builder
data/prepare_data.py         # A2AJ download + filter + chunk -> corpus.jsonl
index_store/build_index.py   # embed + FAISS IndexFlatIP -> faiss.index + chunk_meta.jsonl
chatbot.py                   # terminal RAG chatbot
chatbot_ui.py                # browser RAG chatbot (Gradio ChatInterface, port 7860)
eval/generate_eval_set.py    # synthetic (question, answer, source_chunk) generation
eval/evaluate.py             # retrieval + answer-quality metrics
start.sh                     # container entrypoint: builds index, starts Ollama, pulls models, launches the UI
```

## Docker / RunPod

The image (`Dockerfile`) bakes in the Ollama binary, all Python deps, and the
already-prepared `data/corpus.jsonl` (deterministic, no GPU needed to produce —
see `data/prepare_data.py`), targeting `linux/amd64` explicitly (RunPod GPU
pods, regardless of the host you build on).

`start.sh` is the container's `CMD` and does everything needed before you can
just test/evaluate:

1. **Builds the FAISS index** (`index_store/build_index.py`) if it isn't
   already there — this is the one step that genuinely needs the pod's GPU
   (embedding ~11k chunks with bge-large-en), so it happens at container
   start rather than at `docker build` time. Skipped automatically on
   restarts once the index exists.
2. Starts `ollama serve` and waits until it actually responds (not just
   "installed").
3. Pulls **qwen2.5:32b** first (the default demo model, ~20GB VRAM) and
   immediately launches `chatbot_ui.py` (Gradio) with it on port **7860** --
   a model dropdown (populated from `ollama list`) and a top-k slider let
   you drive the whole thing from a browser instead of the terminal. The
   demo doesn't wait on the rest of the lineup to become usable.
4. Pulls the remaining models in the background, so they show up in the
   dropdown once ready without blocking startup:

   ```
   llama3.2:3b   (small,  ~3GB VRAM)
   llama3.1:8b   (medium, ~6GB VRAM)
   phi4:14b      (large,  ~10GB VRAM)
   ```

5. Stays alive (`sleep infinity`) so RunPod/`docker exec` can attach.

Expect the first minute or two of pod uptime to be spent embedding the
corpus and pulling qwen2.5:32b before the UI is actually reachable — check
`docker logs` (or just watch the attached terminal) for `[start.sh] Ready.`
The other 3 models keep downloading afterward in the background (~13GB
more). Model weights are pulled fresh on every container start unless you
mount a RunPod persistent volume at `/root/.ollama`.

To reach the UI from outside the container, expose port 7860 when you create
the pod (RunPod's "Expose HTTP Ports" field), then open the pod's proxy URL,
e.g. `https://<pod-id>-7860.proxy.runpod.net`. Locally, `docker run -p
7860:7860 ...` and open `http://localhost:7860`.

Build + push:

```bash
docker build -t alyahmed1/a2aj-rag-demo:latest .
docker push alyahmed1/a2aj-rag-demo:latest
```

Once you're in the pod's terminal and `start.sh` has finished (index built,
models pulled, UI launched), all that's left is testing and evaluation --
either through the browser UI above, or the CLI:

```bash
export OLLAMA_HOST=http://localhost:11434
cd /app
python3 chatbot.py --model llama3.2:3b --host http://localhost:11434
python3 eval/generate_eval_set.py --n 150 --model llama3.2:3b --host http://localhost:11434
python3 eval/evaluate.py --model llama3.2:3b --k 5 --host http://localhost:11434
```
