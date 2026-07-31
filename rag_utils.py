"""Shared building blocks: embedding model, hybrid retriever, Ollama client, prompting."""
import json
import re

import numpy as np
import requests

import config

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


class Embedder:
    """Wraps BAAI/bge-large-en. Passages get no prefix; queries get the bge
    instruction prefix. All vectors are L2-normalized so that FAISS inner
    product search is equivalent to cosine similarity."""

    def __init__(self, model_name: str = config.EMBED_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode_passages(self, texts, batch_size=config.EMBED_BATCH_SIZE, show_progress_bar=True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_query(self, text):
        prefixed = config.BGE_QUERY_PREFIX + text
        vec = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return vec[0]


class HybridRetriever:
    """Dense (FAISS IndexFlatIP over normalized bge embeddings) + sparse
    (BM25) retrieval, fused with Reciprocal Rank Fusion. BM25 catches exact
    legal terms (citations, section numbers, party names) that dense
    embeddings tend to under-weight; RRF avoids having to reconcile the two
    retrievers' incompatible score scales."""

    def __init__(self, index_path=config.FAISS_INDEX_PATH, meta_path=config.CHUNK_META_PATH):
        import faiss
        from rank_bm25 import BM25Okapi

        self.index = faiss.read_index(index_path)
        self.chunks = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/metadata size mismatch: index has {self.index.ntotal} vectors, "
                f"metadata has {len(self.chunks)} rows. Rebuild the index."
            )

        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def _chunk_result(self, idx: int, score: float):
        chunk = self.chunks[idx]
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "tribunal": chunk["tribunal"],
            "text": chunk["text"],
            "score": score,
        }

    def search(self, query_vec: np.ndarray, query_text: str, k: int):
        n_candidates = max(config.HYBRID_CANDIDATE_K, k)

        _, ids = self.index.search(query_vec.reshape(1, -1), n_candidates)
        dense_ranked = [int(idx) for idx in ids[0] if idx != -1]

        bm25_scores = self.bm25.get_scores(_tokenize(query_text))
        sparse_ranked = list(np.argsort(bm25_scores)[::-1][:n_candidates])

        rrf_scores = {}
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        for rank, idx in enumerate(sparse_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)

        top_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]
        return [self._chunk_result(idx, rrf_scores[idx]) for idx in top_ids]


class OllamaClient:
    """Thin wrapper around the local Ollama REST API. Works with any model
    already pulled locally (`ollama list` to see what's available)."""

    def __init__(self, host: str = config.OLLAMA_HOST, model: str = config.DEFAULT_OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    def chat(self, messages, stream=False, temperature=0.2):
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if not stream:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        resp = requests.post(url, json=payload, timeout=300, stream=True)
        if not resp.ok:
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} error from {url}: {resp.text}", response=resp
            )
        full = []
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                full.append(piece)
                print(piece, end="", flush=True)
            if data.get("done"):
                break
        print()
        return "".join(full)

    def chat_stream(self, messages, temperature=0.2):
        """Like chat(stream=True), but yields text pieces instead of printing
        -- for callers (e.g. a web UI) that need to render the stream themselves."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        resp = requests.post(url, json=payload, timeout=300, stream=True)
        if not resp.ok:
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} error from {url}: {resp.text}", response=resp
            )
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break

    def list_models(self):
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


RAG_SYSTEM_PROMPT = (
    "You are a legal research assistant answering questions about Canadian "
    "immigration and refugee tribunal decisions. Every user message includes "
    "numbered context excerpts retrieved for that message -- answer ONLY using "
    "those excerpts and cite the excerpt number(s) you used in square brackets, "
    "e.g. [1]. If the excerpts do not contain the answer, say you don't have "
    "enough information — do not guess."
)


def build_system_message():
    return {"role": "system", "content": RAG_SYSTEM_PROMPT}


def build_user_turn(question: str, retrieved_chunks: list):
    context_blocks = [
        f"[{i}] ({c['tribunal']} — {c['title']})\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)
    content = f"Context excerpts:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return {"role": "user", "content": content}


def build_rag_messages(question: str, retrieved_chunks: list):
    """Single-turn convenience wrapper (used by the eval scripts, which have
    no conversation history -- every question is its own fresh exchange)."""
    return [build_system_message(), build_user_turn(question, retrieved_chunks)]
