"""Embed corpus.jsonl with BAAI/bge-large-en and build a FAISS IndexFlatIP.

Usage:
    python index_store/build_index.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rag_utils import Embedder


def main():
    if not os.path.exists(config.CORPUS_PATH):
        raise SystemExit(f"{config.CORPUS_PATH} not found — run data/prepare_data.py first.")

    os.makedirs(config.INDEX_DIR, exist_ok=True)

    chunks = []
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {config.CORPUS_PATH}")

    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model: {config.EMBED_MODEL_NAME} (first run downloads ~1.3GB)")
    embedder = Embedder()

    print("Embedding passages ...")
    vectors = embedder.encode_passages(texts)
    dim = vectors.shape[1]
    print(f"Embedded {vectors.shape[0]} chunks, dim={dim}")

    import faiss
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    print(f"Wrote FAISS index ({index.ntotal} vectors) to {config.FAISS_INDEX_PATH}")

    with open(config.CHUNK_META_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote chunk metadata to {config.CHUNK_META_PATH}")


if __name__ == "__main__":
    main()
