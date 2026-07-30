"""Terminal RAG chatbot over the A2AJ immigration corpus.

Retrieval: BAAI/bge-large-en dense embeddings + BM25, fused via Reciprocal
Rank Fusion, over a FAISS IndexFlatIP. Generation: any locally-pulled Ollama
model. Retrieval runs on every turn; the model also keeps the full
conversation so far, so follow-ups can build on earlier answers.

Usage:
    python chatbot.py --model llama3.2:3b
    python chatbot.py --model mistral:7b --k 8
"""
import argparse
import time

import config
from rag_utils import Embedder, HybridRetriever, OllamaClient, build_system_message, build_user_turn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.DEFAULT_OLLAMA_MODEL,
                         help="Any model name from `ollama list` (default: %(default)s)")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--host", default=config.OLLAMA_HOST, help="Ollama host URL")
    args = parser.parse_args()

    print(f"Loading FAISS + BM25 hybrid index from {config.FAISS_INDEX_PATH} ...")
    store = HybridRetriever()
    print(f"Loading embedding model {config.EMBED_MODEL_NAME} ...")
    embedder = Embedder()
    llm = OllamaClient(host=args.host, model=args.model)

    print(f"\nReady. Model={args.model}  top_k={args.k}  corpus_chunks={store.index.ntotal}")
    print("Ask a question about the A2AJ immigration/refugee decisions in the corpus.")
    print("Type 'exit' or 'quit' to leave.\n")

    conversation = [build_system_message()]

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        t_start = time.perf_counter()

        t0 = time.perf_counter()
        query_vec = embedder.encode_query(question)
        retrieved = store.search(query_vec, question, args.k)
        retrieval_time = time.perf_counter() - t0

        if not retrieved:
            print("Bot: No relevant material found in the corpus.\n")
            continue

        conversation.append(build_user_turn(question, retrieved))

        print("Bot: ", end="")
        t0 = time.perf_counter()
        answer = llm.chat(conversation, stream=True)
        generation_time = time.perf_counter() - t0
        conversation.append({"role": "assistant", "content": answer})

        total_time = time.perf_counter() - t_start

        print("\nSources:")
        for i, c in enumerate(retrieved, start=1):
            print(f"  [{i}] {c['tribunal']} — {c['title']} (score={c['score']:.3f})")
        print(
            f"\n[timing] retrieval={retrieval_time:.2f}s  "
            f"generation={generation_time:.2f}s  total={total_time:.2f}s"
        )
        print()


if __name__ == "__main__":
    main()
