"""Browser UI for the A2AJ RAG demo, built with Gradio's ChatInterface.

Same retrieval + generation pipeline as chatbot.py, just with a web front end
instead of a terminal loop. Any pulled Ollama model can be picked from the
dropdown. Retrieval runs on every turn; the model also sees the full
conversation so far, so follow-ups can build on earlier answers.

Usage:
    python3 chatbot_ui.py
"""
import argparse

import gradio as gr

import config
from rag_utils import Embedder, HybridRetriever, OllamaClient, build_system_message, build_user_turn

_store = None
_embedder = None


def get_default_models():
    probe = OllamaClient(host=config.OLLAMA_HOST, model=config.DEFAULT_OLLAMA_MODEL)
    models = probe.list_models()
    return models or [
        "llama3.2:3b",
        "llama3.1:8b",
        "phi4:14b",
        "qwen2.5:32b",
    ]


def respond(message, history, model_name, k):
    llm = OllamaClient(host=config.OLLAMA_HOST, model=model_name)

    message = message.strip()

    # Gradio's ChatInterface (type="messages") hands history as a flat list
    # of {"role": ..., "content": ...} dicts -- re-emit just those two keys
    # in case a given gradio version tucks extra metadata into the dict.
    conversation = [build_system_message()]
    for turn in history:
        conversation.append({"role": turn["role"], "content": turn["content"]})

    query_vec = _embedder.encode_query(message)
    retrieved = _store.search(query_vec, message, int(k))
    if not retrieved:
        yield "No relevant material found in the corpus."
        return

    conversation.append(build_user_turn(message, retrieved))

    partial = ""
    for piece in llm.chat_stream(conversation):
        partial += piece
        yield partial

    sources = "\n\n**Sources:**\n" + "\n".join(
        f"[{i}] {c['tribunal']} — {c['title']} (score={c['score']:.3f})"
        for i, c in enumerate(retrieved, start=1)
    )
    yield partial + sources


def main():
    global _store, _embedder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind the web server to")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default=None,
                         help="Ollama model preselected in the dropdown (default: first available). "
                              "Run separate instances on different --port values to compare models "
                              "side by side in different browser tabs.")
    args = parser.parse_args()

    print(f"Loading FAISS + BM25 hybrid index from {config.FAISS_INDEX_PATH} ...")
    _store = HybridRetriever()
    print(f"Loading embedding model {config.EMBED_MODEL_NAME} ...")
    _embedder = Embedder()

    models = get_default_models()
    if args.model and args.model not in models:
        models = [args.model] + models
    default_model = args.model or models[0]

    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        title=f"A2AJ Immigration RAG Demo ({default_model})",
        description=(
            f"Hybrid (FAISS + BM25) retrieval over {_store.index.ntotal} chunks, "
            "answered by any local Ollama model."
        ),
        additional_inputs=[
            gr.Dropdown(choices=models, value=default_model, label="Ollama model"),
            gr.Slider(minimum=1, maximum=10, value=config.TOP_K, step=1, label="Top-k retrieved chunks"),
        ],
    )

    demo.queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
