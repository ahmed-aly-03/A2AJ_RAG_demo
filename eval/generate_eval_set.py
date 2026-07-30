"""Auto-generate a retrieval eval set: sample chunks from the indexed corpus
and ask a local Ollama model to write one question answerable ONLY from that
chunk. The source chunk becomes the ground-truth relevant document for that
question, which is what evaluate.py checks retrieval against.

Usage:
    python eval/generate_eval_set.py --n 150 --model llama3.2:3b
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rag_utils import OllamaClient

GEN_PROMPT = """You are creating a retrieval-evaluation question from a legal excerpt.

Excerpt (from a Canadian immigration/refugee tribunal decision, "{title}"):
\"\"\"
{text}
\"\"\"

Write ONE specific, self-contained factual question that:
- can be answered using ONLY this excerpt
- does NOT require knowing which case/document it came from (don't ask "what did the tribunal decide in this case" style vague questions)
- has a short, specific answer (a fact, name, date, outcome, or reason found in the excerpt)

Respond with ONLY a JSON object, no other text, in this exact form:
{{"question": "...", "answer": "..."}}
"""


def extract_json(raw: str):
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "question" not in obj or "answer" not in obj:
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150, help="Number of eval questions to generate")
    parser.add_argument("--model", default=config.DEFAULT_EVAL_GEN_MODEL,
                         help="Ollama model used to write questions (default: %(default)s)")
    parser.add_argument("--host", default=config.OLLAMA_HOST)
    args = parser.parse_args()

    random.seed(config.RANDOM_SEED)

    if not os.path.exists(config.CHUNK_META_PATH):
        raise SystemExit(f"{config.CHUNK_META_PATH} not found — run index_store/build_index.py first.")

    chunks = []
    with open(config.CHUNK_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    sample_size = min(args.n, len(chunks))
    sampled = random.sample(chunks, sample_size)

    llm = OllamaClient(host=args.host, model=args.model)
    os.makedirs(config.EVAL_DIR, exist_ok=True)

    written = 0
    skipped = 0
    with open(config.EVAL_QA_PATH, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(sampled, start=1):
            prompt = GEN_PROMPT.format(title=chunk["title"], text=chunk["text"][:2500])
            try:
                raw = llm.chat([{"role": "user", "content": prompt}], stream=False, temperature=0.3)
            except Exception as e:
                print(f"  [{i}/{sample_size}] generation error: {e}")
                skipped += 1
                continue

            parsed = extract_json(raw)
            if not parsed or not parsed["question"].strip() or not parsed["answer"].strip():
                skipped += 1
                continue

            record = {
                "question": parsed["question"].strip(),
                "expected_answer": parsed["answer"].strip(),
                "source_chunk_id": chunk["chunk_id"],
                "source_doc_id": chunk["doc_id"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if i % 10 == 0:
                print(f"  [{i}/{sample_size}] generated={written} skipped={skipped}")

    print(f"\nWrote {written} eval questions to {config.EVAL_QA_PATH} ({skipped} skipped/failed)")


if __name__ == "__main__":
    main()
