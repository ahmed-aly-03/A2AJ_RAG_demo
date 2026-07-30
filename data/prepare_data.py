"""Download a subset of the A2AJ Canadian case-law dataset, filter it down to
immigration/refugee material, chunk the decision text, and write corpus.jsonl.

Usage:
    python data/prepare_data.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def is_immigration_relevant(name_en: str, citation_en: str) -> bool:
    haystack = f"{name_en or ''} {citation_en or ''}".lower()
    return any(kw in haystack for kw in config.FC_IMMIGRATION_KEYWORDS)


def load_tribunal(code: str):
    from datasets import load_dataset
    print(f"  loading {code} ...")
    ds = load_dataset(config.HF_DATASET_REPO, data_dir=code, split="train")
    return ds


def chunk_text(text: str, doc_id: str, words=config.CHUNK_WORDS, overlap=config.CHUNK_OVERLAP_WORDS):
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    start = 0
    idx = 0
    step = max(words - overlap, 1)
    while start < len(tokens):
        piece = tokens[start:start + words]
        if len(piece) < 30:  # drop tiny tail fragments
            break
        chunks.append({
            "chunk_id": f"{doc_id}::{idx}",
            "doc_id": doc_id,
            "text": " ".join(piece),
        })
        idx += 1
        start += step
    return chunks


def main():
    random.seed(config.RANDOM_SEED)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    all_rows = []
    seen_citations = set()

    for code in config.TRIBUNAL_CODES:
        ds = load_tribunal(code)
        kept = 0
        for row in ds:
            citation = row.get("citation_en") or row.get("url_en") or row.get("name_en")
            if not citation or citation in seen_citations:
                continue
            if code == "FC" and not is_immigration_relevant(row.get("name_en"), row.get("citation_en")):
                continue
            text = row.get("unofficial_text_en") or ""
            if len(text.split()) < 50:
                continue
            seen_citations.add(citation)
            date_val = row.get("document_date_en")
            all_rows.append({
                "doc_id": citation,
                "title": row.get("name_en") or citation,
                "tribunal": code,
                "date": date_val.isoformat() if hasattr(date_val, "isoformat") else date_val,
                "text": text,
            })
            kept += 1
        print(f"  {code}: kept {kept} candidate decisions")

    print(f"Total candidate decisions after dedup/filtering: {len(all_rows)}")

    if len(all_rows) > config.TARGET_CORPUS_SIZE:
        # stratified-ish sample: shuffle then cap
        random.shuffle(all_rows)
        all_rows = all_rows[:config.TARGET_CORPUS_SIZE]
    print(f"Sampled down to {len(all_rows)} decisions (TARGET_CORPUS_SIZE={config.TARGET_CORPUS_SIZE})")

    n_chunks = 0
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            chunks = chunk_text(row["text"], row["doc_id"])
            for c in chunks:
                c["title"] = row["title"]
                c["tribunal"] = row["tribunal"]
                c["date"] = row["date"]
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {len(all_rows)} decisions to {config.CORPUS_PATH}")


if __name__ == "__main__":
    main()
