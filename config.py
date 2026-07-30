"""Shared configuration for the A2AJ RAG demo."""
import os

# --- Data source (A2AJ / HuggingFace) ---
HF_DATASET_REPO = "a2aj/canadian-case-law"
TRIBUNAL_CODES = ["RAD", "RPD", "RLLR", "FC"]

# FC ("Federal Court") is not immigration-only, so rows from it are kept only
# if they match one of these keywords (checked against name_en + citation_en).
FC_IMMIGRATION_KEYWORDS = [
    "immigration", "refugee", "irpa", "irb", "prra", "removal order",
    "permanent resident", "deportation", "inadmissib", "humanitarian and compassionate",
    "h&c", "visa officer", "citizenship", "asylum",
]

TARGET_CORPUS_SIZE = 1000  # number of *decisions* sampled before chunking
RANDOM_SEED = 42

# --- Chunking ---
CHUNK_WORDS = 350
CHUNK_OVERLAP_WORDS = 60

# --- Embedding model ---
EMBED_MODEL_NAME = "BAAI/bge-large-en"
EMBED_BATCH_SIZE = 32
# bge-large-en convention: prefix queries (not passages) with this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index_store")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNK_META_PATH = os.path.join(INDEX_DIR, "chunk_meta.jsonl")

EVAL_QA_PATH = os.path.join(EVAL_DIR, "eval_qa.jsonl")
EVAL_RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results.jsonl")

# --- Ollama ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_EVAL_GEN_MODEL = os.environ.get("OLLAMA_EVAL_MODEL", DEFAULT_OLLAMA_MODEL)

# --- Retrieval / chatbot defaults ---
TOP_K = 5

# --- Hybrid retrieval (dense FAISS + sparse BM25, fused via Reciprocal Rank
# Fusion) ---
HYBRID_CANDIDATE_K = 30  # candidates pulled from each retriever before fusion
RRF_K = 60               # RRF damping constant (standard default)
