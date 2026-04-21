# src/config.py
"""
Central configuration — single source of truth for model names and constants.
Change the model name here and it updates everywhere automatically.
"""

# ── Gemini model ──────────────────────────────────────────────────────────────
# gemini-2.0-flash  = current free tier (replaces deprecated gemini-1.5-flash)
# Free tier limits  = 15 RPM, 1M tokens/day, 1500 requests/day
GEMINI_MODEL = "gemini-2.0-flash"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# ── Reranker model ────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "ather_manuals"

# ── Retrieval ─────────────────────────────────────────────────────────────────
DENSE_TOP_K  = 10
BM25_TOP_K   = 10
FINAL_TOP_K  = 5
RRF_K        = 60
CONFIDENCE_FLOOR = 0.4

# ── Memory ────────────────────────────────────────────────────────────────────
MAX_TURNS = 3

# ── Ingestion ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 50
BM25_CORPUS_FILE = "data/chunks/bm25_corpus.pkl"