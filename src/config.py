# src/config.py
"""
Central configuration — single source of truth for model names and constants.
Change the model name here and it updates everywhere automatically.
"""

# ── Gemini model ──────────────────────────────────────────────────────────────
# The model chain lives in ONE place: src/generation/gemini_client.py
# (MODEL_FALLBACK_CHAIN). It used to be repeated here as "gemini-2.0-flash",
# which nothing read and which no longer matched what the app calls.

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

# ── Confidence (ONE scale: probability, 0.0 - 1.0) ────────────────────────────
# The cross-encoder outputs a raw logit. src/retrieval/scoring.py turns it into a
# probability with a sigmoid, and every threshold below is compared on that
# scale. These are the only definitions; do not redefine them elsewhere.
#
#   confidence  = sigmoid(top reranker logit)
#   < CONFIDENCE_FLOOR     -> answer is prefixed "I'm not fully confident ..."
#   < NO_MATCH_CONFIDENCE  -> nothing in the manual matches: exact fallback, no
#                             Gemini call (calibrated in ADR-004: off-topic
#                             questions score logit -6 = 0.002, every answerable
#                             one scores logit +3.6 = 0.97 or more)
#   < GROUP_MIN_CONFIDENCE -> a non-best scooter group contributes no chunks
CONFIDENCE_FLOOR       = 0.4
NO_MATCH_CONFIDENCE    = 0.05
GROUP_MIN_CONFIDENCE   = 0.5

# ── Memory ────────────────────────────────────────────────────────────────────
MAX_TURNS = 3

# ── Ingestion ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 50
BM25_CORPUS_FILE = "data/chunks/bm25_corpus.pkl"