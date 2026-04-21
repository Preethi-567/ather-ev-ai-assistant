# src/pipeline/ingestion.py
"""
Orchestrate: V2 Chunks → ChromaDB (dense) + BM25 corpus (keyword)

V2 Chunker changes handled here:
  - keywords (list)        → stored as pipe-joined string in ChromaDB metadata
  - component_names (list) → stored as pipe-joined string in ChromaDB metadata
  - components (list[dict])→ stored as JSON string in ChromaDB metadata
  - summary (str)          → prepended to document text for richer embedding
"""

import json
import pickle
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL      = "BAAI/bge-large-en-v1.5"
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME  = "ather_manuals"
BATCH_SIZE       = 50
BM25_CORPUS_FILE = "data/chunks/bm25_corpus.pkl"


def load_all_chunks(chunks_dir: str) -> list:
    """Load all *_chunks.json files from the chunks directory."""
    chunks_dir = Path(chunks_dir)
    all_chunks = []
    for chunk_file in sorted(chunks_dir.glob("*_chunks.json")):
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        logger.info(f"Loaded {len(chunks)} chunks from {chunk_file.name}")
    return all_chunks


def setup_chroma_collection():
    """Create or reconnect to the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection


def serialize_metadata_for_chroma(chunk: dict) -> dict:
    """
    ChromaDB only accepts: str, int, float, bool.
    V2 chunks add list fields that must be converted.

    Serialization map:
      keywords        (list[str])  → pipe-joined string  "charging|battery|volt"
      component_names (list[str])  → pipe-joined string  "Headlamp|Rear mirror"
      components      (list[dict]) → JSON string          '[{"component_id":1,...}]'
      summary         (str)        → kept as-is (already str)
      All other metadata fields   → kept as-is (already safe types)
    """
    meta = {}

    for k, v in chunk["metadata"].items():
        if v is None:
            meta[k] = ""
        elif isinstance(v, bool):
            meta[k] = v
        elif isinstance(v, (int, float)):
            meta[k] = v
        else:
            meta[k] = str(v)

    # ── V2 new fields from top-level chunk (not inside metadata yet) ─────────
    # keywords: list[str] → pipe-separated so ChromaDB can store + filter later
    keywords = chunk.get("keywords", [])
    meta["keywords"] = "|".join(keywords) if keywords else ""

    # component_names: list[str] → pipe-separated for fast string matching
    component_names = chunk.get("component_names", [])
    meta["component_names"] = "|".join(component_names) if component_names else ""

    # components: list[dict] → JSON string (full structured data, for retrieval)
    components = chunk.get("components", [])
    meta["components_json"] = json.dumps(components) if components else "[]"

    # summary: first 200 chars — store so we can return it in search results
    meta["summary"] = chunk.get("summary", "")[:500]

    # has_components flag — comes from metadata in V2 chunker
    meta["has_components"] = bool(components)

    return meta


def build_embedding_document(chunk: dict) -> str:
    """
    Build the text we actually embed.

    Why augment with keywords?
    - BGE embeddings work on semantic meaning, but exact terms like
      'IP67', 'Gen 4', 'BLDC motor' have sparse semantic signal.
    - Appending extracted keywords makes the embedding capture both
      semantic AND lexical signal from the V2 chunker's keyword extraction.
    - summary prepended gives the model a "topic sentence" to anchor on.

    Format:
      [summary]. [full text]. [keywords joined as sentence]
    """
    parts = []

    summary = chunk.get("summary", "").strip()
    if summary:
        parts.append(summary)

    parts.append(chunk["text"].strip())

    keywords = chunk.get("keywords", [])
    if keywords:
        # Present as a natural phrase so the embedding model doesn't treat
        # this as noise — "Keywords: charging battery voltage" reads better
        # than a raw comma list
        parts.append("Keywords: " + ", ".join(keywords))

    component_names = chunk.get("component_names", [])
    if component_names:
        parts.append("Components: " + ", ".join(component_names))

    return " ".join(parts)


def ingest_to_chromadb(chunks: list, collection) -> None:
    """Embed all V2 chunks and upsert into ChromaDB."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    logger.info(f"Embedding and ingesting {len(chunks)} chunks...")

    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Ingesting"):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        ids        = [c["chunk_id"] for c in batch]
        # Use augmented document (summary + text + keywords + components)
        documents  = [build_embedding_document(c) for c in batch]
        metadatas  = [serialize_metadata_for_chroma(c) for c in batch]

        embeddings = embedder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    logger.info(f"ChromaDB collection count: {collection.count()}")


def save_bm25_corpus(chunks: list) -> None:
    """
    Save the BM25 corpus.

    V2 upgrade: BM25 indexed text = chunk text + keywords.
    This means a query for 'IP67' will hit chunks where Gemini
    Vision extracted 'IP67' as a keyword, even if it appears
    only once in the raw text.
    """
    bm25_texts = []
    for c in chunks:
        base_text  = c["text"]
        keywords   = c.get("keywords", [])
        comp_names = c.get("component_names", [])

        # Augment with keywords and component names for richer BM25 matching
        augmented = base_text
        if keywords:
            augmented += " " + " ".join(keywords)
        if comp_names:
            augmented += " " + " ".join(comp_names)

        bm25_texts.append(augmented)

    corpus_data = {
        "texts":      bm25_texts,           # augmented text for BM25 tokenisation
        "raw_texts":  [c["text"] for c in chunks],   # original text for display
        "chunk_ids":  [c["chunk_id"] for c in chunks],
        "metadatas":  [c["metadata"] for c in chunks],
        # V2 extras stored for retrieval layer use
        "summaries":       [c.get("summary", "") for c in chunks],
        "keywords_list":   [c.get("keywords", []) for c in chunks],
        "component_names": [c.get("component_names", []) for c in chunks],
        "components":      [c.get("components", []) for c in chunks],
    }

    Path(BM25_CORPUS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_FILE, "wb") as f:
        pickle.dump(corpus_data, f)

    logger.info(f"BM25 corpus saved: {len(chunks)} docs → {BM25_CORPUS_FILE}")


def run_ingestion(chunks_dir: str = "data/chunks") -> dict:
    """Full ingestion pipeline entry point."""
    chunks = load_all_chunks(chunks_dir)

    if not chunks:
        logger.error(f"No chunks found in {chunks_dir}")
        return {"error": "No chunks found"}

    logger.info(f"Total chunks: {len(chunks)}")

    # Log V2 field presence so you can confirm they were picked up
    sample = chunks[0]
    logger.info(
        f"V2 field check on first chunk — "
        f"keywords: {bool(sample.get('keywords'))}, "
        f"summary: {bool(sample.get('summary'))}, "
        f"components: {bool(sample.get('components'))}, "
        f"component_names: {bool(sample.get('component_names'))}"
    )

    client, collection = setup_chroma_collection()

    existing = collection.count()
    if existing > 0:
        logger.warning(f"Collection already has {existing} docs — upserting")

    ingest_to_chromadb(chunks, collection)
    save_bm25_corpus(chunks)

    return {
        "total_chunks":    len(chunks),
        "chroma_count":    collection.count(),
        "bm25_corpus":     BM25_CORPUS_FILE,
        "collection_name": COLLECTION_NAME,
    }


if __name__ == "__main__":
    summary = run_ingestion("data/chunks")
    print("\n✅ Ingestion complete:")
    print(json.dumps(summary, indent=2))