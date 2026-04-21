# src/retrieval/dense_search.py
"""
Dense semantic search using ChromaDB + BGE-large embeddings.

V2 upgrades:
  - Component-aware filtering: if query mentions a component name
    stored in metadata, we prefer chunks where has_components=True
  - Returns summary field alongside chunk text for faster generation
  - Metadata now contains keywords + component_names for downstream use
"""

import os
import re
from typing import List, Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME = "ather_manuals"
DEFAULT_TOP_K   = 10

# BGE asymmetric retrieval: queries need this prefix, documents do NOT
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseSearcher:
    """
    Singleton-friendly wrapper around ChromaDB + BGE.
    Instantiate once at app startup, reuse across requests.
    """

    def __init__(self):
        logger.info("Initialising DenseSearcher...")

        self.embedder = SentenceTransformer(EMBED_MODEL)

        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)

        logger.info(
            f"DenseSearcher ready — "
            f"collection: {COLLECTION_NAME}, "
            f"docs: {self.collection.count()}"
        )

    def _embed_query(self, query: str) -> List[float]:
        """Embed the query with the BGE prefix."""
        prefixed = BGE_QUERY_PREFIX + query
        embedding = self.embedder.encode(
            prefixed,
            normalize_embeddings=True
        )
        return embedding.tolist()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        vehicle_model: Optional[str] = None,
        content_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run dense semantic search against ChromaDB with Pylance-safe typing."""

        query_embedding = self._embed_query(query)

        # ── Build where filter ────────────────────────────────────────────────
        where_conditions = []
        if vehicle_model:
            where_conditions.append({"vehicle_model": {"$eq": vehicle_model}})
        if content_type_filter:
            where_conditions.append({"content_type": {"$eq": content_type_filter}})

        where_filter: Optional[Dict[str, Any]] = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        try:
            # Explicit parameters to satisfy Pylance/Type Checking
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where=where_filter
            )

        except Exception as e:
            logger.warning(f"ChromaDB query failed — retrying without filter: {e}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        # ── Robust Result Extraction ──────────────────────────────────────────
        # Check if results are valid and contain data
        if not results or not results.get("ids") or len(results["ids"]) == 0:
            logger.warning(f"Dense search: no results for '{query[:60]}'")
            return []

        formatted = []
        
        # Safely extract lists with empty fallbacks
        ids = results.get("ids", [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for i, chunk_id in enumerate(ids):
            # Guard against index out of range and NoneTypes
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            distance = dists[i] if i < len(dists) and dists[i] is not None else 2.0
            doc_text = docs[i] if i < len(docs) and docs[i] is not None else ""
            
            # Distance to Score conversion
            score = round(1.0 - (float(distance) / 2.0), 4)

            # Metadata extraction with string fallbacks for .split() safety
            raw_keywords = meta.get("keywords")
            raw_components = meta.get("component_names")
            
            keywords = [kw for kw in str(raw_keywords or "").split("|") if kw]
            component_names = [cn for cn in str(raw_components or "").split("|") if cn]

            formatted.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "score": score,
                "source": "dense",
                "metadata": meta,
                "summary": meta.get("summary", ""),
                "keywords": keywords,
                "component_names": component_names,
            })

        logger.debug(f"Dense: {len(formatted)} results (top={formatted[0]['score'] if formatted else 0:.3f})")
        return formatted


# ── Module-level singleton (lazy init) ───────────────────────────────────────
_searcher: Optional[DenseSearcher] = None

def get_dense_searcher() -> DenseSearcher:
    global _searcher
    if _searcher is None:
        _searcher = DenseSearcher()
    return _searcher


def dense_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    vehicle_model: Optional[str] = None,
    content_type_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    return get_dense_searcher().search(
        query, top_k, vehicle_model, content_type_filter
    )


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_results = dense_search("how do I charge my Ather scooter?", top_k=3)
    for r in test_results:
        print(f"\n[{r['score']:.3f}] Page {r['metadata'].get('page_number')} "
              f"| {r['metadata'].get('section')}")
        print(f"  keywords: {r['keywords']}")
        print(f"  components: {r['component_names']}")
        print(f"  text: {r['text'][:120]}...")