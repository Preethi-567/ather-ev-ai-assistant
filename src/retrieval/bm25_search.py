# src/retrieval/bm25_search.py
"""
BM25 keyword search using rank_bm25.

V2 upgrades:
  - Corpus is now indexed on augmented text (text + keywords + component_names)
    giving BM25 the benefit of Gemini Vision's domain keyword extraction
  - Returns keywords and component_names in results for the reranker
  - Handles component name queries with a direct name-matching boost
"""

import pickle
import re
from pathlib import Path
from typing import Sequence
from loguru import logger
from rank_bm25 import BM25Okapi

BM25_CORPUS_FILE = "data/chunks/bm25_corpus.pkl"


class BM25Searcher:
    """
    Loads the pre-built BM25 corpus and provides fast keyword search.
    Instantiate once at app startup.
    """

    def __init__(self, corpus_file: str = BM25_CORPUS_FILE):
        if not Path(corpus_file).exists():
            raise FileNotFoundError(
                f"BM25 corpus not found at {corpus_file}. "
                "Run scripts/run_ingestion.py first."
            )

        logger.info(f"Loading BM25 corpus from {corpus_file}...")

        with open(corpus_file, "rb") as f:
            corpus_data = pickle.load(f)

        # V2 corpus has both augmented texts (for indexing) and raw texts
        self.texts          = corpus_data["texts"]       # augmented (keywords appended)
        self.raw_texts      = corpus_data["raw_texts"]   # original text for display
        self.chunk_ids      = corpus_data["chunk_ids"]
        self.metadatas      = corpus_data["metadatas"]

        # V2 extras
        self.summaries       = corpus_data.get("summaries", [""] * len(self.texts))
        self.keywords_list   = corpus_data.get("keywords_list", [[] for _ in self.texts])
        self.component_names = corpus_data.get("component_names", [[] for _ in self.texts])
        self.components      = corpus_data.get("components", [[] for _ in self.texts])

        # Tokenise the augmented texts for BM25
        # Simple whitespace + lowercase tokenisation works well for technical manuals
        tokenised = [self._tokenise(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenised)

        logger.info(f"BM25Searcher ready — {len(self.texts)} documents indexed")

    def _tokenise(self, text: str) -> list[str]:
        """
        Tokenise text for BM25.
        Keeps alphanumeric tokens, lowercased.
        Splits on whitespace and common punctuation.
        Special: preserves tokens like 'IP67', '450X', 'Gen4' intact.
        """
        # Replace punctuation with space (but keep alphanumerics together)
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens  = cleaned.split()
        # Filter very short tokens (reduces noise from single chars)
        return [t for t in tokens if len(t) > 1]

    def _component_name_match(self, query: str) -> list[int]:
        """
        Direct component name matching.
        If the query contains a component name that appears in component_names,
        return indices of chunks that have that component.

        Example: query 'what is the headlamp' → finds all chunks where
        component_names contains 'Headlamp'.
        """
        query_lower = query.lower()
        matched_indices = []

        for i, comp_names in enumerate(self.component_names):
            for name in comp_names:
                if name.lower() in query_lower or query_lower in name.lower():
                    matched_indices.append(i)
                    break  # one match per chunk is enough

        return matched_indices

    def search(
        self,
        query: str,
        top_k: int = 10,
        vehicle_models: "Sequence[str] | str | None" = None
    ) -> list[dict]:
        """
        Run BM25 keyword search.

        Strategy:
        1. Run standard BM25 on tokenised query
        2. Run component name direct matching
        3. Merge: component matches get a score bonus and appear at top

        Args:
            query:         User query string
            top_k:         Number of results to return
            vehicle_models: Restrict to these `vehicle_model` tags. BM25 has no
                            metadata filter, so this is applied while walking the
                            ranked list — i.e. BEFORE the top_k cut, so a model
                            filter never leaves fewer than top_k results when
                            enough matching chunks exist.

        Returns:
            List of result dicts with chunk data + V2 fields
        """
        if isinstance(vehicle_models, str):
            vehicle_models = [vehicle_models]
        allowed = {m for m in (vehicle_models or []) if m}

        query_tokens = self._tokenise(query)

        if not query_tokens:
            logger.warning("BM25: empty query after tokenisation")
            return []

        # ── BM25 scores ──────────────────────────────────────────────────────
        scores = self.bm25.get_scores(query_tokens)

        # ── Component name boost ─────────────────────────────────────────────
        # V2 feature: exact component name matches get a score boost
        # This handles queries like "what does the headlamp do" where
        # BM25 might miss if 'headlamp' only appears in component_names
        component_matches = self._component_name_match(query)
        for idx in component_matches:
            scores[idx] += 5.0   # empirical boost value — tune if needed

        # ── Rank and filter ──────────────────────────────────────────────────
        # Get indices sorted by descending score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []
        seen_ids = set()

        for idx in ranked_indices:
            if len(results) >= top_k:
                break

            score = scores[idx]
            if score <= 0:
                break   # BM25 score of 0 means no keyword overlap at all

            chunk_id = self.chunk_ids[idx]
            meta     = self.metadatas[idx]

            # Model scope (see docstring: applied before the top_k cut)
            if allowed and meta.get("vehicle_model") not in allowed:
                continue

            # Deduplicate (component boost can cause duplicates)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            results.append({
                "chunk_id":        chunk_id,
                "text":            self.raw_texts[idx],    # display original text
                "score":           round(float(score), 4),
                "source":          "bm25",
                "metadata":        meta,
                # V2 fields
                "summary":         self.summaries[idx],
                "keywords":        self.keywords_list[idx],
                "component_names": self.component_names[idx],
                "components":      self.components[idx],
                # Flag if this result came from a component boost
                "component_boosted": idx in component_matches,
            })

        logger.debug(
            f"BM25 search: {len(results)} results "
            f"(top score: {results[0]['score']:.3f} "
            f"for '{query[:40]}')" if results else
            f"BM25 search: 0 results for '{query[:40]}'"
        )

        return results


# ── Module-level singleton ────────────────────────────────────────────────────
_searcher: BM25Searcher | None = None

def get_bm25_searcher() -> BM25Searcher:
    global _searcher
    if _searcher is None:
        _searcher = BM25Searcher()
    return _searcher


def bm25_search(
    query: str,
    top_k: int = 10,
    vehicle_models: "Sequence[str] | str | None" = None
) -> list[dict]:
    """Convenience function — uses the module-level singleton."""
    return get_bm25_searcher().search(query, top_k, vehicle_models)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = bm25_search("IP67 water resistance charging port", top_k=3)
    for r in results:
        print(f"\n[BM25 {r['score']:.2f}] Page {r['metadata'].get('page_number')}")
        print(f"  keywords: {r['keywords']}")
        print(f"  boosted:  {r.get('component_boosted', False)}")
        print(f"  text: {r['text'][:120]}...")