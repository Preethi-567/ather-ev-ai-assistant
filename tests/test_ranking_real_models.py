# tests/test_ranking_real_models.py
"""
Regression test against the REAL embedding model, reranker, ChromaDB index and
BM25 corpus. Opt-in, because it needs the downloaded models and a built index:

    PowerShell:   $env:RUN_REAL_MODEL_TESTS = "1"; python -m pytest tests/test_ranking_real_models.py -q

Each case is a query that was measured to rank the wrong page first before the
ranking-text fix (see ADR-003). The assertion is deliberately loose: the page
that answers the question must be #1 once the model name is stripped. It does
not pin scores, which move with model versions.
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_TESTS") != "1"
    or not (ROOT / "chroma_db").exists()
    or not (ROOT / "data" / "chunks" / "bm25_corpus.pkl").exists(),
    reason="set RUN_REAL_MODEL_TESTS=1 and run ingestion first",
)

# (query, scope, pages any of which is an acceptable #1)
CASES = [
    ("how do I charge my Rizta Z?",                           "Ather Rizta Z", {35, 36}),
    ("what do the charger light colours mean on my Rizta Z?", "Ather Rizta Z", {39}),
    ("where is the Joey Pocket on the Rizta Z?",              "Ather Rizta Z", {16}),
    ("what is the tyre pressure on my 450S?",                 "Ather 450S",    {20}),
]


@pytest.fixture(scope="module")
def searcher():
    os.chdir(ROOT)                    # chroma_db/ and data/ are relative paths
    from src.retrieval.hybrid_search import get_hybrid_searcher
    return get_hybrid_searcher()


@pytest.mark.parametrize("query, model, ok_pages", CASES)
def test_answering_page_ranks_first(searcher, query, model, ok_pages):
    res = searcher.search(query, [model])
    pages = [c["metadata"]["page_number"] for c in res["chunks"]]
    assert pages[0] in ok_pages, f"top-5 pages were {pages}, wanted one of {sorted(ok_pages)} first"


def test_charging_query_surfaces_the_plugging_in_page_in_top_three(searcher):
    res = searcher.search("how do I charge my Rizta Z?", ["Ather Rizta Z"])
    pages = [c["metadata"]["page_number"] for c in res["chunks"]][:3]
    assert {35, 36} & set(pages), pages
