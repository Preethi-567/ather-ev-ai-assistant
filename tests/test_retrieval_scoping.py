# tests/test_retrieval_scoping.py
"""
Model scoping exercised against the REAL ChromaDB and rank_bm25 libraries.
Only the embedding model and the cross-encoder are faked (deterministic, no
download), so what is under test is the filter syntax / ordering, not ML quality.
"""
import pickle
import re
import uuid
import zlib

import chromadb
import numpy as np
import pytest

from src.retrieval import hybrid_search as hs
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.dense_search import DenseSearcher

MODELS = ["Ather 450X", "Ather 450S", "Ather Rizta S", "Ather Rizta Z"]


class FakeEmbedder:
    """Hashed bag-of-words → 64-d unit vector. Deterministic across processes."""
    def encode(self, text, normalize_embeddings=True):
        v = np.zeros(64)
        for tok in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(tok.encode()) % 64] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


def _corpus():
    docs = []
    for i, m in enumerate(MODELS):
        docs.append((f"{m}-tyre",  f"Tyre pressure front 30 PSI rear 28 PSI solo {m}", m, 20 - i, "procedure"))
        docs.append((f"{m}-charge", f"Charging the battery with the portable charger {m}", m, 30 + i, "procedure"))
    return docs


@pytest.fixture()
def dense():
    client = chromadb.EphemeralClient()
    coll = client.create_collection(f"t_{uuid.uuid4().hex[:8]}")
    emb = FakeEmbedder()
    docs = _corpus()
    coll.add(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
        embeddings=[emb.encode(d[1]).tolist() for d in docs],
        metadatas=[{"vehicle_model": d[2], "page_number": d[3], "content_type": d[4],
                    "section": "s", "keywords": "", "component_names": "", "summary": ""} for d in docs],
    )
    s = DenseSearcher.__new__(DenseSearcher)
    s.embedder, s.collection = emb, coll
    return s


def models_of(results):
    return {r["metadata"]["vehicle_model"] for r in results}


# ── dense ─────────────────────────────────────────────────────────────────────

def test_dense_no_filter_sees_every_scooter(dense):
    assert models_of(dense.search("tyre pressure PSI", top_k=8)) == set(MODELS)


def test_dense_single_model_uses_eq(dense):
    out = dense.search("tyre pressure PSI", top_k=10, vehicle_models=["Ather 450S"])
    assert models_of(out) == {"Ather 450S"}
    assert len(out) == 2      # only 2 chunks exist for this model; top_k=10 must not error


def test_dense_multiple_models_use_in(dense):
    out = dense.search("tyre pressure PSI", top_k=10, vehicle_models=["Ather 450X", "Ather Rizta Z"])
    assert models_of(out) == {"Ather 450X", "Ather Rizta Z"}


def test_dense_bare_string_is_accepted_for_old_callers(dense):
    assert models_of(dense.search("tyre", vehicle_models="Ather Rizta S")) == {"Ather Rizta S"}


def test_dense_empty_list_means_unrestricted(dense):
    assert len(models_of(dense.search("tyre", top_k=8, vehicle_models=[]))) == 4


def test_dense_filter_failure_never_leaks_other_scooters(dense):
    """Old behaviour: on a filter error, retry unfiltered and KEEP everything."""
    real = dense.collection

    class Flaky:
        def query(self, **kw):
            if kw.get("where"):
                raise ValueError("simulated chroma filter failure")
            return real.query(**kw)
        def count(self): return real.count()

    dense.collection = Flaky()
    out = dense.search("tyre pressure PSI", top_k=3, vehicle_models=["Ather Rizta Z"])
    assert out and models_of(out) == {"Ather Rizta Z"}


def test_dense_content_filter_and_model_filter_combine(dense):
    out = dense.search("charging", top_k=10, vehicle_models=["Ather 450X"], content_type_filter="procedure")
    assert models_of(out) == {"Ather 450X"}


# ── BM25 ──────────────────────────────────────────────────────────────────────

@pytest.fixture()
def bm25(tmp_path):
    docs = []
    # 450X chunks score highest for the query, 450S chunks lowest. If the model
    # filter were applied AFTER the top_k cut, filtering to 450S would return [].
    for i in range(6):
        docs.append((f"x{i}", "tyre pressure PSI tyre pressure PSI tyre pressure", "Ather 450X", i + 1))
    for i in range(3):
        docs.append((f"s{i}", "tyre pressure PSI", "Ather 450S", i + 10))
    # Filler. BM25Okapi gives a term a NEGATIVE idf when it appears in more than
    # half the corpus, which would zero every score in a tiny all-on-topic test
    # corpus (a test artefact; the real 158-chunk corpus is not affected for
    # ordinary terms). Unrelated documents keep the query terms rare.
    for i in range(30):
        docs.append((f"f{i}", f"charging port headlamp mirror seat storage item{i}", "Ather Rizta Z", 40 + i))
    corpus = {
        "texts": [d[1] for d in docs], "raw_texts": [d[1] for d in docs],
        "chunk_ids": [d[0] for d in docs],
        "metadatas": [{"vehicle_model": d[2], "page_number": d[3]} for d in docs],
    }
    f = tmp_path / "bm25.pkl"
    f.write_bytes(pickle.dumps(corpus))
    return BM25Searcher(corpus_file=str(f))


def test_bm25_filter_runs_before_top_k_cut(bm25):
    out = bm25.search("tyre pressure PSI", top_k=2, vehicle_models=["Ather 450S"])
    assert len(out) == 2 and models_of(out) == {"Ather 450S"}


def test_bm25_multi_model_and_unrestricted(bm25):
    assert models_of(bm25.search("tyre pressure", top_k=20)) == {"Ather 450X", "Ather 450S"}
    assert models_of(bm25.search("tyre pressure", top_k=20, vehicle_models="Ather 450X")) == {"Ather 450X"}


# ── hybrid: grouped search ────────────────────────────────────────────────────

class FakeReranker:
    """Scores a document by how many query words it contains, minus a base."""
    def predict(self, pairs):
        out = []
        for q, d in pairs:
            qw = set(re.findall(r"\w+", q.lower()))
            hits = sum(1 for w in qw if w in d.lower())
            out.append(float(hits) * 3 - 5)
        return np.array(out)


@pytest.fixture()
def hybrid(monkeypatch):
    calls: list[list[str] | None] = []
    docs = _corpus() + [
        ("rizta-joey", "Joey pocket two litre storage under the front", "Ather Rizta S", 15, "feature"),
    ]

    def fake_dense(query, top_k=10, vehicle_models=None, content_type_filter=None):
        calls.append(list(vehicle_models) if vehicle_models else None)
        return [
            {"chunk_id": d[0], "text": d[1], "score": 0.5, "source": "dense",
             "metadata": {"vehicle_model": d[2], "page_number": d[3], "content_type": d[4]},
             "summary": "", "keywords": [], "component_names": []}
            for d in docs if not vehicle_models or d[2] in vehicle_models
        ][:top_k]

    monkeypatch.setattr(hs, "dense_search", fake_dense)
    monkeypatch.setattr(hs, "bm25_search", lambda *a, **k: [])
    s = hs.HybridSearcher.__new__(hs.HybridSearcher)
    s.reranker = FakeReranker()
    s.calls = calls
    return s


def test_single_group_is_a_plain_scoped_search(hybrid):
    out = hybrid.search_grouped("tyre pressure PSI", [["Ather 450S"]])
    assert hybrid.calls == [["Ather 450S"]]
    assert models_of(out["chunks"]) == {"Ather 450S"} and out["groups_searched"] == 1


def test_no_groups_is_unrestricted(hybrid):
    hybrid.search_grouped("tyre pressure PSI", [])
    assert hybrid.calls == [None]


def test_compare_two_scooters_gets_chunks_from_both(hybrid):
    out = hybrid.search_grouped("tyre pressure PSI", [["Ather 450X"], ["Ather Rizta Z"]])
    assert hybrid.calls == [["Ather 450X"], ["Ather Rizta Z"]]
    assert {"Ather 450X", "Ather Rizta Z"} <= models_of(out["chunks"])


def test_series_fallback_drops_the_irrelevant_series(hybrid):
    """Joey pocket exists only in the Rizta guides: the 450 series must not pad
    the prompt with unrelated chunks."""
    series = [["Ather 450X", "Ather 450S"], ["Ather Rizta S", "Ather Rizta Z"]]
    out = hybrid.search_grouped("Joey pocket storage", series)
    assert hybrid.calls == series
    assert out["chunks"][0]["chunk_id"] == "rizta-joey"
    assert not (models_of(out["chunks"]) & {"Ather 450X", "Ather 450S"})


def test_series_fallback_keeps_both_series_when_both_relevant(hybrid):
    series = [["Ather 450X", "Ather 450S"], ["Ather Rizta S", "Ather Rizta Z"]]
    out = hybrid.search_grouped("tyre pressure PSI", series)
    assert models_of(out["chunks"]) & {"Ather 450X", "Ather 450S"}
    assert models_of(out["chunks"]) & {"Ather Rizta S", "Ather Rizta Z"}
    assert out["groups_searched"] == 2


# ── ranking text: model name removed once the scope is pinned ────────────────

@pytest.fixture()
def recording(monkeypatch):
    """Records the query each ranker receives and the (query, doc) pairs the
    reranker scores."""
    seen = {"dense": [], "bm25": [], "pairs": []}
    chunk = {"chunk_id": "c1", "text": "Plug in the charging cable into a 5A socket",
             "score": 0.5, "source": "dense",
             "metadata": {"vehicle_model": "Ather Rizta Z", "page_number": 35, "content_type": "procedure"},
             "summary": "SUMMARY-SHOULD-NOT-BE-SCORED", "keywords": [],
             "component_names": ["COMPONENT-SHOULD-NOT-BE-SCORED"]}

    def fake_dense(query, top_k=10, vehicle_models=None, content_type_filter=None):
        seen["dense"].append(query)
        return [dict(chunk)]

    def fake_bm25(query, top_k=10, vehicle_models=None):
        seen["bm25"].append(query)
        return []

    class RecordingReranker:
        def predict(self, pairs):
            seen["pairs"].extend(pairs)
            return np.array([1.0] * len(pairs))

    monkeypatch.setattr(hs, "dense_search", fake_dense)
    monkeypatch.setattr(hs, "bm25_search", fake_bm25)
    s = hs.HybridSearcher.__new__(hs.HybridSearcher)
    s.reranker = RecordingReranker()
    s.seen = seen
    return s


def test_scoped_search_ranks_on_text_without_the_model_name(recording):
    out = recording.search("how do I charge my Rizta Z?", ["Ather Rizta Z"])
    want = "how do I charge my scooter?"
    assert recording.seen["dense"] == [want]
    assert recording.seen["bm25"] == [want]
    assert [q for q, _ in recording.seen["pairs"]] == [want]
    assert out["ranking_query"] == want


def test_unscoped_search_leaves_the_query_alone(recording):
    """No scope → nothing was filtered by metadata, so nothing may be stripped."""
    q = "how do I charge my Rizta Z?"
    recording.search(q, None)
    assert recording.seen["dense"] == [q]


def test_reranker_sees_chunk_text_only(recording):
    """Not summary, not component fragments, and the full text rather than 400 chars."""
    recording.search("how do I charge my Rizta Z?", ["Ather Rizta Z"])
    (_, doc), = recording.seen["pairs"]
    assert doc == "Plug in the charging cable into a 5A socket"
    assert "SHOULD-NOT-BE-SCORED" not in doc


def test_long_chunk_is_bounded_not_cut_at_400(recording):
    chunk = {"text": "word " * 1000}
    doc = recording._build_reranker_input("q", chunk)
    assert len(doc) == hs.RERANK_DOC_CHARS and hs.RERANK_DOC_CHARS > 400
