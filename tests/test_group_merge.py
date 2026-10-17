# tests/test_group_merge.py
from src.retrieval.group_merge import merge_group_results


def chunk(cid, score):
    return {"chunk_id": cid, "reranker_score": score}


def res(chunks, top=None):
    return {"chunks": chunks,
            "top_score": top if top is not None else (chunks[0]["reranker_score"] if chunks else 0.0)}


def test_empty_input():
    out = merge_group_results([res([]), res([])], per_group=3, min_score=0.0)
    assert out["chunks"] == [] and out["best_group"] is None


def test_weak_group_is_dropped_but_best_group_kept():
    """Rizta-only question: the 450 group must not pad the prompt with noise."""
    rizta = res([chunk("r1", 8.2), chunk("r2", 5.0), chunk("r3", -9.0)])
    s450  = res([chunk("x1", -7.5), chunk("x2", -8.0)])
    out = merge_group_results([s450, rizta], per_group=3, min_score=0.0)
    assert [c["chunk_id"] for c in out["chunks"]] == ["r1", "r2", "r3"]   # best group untouched
    assert out["best_group"] == 1 and out["top_score"] == 8.2


def test_both_groups_contribute_when_both_relevant():
    a = res([chunk("a1", 6.0), chunk("a2", 4.0), chunk("a3", 3.0), chunk("a4", 2.0)])
    b = res([chunk("b1", 5.5), chunk("b2", 1.0), chunk("b3", 0.5)])
    out = merge_group_results([a, b], per_group=3, min_score=0.0)
    assert [c["chunk_id"] for c in out["chunks"]] == ["a1", "b1", "a2", "a3", "b2", "b3"]


def test_per_group_cap():
    a = res([chunk(f"a{i}", 9 - i) for i in range(6)])
    b = res([chunk("b1", 1.0)])
    out = merge_group_results([a, b], per_group=2, min_score=0.0)
    assert len([c for c in out["chunks"] if c["chunk_id"].startswith("a")]) == 2


def test_duplicate_chunk_ids_keep_highest_score_once():
    a = res([chunk("same", 3.0)])
    b = res([chunk("same", 7.0)])
    out = merge_group_results([a, b], per_group=3, min_score=0.0)
    assert len(out["chunks"]) == 1 and out["chunks"][0]["reranker_score"] == 7.0


def test_all_low_scores_still_returns_best_group_for_low_confidence_path():
    a = res([chunk("a1", -6.0)])
    b = res([chunk("b1", -8.0)])
    out = merge_group_results([a, b], per_group=3, min_score=0.0)
    assert [c["chunk_id"] for c in out["chunks"]] == ["a1"]
    assert out["top_score"] == -6.0
