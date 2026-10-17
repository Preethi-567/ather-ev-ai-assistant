# tests/test_api_flow.py
"""
End-to-end /query flow through FastAPI's TestClient.

Real:  intent classification (keyword path), model-scope resolution, grouped-search
       plumbing, prompt/context building, citation rendering, support-contact
       resolution, SQLite logging, Pydantic response models.
Fake:  Gemini (rewriter + sufficiency + generation) and the retriever, so no
       network and no model downloads.

The retriever fake returns canned chunks AND records the scope it was asked for,
which is what proves the model filter actually reaches retrieval.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as main
from src.db import feedback_store
from src.generation import prompt_chain
from src.generation.prompts import FALLBACK_RESPONSE
from src.query.model_detector import SERIES_GROUPS


def mk(model, page, text, file="m.pdf", cid=None, score=6.0):
    return {
        "chunk_id": cid or f"{model}-{page}", "text": text, "summary": text[:40],
        "reranker_score": score, "component_names": [],
        "metadata": {"vehicle_model": model, "page_number": page, "section": "Tyres",
                     "content_type": "procedure", "source_file": file},
    }


CHUNKS = {
    "Ather 450S":    mk("Ather 450S", 20, "Front 30 PSI rear 28 PSI", "450S_User_Manual_Jan_24.pdf"),
    "Ather 450X":    mk("Ather 450X", 20, "Front 30 PSI rear 28 PSI", "2025-Ather_450X-QSG.pdf"),
    "Ather Rizta S": mk("Ather Rizta S", 18, "Front 30 PSI rear 28 PSI", "Rizta S Quick Start Guide.pdf"),
    "Ather Rizta Z": mk("Ather Rizta Z", 19, "Front 30 PSI rear 28 PSI", "Rizta Z Quick Start Guide.pdf"),
}


@pytest.fixture()
def api(tmp_path, monkeypatch):
    # Never touch the repo's tracked data/feedback.db.
    monkeypatch.setattr(feedback_store, "DB_PATH", str(tmp_path / "fb.db"))
    feedback_store.init_db()

    state = {"groups": None, "gemini_prompts": [], "generation": "ok [C1]", "sufficient": "YES"}

    def fake_search(query, groups, content_type_filter=None):
        state["groups"] = [list(g) for g in groups]
        chunks = [CHUNKS[m] for g in groups for m in g if m in CHUNKS][:6]
        return {"chunks": chunks, "top_score": 6.0, "low_confidence": False}

    def fake_gemini(prompt, temperature=0.1, max_output_tokens=800, **kw):
        state["gemini_prompts"].append(prompt)
        if max_output_tokens <= 5:                       # sufficiency check
            return state["sufficient"]
        return state["generation"]

    monkeypatch.setattr(main, "hybrid_search_grouped", fake_search)
    monkeypatch.setattr(prompt_chain, "call_gemini", fake_gemini)
    # No rewriter call: pass the query straight through.
    monkeypatch.setattr(main, "rewrite_query", lambda q, previous_query=None: {"rewritten": q})
    monkeypatch.setattr(main, "classify_intent",
                        lambda q: {"intent": "spec_lookup", "method": "test", "skip_rag": False})

    client = TestClient(main.app)          # no `with`: lifespan (model loading) is not run
    client.state = state
    return client


def ask(api, q, sid=None):
    r = api.post("/query", json={"query": q, "session_id": sid or f"s-{uuid.uuid4().hex[:6]}"})
    assert r.status_code == 200, r.text
    return r.json()


# ── scope reaches retrieval ───────────────────────────────────────────────────

def test_named_model_scopes_retrieval_and_citation(api):
    api.state["generation"] = "Front tyre pressure is 30 PSI [C1]."
    body = ask(api, "what is the tyre pressure on my 450S?")

    assert api.state["groups"] == [["Ather 450S"]]
    assert body["model_scope"] == "explicit" and body["models"] == ["Ather 450S"]
    assert body["answer"] == "Front tyre pressure is 30 PSI (Ather 450S manual, page 20)."
    assert body["sources"] == [{"model": "Ather 450S", "page": 20,
                                "source_file": "450S_User_Manual_Jan_24.pdf"}]
    assert body["cited_pages"] == [20]


def test_no_model_searches_both_series(api):
    api.state["generation"] = "**Ather 450X** 30 PSI [C1]\n**Ather Rizta S** 30 PSI [C3]"
    body = ask(api, "what is the tyre pressure?")

    assert api.state["groups"] == SERIES_GROUPS
    assert body["model_scope"] == "none"
    assert [s["model"] for s in body["sources"]] == ["Ather 450X", "Ather Rizta S"]


def test_comparison_query_searches_each_named_model_separately(api):
    api.state["generation"] = "a [C1] b [C2]"
    ask(api, "tyre pressure 450X vs Rizta Z")
    assert api.state["groups"] == [["Ather 450X"], ["Ather Rizta Z"]]


def test_follow_up_inherits_the_model_from_the_session(api):
    sid = "follow-up-session"
    ask(api, "tyre pressure on the Rizta Z", sid)
    body = ask(api, "and for the rear?", sid)
    assert api.state["groups"] == [["Ather Rizta Z"]]
    assert body["model_scope"] == "session"


def test_new_model_overrides_the_session(api):
    sid = "override-session"
    ask(api, "tyre pressure on the Rizta Z", sid)
    ask(api, "what about the 450S?", sid)
    assert api.state["groups"] == [["Ather 450S"]]


def test_model_is_visible_to_the_generator_but_page_numbers_are_not(api):
    api.state["generation"] = "30 PSI [C1]."
    ask(api, "tyre pressure 450S")
    gen_prompt = next(p for p in api.state["gemini_prompts"] if "RETRIEVED CONTEXT CHUNKS" in p)
    assert "Model     : Ather 450S" in gen_prompt and "[C1]" in gen_prompt
    assert "Page" not in gen_prompt.split("RETRIEVED CONTEXT CHUNKS:")[1]     # nothing to misquote


# ── citation guarantees ───────────────────────────────────────────────────────

def test_hallucinated_tag_and_free_text_page_never_reach_the_user(api):
    api.state["generation"] = "30 PSI [C1][C9]. See page 77 for more."
    body = ask(api, "tyre pressure 450S")
    assert "page 77" not in body["answer"] and "C9" not in body["answer"]
    assert [s["page"] for s in body["sources"]] == [20]


# ── support contact ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected, src", [
    ("tyre pressure 450S",  "+91 7676 811 777", "450S_User_Manual_Jan_24.pdf, page 48"),
    ("tyre pressure 450X",  "+91 7676 811 777", "2025-Ather_450X-QSG.pdf, page 49"),
    ("tyre pressure Rizta Z", None, None),
    ("tyre pressure Rizta S", None, None),
])
def test_support_contact_only_where_the_manual_prints_one(api, query, expected, src):
    body = ask(api, query)
    assert body["support_contact"] == expected
    assert body["support_contact_source"] == src


def test_support_contact_is_withheld_when_scooters_are_mixed(api):
    assert ask(api, "tyre pressure 450S vs Rizta Z")["support_contact"] is None


def test_series_fallback_with_only_450_chunks_may_return_a_contact(api, monkeypatch):
    # Rizta group scored too low and was dropped by the merge → answer is 450-only.
    monkeypatch.setattr(main, "hybrid_search_grouped", lambda query, groups, content_type_filter=None: {
        "chunks": [CHUNKS["Ather 450X"], CHUNKS["Ather 450S"]], "top_score": 6.0, "low_confidence": False})
    api.state["generation"] = "30 PSI [C1]"
    assert ask(api, "tyre pressure?")["support_contact"] == "+91 7676 811 777"


# ── the fallback sentence stays exact, contact stays separate ─────────────────

def test_fallback_is_byte_exact_and_contact_is_a_separate_field(api):
    api.state["sufficient"] = "NO"
    body = ask(api, "does the 450S have a sunroof?")
    assert body["answer"] == FALLBACK_RESPONSE
    assert body["fallback_used"] is True
    assert body["support_contact"] == "+91 7676 811 777"
    assert "7676" not in body["answer"]


def test_fallback_for_a_rizta_has_no_contact_rather_than_a_borrowed_one(api):
    api.state["sufficient"] = "NO"
    body = ask(api, "does the Rizta Z have a sunroof?")
    assert body["answer"] == FALLBACK_RESPONSE and body["support_contact"] is None


def test_model_written_fallback_is_still_detected(api):
    api.state["generation"] = FALLBACK_RESPONSE
    body = ask(api, "tyre pressure 450S")
    assert body["fallback_used"] is True and body["sources"] == []
