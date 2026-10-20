# tests/test_prompt_chain_gates.py
"""The two Gemini-call gates inside PromptChain.run (see NO_MATCH_FLOOR)."""
import pytest

from src.generation import prompt_chain
from src.generation.prompt_chain import NO_MATCH_FLOOR, PromptChain
from src.generation.prompts import FALLBACK_RESPONSE


def chunk(score=6.0):
    return {
        "chunk_id": "c1", "text": "Front 30 PSI rear 28 PSI", "summary": "tyre pressure",
        "reranker_score": score, "component_names": [],
        "metadata": {"vehicle_model": "Ather 450S", "page_number": 20, "section": "Tyres",
                     "content_type": "procedure", "source_file": "m.pdf"},
    }


@pytest.fixture()
def chain(monkeypatch):
    calls = []

    def fake(prompt, temperature=0.1, max_output_tokens=800, **kw):
        calls.append(max_output_tokens)
        return "YES" if max_output_tokens <= 5 else "30 PSI front [C1]."

    monkeypatch.setattr(prompt_chain, "call_gemini", fake)
    c = PromptChain()
    c.calls = calls
    return c


def run(chain, top_score):
    return chain.run("tyre pressure?", {"chunks": [chunk(top_score)], "top_score": top_score,
                                        "low_confidence": top_score < 0}, session_id="gate-test")


def test_score_below_the_floor_is_rejected_with_zero_gemini_calls(chain):
    out = run(chain, NO_MATCH_FLOOR - 0.1)
    assert out["answer"] == FALLBACK_RESPONSE and out["fallback_used"] is True
    assert out["sources"] == [] and out["low_confidence"] is True
    assert chain.calls == []


def test_score_at_the_floor_is_not_rejected(chain):
    """The floor is a strict '<': a borderline score still gets a chance."""
    out = run(chain, NO_MATCH_FLOOR)
    assert out["fallback_used"] is False and len(chain.calls) == 2


def test_default_runs_sufficiency_then_generation(chain, monkeypatch):
    monkeypatch.setattr(prompt_chain, "SUFFICIENCY_CHECK_ENABLED", True)
    run(chain, 6.0)
    assert len(chain.calls) == 2 and chain.calls[0] <= 5


def test_switched_off_runs_generation_only(chain, monkeypatch):
    monkeypatch.setattr(prompt_chain, "SUFFICIENCY_CHECK_ENABLED", False)
    out = run(chain, 6.0)
    assert len(chain.calls) == 1 and chain.calls[0] > 5
    assert out["sources"][0].page == 20


def test_a_no_answer_from_the_sufficiency_check_still_wins_when_enabled(chain, monkeypatch):
    monkeypatch.setattr(prompt_chain, "call_gemini", lambda *a, **k: "NO")
    monkeypatch.setattr(prompt_chain, "SUFFICIENCY_CHECK_ENABLED", True)
    out = run(chain, 6.0)
    assert out["answer"] == FALLBACK_RESPONSE
