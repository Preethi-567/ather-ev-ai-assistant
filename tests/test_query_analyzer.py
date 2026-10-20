# tests/test_query_analyzer.py
"""
How many Gemini calls does understanding one query cost? Each row of the decision
table in query_analyzer.py is pinned here, plus the failure paths.
"""
import pytest

from src.query import intent_classifier, query_analyzer, query_rewriter
from src.query.query_analyzer import _build_prompt, _parse_merged, analyze_query


@pytest.fixture()
def gemini(monkeypatch):
    """One fake for every module that imported call_gemini by name."""
    state = {"prompts": [], "replies": []}

    def fake(prompt, temperature=0.1, max_output_tokens=1000, **kw):
        state["prompts"].append(prompt)
        return state["replies"].pop(0) if state["replies"] else ""

    for mod in (query_analyzer, intent_classifier, query_rewriter):
        monkeypatch.setattr(mod, "call_gemini", fake)
    return state


# ── the decision table ────────────────────────────────────────────────────────

def test_out_of_scope_keyword_costs_nothing_and_skips_the_rewrite(gemini):
    intent, rewrite = analyze_query("who won the IPL?")
    assert intent["intent"] == "out_of_scope" and intent["skip_rag"] is True
    assert rewrite is None and gemini["prompts"] == []


def test_keyword_intent_and_clear_query_cost_nothing(gemini):
    intent, rewrite = analyze_query("what is the tyre pressure on my 450S?")
    assert intent["intent"] == "spec_lookup" and intent["method"] == "keyword"
    assert rewrite["method"] == "passthrough" and rewrite["changed"] is False
    assert gemini["prompts"] == []


def test_fast_rewrite_with_unknown_intent_costs_one_classify_call(gemini):
    gemini["replies"] = ["spec_lookup"]
    intent, rewrite = analyze_query("warp mode")          # in the fast-rewrite table, no keyword
    assert rewrite["method"] == "fast" and len(gemini["prompts"]) == 1
    assert intent["method"] == "gemini"


def test_keyword_intent_with_short_query_costs_one_rewrite_call(gemini):
    gemini["replies"] = ["What does a red warning light on the Ather dashboard mean?"]
    intent, rewrite = analyze_query("red light blinking")
    assert intent["intent"] == "troubleshooting" and intent["method"] == "keyword"
    assert rewrite["method"] == "gemini" and len(gemini["prompts"]) == 1


def test_unknown_intent_and_clear_query_cost_one_classify_call(gemini):
    gemini["replies"] = ["spec_lookup"]
    q = "can you tell me where the Joey pocket is located on this scooter?"
    intent, rewrite = analyze_query(q)
    assert intent["method"] == "gemini" and rewrite["method"] == "passthrough"
    assert len(gemini["prompts"]) == 1


def test_unknown_intent_and_rewrite_needed_is_ONE_merged_call_not_two(gemini):
    """The follow-up case that used to cost two sequential calls."""
    gemini["replies"] = ['{"intent": "charging", "rewritten": "How long does it take to charge the Rizta Z?"}']
    intent, rewrite = analyze_query("how long does it take?", previous_query="how do I charge my Rizta Z?")
    assert len(gemini["prompts"]) == 1
    assert intent == {"intent": "charging", "method": "gemini_merged", "skip_rag": False}
    assert rewrite["rewritten"] == "How long does it take to charge the Rizta Z?"
    assert rewrite["original"] == "how long does it take?" and rewrite["changed"] is True


def test_merged_call_that_says_out_of_scope_returns_no_rewrite(gemini):
    gemini["replies"] = ['{"intent": "out_of_scope", "rewritten": "best pizza near me"}']
    intent, rewrite = analyze_query("best pizzeria near me")
    assert intent["skip_rag"] is True and rewrite is None and len(gemini["prompts"]) == 1


# ── failure paths ─────────────────────────────────────────────────────────────

def test_gemini_outage_uses_defaults_and_spends_no_further_calls(gemini):
    gemini["replies"] = [""]                               # call_gemini returns "" when everything failed
    intent, rewrite = analyze_query("how long does it take?")
    assert len(gemini["prompts"]) == 1                     # did not pile two more calls onto an outage
    assert intent["intent"] == "spec_lookup" and intent["skip_rag"] is False
    assert rewrite["rewritten"] == "how long does it take?"


def test_unusable_reply_falls_back_to_the_two_original_calls(gemini):
    gemini["replies"] = [
        "Sorry, I can't do that",                           # merged call: not JSON
        "charging",                                         # classify call
        "How long does it take to charge the scooter?",     # rewrite call
    ]
    intent, rewrite = analyze_query("how long does it take?", previous_query="how do I charge it?")
    assert len(gemini["prompts"]) == 3
    assert intent["intent"] == "charging" and rewrite["rewritten"].startswith("How long")


def test_unknown_intent_label_is_not_trusted(gemini):
    gemini["replies"] = ['{"intent": "banana", "rewritten": "x"}', "charging", "A clear question about charging time?"]
    intent, _ = analyze_query("how long does it take?")
    assert intent["intent"] == "charging" and len(gemini["prompts"]) == 3


def test_overlong_or_empty_rewrite_keeps_the_original_query(gemini):
    gemini["replies"] = ['{"intent": "charging", "rewritten": ""}']
    _, rewrite = analyze_query("how long does it take?")
    assert rewrite["rewritten"] == "how long does it take?"


# ── parsing and prompt building ───────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ('{"intent": "charging", "rewritten": "Q?"}',                       ("charging", "Q?")),
    ('```json\n{"intent": "safety", "rewritten": "Q?"}\n```',            ("safety", "Q?")),
    ('Here you go: {"intent": "Charging", "rewritten": "\\"Q?\\""}',     ("charging", "Q?")),
    ('{"intent": "banana", "rewritten": "Q?"}',                         None),
    ('{"rewritten": "Q?"}',                                             None),
    ('[1, 2]',                                                          None),
    ('not json',                                                        None),
    ('{broken',                                                         None),
])
def test_parse_merged(raw, expected):
    assert _parse_merged(raw) == expected


def test_previous_query_is_offered_whenever_there_is_one():
    """'how long does it take?' (5 words) is a follow-up the old heuristic missed."""
    follow = _build_prompt("how long does it take?", "how do I charge my Rizta Z?")
    alone  = _build_prompt("how long does it take?", None)
    assert "previous query: how do I charge my Rizta Z?" in follow
    assert "previous query" not in alone.lower().replace("the previous query named", "")


def test_a_self_contained_question_is_told_to_ignore_the_previous_one():
    prompt = _build_prompt("what is the tyre pressure on my 450S?", "how do I charge my Rizta Z?")
    assert "IGNORE the previous query" in prompt


def test_user_text_with_braces_cannot_break_the_prompt():
    prompt = _build_prompt("what is {this} {query}?", None)
    assert "what is {this} {query}?" in prompt
