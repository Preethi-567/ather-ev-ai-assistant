# tests/test_safety.py
"""Rule 4 / Rule 6 notices: appended by code, verbatim, once."""
import pytest

from src.generation import prompt_chain
from src.generation.prompt_chain import PromptChain
from src.generation.prompts import FALLBACK_RESPONSE, LOW_CONFIDENCE_PREFIX, SYSTEM_PROMPT
from src.generation.safety import (
    ESCALATION_NOTICE, TECHNICIAN_NOTICE, append_notices, needs_escalation_notice,
    needs_technician_notice, strip_model_notices,
)

R4 = "⚠️ This should only be serviced by an Ather-certified technician."
R6 = "If this persists, please contact Ather Support or book a service via the Ather app."


def test_notice_wording_is_the_project_spec_exactly():
    assert TECHNICIAN_NOTICE == R4 and ESCALATION_NOTICE == R6


@pytest.mark.parametrize("q", ["how do I charge my scooter", "battery health", "is the motor covered",
                               "what voltage is the charger", "recharge time", "discharge rate"])
def test_technician_notice_triggers(q):
    assert needs_technician_notice("spec_lookup", q)


def test_charging_intent_alone_triggers():
    assert needs_technician_notice("charging", "how long does it take?")


@pytest.mark.parametrize("q", ["what is the tyre pressure", "how do I park", "what is the boot space"])
def test_technician_notice_not_added_to_unrelated(q):
    assert not needs_technician_notice("spec_lookup", q)


@pytest.mark.parametrize("q", ["scooter won't start", "strange noise from the wheel", "red light blinking",
                               "there is a burning smell", "display not working"])
def test_escalation_triggers_on_symptoms(q):
    assert needs_escalation_notice("spec_lookup", q)


@pytest.mark.parametrize("intent", ["troubleshooting", "safety"])
def test_escalation_triggers_on_intent(intent):
    assert needs_escalation_notice(intent, "anything")


@pytest.mark.parametrize("q", ["does it have a sound when locking", "how hot can it get outside", "what is the range"])
def test_escalation_not_added_to_ordinary_questions(q):
    assert not needs_escalation_notice("spec_lookup", q)


def test_both_notices_appended_technician_first_escalation_last():
    out, added = append_notices("Do X [C1].", "troubleshooting", "my battery is not working")
    assert out == f"Do X [C1].\n\n{R4}\n\n{R6}" and added == ["technician", "escalation"]


def test_no_notice_needed_leaves_answer_untouched():
    assert append_notices("30 PSI [C1].", "spec_lookup", "tyre pressure") == ("30 PSI [C1].", [])


def test_model_written_notices_are_replaced_not_duplicated():
    raw = ("Steps [C1].\n⚠️ For electrical or battery-related issues, always consult an Ather-certified technician.\n"
           "If this persists, please contact Ather Support or book a service appointment via the Ather app.")
    out, _ = append_notices(raw, "troubleshooting", "charger not working")
    assert out.count("technician") == 1 and out.count("If this persists") == 1
    assert out.endswith(R6) and R4 in out


def test_a_real_manual_sentence_about_technicians_is_kept():
    raw = "Battery replacement must be done by an Ather-certified technician [C2]."
    assert strip_model_notices(raw) == raw


def test_notices_are_idempotent():
    once, _ = append_notices("A [C1].", "charging", "charge")
    twice, _ = append_notices(once, "charging", "charge")
    assert once == twice


def test_prompt_no_longer_asks_the_model_to_write_notices():
    assert "always append" not in SYSTEM_PROMPT and "always end with" not in SYSTEM_PROMPT
    assert "Do NOT write warnings" in SYSTEM_PROMPT


# ── through PromptChain.run ───────────────────────────────────────────────────
def chunk(score=6.0):
    return {"chunk_id": "c1", "text": "Plug the charger in", "summary": "charging", "reranker_score": score,
            "component_names": [], "metadata": {"vehicle_model": "Ather 450S", "page_number": 12,
            "section": "Charging", "content_type": "procedure", "source_file": "m.pdf"}}


@pytest.fixture()
def chain(monkeypatch):
    c = PromptChain()
    c.reply = "Plug in the charger [C1]."
    monkeypatch.setattr(prompt_chain, "SUFFICIENCY_CHECK_ENABLED", False)
    monkeypatch.setattr(prompt_chain, "call_gemini", lambda *a, **k: c.reply)
    return c


def run(chain, **kw):
    res = {"chunks": [chunk()], "top_score": 6.0, "low_confidence": kw.pop("low", False)}
    return chain.run("how do I charge?", res, session_id="safety-test", **kw)


def test_chain_appends_notices_for_a_charging_answer(chain):
    out = run(chain, intent="charging", original_query="how do I charge")
    assert out["answer"].endswith(R4) and out["notices"] == ["technician"]


def test_chain_never_adds_notices_to_the_fallback(chain):
    chain.reply = FALLBACK_RESPONSE + "\n" + R6
    out = run(chain, intent="troubleshooting", original_query="battery fault")
    assert out["answer"] == FALLBACK_RESPONSE and out["fallback_used"] and out["notices"] == []


def test_fast_reject_has_no_notices(chain):
    res = {"chunks": [chunk(-6)], "top_score": -6.0, "low_confidence": True}
    out = chain.run("battery fault", res, session_id="s", intent="troubleshooting", original_query="battery fault")
    assert out["answer"] == FALLBACK_RESPONSE and out["notices"] == []


def test_low_confidence_prefix_goes_first_and_notices_last(chain):
    out = run(chain, intent="charging", original_query="charge", low=True)
    assert out["answer"].startswith(LOW_CONFIDENCE_PREFIX) and out["answer"].endswith(R4)


def test_memory_keeps_the_body_not_the_boilerplate(chain):
    run(chain, intent="charging", original_query="charge")
    stored = chain.memory.get_turns("safety-test")[-1].response
    assert "serviced by" not in stored
