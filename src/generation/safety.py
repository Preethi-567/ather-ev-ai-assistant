# src/generation/safety.py
"""
Safety and escalation notices, appended by CODE (not by the model).

WHY CODE AND NOT THE PROMPT
---------------------------
Rules 4 and 6 are "always" rules. A model that is asked to write a sentence
"always" will paraphrase it, drop it, or add it to answers that do not need it,
and a safety sentence that varies run to run cannot be tested. Whether the notice
is required is a property of the QUESTION (is it about the battery? does it
describe a fault?), so it is decided here from the question and appended
verbatim, once. The prompt (v1.2) tells the model not to write them.

TWO NOTICES (exact wording from the project rules)
    TECHNICIAN_NOTICE  Rule 4 - battery / charging / electrical / motor queries
    ESCALATION_NOTICE  Rule 6 - something not working, noise, safety concern

NOT APPLIED to the fixed fallback or out-of-scope reply. Those replies must stay
exactly as specified, and the fallback already sends the user to Ather Support.
"""

import re

TECHNICIAN_NOTICE = "⚠️ This should only be serviced by an Ather-certified technician."
ESCALATION_NOTICE = (
    "If this persists, please contact Ather Support or book a service via the Ather app."
)

# ── What the question is about ────────────────────────────────────────────────
# `charg\w*` covers charge/charging/charger; recharge and discharge are listed
# separately because \b does not match inside a word.
_ELECTRICAL = re.compile(
    r"\b(?:batter\w*|charg\w*|recharg\w*|discharg\w*|electric(?:al|ity)?|"
    r"motor|voltage|volts?|wiring|wires?|fuses?|short[- ]circuit\w*|bms)\b",
    re.IGNORECASE,
)

# Something wrong, not merely something to look up. Words that are common in
# ordinary questions ("sound", "hot", "slow") are left out on purpose: a notice
# on every harmless question teaches users to ignore it.
_SYMPTOM = re.compile(
    r"\b(?:not working|won'?t|doesn'?t|does not|isn'?t working|stopped|stuck|"
    r"noises?|noisy|beep\w*|buzz\w*|rattl\w*|grind\w*|vibrat\w*|"
    r"blink\w*|flicker\w*|warning|error|fault\w*|fail\w*|problem|issue|broken|"
    r"smell\w*|smok\w*|burn\w*|overheat\w*|leak\w*|fire|shock\w*|spark\w*|"
    r"swollen|dead|jerk\w*|unsafe|danger\w*|accident|crash\w*)\b",
    re.IGNORECASE,
)

_ESCALATION_INTENTS = {"troubleshooting", "safety"}


def needs_technician_notice(intent: str | None, *texts: str | None) -> bool:
    """Rule 4: battery, charging, electrical or motor question."""
    if intent == "charging":
        return True
    return any(t and _ELECTRICAL.search(t) for t in texts)


def needs_escalation_notice(intent: str | None, *texts: str | None) -> bool:
    """Rule 6: the question describes something wrong or a safety concern."""
    if intent in _ESCALATION_INTENTS:
        return True
    return any(t and _SYMPTOM.search(t) for t in texts)


# ── Remove notices the model wrote anyway ─────────────────────────────────────
# Deliberately NARROW. A real sentence from the manual such as "Battery
# replacement must be done by an Ather-certified technician [C2]." is an answer,
# not a notice, and must survive. Only (a) a line that starts with the warning
# sign and mentions a technician, and (b) the escalation boilerplate, which
# always begins "If this persists", are removed.
_MODEL_WARNING_LINE = re.compile(r"^[ \t]*⚠️?[^\n]*technician[^\n]*\n?", re.IGNORECASE | re.MULTILINE)
_MODEL_ESCALATION = re.compile(r"[ \t]*If this persists,[^\n]*?(?:Ather app|Ather Support)[^\n]*", re.IGNORECASE)


def strip_model_notices(answer: str) -> str:
    """Drop notices the model wrote itself, so the code adds each exactly once."""
    cleaned = _MODEL_WARNING_LINE.sub("", answer)
    cleaned = _MODEL_ESCALATION.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def append_notices(
    answer: str,
    intent: str | None,
    original_query: str | None = None,
    rewritten_query: str | None = None,
) -> tuple[str, list[str]]:
    """
    Return (answer with required notices, names of the notices added).

    The caller must NOT pass the fixed fallback or out-of-scope reply here.
    Order: technician notice first, escalation notice LAST ("always end with").
    """
    body = strip_model_notices(answer)
    texts = (original_query, rewritten_query)

    added: list[str] = []
    parts = [body]
    if needs_technician_notice(intent, *texts):
        parts.append(TECHNICIAN_NOTICE)
        added.append("technician")
    if needs_escalation_notice(intent, *texts):
        parts.append(ESCALATION_NOTICE)
        added.append("escalation")

    return "\n\n".join(parts), added
