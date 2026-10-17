# tests/test_support_contacts.py
"""
The numbers in src/support_contacts.py are checked against the real manuals on
every run, so a typo here — or a manual that changes — fails the build instead of
putting a wrong number in front of a rider.
"""
import re
from pathlib import Path

import pytest

from src.support_contacts import SUPPORT_CONTACTS, resolve_support_contact

MANUALS = Path(__file__).resolve().parent.parent / "data" / "manuals"
EXACT_FALLBACK = (
    "I couldn't find that in your Ather manual. For this, I'd recommend "
    "contacting Ather Support or visiting your nearest Ather service centre."
)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s)


@pytest.mark.parametrize("model", sorted(SUPPORT_CONTACTS))
def test_number_is_printed_on_the_registered_page(model):
    fitz = pytest.importorskip("fitz")
    c = SUPPORT_CONTACTS[model]
    pdf = MANUALS / c.source_file
    if not pdf.exists():
        pytest.skip(f"{pdf} not present")
    with fitz.open(str(pdf)) as doc:
        text = _squash(doc[c.page - 1].get_text())
    assert c.number in text, f"{c.number} not found on {c.source_file} page {c.page}"


def test_only_models_with_a_printed_number_are_registered():
    assert set(SUPPORT_CONTACTS) == {"Ather 450X", "Ather 450S"}


@pytest.mark.parametrize("models, expected", [
    (["Ather 450X"],                      "+91 7676 811 777"),
    (["Ather 450S"],                      "+91 7676 811 777"),
    (["Ather 450X", "Ather 450S"],        "+91 7676 811 777"),
    (["Ather Rizta S"],                   None),
    (["Ather Rizta Z"],                   None),
    (["Ather 450S", "Ather Rizta Z"],     None),   # mixed → never borrow a number
    ([],                                  None),
    (["Ather Unknown"],                   None),
])
def test_resolution(models, expected):
    got = resolve_support_contact(models)
    assert (got.number if got else None) == expected


def test_fallback_sentence_is_exact_and_number_free():
    from src.generation.prompts import FALLBACK_RESPONSE, SYSTEM_PROMPT
    assert FALLBACK_RESPONSE == EXACT_FALLBACK
    assert EXACT_FALLBACK in SYSTEM_PROMPT          # rule 2 quotes it verbatim
    assert not re.search(r"\d{4}", FALLBACK_RESPONSE)
