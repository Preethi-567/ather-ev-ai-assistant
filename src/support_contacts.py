# src/support_contacts.py
"""
Verified support contacts, kept OUT of the fallback sentence on purpose.

The project's fallback sentence is fixed and must stay byte-for-byte identical:

    "I couldn't find that in your Ather manual. For this, I'd recommend
     contacting Ather Support or visiting your nearest Ather service centre."

A phone number is therefore never appended to it. Instead the API returns it in a
separate `support_contact` field, and only when ALL of these hold:

  1. the manual for that scooter prints a number (we do not infer one), and
  2. the number is registered below with its source file + page, and
  3. every model in play has the same verified number.

If any model in the answer has no verified number, the field is None. Absence is
safe; a wrong number on a safety/escalation message is not.

VERIFIED AGAINST THE PDFs (tests/test_support_contacts.py re-checks this on every
test run, so a manual update or a typo here fails CI instead of reaching a user):

    Ather 450X     2025-Ather_450X-QSG.pdf        page 49   +91 7676 811 777
    Ather 450S     450S_User_Manual_Jan_24.pdf    page 48   +91 7676 811 777

NOT included (no number in the manual):
    Ather Rizta S / Rizta Z — the guides point to "24x7 Roadside Assistance help
    line through Ather App" and a URL, but print no phone number. Adding the
    450-series number to a Rizta answer would be an ungrounded claim.

The 450X/450S manuals also say: "In case this number is unreachable, please
check for the updated number on atherenergy.com". Treat these entries as
needing a re-check whenever the manuals are replaced.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SupportContact:
    number: str
    source_file: str      # file under data/manuals/
    page: int             # 1-indexed page where the number is printed


SUPPORT_CONTACTS: dict[str, SupportContact] = {
    "Ather 450X": SupportContact(
        number="+91 7676 811 777",
        source_file="2025-Ather_450X-QSG.pdf",
        page=49,
    ),
    "Ather 450S": SupportContact(
        number="+91 7676 811 777",
        source_file="450S_User_Manual_Jan_24.pdf",
        page=48,
    ),
}


def resolve_support_contact(models: Iterable[str]) -> SupportContact | None:
    """
    Return a contact only if every model has a verified one and they agree.

    >>> resolve_support_contact(["Ather 450S"]).number
    '+91 7676 811 777'
    >>> resolve_support_contact(["Ather 450S", "Ather Rizta Z"]) is None
    True
    """
    wanted = {m for m in models if m}
    if not wanted:
        return None

    contacts = [SUPPORT_CONTACTS.get(m) for m in wanted]
    if any(c is None for c in contacts):
        return None

    numbers = {c.number for c in contacts if c is not None}
    if len(numbers) != 1:
        return None

    # Deterministic pick when several models share one number.
    first_model = sorted(wanted)[0]
    return SUPPORT_CONTACTS[first_model]
