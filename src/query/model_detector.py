# src/query/model_detector.py
"""
Scooter-model detection + per-session "sticky" model.

WHY THIS FILE EXISTS
--------------------
The index holds five documents covering four scooters:

    Ather 450X     (QSG + spec sheet)
    Ather 450S     (user manual)
    Ather Rizta S  (quick start guide)
    Ather Rizta Z  (quick start guide)

Before this module, /query never filtered by model, so a 450S question could be
answered from a Rizta chunk (and tyre-pressure pages exist, near-identically, in
four of the five documents). Retrieval is now *scoped*:

    explicit mention in the query   → filter to exactly those models
    else, model named earlier       → reuse it for this session ("sticky")
    else                            → no scope; caller retrieves per series

Resolution order is deliberate: an explicit mention always beats session state,
and session state always beats "ask both series".

OUTPUT SHAPE
------------
A scope is a list of GROUPS. Each group is the set of model tags that one
mention of the user's query maps to:

    "450X"          → [["Ather 450X"]]
    "Rizta"         → [["Ather Rizta S", "Ather Rizta Z"]]     (one group)
    "450X vs Rizta Z" → [["Ather 450X"], ["Ather Rizta Z"]]    (two groups)

Retrieval runs once per group so that one model cannot crowd another out of the
top-k (the "compare two scooters" case).

This module is pure Python (no ML imports) so it is cheap to unit-test.
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Literal

# ── Canonical model tags ──────────────────────────────────────────────────────
# These MUST match the `vehicle_model` strings written at ingestion time
# (scripts/run_ingestion.py → FILENAME_TO_MODEL).
M_450X    = "Ather 450X"
M_450S    = "Ather 450S"
M_RIZTA_S = "Ather Rizta S"
M_RIZTA_Z = "Ather Rizta Z"

ALL_MODELS = (M_450X, M_450S, M_RIZTA_S, M_RIZTA_Z)

# Series used when the user names no model at all. One retrieval per series.
SERIES_GROUPS: list[list[str]] = [
    [M_450X, M_450S],
    [M_RIZTA_S, M_RIZTA_Z],
]

# ── Mention patterns ──────────────────────────────────────────────────────────
# Order = specificity. A matched span is blanked out before the next pattern
# runs, so "Rizta Z" is never *also* counted as a bare "Rizta".
#
# Deliberate edge cases (all covered in tests/test_model_detector.py):
#   - "Rizta Zip"   : "Zip" is a riding mode, not the Z variant → \b after z
#   - "450 seconds" : must not read as 450S                      → no space before s
#   - "Ather450S"   : people type it without a space             → lookbehind (?<!\d),
#                     not \b (there is no word boundary between "r" and "4")
#   - "450 Plus"    : the spec sheet covers it but there is no 450 Plus manual;
#                     it maps to the 450X document set until P2 splits the
#                     spec table per model (see TODO below).
_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\brizta[\s\-]*z\b", re.I), [M_RIZTA_Z]),
    (re.compile(r"\brizta[\s\-]*s\b", re.I), [M_RIZTA_S]),
    (re.compile(r"(?<!\d)450[\s\-]?x\b", re.I),   [M_450X]),
    (re.compile(r"(?<!\d)450-?s\b", re.I),        [M_450S]),
    # TODO(P2): 450 Plus columns are currently tagged "Ather 450X" in the spec
    # sheet chunk. Until the table is split per model, answers about the
    # 450 Plus can only come from that mixed chunk.
    (re.compile(r"(?<!\d)450[\s\-]?(?:plus|\+)", re.I), [M_450X]),
    (re.compile(r"\brizta\b", re.I),         [M_RIZTA_S, M_RIZTA_Z]),
    (re.compile(r"(?<!\d)450(?!\d)", re.I),           [M_450X, M_450S]),
]


@dataclass(frozen=True)
class ModelScope:
    """Result of resolving which scooter(s) a query is about."""
    groups: tuple[tuple[str, ...], ...] = ()
    source: Literal["explicit", "session", "none"] = "none"

    @property
    def models(self) -> list[str]:
        """Flat, de-duplicated model list (stable order)."""
        seen: dict[str, None] = {}
        for g in self.groups:
            for m in g:
                seen.setdefault(m, None)
        return list(seen)

    @property
    def is_scoped(self) -> bool:
        return bool(self.groups)


def detect_models(text: str) -> list[list[str]]:
    """
    Return one group per mention, in order of first appearance.

    >>> detect_models("tyre pressure on the 450S?")
    [['Ather 450S']]
    >>> detect_models("450X vs Rizta Z boot space")
    [['Ather 450X'], ['Ather Rizta Z']]
    """
    if not text:
        return []

    work = text
    hits: list[tuple[int, list[str]]] = []

    for pattern, group in _PATTERNS:
        for m in pattern.finditer(work):
            hits.append((m.start(), list(group)))
        # Blank matched spans (same length keeps later offsets valid) so a more
        # specific mention is not re-matched by a more generic pattern.
        work = pattern.sub(lambda mm: " " * len(mm.group(0)), work)

    hits.sort(key=lambda h: h[0])

    groups: list[list[str]] = []
    for _, g in hits:
        if g not in groups:
            groups.append(g)
    return groups


# ── Session stickiness ────────────────────────────────────────────────────────
@dataclass
class SessionModelStore:
    """
    Remembers the last explicitly named scope for each session.

    RACE: two requests with the same session_id can be in flight at once (a
    double-submit, or two browser tabs sharing a session). Both may call set();
    the last writer wins. That is acceptable — a session is one human, and the
    worst case is that a follow-up inherits the other tab's model. The lock
    protects the dict itself so this stays safe if FastAPI endpoints are moved
    from `async def` to `def` (threadpool), where mutation really is concurrent.

    Not evicted: one small tuple per session, same lifetime policy as
    ConversationMemory._sessions. Revisit if sessions are ever persisted.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _scopes: dict[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)

    def get(self, session_id: str) -> tuple[tuple[str, ...], ...]:
        with self._lock:
            return self._scopes.get(session_id, ())

    def set(self, session_id: str, groups: tuple[tuple[str, ...], ...]) -> None:
        with self._lock:
            self._scopes[session_id] = groups

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._scopes.pop(session_id, None)


_store = SessionModelStore()


def get_session_model_store() -> SessionModelStore:
    return _store


def resolve_scope(
    original_query: str,
    rewritten_query: str,
    session_id: str,
    store: SessionModelStore | None = None,
    sticky: bool = True,
) -> ModelScope:
    """
    Decide which scooter(s) this query is about.

    Detection runs on BOTH texts. The original is authoritative (it is what the
    user typed). The rewritten query is consulted too because the rewriter
    resolves follow-ups ("what about the Z model?" → "...Ather Rizta Z...").
    Original mentions are taken first; rewritten-only mentions are appended.

    DESIGN NOTE (interview framing): the rewriter is an LLM, so it is allowed to
    *add* a model only when the original had none. If the user typed an explicit
    model, the rewriter cannot change it.
    """
    store = store or _store

    from_original = detect_models(original_query)
    if from_original:
        groups = tuple(tuple(g) for g in from_original)
        if sticky:
            store.set(session_id, groups)
        return ModelScope(groups=groups, source="explicit")

    from_rewritten = detect_models(rewritten_query)
    if from_rewritten:
        groups = tuple(tuple(g) for g in from_rewritten)
        if sticky:
            store.set(session_id, groups)
        return ModelScope(groups=groups, source="explicit")

    if sticky:
        remembered = store.get(session_id)
        if remembered:
            return ModelScope(groups=remembered, source="session")

    return ModelScope(groups=(), source="none")
