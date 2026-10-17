# src/generation/citations.py
"""
Deterministic citation rendering.

THE PROBLEM THIS REPLACES
-------------------------
The old validator asked the LLM to write "page 12" and then regex-checked the
number against the retrieved chunks. Two weaknesses:

  1. Page numbers are not unique. Five documents, four scooters: "page 20" is the
     tyre-pressure page in the 450X, 450S, Rizta S and Rizta Z guides alike
     (pages 18-20). A bare "page 20" could be checked as "exists" while pointing
     at the wrong scooter.
  2. It validated *after* generation. A wrong citation was repaired, not prevented.

THE DESIGN
----------
The model never sees page numbers and never writes them. Each retrieved chunk is
shown as a tag, [C1] .. [Cn]. The model cites by tag. This module maps a tag to the
chunk's (model, page) in code. A tag that is not in the retrieved set cannot map to
anything, so it is stripped. The only way a citation reaches the user is through a
chunk that was actually retrieved.

    LLM output:   "Front tyre pressure is 30 PSI [C1]."
    Rendered:     "Front tyre pressure is 30 PSI (Ather 450S manual, page 20)."

A second layer still guards free text: if the model writes "page N" / "pages N, M"
on its own despite the prompt, any N that is not a retrieved page is removed.
Valid-looking free-text pages are left alone (they cannot be attributed to a
model, so they are never added to `sources`).

Pure Python; no LLM or retrieval imports.
"""

import re
from dataclasses import dataclass

# What a citation looks like to the user. One line to change if the project
# prefers the literal "According to your … manual, page …" wording.
CITATION_TEMPLATE = "{model} manual, page {page}"

# A run of one or more tag groups:  [C1]  [C1][C3]  [C1, C3]  [C1] [C2]
_TAG_GROUP = r"\[\s*[Cc]\d+(?:\s*[,;/&]\s*[Cc]?\d+)*\s*\]"
_TAG_RUN = re.compile(rf"(?:{_TAG_GROUP}\s*)+")
_DIGITS = re.compile(r"\d+")

# Free-text page references:  "page 12", "pages 12, 14", "pages 12 and 14"
_FREE_PAGE_RUN = re.compile(
    r"\b[Pp]ages?\s+\d+(?:\s*(?:,|and|&|–|-)\s*\d+)*"
)

REMOVED_MARK = "[page reference removed]"


@dataclass(frozen=True)
class Source:
    """One verified citation target."""
    model: str
    page: int
    source_file: str

    def label(self) -> str:
        return CITATION_TEMPLATE.format(model=self.model, page=self.page)


def source_for_chunk(chunk: dict) -> Source | None:
    """Build a Source from a retrieved chunk; None if it lacks a page."""
    meta = chunk.get("metadata", {}) or {}
    page = meta.get("page_number")
    if page in (None, ""):
        return None
    return Source(
        model=str(meta.get("vehicle_model") or "Ather"),
        page=int(page),
        source_file=str(meta.get("source_file") or ""),
    )


def build_tag_map(chunks: list[dict]) -> dict[int, Source]:
    """
    Tag number → Source. Tag numbers are 1-based positions in `chunks`, matching
    the [C1].. headers written into the prompt by the context builder.
    A chunk with no page gets no entry, so a tag pointing at it is stripped.
    """
    tag_map: dict[int, Source] = {}
    for i, chunk in enumerate(chunks, 1):
        src = source_for_chunk(chunk)
        if src is not None:
            tag_map[i] = src
    return tag_map


def strip_invalid_free_text_pages(answer: str, chunks: list[dict]) -> tuple[str, list[int]]:
    """
    Layer 2. Remove any free-text page number that is not a retrieved page.
    Handles lists ("pages 12, 14"), which the old single-number regex missed.

    Returns (cleaned_answer, removed_page_numbers).
    """
    valid = {
        int(c["metadata"]["page_number"])
        for c in chunks
        if c.get("metadata", {}).get("page_number") not in (None, "")
    }
    removed: list[int] = []

    def _fix(m: re.Match[str]) -> str:
        nums = [int(n) for n in _DIGITS.findall(m.group(0))]
        bad = [n for n in nums if n not in valid]
        if not bad:
            return m.group(0)
        removed.extend(bad)
        return REMOVED_MARK

    return _FREE_PAGE_RUN.sub(_fix, answer), removed


def render_citations(
    answer: str,
    chunks: list[dict],
) -> tuple[str, list[Source], list[int]]:
    """
    Replace tag runs with rendered citations.

    Returns:
        rendered_answer
        sources           de-duplicated, in order of first appearance
        unknown_tags      tag numbers the model used that map to nothing
    """
    tag_map = build_tag_map(chunks)
    sources: list[Source] = []
    unknown: list[int] = []

    def _replace(m: re.Match[str]) -> str:
        run = m.group(0)
        trailing = run[len(run.rstrip()):]          # keep original spacing

        labels: list[str] = []
        seen_here: set[Source] = set()
        for n in (int(d) for d in _DIGITS.findall(run)):
            src = tag_map.get(n)
            if src is None:
                unknown.append(n)
                continue
            if src in seen_here:
                continue
            seen_here.add(src)
            labels.append(src.label())
            if src not in sources:
                sources.append(src)

        if not labels:
            return trailing                          # all tags invalid → drop them
        return f"({'; '.join(labels)})" + trailing

    rendered = _TAG_RUN.sub(_replace, answer)

    # Tidy: a dropped tag can leave "word ." or double spaces.
    rendered = re.sub(r"[ \t]+([.,;:!?])", r"\1", rendered)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    return rendered.strip(), sources, unknown
