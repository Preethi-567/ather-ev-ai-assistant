# tests/test_citations.py
from src.generation.citations import (
    Source, build_tag_map, render_citations, strip_invalid_free_text_pages,
)


def mk(model, page, file="x.pdf"):
    return {"metadata": {"vehicle_model": model, "page_number": page, "source_file": file}}


# The real-world ambiguity this module exists for: the tyre-pressure page
# is page 20 / 20 / 18 / 19 across four different scooters' documents.
TYRE_CHUNKS = [
    mk("Ather 450X", 20, "2025-Ather_450X-QSG.pdf"),
    mk("Ather 450S", 20, "450S_User_Manual_Jan_24.pdf"),
    mk("Ather Rizta S", 18, "Rizta S Quick Start Guide.pdf"),
    mk("Ather Rizta Z", 19, "Rizta Z Quick Start Guide.pdf"),
]


def test_same_page_number_resolves_to_the_right_scooter():
    out, sources, unknown = render_citations("Front 30 PSI [C2].", TYRE_CHUNKS)
    assert out == "Front 30 PSI (Ather 450S manual, page 20)."
    assert sources == [Source("Ather 450S", 20, "450S_User_Manual_Jan_24.pdf")]
    assert unknown == []


def test_per_model_sections_each_get_their_own_source():
    ans = "**Ather 450X** 30 PSI [C1]\n**Ather Rizta Z** 30 PSI [C4]"
    out, sources, _ = render_citations(ans, TYRE_CHUNKS)
    assert "(Ather 450X manual, page 20)" in out
    assert "(Ather Rizta Z manual, page 19)" in out
    assert [s.model for s in sources] == ["Ather 450X", "Ather Rizta Z"]


def test_adjacent_and_listed_tags_merge_into_one_parenthesis():
    for tags in ("[C1][C3]", "[C1] [C3]", "[C1, C3]", "[C1,C3]"):
        out, sources, _ = render_citations(f"Fact {tags}.", TYRE_CHUNKS)
        assert out == "Fact (Ather 450X manual, page 20; Ather Rizta S manual, page 18).", tags
        assert len(sources) == 2


def test_duplicate_citation_in_one_run_is_collapsed():
    out, sources, _ = render_citations("x [C1][C1]", TYRE_CHUNKS)
    assert out.count("page 20") == 1 and len(sources) == 1


def test_sources_dedupe_across_the_answer_and_keep_first_seen_order():
    _, sources, _ = render_citations("a [C3]. b [C1]. c [C3].", TYRE_CHUNKS)
    assert [s.model for s in sources] == ["Ather Rizta S", "Ather 450X"]


def test_hallucinated_tag_is_stripped_and_reported():
    out, sources, unknown = render_citations("Fact [C9].", TYRE_CHUNKS)
    assert out == "Fact." and sources == [] and unknown == [9]


def test_valid_and_invalid_tags_in_one_run():
    out, sources, unknown = render_citations("Fact [C1][C7].", TYRE_CHUNKS)
    assert out == "Fact (Ather 450X manual, page 20)." and unknown == [7]


def test_tag_zero_is_invalid():
    _, sources, unknown = render_citations("Fact [C0].", TYRE_CHUNKS)
    assert sources == [] and unknown == [0]


def test_chunk_without_page_cannot_be_cited():
    chunks = [{"metadata": {"vehicle_model": "Ather 450X"}}]
    assert build_tag_map(chunks) == {}
    out, sources, unknown = render_citations("Fact [C1].", chunks)
    assert sources == [] and unknown == [1]


def test_lowercase_tag_is_accepted():
    out, sources, _ = render_citations("Fact [c2].", TYRE_CHUNKS)
    assert len(sources) == 1 and sources[0].model == "Ather 450S"


def test_text_without_tags_is_untouched():
    txt = "I couldn't find that in your Ather manual."
    assert render_citations(txt, TYRE_CHUNKS) == (txt, [], [])


def test_ordinary_brackets_are_not_mistaken_for_tags():
    txt = "Press [Mode] then [OK], see [1] and [C] and [Cx]."
    assert render_citations(txt, TYRE_CHUNKS)[0] == txt


# ── layer 2: free-text page numbers ───────────────────────────────────────────

def test_free_text_list_is_checked_number_by_number():
    """The old regex saw nothing in 'pages 12, 14' — the exact format the v1.0
    prompt asked the model to use."""
    chunks = [mk("Ather 450X", 12)]
    out, removed = strip_invalid_free_text_pages("See pages 12, 14 for details.", chunks)
    assert removed == [14] and "[page reference removed]" in out


def test_free_text_valid_pages_are_left_alone():
    chunks = [mk("Ather 450X", 12), mk("Ather 450X", 14)]
    txt = "See pages 12, 14 and page 12."
    assert strip_invalid_free_text_pages(txt, chunks) == (txt, [])


def test_free_text_single_invalid_page():
    out, removed = strip_invalid_free_text_pages("Refer to Page 99.", [mk("Ather 450X", 12)])
    assert removed == [99] and out == "Refer to [page reference removed]."


def test_free_text_and_tag_pipeline_does_not_revalidate_rendered_citations():
    """Order matters: layer 2 runs on the raw text, THEN tags are rendered, so a
    rendered 'page 20' is never re-checked against the wrong scooter."""
    raw = "Pressure is 30 PSI [C2]."
    raw, removed = strip_invalid_free_text_pages(raw, TYRE_CHUNKS)
    out, _, _ = render_citations(raw, TYRE_CHUNKS)
    assert removed == [] and "page 20" in out
