# # src/pipeline/pdf_extractor.py
# """
# PDF Extractor — Pure PyMuPDF (fitz)

# Extracts text directly from PDF pages using PyMuPDF's built-in text engine.
# No API calls. No rate limits. Runs locally and instantly.

# Output per page:
# {
#     "section_title":  str,
#     "content_type":   "feature" | "specification" | "warning" | "procedure" | "general",
#     "text":           str,
#     "page_number":    int,
#     "vehicle_model":  str,
#     "source_file":    str,
#     "has_warning":    bool,
#     "has_table":      bool,
#     "word_count":     int
# }

# PyMuPDF extraction strategy per page:
#   1. Try structured block extraction (preserves layout, headings, tables)
#   2. Fall back to raw text extraction if blocks are empty
#   3. Skip pages under MIN_WORD_COUNT (blank / decorative)
#   4. Classify content_type using keyword heuristics
#   5. Detect warnings and tables using pattern matching
# """

# import re
# import json
# from pathlib import Path
# from loguru import logger
# from typing import TypedDict, Any, List, Dict

# import fitz  # PyMuPDF

# # ── Config ────────────────────────────────────────────────────────────────────
# MIN_WORD_COUNT = 30     # Pages with fewer words are skipped (blank/decorative)
# DPI            = 150    # Used only if you later add image rendering — not used now


# # ── Content-type classifier ───────────────────────────────────────────────────
# # These keyword sets drive the heuristic content_type assignment.
# # Order matters: WARNING is checked first because it overrides everything else.

# WARNING_KEYWORDS = {
#     "warning", "caution", "danger", "do not", "never", "avoid",
#     "risk", "hazard", "electric shock", "fire", "injury", "death",
#     "important", "note", "attention"
# }

# PROCEDURE_KEYWORDS = {
#     "step", "steps", "press", "hold", "turn", "rotate", "connect",
#     "disconnect", "insert", "remove", "tap", "swipe", "charge",
#     "switch on", "switch off", "power on", "power off", "follow",
#     "proceed", "ensure", "check", "plug", "unplug"
# }

# SPECIFICATION_KEYWORDS = {
#     "specifications", "specs", "weight", "dimension", "capacity",
#     "voltage", "watt", "kilowatt", "kw", "kwh", "battery",
#     "range", "speed", "torque", "motor", "ip67", "ip rating",
#     "wheelbase", "ground clearance", "seat height", "tyre",
#     "mm", "kg", "nm", "rpm", "v ", "ah", "km/h", "kmph"
# }

# FEATURE_KEYWORDS = {
#     "feature", "function", "mode", "display", "screen", "dashboard",
#     "indicator", "light", "lamp", "mirror", "handle", "brake",
#     "throttle", "suspension", "ather", "connected", "app", "bluetooth"
# }


# def classify_content_type(text: str) -> str:
#     """
#     Classify page content into one of five types using keyword heuristics.

#     Why heuristics instead of ML?
#     - We have no labelled training data
#     - The keyword signal in Ather manuals is strong and consistent
#     - Zero compute cost, zero latency

#     Priority order: warning > procedure > specification > feature > general
#     """
#     text_lower = text.lower()

#     # Warning: highest priority — safety-critical content must never be mislabelled
#     warning_hits = sum(1 for kw in WARNING_KEYWORDS if kw in text_lower)
#     if warning_hits >= 2:
#         return "warning"

#     # Procedure: numbered steps or action verbs
#     procedure_hits = sum(1 for kw in PROCEDURE_KEYWORDS if kw in text_lower)
#     # Also check for numbered list patterns like "1. " "2. "
#     has_numbered_steps = bool(re.search(r'\b[1-9]\.\s+\w', text))
#     if procedure_hits >= 2 or (has_numbered_steps and procedure_hits >= 1):
#         return "procedure"

#     # Specification: units and technical numbers
#     spec_hits = sum(1 for kw in SPECIFICATION_KEYWORDS if kw in text_lower)
#     # Also check for patterns like "120 km/h", "3.7 kWh", "IP67"
#     has_spec_pattern = bool(re.search(
#         r'\d+\.?\d*\s*(km|kg|v|w|kw|kwh|nm|mm|rpm|ah|kmph|km/h)',
#         text_lower
#     ))
#     if spec_hits >= 2 or (has_spec_pattern and spec_hits >= 1):
#         return "specification"

#     # Feature: product feature descriptions
#     feature_hits = sum(1 for kw in FEATURE_KEYWORDS if kw in text_lower)
#     if feature_hits >= 2:
#         return "feature"

#     return "general"


# def detect_has_warning(text: str) -> bool:
#     """True if this page contains warning/caution language."""
#     text_lower = text.lower()
#     strong_warning_words = {
#         "warning", "caution", "danger", "do not", "never",
#         "hazard", "electric shock", "injury"
#     }
#     return any(w in text_lower for w in strong_warning_words)


# def detect_has_table(text: str) -> bool:
#     """
#     Heuristic table detection.
#     PyMuPDF often renders table cells with consistent spacing.
#     We look for multiple lines that contain tab characters or
#     multiple consecutive spaces (column separators).
#     """
#     lines = text.split("\n")
#     table_like_lines = 0
#     for line in lines:
#         # Tab-separated or multiple-space-separated columns
#         if "\t" in line or re.search(r'  {3,}', line):
#             table_like_lines += 1
#     return table_like_lines >= 3


# def extract_section_title(blocks: list, page_text: str) -> str:
#     """
#     Extract the most likely section title from a page.

#     Strategy:
#     1. Find the largest font-size text block (PyMuPDF block type 0 = text)
#     2. Fall back to the first non-empty line of the page text
#     3. Fall back to "Untitled Section"
#     """
#     largest_font  = 0
#     likely_title  = ""

#     for block in blocks:
#         if block.get("type") != 0:  # type 0 = text block
#             continue
#         for line in block.get("lines", []):
#             for span in line.get("spans", []):
#                 font_size = span.get("size", 0)
#                 span_text = span.get("text", "").strip()
#                 if font_size > largest_font and len(span_text) > 3:
#                     largest_font = font_size
#                     likely_title = span_text

#     if likely_title:
#         return likely_title[:100]   # cap at 100 chars

#     # Fallback: first non-empty line
#     for line in page_text.split("\n"):
#         line = line.strip()
#         if len(line) > 5:
#             return line[:100]

#     return "Untitled Section"


# def extract_text_from_blocks(page: fitz.Page) -> tuple[str, list, str]:
#     """
#     Extract text using PyMuPDF's structured block API.

#     Returns:
#         (clean_text, raw_blocks, section_title)

#     Block structure from PyMuPDF get_text("dict"):
#       page → blocks → lines → spans → text/font/size

#     Why structured blocks?
#     - Preserves reading order better than raw text dump
#     - Gives us font size for title detection
#     - Separates image blocks from text blocks
#     """
#     page_dict  = page.get_text("dict", sort=True)   # sort=True = reading order
#     raw_blocks = page_dict.get("blocks", [])

#     text_lines = []

#     for block in raw_blocks:
#         if block.get("type") != 0:  # skip image blocks (type 1)
#             continue

#         for line in block.get("lines", []):
#             line_text = ""
#             for span in line.get("spans", []):
#                 span_text = span.get("text", "").strip()
#                 if span_text:
#                     line_text += span_text + " "
#             line_text = line_text.strip()
#             if line_text:
#                 text_lines.append(line_text)

#     clean_text    = "\n".join(text_lines).strip()
#     section_title = extract_section_title(raw_blocks, clean_text)

#     return clean_text, raw_blocks, section_title


# def extract_text_fallback(page: fitz.Page) -> tuple[str, str]:
#     """
#     Fallback: raw text extraction when block extraction produces nothing.
#     Uses PyMuPDF's plain text mode.

#     Returns:
#         (clean_text, section_title)
#     """
#     raw_text = page.get_text("text", sort=True).strip()

#     # Clean up excessive whitespace
#     lines      = [line.strip() for line in raw_text.split("\n") if line.strip()]
#     clean_text = "\n".join(lines)

#     # Title = first non-trivial line
#     section_title = "Untitled Section"
#     for line in lines:
#         if len(line) > 5:
#             section_title = line[:100]
#             break

#     return clean_text, section_title


# def extract_page_data(
#     page: fitz.Page,
#     page_number: int,
#     vehicle_model: str,
#     source_file: str
# ) -> dict | None:
#     """
#     Extract all data from a single page and return a structured dict.

#     Returns None if the page should be skipped (blank / too short).
#     """

#     # ── Primary extraction: structured blocks ────────────────────────────────
#     text, raw_blocks, section_title = extract_text_from_blocks(page)

#     # ── Fallback: raw text if blocks gave nothing ────────────────────────────
#     if not text or len(text.split()) < 5:
#         text, section_title = extract_text_fallback(page)

#     # ── Skip check ───────────────────────────────────────────────────────────
#     word_count = len(text.split())
#     if word_count < MIN_WORD_COUNT:
#         logger.debug(
#             f"Page {page_number}: {word_count} words — below minimum, skipping"
#         )
#         return None

#     # ── Classify and detect ──────────────────────────────────────────────────
#     content_type = classify_content_type(text)
#     has_warning  = detect_has_warning(text)
#     has_table    = detect_has_table(text)

#     return {
#         "section_title": section_title,
#         "content_type":  content_type,
#         "text":          text,
#         "page_number":   page_number,
#         "vehicle_model": vehicle_model,
#         "source_file":   source_file,
#         "has_warning":   has_warning,
#         "has_table":     has_table,
#         "word_count":    word_count,
#     }


# def extract_pdf(
#     pdf_path: str,
#     output_dir: str,
#     vehicle_model: str = "Ather 450X"
# ) -> str:
#     """
#     Extract all pages from a PDF using PyMuPDF only.

#     Args:
#         pdf_path:      Path to the PDF file.
#         output_dir:    Directory where extracted JSON will be saved.
#         vehicle_model: Model name injected into every page's metadata.
#                        Used by dense_search for ChromaDB filtering.

#     Returns:
#         Path to the saved JSON file (used by run_ingestion.py).
#     """
#     pdf_path   = Path(pdf_path)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Output filenames match what chunker.py expects
#     output_file = output_dir / (pdf_path.stem + "_extracted.json")
#     failed_log  = output_dir / (pdf_path.stem + "_failed_pages.json")

#     doc   = fitz.open(str(pdf_path))
#     total = len(doc)

#     logger.info(f"Extracting: {pdf_path.name} | {total} pages | model: {vehicle_model}")

#     extracted_pages: list[dict] = []
#     failed_pages:    list[int]  = []
#     skipped_pages:   list[int]  = []

#     for page_index in range(total):
#         page_number = page_index + 1

#         try:
#             page      = doc[page_index]
#             page_data = extract_page_data(
#                 page         = page,
#                 page_number  = page_number,
#                 vehicle_model= vehicle_model,
#                 source_file  = pdf_path.name
#             )

#             if page_data is None:
#                 skipped_pages.append(page_number)
#             else:
#                 extracted_pages.append(page_data)
#                 logger.debug(
#                     f"  Page {page_number:>3}/{total} | "
#                     f"{page_data['content_type']:>13} | "
#                     f"{page_data['word_count']:>4} words | "
#                     f"{page_data['section_title'][:40]}"
#                 )

#         except Exception as e:
#             logger.error(f"  Page {page_number} failed: {e}")
#             failed_pages.append(page_number)

#     doc.close()

#     # ── Save extracted pages ─────────────────────────────────────────────────
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(extracted_pages, f, indent=2, ensure_ascii=False)

#     # ── Save failed pages log ────────────────────────────────────────────────
#     if failed_pages:
#         with open(failed_log, "w", encoding="utf-8") as f:
#             json.dump({"failed_pages": failed_pages}, f, indent=2)
#         logger.warning(f"  {len(failed_pages)} page(s) failed → {failed_log.name}")

#     logger.success(
#         f"Done: {pdf_path.name} → "
#         f"{len(extracted_pages)} pages extracted, "
#         f"{len(skipped_pages)} skipped, "
#         f"{len(failed_pages)} failed"
#         f" → {output_file.name}"
#     )

#     return str(output_file)


# # ── Quick test ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python pdf_extractor.py path/to/manual.pdf")
#         sys.exit(1)

#     result = extract_pdf(
#         pdf_path      = sys.argv[1],
#         output_dir    = "data/extracted",
#         vehicle_model = "Ather 450X"
#     )
#     print(f"\nSaved to: {result}")

#     # Print a sample page to verify structure
#     with open(result) as f:
#         pages = json.load(f)

#     if pages:
#         print(f"\nTotal pages extracted: {len(pages)}")
#         print("\nSample page (first):")
#         sample = pages[0]
#         print(f"  section_title : {sample['section_title']}")
#         print(f"  content_type  : {sample['content_type']}")
#         print(f"  vehicle_model : {sample['vehicle_model']}")
#         print(f"  source_file   : {sample['source_file']}")
#         print(f"  page_number   : {sample['page_number']}")
#         print(f"  word_count    : {sample['word_count']}")
#         print(f"  has_warning   : {sample['has_warning']}")
#         print(f"  has_table     : {sample['has_table']}")
#         print(f"  text preview  : {sample['text'][:200]}")

import re
import json
from pathlib import Path
from loguru import logger
from typing import TypedDict, List, Dict, Any

import fitz  # PyMuPDF


# ── TypedDicts for PyMuPDF structure ──────────────────────────────────────────
class Span(TypedDict, total=False):
    text: str
    size: float

class Line(TypedDict, total=False):
    spans: List[Span]

class Block(TypedDict, total=False):
    type: int
    lines: List[Line]


# ── Config ────────────────────────────────────────────────────────────────────
MIN_WORD_COUNT = 30


# ── Keyword sets ──────────────────────────────────────────────────────────────
WARNING_KEYWORDS = {
    "warning", "caution", "danger", "do not", "never", "avoid",
    "risk", "hazard", "electric shock", "fire", "injury", "death",
    "important", "note", "attention"
}

PROCEDURE_KEYWORDS = {
    "step", "steps", "press", "hold", "turn", "rotate", "connect",
    "disconnect", "insert", "remove", "tap", "swipe", "charge",
    "switch on", "switch off", "power on", "power off", "follow",
    "proceed", "ensure", "check", "plug", "unplug"
}

SPECIFICATION_KEYWORDS = {
    "specifications", "specs", "weight", "dimension", "capacity",
    "voltage", "watt", "kilowatt", "kw", "kwh", "battery",
    "range", "speed", "torque", "motor", "ip67", "ip rating",
    "wheelbase", "ground clearance", "seat height", "tyre",
    "mm", "kg", "nm", "rpm", "v ", "ah", "km/h", "kmph"
}

FEATURE_KEYWORDS = {
    "feature", "function", "mode", "display", "screen", "dashboard",
    "indicator", "light", "lamp", "mirror", "handle", "brake",
    "throttle", "suspension", "ather", "connected", "app", "bluetooth"
}


# ── Classifier ────────────────────────────────────────────────────────────────
def classify_content_type(text: str) -> str:
    text_lower = text.lower()

    if sum(1 for kw in WARNING_KEYWORDS if kw in text_lower) >= 2:
        return "warning"

    procedure_hits = sum(1 for kw in PROCEDURE_KEYWORDS if kw in text_lower)
    if procedure_hits >= 2 or (re.search(r'\b[1-9]\.\s+\w', text) and procedure_hits >= 1):
        return "procedure"

    spec_hits = sum(1 for kw in SPECIFICATION_KEYWORDS if kw in text_lower)
    if spec_hits >= 2 or re.search(r'\d+\.?\d*\s*(km|kg|v|w|kw|kwh|nm|mm|rpm|ah|kmph|km/h)', text_lower):
        return "specification"

    if sum(1 for kw in FEATURE_KEYWORDS if kw in text_lower) >= 2:
        return "feature"

    return "general"


def detect_has_warning(text: str) -> bool:
    return any(w in text.lower() for w in {
        "warning", "caution", "danger", "do not", "never",
        "hazard", "electric shock", "injury"
    })


def detect_has_table(text: str) -> bool:
    lines = text.split("\n")
    count = 0
    for line in lines:
        if "\t" in line or re.search(r'  {3,}', line):
            count += 1
    return count >= 3


# ── Section title extraction ──────────────────────────────────────────────────
def extract_section_title(blocks: List[Block], page_text: str) -> str:
    largest_font = 0
    likely_title = ""

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != 0:
            continue

        lines = block.get("lines", [])
        if not isinstance(lines, list):
            continue

        for line in lines:
            if not isinstance(line, dict):
                continue

            spans = line.get("spans", [])
            if not isinstance(spans, list):
                continue

            for span in spans:
                if not isinstance(span, dict):
                    continue

                font_size = span.get("size", 0)
                span_text = span.get("text", "").strip()

                if font_size > largest_font and len(span_text) > 3:
                    largest_font = font_size
                    likely_title = span_text

    if likely_title:
        return likely_title[:100]

    for line in page_text.split("\n"):
        line = line.strip()
        if len(line) > 5:
            return line[:100]

    return "Untitled Section"


# ── Extraction ────────────────────────────────────────────────────────────────
def extract_text_from_blocks(page: fitz.Page) -> tuple[str, List[Block], str]:
    page_dict = page.get_text("dict", sort=True)

    if not isinstance(page_dict, dict):
        return "", [], "Untitled Section"
    
    raw_blocks: List[Block] = page_dict.get("blocks", [])

    text_lines = []

    for block in raw_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != 0:
            continue

        lines = block.get("lines", [])
        if not isinstance(lines, list):
            continue

        for line in lines:
            if not isinstance(line, dict):
                continue

            spans = line.get("spans", [])
            if not isinstance(spans, list):
                continue

            line_text = ""
            for span in spans:
                if not isinstance(span, dict):
                    continue

                span_text = span.get("text", "").strip()
                if span_text:
                    line_text += span_text + " "

            if line_text.strip():
                text_lines.append(line_text.strip())

    clean_text = "\n".join(text_lines).strip()
    section_title = extract_section_title(raw_blocks, clean_text)

    return clean_text, raw_blocks, section_title


def extract_text_fallback(page: fitz.Page) -> tuple[str, str]:
    raw_text = page.get_text("text", sort=True)

    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    raw_text = raw_text.strip()
    
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    clean_text = "\n".join(lines)

    section_title = next((l[:100] for l in lines if len(l) > 5), "Untitled Section")
    return clean_text, section_title


def extract_page_data(
    page: fitz.Page,
    page_number: int,
    vehicle_model: str,
    source_file: str
) -> Dict[str, Any] | None:

    text, _, section_title = extract_text_from_blocks(page)

    if not text or len(text.split()) < 5:
        text, section_title = extract_text_fallback(page)

    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        logger.debug(f"Page {page_number}: skipped ({word_count} words)")
        return None

    return {
        "section_title": section_title,
        "content_type": classify_content_type(text),
        "text": text,
        "page_number": page_number,
        "vehicle_model": vehicle_model,
        "source_file": source_file,
        "has_warning": detect_has_warning(text),
        "has_table": detect_has_table(text),
        "word_count": word_count,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def extract_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    vehicle_model: str = "Ather 450X"
) -> str:

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (pdf_path.stem + "_extracted.json")
    failed_log = output_dir / (pdf_path.stem + "_failed_pages.json")

    doc = fitz.open(str(pdf_path))

    extracted_pages: List[Dict[str, Any]] = []
    failed_pages: List[int] = []
    skipped_pages: List[int] = []

    for i in range(len(doc)):
        try:
            page_data = extract_page_data(
                doc[i], i + 1, vehicle_model, pdf_path.name
            )
            if page_data:
                extracted_pages.append(page_data)
            else:
                skipped_pages.append(i + 1)
        except Exception:
            failed_pages.append(i + 1)

    doc.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_pages, f, indent=2, ensure_ascii=False)

    if failed_pages:
        with open(failed_log, "w", encoding="utf-8") as f:
            json.dump({"failed_pages": failed_pages}, f, indent=2)

    logger.success(f"Extracted {len(extracted_pages)} pages")

    return str(output_file)