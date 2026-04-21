"""
SMART CHUNKER V2 — Production Ready

Upgrades:
- Section-aware chunking
- Component extraction (for diagram pages)
- Better metadata for hybrid retrieval
- Embedding-ready fields
- Retrieval optimization (keywords + summaries)
"""

import json
import re
from pathlib import Path
from loguru import logger
from typing import List, Dict

# ── CONFIG ─────────────────────────────────────────
MAX_CHUNK_WORDS = 300
MIN_CHUNK_WORDS = 25
OVERLAP_SENTENCES = 2


# ── TEXT SPLITTER ─────────────────────────────────
def split_long_text(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks, current, count = [], [], 0

    for sentence in sentences:
        words = len(sentence.split())

        if count + words > MAX_CHUNK_WORDS and current:
            chunks.append(" ".join(current))

            # overlap
            current = current[-OVERLAP_SENTENCES:]
            count = sum(len(s.split()) for s in current)

        current.append(sentence)
        count += words

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── COMPONENT EXTRACTOR (🔥 IMPORTANT) ─────────────
def extract_components(text: str) -> List[Dict]:
    """
    Extract numbered components like:
    1 Rear view mirror
    2 Headlamp
    """

    components = []
    matches = re.findall(r'(\d+)\s+([A-Za-z0-9 \-]+)', text)

    for num, name in matches:
        components.append({
            "component_id": int(num),
            "component_name": name.strip()
        })

    return components


# ── KEYWORD EXTRACTOR ─────────────────────────────
def extract_keywords(text: str) -> List[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return list(set(words))[:20]  # top 20 keywords


# ── CHUNK CREATOR ─────────────────────────────────
def create_chunk(text, page_data, chunk_index, total_chunks):
    source = Path(page_data["source_file"]).stem

    components = extract_components(text)

    return {
        "chunk_id": f"{source}_p{page_data['page_number']}_c{chunk_index}",

        # 🔹 Core content
        "text": text.strip(),

        # 🔹 Retrieval helpers
        "keywords": extract_keywords(text),
        "summary": text[:200],  # simple summary (can upgrade later)

        # 🔹 Component awareness
        "components": components,
        "component_names": [c["component_name"] for c in components],

        # 🔹 Metadata
        "metadata": {
            "vehicle_model": page_data.get("vehicle_model"),
            "section": page_data.get("section_title"),
            "page_number": page_data.get("page_number"),
            "content_type": page_data.get("content_type"),
            "source_file": page_data.get("source_file"),

            "has_warning": page_data.get("has_warning", False),
            "has_table": page_data.get("has_table", False),
            "has_components": len(components) > 0,

            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "word_count": len(text.split())
        }
    }


# ── MAIN CHUNKER ─────────────────────────────────
def chunk_extracted_pages(input_path: str, output_dir: str):

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        word_count = len(text.split())
        content_type = page.get("content_type", "general")

        # ── STRATEGY ─────────────────────────────

        # 1️⃣ COMPONENT PAGES (🔥 highest priority)
        components = extract_components(text)
        if components:
            chunks_text = [text]

        # 2️⃣ WARNING (never split)
        elif content_type == "warning":
            chunks_text = [text]

        # 3️⃣ PROCEDURE
        elif content_type == "procedure":
            if word_count <= MAX_CHUNK_WORDS * 1.5:
                chunks_text = [text]
            else:
                chunks_text = split_long_text(text)

        # 4️⃣ SPECIFICATIONS
        elif content_type == "specification":
            chunks_text = [text]

        # 5️⃣ GENERAL
        else:
            if word_count <= MAX_CHUNK_WORDS:
                chunks_text = [text]
            else:
                chunks_text = split_long_text(text)

        # ── CREATE CHUNKS ────────────────────────
        for i, chunk_text in enumerate(chunks_text):

            if len(chunk_text.split()) < MIN_CHUNK_WORDS:
                continue

            chunk = create_chunk(
                chunk_text,
                page,
                i,
                len(chunks_text)
            )

            all_chunks.append(chunk)

    # ── SAVE ────────────────────────────────────
    output_file = Path(output_dir) / f"{Path(input_path).stem}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Created {len(all_chunks)} chunks → {output_file}")

    return all_chunks


# ── RUN ─────────────────────────────────────────
if __name__ == "__main__":
    chunk_extracted_pages(
        "data/extracted/sample_extracted.json",
        "data/chunks"
    )
