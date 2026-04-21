# scripts/run_ingestion.py
"""
Full pipeline runner: PDF → Extract → Chunk → Ingest

Usage:
    python scripts/run_ingestion.py

Put all Ather PDF manuals in data/manuals/ before running.
"""

import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.pdf_extractor import extract_pdf
from src.pipeline.chunker import chunk_extracted_pages
from src.pipeline.ingestion import run_ingestion

logger.remove()
logger.add("logs/ingestion.log", rotation="10 MB", level="DEBUG")
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")


# ── Vehicle model mapping ─────────────────────────────────────────────────────
# Maps filename keywords → vehicle_model string written into chunk metadata.
# Edit this if you add more PDFs later.
FILENAME_TO_MODEL = {
    "450x":   "Ather 450X",
    "450s":   "Ather 450S",
    "rizta s": "Ather Rizta S",
    "rizta z": "Ather Rizta Z",
    "rizta":  "Ather Rizta",       # fallback if no S/Z in name
    "spec":   "Ather 450X",        # spec sheet is 450X
}

def detect_vehicle_model(filename: str) -> str:
    """
    Detect vehicle model from filename using keyword matching.
    Case-insensitive. Returns a human-readable model string.
    """
    name_lower = filename.lower()

    # Check specific combos first (order matters — 'rizta s' before 'rizta')
    for keyword, model in FILENAME_TO_MODEL.items():
        if keyword in name_lower:
            return model

    # Fallback — unknown PDF gets a generic label rather than crashing
    logger.warning(
        f"Could not detect vehicle model from filename '{filename}'. "
        f"Defaulting to 'Ather Unknown'. Edit FILENAME_TO_MODEL in run_ingestion.py."
    )
    return "Ather Unknown"


def main():
    manuals_dir   = Path("data/manuals")
    extracted_dir = Path("data/extracted")
    chunks_dir    = Path("data/chunks")

    # Create dirs if missing
    extracted_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    pdf_files = sorted(manuals_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {manuals_dir}/")
        logger.info("Place your Ather manual PDFs in data/manuals/ and re-run.")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

    # ── STEP 1: Extract all PDFs ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1: PDF Extraction (Gemini Vision)")
    logger.info("=" * 50)

    extraction_summaries = []

    for pdf_path in pdf_files:

        # Skip PDFs that are already extracted (safe re-run behaviour)
        expected_output = extracted_dir / f"{pdf_path.stem}_extracted.json"
        if expected_output.exists():
            logger.info(
                f"Skipping '{pdf_path.name}' — already extracted at "
                f"{expected_output.name}. Delete the file to re-extract."
            )
            continue

        vehicle_model = detect_vehicle_model(pdf_path.name)
        logger.info(f"\nProcessing : {pdf_path.name}")
        logger.info(f"Model tag  : {vehicle_model}")

        try:
            # ✅ Correct call — matches extract_pdf(pdf_path, output_dir, vehicle_model)
            summary = extract_pdf(
                pdf_path=str(pdf_path),
                output_dir=str(extracted_dir),
                vehicle_model=vehicle_model
            )
            extraction_summaries.append(summary)
            logger.info(f"Done: {summary}")

        except Exception as e:
            logger.error(f"Failed to extract '{pdf_path.name}': {e}")
            logger.info("Continuing with remaining PDFs...")

    # ── STEP 2: Chunk all extracted JSONs ─────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 2: Chunking")
    logger.info("=" * 50)

    for extracted_file in sorted(extracted_dir.glob("*_extracted.json")):

        # Skip if chunks already exist
        chunk_output = chunks_dir / f"{extracted_file.stem}_chunks.json"
        if chunk_output.exists():
            logger.info(
                f"Skipping '{extracted_file.name}' — chunks already exist. "
                "Delete the chunks file to re-chunk."
            )
            continue

        logger.info(f"\nChunking: {extracted_file.name}")
        try:
            chunks = chunk_extracted_pages(
                input_path=str(extracted_file),   # ✅ matches V2 chunker param name
                output_dir=str(chunks_dir)
            )
            logger.info(f"Created {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to chunk '{extracted_file.name}': {e}")

    # ── STEP 3: Ingest into ChromaDB + BM25 ──────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 3: Ingestion (ChromaDB + BM25)")
    logger.info("=" * 50)

    try:
        summary = run_ingestion(str(chunks_dir))
        logger.info(f"\n✅ Pipeline complete! Summary: {summary}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()