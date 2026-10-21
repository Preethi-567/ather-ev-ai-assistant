# Ather EV Assistant

A retrieval-augmented chatbot that answers Ather scooter owners' questions **only** from the official manuals in `data/manuals/`, with a model-and-page citation on every answer. Built on free, local tools plus the Gemini free tier.

## What it does

- Detects which scooter you mean (450X, 450S, Rizta S, Rizta Z) and searches that manual only.
- Hybrid retrieval: dense (BGE) + BM25, fused with RRF, reranked by a cross-encoder.
- Three-step answer chain: sufficiency check (optional), grounded generation, deterministic citations.
- If the manual does not contain the answer it says so with one fixed sentence instead of guessing.
- Safety notices (battery/charging/electrical, and "something is wrong") are appended by code, verbatim.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env        # then put your GEMINI_API_KEY in .env
python scripts/run_ingestion.py   # builds chroma_db/ and the BM25 corpus from data/manuals
```

## Run

```powershell
uvicorn api.main:app --port 8000        # API; logs to logs/ather_assistant.log
streamlit run frontend/app.py           # chat UI on http://localhost:8501
python -m pytest                        # unit tests (no network, no model downloads)
```

`RUN_REAL_MODEL_TESTS=1 python -m pytest` also runs the opt-in tests that load the real embedding and reranker models.

## Configuration

| Setting | Where | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | `.env` | required |
| `GEMINI_MAX_CALLS_PER_MIN` | `.env` | rolling-minute call budget (default 10) |
| `ATHER_SUFFICIENCY_CHECK` | `.env` | `0` skips the YES/NO check |
| `CONFIDENCE_FLOOR`, `NO_MATCH_CONFIDENCE`, `GROUP_MIN_CONFIDENCE` | `src/config.py` | confidence thresholds, 0-1 scale (the only place they are defined) |
| Gemini model chain | `src/generation/gemini_client.py` | `MODEL_FALLBACK_CHAIN` |

## API

`POST /query` · `POST /feedback` · `GET /analytics` · `GET /health`. `/query` returns the answer, `sources` (model, page, file), `confidence` (0-1), `notices`, `gemini_calls`, `low_confidence` and `fallback_used`.

## Design notes

Architecture decisions are in `docs/adr/` (model-scoped retrieval, citation by chunk tag, ranking text, latency and quota, safety notices and confidence).

## Status

Done: extraction/ingestion, hybrid retrieval, model scoping, generation chain, API, Streamlit chat and analytics. Planned: spec-sheet tables, evaluation set and RAGAS, voice, maintenance recommender.

## Limits

Answers come from the PDFs only; the assistant is not a diagnostic tool. Contact numbers are shown only where a manual prints one.
