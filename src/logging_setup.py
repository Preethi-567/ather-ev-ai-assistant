# src/logging_setup.py
"""
File logging for the API: logs/ather_assistant.log (loguru).

Two things go there:
  * every ordinary log line (INFO and above, plus the DEBUG lines of this app);
  * one structured line per request, written by log_request(), in the shape the
    project rules ask for: query, intent, rewritten_query, retrieved_pages,
    rerank_score, latency_ms, response_length (+ confidence, gemini_calls,
    session_id, which P5 needs for calibration).

The request line is "REQUEST " followed by ONE JSON object so it can be grepped
and parsed later:
    grep REQUEST logs/ather_assistant.log | cut -d' ' -f... | ...
    or: json.loads(line.split("REQUEST ", 1)[1])

setup_logging() is called from the API lifespan, not at import time, so merely
importing the module (tests, scripts) never creates a file. It is idempotent:
calling it twice does not duplicate lines.
"""

import json
import sys
from pathlib import Path

from loguru import logger

# Anchored to the repo root, not the current directory, so the file is the same
# wherever uvicorn is started from.
LOG_DIR  = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "ather_assistant.log"

_file_sink_id: int | None = None


def setup_logging(log_file: Path | str | None = None, level: str = "INFO") -> Path:
    """Add the rotating file sink once. Returns the path being written."""
    global _file_sink_id
    path = Path(log_file) if log_file else LOG_FILE

    if _file_sink_id is not None:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    _file_sink_id = logger.add(
        str(path),
        level=level,
        rotation="5 MB",
        retention=5,
        encoding="utf-8",        # the answers contain the warning sign emoji
        enqueue=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {message}",
    )
    return path


def log_request(
    *,
    query: str,
    intent: str,
    rewritten_query: str,
    retrieved_pages: list,
    rerank_score: float | None,
    confidence: float,
    latency_ms: int,
    response_length: int,
    gemini_calls: int,
    session_id: str,
) -> None:
    """One structured line per request."""
    record = {
        "query": query,
        "intent": intent,
        "rewritten_query": rewritten_query,
        "retrieved_pages": retrieved_pages,
        "rerank_score": rerank_score,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "response_length": response_length,
        "gemini_calls": gemini_calls,
        "session_id": session_id,
    }
    logger.info("REQUEST " + json.dumps(record, ensure_ascii=False))
