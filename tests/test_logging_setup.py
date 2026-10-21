# tests/test_logging_setup.py
import json

from loguru import logger

from src import logging_setup


def test_import_creates_no_file_and_setup_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_setup, "_file_sink_id", None)
    target = tmp_path / "logs" / "ather_assistant.log"
    assert not target.exists()                       # nothing at import time

    logging_setup.setup_logging(target)
    logging_setup.setup_logging(target)              # second call must not add a second sink
    logger.info("hello once")
    logger.remove(logging_setup._file_sink_id)

    assert target.read_text(encoding="utf-8").count("hello once") == 1


def test_request_line_has_the_specified_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_setup, "_file_sink_id", None)
    target = tmp_path / "a.log"
    logging_setup.setup_logging(target)
    logging_setup.log_request(
        query="q ⚠️", intent="charging", rewritten_query="rq", retrieved_pages=[35, 36],
        rerank_score=7.9, confidence=0.9996, latency_ms=2300, response_length=120,
        gemini_calls=1, session_id="s1",
    )
    logger.remove(logging_setup._file_sink_id)

    line = next(l for l in target.read_text(encoding="utf-8").splitlines() if "REQUEST " in l)
    rec = json.loads(line.split("REQUEST ", 1)[1])
    for key in ("query", "intent", "rewritten_query", "retrieved_pages", "rerank_score",
                "latency_ms", "response_length"):
        assert key in rec
    assert rec["retrieved_pages"] == [35, 36] and rec["query"] == "q ⚠️"


def test_default_log_path_is_under_the_repo_logs_folder():
    assert logging_setup.LOG_FILE.name == "ather_assistant.log"
    assert logging_setup.LOG_FILE.parent.name == "logs"
