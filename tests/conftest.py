# tests/conftest.py
"""
Test bootstrap.

1. Puts the repo root on sys.path so `import src...` works under plain `pytest`.
2. If sentence-transformers is not installed (e.g. a slim CI box), registers a
   stub so modules that import it at the top can still be imported. Tests never
   load a real model; they replace the searchers/rerankers with fakes. On a
   machine that HAS sentence-transformers, the real package is used untouched.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType("sentence_transformers")

    class _NoModel:
        def __init__(self, *a, **k):
            raise RuntimeError("sentence_transformers is stubbed in tests; inject a fake")

    stub.SentenceTransformer = _NoModel
    stub.CrossEncoder = _NoModel
    sys.modules["sentence_transformers"] = stub


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _log_file_in_tmp(tmp_path, monkeypatch):
    """
    Tests that run the API lifespan call setup_logging(); keep that file out of the
    repo's real logs/ folder, and remove the sink afterwards so tests stay isolated.
    """
    from loguru import logger
    from src import logging_setup

    monkeypatch.setattr(logging_setup, "LOG_FILE", tmp_path / "logs" / "ather_assistant.log")
    monkeypatch.setattr(logging_setup, "_file_sink_id", None)
    yield
    if logging_setup._file_sink_id is not None:
        try:
            logger.remove(logging_setup._file_sink_id)
        except ValueError:          # the test already removed its own sink
            pass
