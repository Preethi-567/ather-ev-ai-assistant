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
