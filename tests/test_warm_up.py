# tests/test_warm_up.py
"""Startup warm-up: nothing may be left to load lazily on the first user's request."""
import asyncio

import pytest

import api.main as main
from src.retrieval import hybrid_search as hs


@pytest.fixture()
def fakes(monkeypatch):
    log: list[str] = []

    class FakeHybrid:
        def search(self, query, vehicle_models=None, content_type_filter=None):
            log.append(f"search:{query}")
            return {"chunks": []}

    monkeypatch.setattr(hs, "get_dense_searcher", lambda: log.append("dense"))
    monkeypatch.setattr(hs, "get_bm25_searcher", lambda: log.append("bm25"))
    monkeypatch.setattr(hs, "get_hybrid_searcher", lambda: (log.append("hybrid"), FakeHybrid())[1])
    return log


def test_warm_up_loads_every_component_and_runs_one_search(fakes):
    hs.warm_up()
    assert {"dense", "bm25", "hybrid"} <= set(fakes)
    assert fakes.count("search:warm up") == 1          # the full path ran once, unscoped


def _run_lifespan():
    async def go():
        async with main.lifespan(main.app):
            return "started"
    return asyncio.run(go())


def test_startup_calls_warm_up_once(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "get_prompt_chain", lambda: None)
    monkeypatch.setattr(main, "warm_up", lambda: calls.append("warm"))
    assert _run_lifespan() == "started" and calls == ["warm"]


def test_a_failed_warm_up_does_not_stop_the_api_from_starting(monkeypatch):
    """/health must stay reachable to report the problem."""
    def boom():
        raise RuntimeError("collection 'ather_manuals' does not exist")

    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "get_prompt_chain", lambda: None)
    monkeypatch.setattr(main, "warm_up", boom)
    assert _run_lifespan() == "started"
