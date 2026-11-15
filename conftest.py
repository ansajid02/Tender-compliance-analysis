"""Test isolation: no network, no .env influence, nothing written outside tmp dirs.

- agentcore settings pinned: Gemini backend (faked), hash embeddings, rerank off, traces in tmp
- tender data dir in tmp; vector store = in-memory Qdrant
- tests that need an LLM register a fake one: monkeypatch.setitem(llm.BACKENDS, "gemini", fake)
"""

import dataclasses

import pytest


@pytest.fixture(autouse=True)
def tmp_traces(tmp_path, monkeypatch):
    from agentcore import config, embed, llm, rerank, trace

    s = config.settings()
    patched = dataclasses.replace(s, trace_dir=tmp_path / "traces", llm_backend="gemini", embed_backend="hash",
                                  embed_model="hash", embed_dim=256, rerank_backend="none",
                                  gemini_model_flash="test-flash", gemini_model_pro="test-pro",
                                  gemini_api_key="test-key", gemini_thinking="low",
                                  bedrock_api_key="test-key")
    for mod in (config, trace, embed, rerank, llm):
        monkeypatch.setattr(mod, "settings", lambda: patched)

    def no_network(*a, **kw):
        raise AssertionError("a test tried to call a real LLM: register a fake backend")

    monkeypatch.setitem(llm.BACKENDS, "gemini", no_network)
    monkeypatch.setitem(llm.BACKENDS, "bedrock", no_network)
    return tmp_path / "traces"


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    from tender import config, store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "data" / "projects")
    s = store.Store(client=QdrantClient(":memory:"), collection="test")
    store.set_store(s)
    yield s
    store.set_store(None)
