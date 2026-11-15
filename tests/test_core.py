"""agentcore: LLM layer (Gemini default; Bedrock Claude + OpenAI-compatible routes), structured output,
tracing, embeddings."""

import json
from types import SimpleNamespace as NS

import httpx
import numpy as np
import pytest
from pydantic import BaseModel

from agentcore import embed, evals, llm, rerank, trace
from agentcore.llm import RawResult


class Answer(BaseModel):
    value: int


def fake_backend(replies, finish=None):
    calls = []

    def call(model, prompt, system, json_schema, temperature, max_tokens):
        calls.append({"prompt": prompt, "schema": json_schema})
        return RawResult(text=replies[len(calls) - 1], model=model, input_tokens=10, output_tokens=5,
                         finish_reason=finish)

    return call, calls


def test_structured_output_and_trace(monkeypatch, tmp_traces):
    call, calls = fake_backend(['```json\n{"value": 7}\n```'])
    monkeypatch.setitem(llm.BACKENDS, "gemini", call)
    with trace.run("t") as run_id, trace.collect() as events:
        out = llm.generate("q", schema=Answer, agent="tester")
    assert out == Answer(value=7)
    assert calls[0]["schema"]["properties"]["value"]["type"] == "integer"
    assert events[0]["model"] == "test-flash" and events[0]["ok"]
    on_disk = [json.loads(l) for l in (tmp_traces / f"{run_id}.jsonl").read_text().splitlines()]
    assert on_disk[0]["name"] == "tester"


def test_retry_on_invalid_json(monkeypatch):
    call, calls = fake_backend(['{"value": "seven"}', '{"value": 7}'])
    monkeypatch.setitem(llm.BACKENDS, "gemini", call)
    assert llm.generate("q", schema=Answer).value == 7
    assert "did not match the required JSON schema" in calls[1]["prompt"]


def test_cut_off_reply_retries_without_constrained_decoding(monkeypatch):
    calls = []

    def call(model, prompt, system, json_schema, temperature, max_tokens):
        calls.append(json_schema)
        if json_schema:
            return RawResult(text='{"value": 7' + " " * 50, model=model, finish_reason="length")
        return RawResult(text='{"value": 7}', model=model)

    monkeypatch.setitem(llm.BACKENDS, "gemini", call)
    assert llm.generate("q", schema=Answer).value == 7
    assert calls[0] is not None and calls[1] is None


def test_gives_up_after_retries(monkeypatch):
    call, _ = fake_backend(["nope", "still nope"])
    monkeypatch.setitem(llm.BACKENDS, "gemini", call)
    with pytest.raises(llm.LLMError):
        llm.generate("q", schema=Answer)


def test_claude_route_request(monkeypatch):
    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            return NS(stop_reason="end_turn", stop_details=None, content=[NS(type="text", text='{"value": 42}')],
                      usage=NS(input_tokens=10, cache_read_input_tokens=90, cache_creation_input_tokens=0,
                               output_tokens=5))

    monkeypatch.setattr(llm, "_bedrock_client", lambda: NS(messages=FakeMessages()))
    with trace.collect() as events:
        out = llm._call_bedrock("anthropic.claude-haiku-5-5", "q", "sys", Answer.model_json_schema(), 0.0, 100)
    assert json.loads(out.text)["value"] == 42
    assert sent["system"] == "sys" and sent["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in sent and sent["output_config"]["format"]["type"] == "json_schema"
    assert out.input_tokens == 100 and out.cache_read_tokens == 90


def test_claude_refusal_raises(monkeypatch):
    resp = NS(stop_reason="refusal", stop_details=NS(category="cyber"), content=[], usage=None)
    monkeypatch.setattr(llm, "_bedrock_client", lambda: NS(messages=NS(create=lambda **kw: resp)))
    with pytest.raises(llm.LLMError, match="refused"):
        llm._call_bedrock("anthropic.test", "q", None, None, 0.0, 100)


def test_non_claude_uses_openai_compatible_route(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        assert req.url.path == "/v1/chat/completions" and body["model"] == "qwen.test"
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"value": 7}'}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    client = httpx.Client(base_url="https://bedrock-mantle.test/v1", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm, "_bedrock_openai_client", lambda: client)
    out = llm._call_bedrock("qwen.test", "q", "sys", Answer.model_json_schema(), 0.0, 100)
    assert json.loads(out.text) == {"value": 7} and out.input_tokens == 3


def test_hash_embeddings_are_normalised_and_deterministic():
    a = embed.embed(["Zakat certificate", "Zakat certificate", "ISO 27001"])
    assert a.shape == (3, 256)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert a[0] @ a[1] == pytest.approx(1.0) and a[0] @ a[2] < 0.5


def test_cohere_embedding_request(monkeypatch):
    seen = {}

    def invoke(model, body, region=None):
        seen.update(model=model, body=body)
        return {"embeddings": [[1.0, 0.0], [0.0, 2.0]]}, 7

    monkeypatch.setattr(embed, "_invoke", invoke)
    vecs, tokens = embed._bedrock("cohere.embed-multilingual-v3", ["a", "b"], "query")
    assert seen["body"]["input_type"] == "search_query" and vecs.shape == (2, 2) and tokens == 7


def test_rerank_completes_partial_orders():
    assert rerank._complete([2, 2, 9, 0], 4) == [2, 0, 1, 3]
    assert rerank.rerank("q", ["a", "b", "c"], backend="none") == [0, 1, 2]


def test_classification_report():
    r = evals.classification_report(["A", "A", "B", "B"], ["A", "B", "B", "ERROR"], ["A", "B"])
    assert r["accuracy"] == 0.5 and r["confusion"]["B"]["ERROR"] == 1


# ---------------------------------------------------------------- Gemini


class FakeGeminiModels:
    def __init__(self, resp=None):
        self.resp, self.sent = resp, {}

    def generate_content(self, model, contents, config):
        self.sent.update(model=model, contents=contents, config=config)
        return self.resp

    def embed_content(self, model, contents, config):
        self.sent.update(model=model, contents=contents, config=config)
        return NS(embeddings=[NS(values=[float(i + 1), 0.0], statistics=None) for i, _ in enumerate(contents)])


def _gemini_resp(text, finish="STOP", block=None):
    return NS(text=text, prompt_feedback=NS(block_reason=block) if block else None,
              candidates=[NS(finish_reason=NS(name=finish))],
              usage_metadata=NS(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30,
                                cached_content_token_count=0))


def test_gemini_request_and_usage(monkeypatch):
    models = FakeGeminiModels(_gemini_resp('{"value": 5}'))
    monkeypatch.setattr(llm, "_gemini_client", lambda: NS(models=models))
    monkeypatch.setitem(llm.BACKENDS, "gemini", llm._call_gemini)  # real backend code, fake client
    with trace.collect() as events:
        out = llm.generate("q", schema=Answer, system="sys", tier="pro", agent="judge")
    assert out.value == 5
    cfg = models.sent["config"]
    assert models.sent["model"] == "test-pro" and cfg.system_instruction == "sys"
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema["title"] == "Answer"
    assert cfg.thinking_config.thinking_level.name == "LOW"
    e = events[0]
    assert e["backend"] == "gemini" and e["input_tokens"] == 100 and e["output_tokens"] == 50  # thinking billed


def test_gemini_3_keeps_default_temperature(monkeypatch):
    models = FakeGeminiModels(_gemini_resp("hi"))
    monkeypatch.setattr(llm, "_gemini_client", lambda: NS(models=models))
    llm._call_gemini("gemini-3.8-flash", "q", None, None, 0.0, 100)
    assert models.sent["config"].temperature is None
    llm._call_gemini("gemini-2.5-flash", "q", None, None, 0.0, 100)
    assert models.sent["config"].temperature == 0.0


def test_gemini_blocked_and_cut_off(monkeypatch):
    monkeypatch.setattr(llm, "_gemini_client", lambda: NS(models=FakeGeminiModels(_gemini_resp("", "SAFETY"))))
    with pytest.raises(llm.LLMError, match="SAFETY"):
        llm._call_gemini("m", "q", None, None, 0.0, 100)
    monkeypatch.setattr(llm, "_gemini_client", lambda: NS(models=FakeGeminiModels(_gemini_resp("", block="OTHER"))))
    with pytest.raises(llm.LLMError, match="blocked the prompt"):
        llm._call_gemini("m", "q", None, None, 0.0, 100)
    assert llm._was_cut_off("MAX_TOKENS")


def test_gemini_embeddings_task_types(monkeypatch):
    from agentcore import config

    models = FakeGeminiModels()
    monkeypatch.setattr(llm, "_gemini_client", lambda: NS(models=models))
    vecs, _ = embed._gemini("gemini-embedding-001", ["a", "b"], "query")
    assert vecs.shape == (2, 2) and models.sent["config"].task_type == "RETRIEVAL_QUERY"
    assert models.sent["config"].output_dimensionality == config.settings().embed_dim
    embed._gemini("gemini-embedding-001", ["a"], "document")
    assert models.sent["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_missing_gemini_credentials_message(monkeypatch):
    import dataclasses

    from agentcore import config

    bare = dataclasses.replace(config.settings(), gemini_api_key="", gcp_project="")
    monkeypatch.setattr(llm, "settings", lambda: bare)
    llm._gemini_client.cache_clear()
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm._gemini_client()
    llm._gemini_client.cache_clear()


def test_gemini_429_waits_for_retry_delay_then_succeeds(monkeypatch):
    from google.genai import errors

    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "RequestsPerMinute"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}}
    calls = {"n": 0}

    class Models:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise errors.ClientError(429, body)
            return _gemini_resp("ok")

    monkeypatch.setattr(llm, "_gemini_client", lambda: NS(models=Models()))
    assert llm._call_gemini("m", "q", None, None, 0.0, 10).text == "ok"
    assert calls["n"] == 2 and 17 <= sleeps[0] <= 19


def test_gemini_daily_quota_fails_fast(monkeypatch):
    from google.genai import errors

    body = {"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                                "violations": [{"quotaId": "GenerateRequestsPerDayPerProject"}]}]}}

    class Models:
        def generate_content(self, model, contents, config):
            raise errors.ClientError(429, body)

    monkeypatch.setattr(llm, "_gemini_client", lambda: NS(models=Models()))
    with pytest.raises(llm.LLMError, match="daily quota"):
        llm._call_gemini("m", "q", None, None, 0.0, 10)


def test_budget_cap_stops_new_calls(monkeypatch):
    monkeypatch.setenv("LLM_BUDGET_USD", "0.002")
    monkeypatch.setattr(llm, "cost_usd", lambda *a: 0.001)
    llm.reset_spend()
    call, calls = fake_backend(['{"value": 1}'] * 5)
    monkeypatch.setitem(llm.BACKENDS, "gemini", call)
    llm.generate("q", schema=Answer)
    llm.generate("q", schema=Answer)
    with pytest.raises(llm.BudgetExceeded, match="budget"):
        llm.generate("q", schema=Answer)
    assert len(calls) == 2 and llm.spent_usd() == pytest.approx(0.002)
    llm.reset_spend()


def test_run_budget_is_per_block_and_shared_by_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import contextvars

    monkeypatch.delenv("LLM_BUDGET_USD", raising=False)
    monkeypatch.setattr(llm, "cost_usd", lambda *a: 0.01)
    call, _ = fake_backend(['{"value": 1}'] * 50)
    monkeypatch.setitem(llm.BACKENDS, "gemini", call)
    with llm.budget(0.03):
        with ThreadPoolExecutor(4) as pool:  # threads share the run's budget (context copied)
            outs = [pool.submit(contextvars.copy_context().run, lambda: llm.generate("q", schema=Answer))
                    for _ in range(3)]
            [f.result() for f in outs]
        with pytest.raises(llm.BudgetExceeded, match="run budget"):
            llm.generate("q", schema=Answer)
    with llm.budget(0.03):  # a new run starts from zero
        llm.generate("q", schema=Answer)
    llm.reset_spend()


def test_gemini_client_is_built_once_under_concurrency(monkeypatch):
    """Parallel graph tasks call _gemini_client() together; a second, uncached client gets garbage-collected
    mid-request and closes its connection."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time as _t

    built = []

    class SlowClient:
        def __init__(self, **kw):
            _t.sleep(0.05)  # widen the race window
            built.append(threading.get_ident())

    import google.genai as genai

    monkeypatch.setattr(genai, "Client", SlowClient)
    llm._gemini_client.cache_clear()
    try:
        with ThreadPoolExecutor(8) as pool:
            clients = list(pool.map(lambda _: llm._gemini_client(), range(16)))
        assert len(built) == 1 and len({id(c) for c in clients}) == 1
    finally:
        llm._gemini_client.cache_clear()
