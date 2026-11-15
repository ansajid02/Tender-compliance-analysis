"""The only way agent code talks to an LLM.

    from agentcore import llm
    verdict = llm.generate(prompt, schema=Verdict, tier="pro", agent="judge")

Backends (LLM_BACKEND in .env, default "gemini"):
- "gemini":  Google Gemini via the google-genai SDK. GEMINI_API_KEY -> Gemini API; otherwise
             GOOGLE_CLOUD_PROJECT -> Vertex AI. tier flash/pro -> GEMINI_MODEL_FLASH / GEMINI_MODEL_PRO.
- "bedrock": AWS Bedrock. anthropic.* ids -> Claude Messages API (AnthropicBedrockMantle); other ids
             (qwen.*, ...) -> Bedrock's OpenAI-compatible endpoint, same API key.
- schema: a Pydantic model -> JSON-schema-constrained output, validated, retried once on failure
- every call is traced (tokens, cost, latency) via agentcore.trace
- tests swap a backend: monkeypatch.setitem(llm.BACKENDS, "gemini", fn)
"""

from __future__ import annotations

import functools
import json
from contextlib import contextmanager
from contextvars import ContextVar
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, TypeVar, overload

import httpx
from pydantic import BaseModel, ValidationError

from . import trace
from .config import settings
from .pricing import cost_usd

T = TypeVar("T", bound=BaseModel)
Tier = Literal["flash", "pro"]


class LLMError(RuntimeError):
    pass


@dataclass
class RawResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    cost_usd: float | None = None
    cache_read_tokens: int = 0


# A backend takes (model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult
Backend = Callable[[str, str, str | None, dict | None, float, int | None], RawResult]

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def _has_aws_creds() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
                or (Path.home() / ".aws" / "credentials").exists() or (Path.home() / ".aws" / "config").exists())


def thread_safe_cache(fn):
    """lru_cache + lock. lru_cache alone lets parallel first calls build several clients; the extra ones are
    garbage-collected mid-request, which closes their HTTP connection ("client has been closed")."""
    cached = lru_cache(maxsize=8)(fn)
    lock = threading.Lock()

    @functools.wraps(fn)
    def wrapper(*args):
        with lock:
            return cached(*args)

    wrapper.cache_clear = cached.cache_clear
    return wrapper


# ---------------------------------------------------------------- Claude on Bedrock (Messages API)


@thread_safe_cache
def _bedrock_client():
    from anthropic import AnthropicBedrockMantle

    s = settings()
    if not s.bedrock_api_key and not _has_aws_creds():
        raise LLMError("AWS_BEARER_TOKEN_BEDROCK is not set in .env (and no AWS credentials found). "
                       "Create a Bedrock API key in the AWS console (Bedrock -> API keys) and paste it there.")
    # With no key, the SDK falls back to the normal AWS credential chain (IAM / SSO profile).
    return AnthropicBedrockMantle(aws_region=s.bedrock_region, api_key=s.bedrock_api_key or None,
                                  timeout=180.0, max_retries=4)


def _call_claude(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    import anthropic

    s = settings()
    kw: dict = {"model": model, "max_tokens": max_output_tokens or 4096,
                "messages": [{"role": "user", "content": prompt}]}
    if system:
        kw["system"] = system
        # The system prompt (agent instructions) is the same on every call of an agent: cache it.
        # Prefixes shorter than the model's minimum are silently not cached (no error, no cost).
        kw["cache_control"] = {"type": "ephemeral"}
    # temperature is not sent: current Claude models reject sampling parameters.
    # Thinking stays at the model default; effort bounds how much it thinks (and costs).
    output_config: dict = {}
    if s.bedrock_effort and s.bedrock_effort != "none":
        output_config["effort"] = s.bedrock_effort
    if json_schema:
        # transform_schema drops constraints the API can't enforce (pydantic still validates them)
        output_config["format"] = {"type": "json_schema", "schema": anthropic.transform_schema(json_schema)}
    if output_config:
        kw["output_config"] = output_config
    try:
        resp = _bedrock_client().messages.create(**kw)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise LLMError(f"Bedrock {e.status_code}: check AWS_BEARER_TOKEN_BEDROCK / AWS_REGION in .env and that "
                       f"model access for '{model}' is enabled in the Bedrock console. {str(e)[:300]}") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Bedrock 404 for model '{model}' in {s.bedrock_region}: {str(e)[:300]}") from e
    except anthropic.APIStatusError as e:  # 429 / 5xx were already retried by the SDK
        raise LLMError(f"Bedrock HTTP {e.status_code}: {str(e)[:500]}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Bedrock connection error: {e}") from e
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        raise LLMError(f"Bedrock: model refused (category={cat})")
    u = resp.usage
    cache_read = u.cache_read_input_tokens or 0
    return RawResult(
        text="".join(b.text for b in resp.content if b.type == "text"),
        model=model,
        input_tokens=(u.input_tokens or 0) + cache_read + (u.cache_creation_input_tokens or 0),
        output_tokens=u.output_tokens or 0,  # includes thinking tokens
        finish_reason=resp.stop_reason,
        cache_read_tokens=cache_read,
    )


# ---------------------------------------------------------------- other Bedrock models (OpenAI-compatible)


@thread_safe_cache
def _bedrock_openai_client() -> httpx.Client:
    s = settings()
    if not s.bedrock_api_key:
        raise LLMError("AWS_BEARER_TOKEN_BEDROCK is not set in .env (needed for non-Claude Bedrock models).")
    return httpx.Client(base_url=f"https://bedrock-mantle.{s.bedrock_region}.api.aws/v1",
                        headers={"Authorization": f"Bearer {s.bedrock_api_key}"}, timeout=httpx.Timeout(180.0))


def _call_openai_compatible(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    """Non-Claude Bedrock models (Qwen, Mistral, Gemma, ...) via Bedrock's OpenAI-compatible endpoint."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body: dict = {"model": model, "messages": messages, "temperature": temperature,
                  "max_tokens": max_output_tokens or 4096}
    if json_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": json_schema.get("title", "output"), "strict": False, "schema": json_schema}}
    delay = 2.0
    for attempt in range(5):
        try:
            r = _bedrock_openai_client().post("/chat/completions", json=body)
        except httpx.TransportError as e:
            if attempt == 4:
                raise LLMError(f"Bedrock connection error for '{model}': {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        if r.status_code not in _RETRY_STATUS or attempt == 4:
            break
        time.sleep(delay)
        delay = min(delay * 2, 30)
    if r.status_code != 200:
        raise LLMError(f"Bedrock HTTP {r.status_code} for '{model}': {r.text[:500]}")
    data = r.json()
    if not data.get("choices"):
        raise LLMError(f"Bedrock error for '{model}': {json.dumps(data)[:500]}")
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return RawResult(
        text=choice.get("message", {}).get("content") or "",
        model=model,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        finish_reason=choice.get("finish_reason"),
    )


def _call_bedrock(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    """anthropic.* -> Claude Messages API; every other Bedrock model -> OpenAI-compatible API."""
    if model.startswith("anthropic."):
        return _call_claude(model, prompt, system, json_schema, temperature, max_output_tokens)
    return _call_openai_compatible(model, prompt, system, json_schema, temperature, max_output_tokens)


# ---------------------------------------------------------------- Google Gemini (google-genai)

_GEMINI_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "MODEL_ARMOR"}
# Gemini counts thinking tokens against max_output_tokens: without headroom a short cap is spent on thinking
# and the reply has no text. Headroom costs nothing unless the model actually thinks that much.
_THINKING_HEADROOM = {"minimal": 1024, "low": 4096, "medium": 8192, "high": 16384}
_GEMINI_MAX_OUTPUT = 65536


@thread_safe_cache
def _gemini_client():
    from google import genai
    from google.genai import types

    s = settings()
    # 5xx / timeouts: the SDK retries with backoff. 429 is handled in _call_gemini (it honours Google's
    # retryDelay: free-tier per-minute windows are longer than the SDK's backoff).
    retry = types.HttpOptions(timeout=180_000, retry_options=types.HttpRetryOptions(  # timeout in ms
        attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[408, 500, 502, 503, 504]))
    if s.gemini_api_key:
        return genai.Client(api_key=s.gemini_api_key, http_options=retry)
    if s.gcp_project:  # Vertex AI: credentials from `gcloud auth application-default login`
        return genai.Client(enterprise=True, project=s.gcp_project, location=s.gcp_location, http_options=retry)
    raise LLMError("No Gemini credentials: set GEMINI_API_KEY in .env (https://aistudio.google.com/apikey), "
                   "or GOOGLE_CLOUD_PROJECT for Vertex AI.")


_rate_lock = threading.Lock()
_next_slot = 0.0
RATE_LIMIT_RETRIES = 6


def _throttle() -> None:
    """GEMINI_RPM > 0: space request starts evenly across all threads (free-tier keys have low limits)."""
    global _next_slot
    rpm = int(os.environ.get("GEMINI_RPM", "0").strip() or 0)
    if rpm <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + 60.0 / rpm
    time.sleep(max(0.0, start - now))


def _quota_info(e) -> tuple[float | None, bool]:
    """From a 429: (Google's suggested retry delay in s, is it a per-day quota?)."""
    delay, daily = None, False
    for d in ((getattr(e, "details", None) or {}).get("error", {}) or {}).get("details", []) or []:
        kind = str(d.get("@type", ""))
        if kind.endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
            delay = float(m.group(1)) if m else None
        if kind.endswith("QuotaFailure"):
            daily = any("PerDay" in str(v.get("quotaId", "")) for v in d.get("violations", []))
    return delay, daily


def _call_gemini(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    from google.genai import errors, types

    s = settings()
    headroom = _THINKING_HEADROOM.get(s.gemini_thinking, 16384)  # "" = model default: assume a lot
    kw: dict = {"system_instruction": system,
                "max_output_tokens": min((max_output_tokens or 4096) + headroom, _GEMINI_MAX_OUTPUT)}
    if not model.startswith("gemini-3"):
        kw["temperature"] = temperature  # Gemini 3 models are tuned for their default temperature
    if s.gemini_thinking:
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=s.gemini_thinking.upper())
    if json_schema:
        kw["response_mime_type"] = "application/json"
        kw["response_json_schema"] = json_schema
    try:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            _throttle()
            try:
                resp = _gemini_client().models.generate_content(model=model, contents=prompt,
                                                                 config=types.GenerateContentConfig(**kw))
                break
            except errors.ClientError as e:
                if e.code != 429 or attempt == RATE_LIMIT_RETRIES:
                    raise
                delay, daily = _quota_info(e)
                if daily:
                    raise LLMError(f"Gemini daily quota exhausted for '{model}' (free tier?). Wait until it resets, "
                                   "use another model, or enable billing on the key.") from e
                time.sleep((delay if delay is not None else min(60.0, 5.0 * 2 ** attempt)) + random.uniform(0, 2))
    except errors.ClientError as e:  # 4xx
        hint = {400: "bad request (model id? thinking level? schema?)", 401: "check GEMINI_API_KEY",
                429: "rate limit: set GEMINI_RPM in .env (free tier: ~10) or lower CONCURRENCY",
                403: "check GEMINI_API_KEY / Vertex permissions", 404: f"model '{model}' not found (`make models`)"}
        raise LLMError(f"Gemini {e.code}: {hint.get(e.code, '')} {str(e)[:400]}") from e
    except errors.APIError as e:  # 5xx after retries
        raise LLMError(f"Gemini HTTP {e.code}: {str(e)[:400]}") from e
    if resp.prompt_feedback and resp.prompt_feedback.block_reason:
        raise LLMError(f"Gemini blocked the prompt ({resp.prompt_feedback.block_reason})")
    finish = None
    if resp.candidates and resp.candidates[0].finish_reason is not None:
        finish = getattr(resp.candidates[0].finish_reason, "name", str(resp.candidates[0].finish_reason))
    if finish in _GEMINI_BLOCKED:
        raise LLMError(f"Gemini stopped the reply (finish_reason={finish})")
    um = resp.usage_metadata
    return RawResult(
        text=resp.text or "",
        model=model,
        input_tokens=(um.prompt_token_count or 0) if um else 0,
        # thinking tokens are billed as output tokens
        output_tokens=((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0,
        finish_reason=finish,
        cache_read_tokens=(um.cached_content_token_count or 0) if um else 0,
    )


BACKENDS: dict[str, Backend] = {"gemini": _call_gemini, "bedrock": _call_bedrock}


def _model_name(backend: str, tier: Tier, model: str | None) -> str:
    if model:
        return model
    if backend in ("gemini", "bedrock"):
        return settings().model_for(backend, tier)
    return f"{backend}-{tier}"  # test / custom backends


# ---------------------------------------------------------------- public API

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _was_cut_off(finish_reason: str | None) -> bool:
    """True if the reply stopped at the token limit (OpenAI 'length', Claude 'max_tokens', Gemini MAX_TOKENS)."""
    reason = str(finish_reason or "").lower()
    return reason == "length" or "max_tokens" in reason


def _clean_json(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text.strip()


# ---------------------------------------------------------------- spending caps
#
# 1. per run:     with llm.budget(0.50): ...   (one check / job; shared by its parallel threads, because
#                 LangGraph and our thread pools copy the context into worker threads)
# 2. per process: LLM_BUDGET_USD in the environment (CLI safety net)


class BudgetExceeded(LLMError):
    pass


class _Budget:
    def __init__(self, limit: float) -> None:
        self.limit, self.spent, self.lock = limit, 0.0, threading.Lock()


_budget_var: ContextVar[_Budget | None] = ContextVar("llm_budget", default=None)
_process_budget = _Budget(0.0)


@contextmanager
def budget(limit_usd: float | None):
    """Cap the traced LLM spend inside this block (None / 0 = no per-run cap)."""
    if not limit_usd:
        yield None
        return
    b = _Budget(float(limit_usd))
    token = _budget_var.set(b)
    try:
        yield b
    finally:
        _budget_var.reset(token)


def spent_usd() -> float:
    b = _budget_var.get()
    return (b or _process_budget).spent


def reset_spend() -> None:
    with _process_budget.lock:
        _process_budget.spent = 0.0


def _check_budget(agent: str) -> None:
    run = _budget_var.get()
    if run is not None and run.spent >= run.limit:
        raise BudgetExceeded(f"run budget of ${run.limit:.2f} reached (spent ${run.spent:.4f}); {agent} call not made")
    limit = float(os.environ.get("LLM_BUDGET_USD", "0").strip() or 0)
    if limit > 0 and _process_budget.spent >= limit:
        raise BudgetExceeded(f"LLM budget of ${limit:.2f} reached (spent ${_process_budget.spent:.4f}); "
                             f"{agent} call not made")


def _add_spend(cost: float | None) -> None:
    if not cost:
        return
    for b in (_budget_var.get(), _process_budget):
        if b is not None:
            with b.lock:
                b.spent += cost


@overload
def generate(prompt: str, schema: type[T], **kw) -> T: ...
@overload
def generate(prompt: str, schema: None = None, **kw) -> str: ...


def generate(
    prompt: str,
    schema: type[T] | None = None,
    *,
    system: str | None = None,
    tier: Tier = "flash",
    backend: str | None = None,
    model: str | None = None,
    agent: str = "unknown",
    temperature: float = 0.0,
    max_output_tokens: int | None = 4096,
    retries: int = 1,
) -> T | str:
    """Call an LLM. With `schema`, returns a validated instance of it."""
    backend = backend or settings().llm_backend
    if backend not in BACKENDS:
        raise LLMError(f"unknown backend '{backend}' (choose from {', '.join(BACKENDS)})")
    call = BACKENDS[backend]
    model_name = _model_name(backend, tier, model)
    json_schema = schema.model_json_schema() if schema else None
    current_prompt = prompt
    last_error: Exception | None = None
    # Some models under constrained JSON decoding (seen with Qwen on Bedrock) get stuck emitting
    # whitespace after a string value until max_tokens, so the JSON is cut off. Retrying the same
    # request repeats the loop, so after a cut-off we retry WITHOUT response_format, schema in the prompt.
    constrained = True

    for attempt in range(retries + 1):
        _check_budget(agent)
        with trace.span("llm", agent, backend=backend, model=model_name, tier=tier,
                        attempt=attempt, schema=schema.__name__ if schema else None,
                        constrained=constrained if schema else None) as sp:
            raw = call(model_name, current_prompt, system, json_schema if constrained else None,
                       temperature, max_output_tokens)
            cost = raw.cost_usd if raw.cost_usd is not None else cost_usd(backend, raw.model, raw.input_tokens,
                                                                           raw.output_tokens)
            _add_spend(cost)
            sp.update(
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                cache_read_tokens=raw.cache_read_tokens,
                cost_usd=cost,
                finish_reason=raw.finish_reason,
            )
            if schema is None:
                if not raw.text:
                    raise LLMError(f"Empty response (finish_reason={raw.finish_reason})")
                return raw.text
            try:
                return schema.model_validate_json(_clean_json(raw.text))
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                sp["schema_error"] = str(e)[:500]
                if _was_cut_off(raw.finish_reason):
                    constrained = False
        if constrained:
            current_prompt = (
                f"{prompt}\n\n---\nYour previous reply did not match the required JSON schema.\n"
                f"Error: {str(last_error)[:1500]}\nReply again with valid JSON only."
            )
        else:
            current_prompt = (
                f"{prompt}\n\n---\nReply with ONE compact JSON object, nothing else, that matches this "
                f"JSON schema:\n{json.dumps(json_schema, separators=(',', ':'))}"
            )

    raise LLMError(f"{agent}: no valid {schema.__name__} after {retries + 1} attempts: {last_error}")
