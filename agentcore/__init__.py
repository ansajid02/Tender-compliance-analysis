"""agentcore: this project's copy of the agent-lab core (LLM layer, structured output, tracing, evals,
embeddings, reranking). Google Gemini by default, AWS Bedrock switchable. Each project has its own copy
so it can be deployed on its own."""

from . import embed, evals, llm, rerank, trace
from .config import settings

__all__ = ["embed", "evals", "llm", "rerank", "trace", "settings"]
