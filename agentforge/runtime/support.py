"""
Runtime support imported by generated systems.

Generated code stays thin and declarative; the real plumbing (LLM client, tool
resolution, agent execution loop) lives here so it can be improved without
regenerating every system.
"""
from __future__ import annotations

from typing import Any

from ..tools.registry import get_tool


class _Response:
    """Minimal response object duck-typing LangChain's AIMessage."""
    def __init__(self, content: str):
        self.content = content


class _StructuredWrapper:
    def __init__(self, llm: "_OpenRouterLLM", schema):
        self._llm = llm
        self._schema = schema

    def invoke(self, messages):
        result = self._llm.invoke(messages)
        return self._schema.model_validate_json(result.content)


class _OpenRouterLLM:
    """Thin OpenAI-SDK wrapper with the LangChain interface used by AgentForge."""

    def __init__(self, model, api_key, base_url, temperature=0.1, tools=None):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._tools = tools or []
        self._client = None

    def _client_(self):
        if self._client is None:
            import httpx
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=5.0),
                max_retries=3,
            )
        return self._client

    def invoke(self, messages):
        normalized = []
        for m in messages:
            if isinstance(m, dict):
                normalized.append(m)
            else:
                role = getattr(m, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                normalized.append({"role": role, "content": getattr(m, "content", str(m))})

        kwargs: dict = {
            "model": self._model,
            "messages": normalized,
            "temperature": self._temperature,
        }
        if self._tools:
            kwargs["tools"] = self._tools

        resp = self._create_with_retry(kwargs)
        return _Response(resp.choices[0].message.content or "")

    def _create_with_retry(self, kwargs, max_attempts: int = 5):
        """Call the completion endpoint, retrying on rate limits (HTTP 429).

        Free/shared endpoints (and busy paid ones) return 429 with a
        Retry-After hint. The OpenAI SDK's own retry doesn't reliably honor
        provider-level 429s forwarded through OpenRouter, so we handle them
        here: sleep for the server-suggested delay (capped), with exponential
        backoff as a fallback, then re-raise if attempts are exhausted.
        """
        import time
        import openai

        for attempt in range(1, max_attempts + 1):
            try:
                return self._client_().chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                if attempt == max_attempts:
                    raise
                delay = self._retry_after_seconds(e)
                if delay is None:
                    delay = min(2 ** attempt, 30)  # 2,4,8,16,30 fallback
                delay = min(delay, 60) + 0.5  # cap + small jitter
                print(f"[agentforge] rate-limited (429); retrying in "
                      f"{delay:.0f}s (attempt {attempt}/{max_attempts})...")
                time.sleep(delay)

    @staticmethod
    def _retry_after_seconds(err) -> float | None:
        """Extract a retry delay from a RateLimitError, if the provider gave one."""
        # 1) Standard HTTP header.
        resp = getattr(err, "response", None)
        if resp is not None:
            hdr = None
            try:
                hdr = resp.headers.get("Retry-After")
            except Exception:
                hdr = None
            if hdr:
                try:
                    return float(hdr)
                except (TypeError, ValueError):
                    pass
        # 2) OpenRouter nests retry_after_seconds in error.metadata.
        body = getattr(err, "body", None)
        if isinstance(body, dict):
            meta = (body.get("error") or {}).get("metadata") or {}
            for key in ("retry_after_seconds", "retry_after_seconds_raw"):
                val = meta.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return None

    def bind_tools(self, tools):
        openai_tools = []
        for t in tools:
            schema = {}
            if hasattr(t, "args_schema") and t.args_schema is not None:
                schema = getattr(t.args_schema, "schema", lambda: {})()
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": getattr(t, "description", ""),
                    "parameters": schema,
                },
            })
        return _OpenRouterLLM(self._model, self._api_key, self._base_url,
                              self._temperature, openai_tools)

    def with_structured_output(self, schema):
        return _StructuredWrapper(self, schema)


def make_llm(
    # NOTE: ':free' uses OpenRouter's SHARED free pool and is throttled upstream
    # (frequent 429s, handled by retry below). For reliable throughput, add
    # credit at openrouter.ai and switch to the paid id 'qwen/qwen3-coder'
    # (drop ':free') here and in cli.py's --model default. To stay fully local
    # instead, pass provider='ollama' with a local model.
    model: str = "openai/gpt-oss-120b:free",
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.1,
    provider: str = "openrouter",
):
    """Return a chat model. Supports openrouter (default) and ollama."""
    import os

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to a .env file or export it as an env var."
        )
    return _OpenRouterLLM(model=model, api_key=api_key, base_url=base_url,
                          temperature=temperature)


def resolve_tools(names: list[str]) -> list:
    """Map tool names from the DSL to callables in the local registry."""
    tools = []
    for n in names:
        tool = get_tool(n)
        if tool is None:
            raise RuntimeError(
                f"tool '{n}' not found in registry; register it or remove from spec"
            )
        tools.append(tool)
    return tools


def build_messages(role: str, memory_kind: str, agent_id: str, state: dict) -> list:
    """Assemble the message list for this agent's LLM call."""
    system = {"role": "system", "content": f"You are an agent. Role: {role}"}
    history = state.get("messages", [])
    if memory_kind == "vector":
        # Placeholder hook: retrieve relevant context and prepend.
        # A real impl queries a local Chroma/LanceDB collection per agent_id.
        pass
    return [system, *history]


def run_agent(llm, tools: list, messages: list, state: dict) -> dict[str, Any]:
    """
    Execute one agent turn. Binds tools if present, invokes the model, and
    returns a normalized result. Router agents are expected to emit a JSON-ish
    'route' signal in their output, surfaced via state['scratch'].
    """
    model = llm.bind_tools(tools) if tools else llm
    response = model.invoke(messages)
    content = getattr(response, "content", str(response))

    # Lightweight route extraction: agents can signal control flow by emitting
    # a line like 'ROUTE: needs_revision'. Deterministic and easy to validate.
    route = None
    for line in str(content).splitlines():
        if line.strip().upper().startswith("ROUTE:"):
            route = line.split(":", 1)[1].strip()
            break

    return {
        "message": {"role": "assistant", "content": content},
        "output": {"text": content, "route": route},
    }