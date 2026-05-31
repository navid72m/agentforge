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

        resp = self._client_().chat.completions.create(**kwargs)
        return _Response(resp.choices[0].message.content or "")

    def bind_tools(self, tools):
        openai_tools = []
        for t in tools:
            name, description, schema = self._tool_schema(t)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            })
        return _OpenRouterLLM(self._model, self._api_key, self._base_url,
                              self._temperature, openai_tools)

    @staticmethod
    def _tool_schema(t):
        """Derive (name, description, json_schema) from a tool that may be a
        LangChain BaseTool OR a plain registered function. The registry stores
        bare functions, so we introspect those rather than assuming .name etc."""
        # LangChain-style tool object.
        if hasattr(t, "name") and not callable(getattr(t, "name", None)):
            name = t.name
            description = getattr(t, "description", "") or ""
            schema = {"type": "object", "properties": {}}
            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None:
                fn = getattr(args_schema, "model_json_schema", None) or \
                     getattr(args_schema, "schema", None)
                if fn:
                    try:
                        schema = fn()
                    except Exception:
                        pass
            return name, description, schema

        # Plain function: introspect name, docstring, and signature.
        import inspect
        name = getattr(t, "__name__", "tool")
        description = (inspect.getdoc(t) or "").strip()
        props, required = {}, []
        try:
            sig = inspect.signature(t)
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                ann = param.annotation
                json_type = "string"
                if ann in (int, float):
                    json_type = "number"
                elif ann is bool:
                    json_type = "boolean"
                props[pname] = {"type": json_type}
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
        except (TypeError, ValueError):
            pass
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return name, description, schema

    def with_structured_output(self, schema):
        return _StructuredWrapper(self, schema)


def make_llm(
    model: str = "qwen/qwen3-coder:free",
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


def build_messages(role: str, memory_kind: str, agent_id: str, state: dict,
                   max_history: int = 8) -> list:
    """Assemble the message list for this agent's LLM call.

    Bounds the history to avoid unbounded context growth: in a cyclic graph,
    every node appends to state['messages'], so passing the full list to each
    call makes the prompt grow without limit and eventually exceeds the model's
    context window. We keep the FIRST message (the original input/task, which
    later agents still need) plus the most recent `max_history` messages, and
    drop the accumulated middle. This caps prompt size per call regardless of
    how many times the graph loops.
    """
    system = {"role": "system", "content": f"You are an agent. Role: {role}"}
    history = state.get("messages", [])
    if memory_kind == "vector":
        # Placeholder hook: retrieve relevant context and prepend.
        # A real impl queries a local Chroma/LanceDB collection per agent_id.
        pass
    if len(history) > max_history + 1:
        # Keep the original input + the most recent max_history messages.
        history = [history[0], *history[-max_history:]]
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