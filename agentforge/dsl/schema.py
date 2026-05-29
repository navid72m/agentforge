"""
AgentForge DSL Schema.

This is the contract of the whole framework. Meta-agents PRODUCE instances of
SystemSpec (as YAML/JSON); the deterministic compiler CONSUMES them and emits
runnable LangGraph code. No LLM is involved in compilation, so output is
reproducible and debuggable.

Keep this schema strict: an invalid spec should fail loudly here, not at runtime
inside generated code.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Reserved node name for graph termination (maps to langgraph END).
END = "END"


class Runtime(str, Enum):
    langgraph = "langgraph"
    # crewai = "crewai"  # reserved for a future backend; not yet supported


class MemoryKind(str, Enum):
    none = "none"
    buffer = "buffer"   # in-state conversation buffer
    vector = "vector"   # local embedded vector store (Chroma/LanceDB)


class LLMConfig(BaseModel):
    provider: Literal["openrouter", "ollama"] = "openrouter"
    model: str = "qwen/qwen3-coder:free"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    base_url: str = "https://openrouter.ai/api/v1"


class AgentSpec(BaseModel):
    id: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    role: str  # natural-language role; becomes the system prompt seed
    tools: list[str] = Field(default_factory=list)
    memory: MemoryKind = MemoryKind.none
    # Optional per-agent model override; falls back to system-level llm.
    model: Optional[str] = None


class Edge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    # If condition is set, this is a conditional edge keyed on a router output.
    condition: Optional[str] = None

    model_config = {"populate_by_name": True}

    @field_validator("condition")
    @classmethod
    def _strip_route_prefix(cls, v: Optional[str]) -> Optional[str]:
        # The runtime matches on the BARE signal; a 'ROUTE:' prefix in the
        # condition is a common generation error that silently breaks matching.
        # Normalize it away so the edge matches what the router actually emits.
        if v is None:
            return v
        v = v.strip()
        if v.upper().startswith("ROUTE:"):
            v = v.split(":", 1)[1].strip()
        return v or None


class GraphSpec(BaseModel):
    entry: str
    edges: list[Edge]


class SystemSpec(BaseModel):
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    runtime: Runtime = Runtime.langgraph
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: list[AgentSpec]
    graph: GraphSpec
    # success_criteria drives the validator's smoke test.
    success_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_graph_integrity(self) -> "SystemSpec":
        agent_ids = {a.id for a in self.agents}
        if len(agent_ids) != len(self.agents):
            raise ValueError("duplicate agent ids")

        valid_targets = agent_ids | {END}
        if self.graph.entry not in agent_ids:
            raise ValueError(f"entry '{self.graph.entry}' is not a defined agent")

        # Reachability + dangling-reference checks.
        referenced: set[str] = set()
        for e in self.graph.edges:
            if e.from_ not in agent_ids:
                raise ValueError(f"edge source '{e.from_}' is not a defined agent")
            if e.to not in valid_targets:
                raise ValueError(f"edge target '{e.to}' is unknown")
            referenced.add(e.from_)

        # Every agent except pure sinks should be reachable from entry.
        reachable = self._reachable_from(self.graph.entry)
        orphans = agent_ids - reachable
        if orphans:
            raise ValueError(f"unreachable agents (orphans): {sorted(orphans)}")

        if END not in {e.to for e in self.graph.edges}:
            raise ValueError("graph has no path to END; it would never terminate")

        return self

    def _reachable_from(self, start: str) -> set[str]:
        adj: dict[str, list[str]] = {}
        for e in self.graph.edges:
            adj.setdefault(e.from_, []).append(e.to)
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in seen or node == END:
                continue
            seen.add(node)
            stack.extend(adj.get(node, []))
        return seen
