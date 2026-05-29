# AgentForge

A **low-code framework for generating agentic AI systems** from a prompt or spec.
You describe what you want; meta-agents design the system and emit runnable
**LangGraph** code. The LLM runs through **OpenRouter** — no local GPU required.

> Runtime backend: **LangGraph only** for now. CrewAI is a planned second backend
> (the DSL and compiler are structured to add it without touching the meta-agents).

---

## The core idea: separate *generation* from *compilation*

```
User prompt
   ↓  (LLM via OpenRouter, meta-agents)
DSL (SystemSpec)            ← humans can read & edit this; it's the contract
   ↓  (deterministic compiler — NO LLM)
LangGraph Python            ← reproducible, byte-identical for a given DSL
   ↓
Running agentic system (SQLite-checkpointed, resumable)
```

LLM creativity is confined to producing the DSL. Turning DSL into code is pure,
testable machinery — so generated systems are reproducible and debuggable.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# 3. Generate a spec from a prompt
python -m agentforge.cli generate "Build a research assistant that reads docs and summarises them"

# 4. Validate the generated spec
python -m agentforge.cli validate research_assistant.yaml

# 5. Compile to a runnable LangGraph app
python -m agentforge.cli compile research_assistant.yaml -o research_app.py

# 6. Run it
python research_app.py "What is retrieval-augmented generation?"
```

---

## CLI reference

| Command | Description | Needs API key? |
|---------|-------------|:--------------:|
| `generate "<prompt>" [-o spec.yaml] [--model MODEL] [--no-review]` | Run meta-pipeline: prompt → DSL spec | ✅ |
| `compile <spec.yaml> [-o out.py]` | Deterministically compile a DSL to LangGraph Python | — |
| `validate <spec.yaml>` | Static-check a DSL (reachability, routing, tools) | — |

```bash
# Override the model (any OpenRouter model ID)
python -m agentforge.cli generate "..." --model qwen/qwen3-coder:free

# Skip human review checkpoints (fully autonomous)
python -m agentforge.cli generate "..." --no-review
```

---

## Pipeline stages

`generate` runs six stages automatically. Three pause for human review.

| Stage | What it does | Human checkpoint |
|-------|-------------|:----------------:|
| **parse** | prompt → structured requirements + open questions | ✅ |
| **architect** | choose topology, agent roles, patterns | ✅ |
| **design** | concrete agents, prompts, graph wiring | — |
| **resolve_tools** | map capabilities to local tool registry, flag gaps | ✅ |
| **compile** | DSL → LangGraph (deterministic) | — |
| **validate** | static checks + convergence smoke test | — |

At each checkpoint the full intermediate artifact is printed as YAML. Paste
edited YAML then press **Enter** twice to override, or just **Enter** to accept.

---

## The DSL

```yaml
name: research_assistant          # snake_case system identifier
runtime: langgraph

llm:
  provider: openrouter            # "openrouter" (default) or "ollama"
  model: qwen/qwen3-coder:free    # any model available on the provider
  temperature: 0.1
  base_url: https://openrouter.ai/api/v1

agents:
  - id: researcher                # snake_case node name
    role: "Gather information and draft a summary."
    tools: [doc_loader]           # names from the tool registry
    memory: vector                # none | buffer | vector
    model: null                   # optional per-agent model override

  - id: critic
    role: >
      Review the draft for accuracy and gaps.
      If revision is needed, emit 'ROUTE: needs_revision'.
      If the draft is good, emit 'ROUTE: approved'.
    memory: buffer

graph:
  entry: researcher
  edges:
    - { from: researcher, to: critic }
    - { from: critic, to: researcher, condition: needs_revision }
    - { from: critic, to: END,        condition: approved }

success_criteria:
  - "Final answer addresses the user's question"
  - "Critic approved the draft"
```

### DSL fields

**`llm`** (system-wide default; overridable per agent via `model:`)

| Field | Default | Notes |
|-------|---------|-------|
| `provider` | `openrouter` | `"openrouter"` or `"ollama"` |
| `model` | `qwen/qwen3-coder:free` | Any model ID valid for the provider |
| `temperature` | `0.1` | 0.0 – 2.0 |
| `base_url` | `https://openrouter.ai/api/v1` | Override for self-hosted or Ollama |

**`agents[]`**

| Field | Required | Notes |
|-------|:--------:|-------|
| `id` | ✅ | `[a-z][a-z0-9_]*` |
| `role` | ✅ | Natural-language role; becomes the system prompt seed |
| `tools` | — | Names from the tool registry (see below) |
| `memory` | — | `none` (default) \| `buffer` \| `vector` |
| `model` | — | Per-agent model override; falls back to `llm.model` |

**`graph`**

| Field | Notes |
|-------|-------|
| `entry` | Starting agent ID |
| `edges[].from` | Source agent ID |
| `edges[].to` | Target agent ID or `END` |
| `edges[].condition` | Routing signal (optional). Matches `ROUTE: <signal>` emitted by a router agent. The `ROUTE:` prefix is stripped automatically. |

### Graph validation (enforced at load time)

- No duplicate agent IDs
- `entry` must name a defined agent
- All edge sources/targets must be defined agents or `END`
- All agents must be reachable from `entry` (no orphans)
- At least one path to `END` (the graph must be able to terminate)

---

## Routing

Conditional control flow uses a lightweight signal convention:

```
Router agent output:  "... ROUTE: needs_revision ..."
Matching edge:        { from: critic, to: researcher, condition: needs_revision }
```

A router agent just emits a line starting with `ROUTE:` anywhere in its response.
The runtime scans for it and dispatches to the matching edge. The `ROUTE:` prefix
in a condition value is stripped automatically so both `needs_revision` and
`ROUTE: needs_revision` work as condition strings.

---

## Memory modes

| Mode | Behaviour |
|------|-----------|
| `none` | No persistent memory; each agent sees the shared message history |
| `buffer` | In-state conversation buffer (full message list in graph state) |
| `vector` | Hook in `build_messages` for local Chroma/LanceDB retrieval (wired but not yet fully implemented) |

---

## Tool registry

Tools are named capabilities that agents can call. Only registered tools can be
referenced in a spec; unknown tools are flagged at the `resolve_tools` checkpoint.

**Built-in tools**

| Name | Signature | What it does |
|------|-----------|-------------|
| `doc_loader` | `doc_loader(path: str) → str` | Reads and returns the contents of a local text/markdown file |
| `calculator` | `calculator(expression: str) → str` | Evaluates a safe arithmetic expression (AST-based; supports `+`, `-`, `*`, `/`, `**`, unary negation) |

**Adding a custom tool**

```python
# agentforge/tools/registry.py
from . import registry

@registry.register
def my_tool(arg: str) -> str:
    """Description shown to the model."""
    ...
```

Then reference it in the DSL: `tools: [my_tool]`.

---

## LLM provider configuration

### OpenRouter (default)

OpenRouter gives access to hundreds of models through one API key.

```bash
# .env
OPENROUTER_API_KEY=sk-or-...
```

The default model is `qwen/qwen3-coder:free` (free tier, rate-limited).
For reliable throughput drop `:free` and add credit at openrouter.ai:

```yaml
llm:
  provider: openrouter
  model: qwen/qwen3-coder   # paid — no upstream throttling
```

Free-tier 429s are handled automatically (up to 5 retries with the
server-suggested `Retry-After` delay).

### Ollama (local)

```yaml
llm:
  provider: ollama
  model: qwen2.5:14b
  base_url: http://localhost:11434
```

Requires a local Ollama instance. Pass `--model <name>` on the CLI too if using
`generate`.

---

## What the compiler generates

`compile` produces a single self-contained Python file:

```
<name>_app.py
├── State (TypedDict)          messages list + scratch dict + last_agent str
├── agent_<id>(state) → dict   one function per agent (LLM call + tool loop)
├── route_<id>(state) → str    one router per set of conditional edges
├── build_graph() → app        wires StateGraph, adds SQLite checkpointer
└── run(prompt, thread_id)     entry point; initialises state and invokes graph
```

Key properties:
- **No LLM during compilation** → output is byte-identical for the same DSL
- **SQLite checkpointing** is automatic; every run is resumable via `thread_id`
- **Tools are resolved at runtime** from the local registry, not hardcoded

---

## Validation and smoke tests

`validate` runs two layers of checks without calling any LLM:

**Static checks** (`validator.py`)
- Schema integrity (duplicate IDs, orphan agents, missing `END`)
- Conditional edge signals match router outputs
- All referenced tools exist in the registry

**Smoke tests** (`smoke.py`)
- Graph convergence (every path eventually reaches `END`)
- Dead-router detection (router emits a signal that no edge handles)
- Progress-discard warnings (cycles that could loop forever)

---

## Project layout

```
agentforge/
  cli.py              entry point (compile / validate / generate)
  dsl/
    schema.py         the DSL contract (Pydantic models)
    loader.py         YAML/JSON → validated SystemSpec
  compiler/
    langgraph_compiler.py   deterministic DSL → LangGraph code
  meta_agents/
    pipeline.py       6-stage generation pipeline + human checkpoints
    validator.py      static checks
    smoke.py          convergence + dead-router detection
  runtime/
    support.py        LLM client (_OpenRouterLLM), tool resolution, agent loop
  tools/
    registry.py       built-in tools (doc_loader, calculator)
examples/
  research_assistant.yaml
tests/
  test_core.py        deterministic-core tests (no LLM required)
```

---

## Extending to CrewAI later

Add a `crewai_compiler.py` alongside the LangGraph one and switch on
`spec.runtime`. The DSL, meta-agents, and validator stay unchanged — only the
compile step branches.
