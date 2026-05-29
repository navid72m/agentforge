# AgentForge

A **local-first, low-code framework for generating agentic AI systems** from a
prompt or spec. You describe what you want; meta-agents design the system and
emit runnable **LangGraph** code. Everything runs locally on **Ollama** — no
cloud dependency.

> Runtime backend: **LangGraph only** for now. CrewAI is a planned second
> backend (the DSL and compiler are structured to add it without touching the
> meta-agents).

## The core idea: separate *generation* from *compilation*

```
User prompt
   ↓  (LLM, meta-agents)
DSL (SystemSpec)            ← humans can read & edit this; it's the contract
   ↓  (deterministic compiler — NO LLM)
LangGraph Python            ← reproducible, byte-identical for a given DSL
   ↓
Running agentic system (SQLite-checkpointed, resumable)
```

LLM creativity is confined to producing the DSL. Turning DSL into code is pure,
testable machinery — so generated systems are reproducible and debuggable.

## Autonomy level: 6 / 10

The meta-pipeline generates the full DSL on its own, but pauses at three
**human checkpoints** — `parse`, `architect`, and `resolve_tools` — where you
review (and optionally edit) the intermediate artifact before the pipeline
continues. Set `--no-review` to run fully hands-off.

## Pipeline stages

| Stage          | What it does                                          | Human checkpoint |
|----------------|-------------------------------------------------------|:---------------:|
| parse          | prompt → structured requirements + open questions     | ✅ |
| architect      | choose topology, agent roles, patterns                | ✅ |
| design         | concrete agents, prompts, graph wiring                | — |
| resolve_tools  | map capabilities to local tool registry, flag gaps    | ✅ |
| compile        | DSL → LangGraph (deterministic)                       | — |
| validate       | static checks + smoke test                            | — |

## Usage

```bash
# Deterministically compile a hand-written or generated DSL (no Ollama needed)
python -m agentforge.cli compile examples/research_assistant.yaml -o research_app.py

# Static-check a DSL
python -m agentforge.cli validate examples/research_assistant.yaml

# Generate a DSL from a prompt (requires local Ollama running)
python -m agentforge.cli generate "Build a system that researches a topic and fact-checks its own draft"

# Run a compiled system
python research_app.py "What is retrieval-augmented generation?"
```

## The DSL

```yaml
name: research_assistant
runtime: langgraph
llm: { provider: ollama, model: qwen2.5:14b }
agents:
  - id: researcher
    role: "Gather information and draft a summary."
    tools: [doc_loader]
    memory: vector
  - id: critic
    role: "Review the draft; emit 'ROUTE: approved' or 'ROUTE: needs_revision'."
graph:
  entry: researcher
  edges:
    - { from: researcher, to: critic }
    - { from: critic, to: researcher, condition: needs_revision }
    - { from: critic, to: END, condition: approved }
```

Control flow uses a lightweight signal: a router agent emits `ROUTE: <signal>`
and conditional edges match on it. Deterministic and easy to validate.

## Local-first stack

- **LLM:** Ollama (Qwen 2.5 / Llama 3.1 / DeepSeek) — `make_llm` in `runtime/support.py`
- **Orchestration:** LangGraph with explicit state graph + checkpointing
- **Persistence:** SQLite checkpointer (resumable threads)
- **Tools:** local registry (`tools/registry.py`); external tools must be
  explicitly registered and are flagged at the `resolve_tools` checkpoint
- **Vector memory:** hook in `build_messages` for embedded Chroma/LanceDB

## Layout

```
agentforge/
  dsl/         schema.py (the contract) + loader
  compiler/    langgraph_compiler.py (deterministic DSL → code)
  meta_agents/ pipeline.py (generation) + validator.py
  runtime/     support.py (LLM client, tool resolution, agent loop)
  tools/       registry.py (local-first tools)
  cli.py
examples/      research_assistant.yaml
tests/
```

## Extending to CrewAI later

Add a `crewai_compiler.py` alongside the LangGraph one and switch on
`spec.runtime`. The DSL, meta-agents, and validator stay unchanged — only the
compile step branches.
