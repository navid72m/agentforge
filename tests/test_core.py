"""Tests for the deterministic core (no Ollama required)."""
import sys
import types

import pytest
from pydantic import ValidationError

from agentforge.compiler.langgraph_compiler import compile_spec
from agentforge.dsl.loader import load_spec_from_str
from agentforge.dsl.schema import SystemSpec
from agentforge.meta_agents.validator import has_errors, static_check

GOOD = """
name: demo
runtime: langgraph
agents:
  - id: planner
    role: plan
  - id: worker
    role: "do work; emit ROUTE done or ROUTE more"
graph:
  entry: planner
  edges:
    - { from: planner, to: worker }
    - { from: worker, to: planner, condition: more }
    - { from: worker, to: END, condition: done }
"""


def test_loads_and_validates():
    spec = load_spec_from_str(GOOD)
    assert spec.name == "demo"
    assert len(spec.agents) == 2


def test_compile_is_deterministic():
    spec = load_spec_from_str(GOOD)
    assert compile_spec(spec) == compile_spec(spec)


def test_generated_code_is_valid_python():
    import ast
    ast.parse(compile_spec(load_spec_from_str(GOOD)))


def test_rejects_no_path_to_end():
    bad = {"name": "x", "agents": [{"id": "a", "role": "r"}],
           "graph": {"entry": "a", "edges": [{"from": "a", "to": "a"}]}}
    with pytest.raises(ValidationError):
        SystemSpec.model_validate(bad)


def test_rejects_orphan():
    bad = {"name": "x",
           "agents": [{"id": "a", "role": "r"}, {"id": "b", "role": "r"}],
           "graph": {"entry": "a", "edges": [{"from": "a", "to": "END"}]}}
    with pytest.raises(ValidationError):
        SystemSpec.model_validate(bad)


def test_static_check_flags_unknown_tool():
    spec = SystemSpec.model_validate({
        "name": "x",
        "agents": [{"id": "a", "role": "r", "tools": ["nonexistent"]}],
        "graph": {"entry": "a", "edges": [{"from": "a", "to": "END"}]},
    })
    diags = static_check(spec)
    assert has_errors(diags)


def test_generated_graph_runs_with_stubbed_runtime():
    mod = types.ModuleType("agentforge.runtime.support")
    state = {"n": 0}

    def run_agent(llm, tools, messages, st):
        aid = st.get("last_agent", "")
        if aid == "planner" and "planner" in st.get("scratch", {}):
            state["n"] += 1
            route = "done" if state["n"] >= 2 else "more"
            return {"message": {"role": "assistant", "content": route},
                    "output": {"text": "w", "route": route}}
        return {"message": {"role": "assistant", "content": "p"},
                "output": {"text": "p", "route": None}}

    mod.make_llm = lambda model, **k: None
    mod.resolve_tools = lambda n: []
    mod.build_messages = lambda **k: []
    mod.run_agent = run_agent
    sys.modules["agentforge.runtime.support"] = mod

    import importlib.util
    code = compile_spec(load_spec_from_str(GOOD))
    path = "/tmp/_test_gen.py"
    open(path, "w").write(code)
    s = importlib.util.spec_from_file_location("_test_gen", path)
    g = importlib.util.module_from_spec(s)
    s.loader.exec_module(g)

    app = g.build_graph(":memory:")
    out = app.invoke(
        {"messages": [{"role": "user", "content": "go"}], "scratch": {}, "last_agent": ""},
        {"configurable": {"thread_id": "t"}},
    )
    assert out["last_agent"] == "worker"
    assert state["n"] == 2


def test_trap_cycle_is_error():
    from agentforge.dsl.schema import SystemSpec
    from agentforge.meta_agents.validator import static_check, has_errors
    spec = SystemSpec.model_validate({
        "name": "trap",
        "agents": [{"id": "entry", "role": "r"}, {"id": "a", "role": "r"}, {"id": "b", "role": "r"}],
        "graph": {"entry": "entry", "edges": [
            {"from": "entry", "to": "a"}, {"from": "a", "to": "b"},
            {"from": "b", "to": "a"}, {"from": "entry", "to": "END", "condition": "done"},
        ]},
    })
    assert has_errors(static_check(spec))


def test_progress_discard_heuristic():
    from agentforge.dsl.schema import SystemSpec
    from agentforge.meta_agents.smoke import smoke_test
    broken = SystemSpec.model_validate({
        "name": "refl", "agents": [
            {"id": "generator_agent", "role": "Generates the initial description."},
            {"id": "critique_agent", "role": "Evaluate. Emit ROUTE: needs_revision or ROUTE: ready."},
            {"id": "revision_agent", "role": "Modifies based on critique."},
        ],
        "graph": {"entry": "generator_agent", "edges": [
            {"from": "generator_agent", "to": "critique_agent"},
            {"from": "critique_agent", "to": "revision_agent", "condition": "needs_revision"},
            {"from": "critique_agent", "to": "END", "condition": "ready"},
            {"from": "revision_agent", "to": "generator_agent"},
        ]},
    })
    assert smoke_test(broken).progress_discard


def test_condition_route_prefix_normalized():
    from agentforge.dsl.schema import Edge
    assert Edge.model_validate({"from": "a", "to": "b", "condition": "ROUTE: clean"}).condition == "clean"


def test_structured_repair_loop_recovers():
    import json
    from agentforge.meta_agents.pipeline import _structured
    from agentforge.dsl.schema import SystemSpec

    class Resp:
        def __init__(self, c): self.content = c

    class Flaky:
        def __init__(self): self.n = 0
        def with_structured_output(self, schema): raise RuntimeError("unsupported")
        def invoke(self, msgs):
            self.n += 1
            if self.n == 1:
                return Resp(json.dumps({"name": "x", "agents": [{"id": "a", "role": "r"}]}))
            return Resp(json.dumps({"name": "x", "agents": [{"id": "a", "role": "r"}],
                                    "graph": {"entry": "a", "edges": [{"from": "a", "to": "END"}]}}))

    spec = _structured(Flaky(), SystemSpec, "sys", "input")
    assert spec.name == "x" and len(spec.graph.edges) == 1


def test_structured_raises_clean_after_retries():
    import json, pytest
    from agentforge.meta_agents.pipeline import _structured, StructuredGenerationError
    from agentforge.dsl.schema import SystemSpec

    class Resp:
        def __init__(self, c): self.content = c

    class Dead:
        def with_structured_output(self, schema): raise RuntimeError("nope")
        def invoke(self, msgs):
            return Resp(json.dumps({"name": "x", "agents": [{"id": "a", "role": "r"}]}))

    with pytest.raises(StructuredGenerationError):
        _structured(Dead(), SystemSpec, "s", "u", max_attempts=2)


def test_design_split_enforces_signal_ownership():
    """A worker emitting a loop-end signal must be rewritten to an unconditional
    edge back to the iterator — the structural fix for the ownership bug."""
    import json
    from agentforge.meta_agents.pipeline import _design_split

    class Resp:
        def __init__(self, c): self.content = c

    agents_json = json.dumps({
        "agents": [
            {"id": "iterator", "role": "Check queue. Emit ROUTE: continue or ROUTE: finish.", "tools": []},
            {"id": "worker", "role": "Do work.", "tools": []},
            {"id": "aggregator", "role": "Report.", "tools": []},
        ],
        "iterator_id": "iterator", "router_ids": [],
    })
    # Worker wrongly emits 'finish' -> must become unconditional edge to iterator.
    edges_json = json.dumps({
        "entry": "iterator",
        "edges": [
            {"from": "iterator", "to": "worker", "condition": "continue"},
            {"from": "iterator", "to": "aggregator", "condition": "finish"},
            {"from": "worker", "to": "aggregator", "condition": "finish"},
            {"from": "aggregator", "to": "END", "condition": None},
        ],
    })

    class TwoCall:
        def __init__(self): self.n = 0
        def with_structured_output(self, s): raise RuntimeError("manual")
        def invoke(self, msgs):
            self.n += 1
            return Resp(agents_json if self.n == 1 else edges_json)

    class R:
        goal = "Process a queue of items"
        success_criteria = ["x"]
        def model_dump(self): return {"goal": self.goal, "success_criteria": self.success_criteria}
    class A:
        def model_dump(self): return {"topology": "cyclic"}

    spec = _design_split(TwoCall(), R(), A())
    w = [e for e in spec.graph.edges if e.from_ == "worker"]
    assert len(w) == 1 and w[0].to == "iterator" and w[0].condition is None


def test_slug_no_trailing_underscore():
    from agentforge.meta_agents.pipeline import _slug
    assert not _slug("Process PDFs!!!   ").endswith("_")
    assert _slug("")  # non-empty fallback


def test_assembly_retry_recovers_from_orphan_edges():
    """Step-2 edges that orphan agents should trigger an edge-retry, not crash."""
    import json
    from agentforge.meta_agents.pipeline import _design_split

    class Resp:
        def __init__(self, c): self.content = c

    agents_json = json.dumps({
        "agents": [
            {"id": "iterator", "role": "Read queue. Emit ROUTE: continue / ROUTE: finish.", "tools": []},
            {"id": "worker", "role": "Work.", "tools": []},
            {"id": "agg", "role": "Report.", "tools": []},
        ],
        "iterator_id": "iterator", "router_ids": [],
    })
    broken = json.dumps({"entry": "iterator", "edges": [
        {"from": "iterator", "to": "agg", "condition": "finish"},
    ]})  # 'worker' orphaned, no continue path
    complete = json.dumps({"entry": "iterator", "edges": [
        {"from": "iterator", "to": "worker", "condition": "continue"},
        {"from": "iterator", "to": "agg", "condition": "finish"},
        {"from": "worker", "to": "iterator", "condition": None},
        {"from": "agg", "to": "END", "condition": None},
    ]})

    class Recovers:
        def __init__(self): self.n = 0
        def with_structured_output(self, s): raise RuntimeError("manual")
        def invoke(self, msgs):
            self.n += 1
            if self.n == 1: return Resp(agents_json)
            if self.n == 2: return Resp(broken)
            return Resp(complete)

    class R:
        goal = "Process a queue"; success_criteria = ["x"]
        def model_dump(self): return {"goal": self.goal, "success_criteria": self.success_criteria}
    class A:
        def model_dump(self): return {"topology": "cyclic"}

    spec = _design_split(Recovers(), R(), A())
    ids = {a.id for a in spec.agents}
    reached = {spec.graph.entry} | {e.to for e in spec.graph.edges}
    assert ids <= reached  # every agent reachable


def test_deterministic_assembly_survives_empty_step2():
    """With complete step-1 metadata (pipeline+routes+terminal), the graph is
    built deterministically and a useless step-2 cannot break it. This is the
    fix for the recurring orphaned-spine failure."""
    import json
    from agentforge.meta_agents.pipeline import _design_split

    class Resp:
        def __init__(self, c): self.content = c

    agents_json = json.dumps({
        "agents": [
            {"id": "iterator", "role": "Read queue. Emit ROUTE: continue / ROUTE: finish.", "tools": []},
            {"id": "parser", "role": "Parse.", "tools": []},
            {"id": "router", "role": "Decide. Emit ROUTE: a / ROUTE: b.", "tools": []},
            {"id": "handler_a", "role": "Handle a.", "tools": []},
            {"id": "handler_b", "role": "Handle b.", "tools": []},
            {"id": "report", "role": "Report.", "tools": []},
        ],
        "iterator_id": "iterator", "router_ids": ["router"],
        "pipeline": ["iterator", "parser", "router"],
        "routes": {"router": {"a": "handler_a", "b": "handler_b"}},
        "terminal_id": "report",
    })

    class LLM:
        def __init__(self): self.n = 0
        def with_structured_output(self, s): raise RuntimeError("m")
        def invoke(self, m):
            self.n += 1
            return Resp(agents_json if self.n == 1 else json.dumps({"entry": "iterator", "edges": []}))

    class R:
        goal = "Process a queue"; success_criteria = ["x"]
        def model_dump(self): return {"goal": self.goal, "success_criteria": self.success_criteria}
    class A:
        def model_dump(self): return {"topology": "cyclic"}

    spec = _design_split(LLM(), R(), A())
    ids = {a.id for a in spec.agents}
    reached = {spec.graph.entry} | {e.to for e in spec.graph.edges}
    assert ids <= reached
    # distinct handlers preserved
    targets = {e.to for e in spec.graph.edges if e.from_ == "router"}
    assert targets == {"handler_a", "handler_b"}
