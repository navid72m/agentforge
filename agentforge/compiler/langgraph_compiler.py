"""
Deterministic DSL -> LangGraph code compiler.

NO LLM is used here. Given the same SystemSpec, this always emits byte-identical
Python. That property is what makes generated systems debuggable and trustworthy:
the LLM's creativity is confined to producing the DSL; turning DSL into code is
pure, testable machinery.

The generated module exposes `build_graph()` returning a compiled LangGraph app
with a SQLite checkpointer, plus a `__main__` block for quick local runs.
"""
from __future__ import annotations

from textwrap import indent

from ..dsl.schema import END, SystemSpec


def _detect_iterator(spec: SystemSpec) -> str | None:
    """Identify the iterator node: the one whose conditional-edge signals are
    exactly the loop-control pair continue/finish. Returns its id, or None for
    non-iterative (purely sequential/branching) systems. Detected structurally
    so the compiler can emit it as a DETERMINISTIC node (real queue logic in
    code) instead of an unreliable LLM 'guess continue or finish' agent."""
    by_source: dict[str, set] = {}
    for e in spec.graph.edges:
        if e.condition:
            by_source.setdefault(e.from_, set()).add(e.condition)
    for node, signals in by_source.items():
        if {"continue", "finish"} <= signals:
            return node
    return None


def _agent_node_fn(spec: SystemSpec, agent) -> str:
    model = agent.model or spec.llm.model
    tools_repr = ", ".join(repr(t) for t in agent.tools)

    # The iterator is emitted as DETERMINISTIC code: it pops the next item off
    # the queue and signals continue, or signals finish when the queue is empty.
    # No LLM call — this is what makes the loop terminate correctly instead of
    # relying on the model to guess continue/finish (which looped forever or
    # exited early in testing).
    if agent.id == _detect_iterator(spec):
        return f'''
def agent_{agent.id}(state: State) -> dict:
    """{agent.role}"""
    queue = list(state.get("queue", []))
    results = state.get("results", [])
    # Record the result of the item just processed (if any) before advancing.
    prev = state.get("current")
    if prev is not None:
        results = results + [{{"item": prev, "scratch": state.get("scratch", {{}})}}]
    if queue:
        nxt = queue.pop(0)
        return {{"queue": queue, "current": nxt, "results": results,
                "messages": [{{"role": "user", "content": str(nxt)}}],
                "scratch": {{"{agent.id}": {{"route": "continue"}}}},
                "last_agent": {agent.id!r}}}
    return {{"queue": [], "current": None, "results": results,
            "scratch": {{"{agent.id}": {{"route": "finish"}}}},
            "last_agent": {agent.id!r}}}
'''

    return f'''
def agent_{agent.id}(state: State) -> dict:
    """{agent.role}"""
    llm = make_llm(model={model!r})
    tools = resolve_tools([{tools_repr}])
    messages = build_messages(
        role={agent.role!r},
        memory_kind={agent.memory.value!r},
        agent_id={agent.id!r},
        state=state,
    )
    result = run_agent(llm, tools, messages, state)
    return {{"messages": (state["messages"] + [result["message"]])[-12:],
            "scratch": {{**state.get("scratch", {{}}), {agent.id!r}: result["output"]}},
            "last_agent": {agent.id!r}}}
'''


def _router_fns(spec: SystemSpec) -> str:
    """Emit one router per source node that has conditional edges."""
    by_source: dict[str, list] = {}
    for e in spec.graph.edges:
        by_source.setdefault(e.from_, []).append(e)

    out = []
    for source, edges in by_source.items():
        conditional = [e for e in edges if e.condition]
        if not conditional:
            continue
        # Map condition string -> target. The fallback (when no ROUTE signal is
        # parsed) prefers an unconditional edge; if all edges are conditional,
        # fall back to the FIRST conditional destination rather than END, so a
        # parse miss continues the flow instead of silently terminating the run.
        mapping = {e.condition: e.to for e in conditional}
        unconditional = [e for e in edges if not e.condition]
        if unconditional:
            default = unconditional[0].to
        else:
            default = conditional[0].to
        out.append(f'''
def route_{source}(state: State) -> str:
    decision = (state.get("scratch", {{}}).get({source!r}, {{}}) or {{}})
    signal = decision.get("route") if isinstance(decision, dict) else None
    mapping = {mapping!r}
    return mapping.get(signal, {default!r})
''')
    return "\n".join(out)


def _graph_wiring(spec: SystemSpec) -> str:
    lines = ["    g = StateGraph(State)"]
    for a in spec.agents:
        lines.append(f"    g.add_node({a.id!r}, agent_{a.id})")
    lines.append(f"    g.set_entry_point({spec.graph.entry!r})")

    by_source: dict[str, list] = {}
    for e in spec.graph.edges:
        by_source.setdefault(e.from_, []).append(e)

    for source, edges in by_source.items():
        has_conditional = any(e.condition for e in edges)
        if has_conditional:
            # The router returns a destination node name (a string). The path map
            # keys are those return values; the values are the actual graph
            # destinations (node-name string, or the END constant). CRITICAL: the
            # map must also include the router's DEFAULT fallback return, or a
            # fall-through (e.g. when no ROUTE signal is parsed) raises KeyError.
            conditional = [e for e in edges if e.condition]
            unconditional = [e for e in edges if not e.condition]
            default = unconditional[0].to if unconditional else END
            targets = set(e.to for e in edges)
            targets.add(default)  # ensure the fallback destination is routable
            pairs = []
            for t in sorted(targets, key=lambda x: (x == END, str(x))):
                value = "END" if t == END else repr(t)   # END constant vs node str
                pairs.append(f"{t!r}: {value}")
            lines.append(
                f"    g.add_conditional_edges({source!r}, route_{source}, "
                f"{{{', '.join(pairs)}}})"
            )
        else:
            for e in edges:
                tgt = "END" if e.to == END else repr(e.to)
                lines.append(f"    g.add_edge({source!r}, {tgt})")
    return "\n".join(lines)


END_CONST = "END"  # how END is referenced in generated wiring


def compile_spec(spec: SystemSpec) -> str:
    if spec.runtime.value != "langgraph":
        raise NotImplementedError(
            f"runtime '{spec.runtime.value}' not supported yet (langgraph only)"
        )

    nodes = "\n".join(_agent_node_fn(spec, a) for a in spec.agents)
    routers = _router_fns(spec)
    wiring = _graph_wiring(spec)

    header = f'''"""
AUTO-GENERATED by AgentForge. Do not edit by hand.
System: {spec.name}
Runtime: langgraph
Regenerate by editing the DSL and recompiling.
"""
from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agentforge.runtime.support import make_llm, resolve_tools, build_messages, run_agent


class State(TypedDict):
    messages: list   # bounded working context (NOT operator.add — see agents)
    scratch: dict
    last_agent: str
    queue: list      # remaining work items (iterative systems)
    current: object  # the item currently being processed
    results: list    # accumulated per-item results
'''

    builder = f'''
def build_graph(checkpoint_path: str = "agentforge/checkpoints/{spec.name}.sqlite",
                use_checkpointer: bool = False):
    """Build the compiled graph. Checkpointing is OFF by default: it persists
    full state on every super-step, which for long iterative runs bloats the
    SQLite store (and can exceed blob size limits). Enable it only when you need
    resumability, and prefer short per-item state when you do."""
{indent(wiring, "")}
    if use_checkpointer:
        import sqlite3
        conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        return g.compile(checkpointer=saver)
    return g.compile()
'''

    has_iterator = _detect_iterator(spec) is not None
    if has_iterator:
        runner = '''
def run(items, thread_id: str = "default", recursion_limit: int = 200) -> dict:
    """Process a list of work items. The iterator pops items deterministically
    and the loop terminates when the queue is empty. `recursion_limit` bounds
    total node steps; raise it for very large batches."""
    if isinstance(items, str):
        items = [items]
    app = build_graph()
    config = {"recursion_limit": recursion_limit}
    init = {"messages": [], "scratch": {}, "last_agent": "",
            "queue": list(items), "current": None, "results": []}
    return app.invoke(init, config=config)


if __name__ == "__main__":
    import sys
    items = sys.argv[1:] if len(sys.argv) > 1 else ["Hello"]
    out = run(items)
    print(f"processed {len(out.get('results', []))} item(s)")
    for r in out.get("results", []):
        print("-", r.get("item"))
'''
    else:
        runner = '''
def run(prompt: str, thread_id: str = "default") -> dict:
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    init = {"messages": [{"role": "user", "content": prompt}],
            "scratch": {}, "last_agent": "",
            "queue": [], "current": None, "results": []}
    return app.invoke(init, config=config)


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    out = run(prompt)
    print(out["messages"][-1])
'''

    return header + "\n" + nodes + "\n" + routers + "\n" + builder + "\n" + runner