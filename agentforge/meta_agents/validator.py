"""
Validator: static checks on the generated spec + a smoke-test hook.

Static checks here are belt-and-suspenders on top of the Pydantic validators in
schema.py — they produce human-readable diagnostics rather than raising, so the
pipeline can decide whether to loop back to a meta-agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..dsl.schema import END, SystemSpec
from ..tools.registry import available_tools


@dataclass
class Diagnostic:
    level: str   # "error" | "warning"
    message: str


def static_check(spec: SystemSpec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    tools = set(available_tools())

    for a in spec.agents:
        for t in a.tools:
            if t not in tools:
                diags.append(Diagnostic("error", f"agent '{a.id}' uses unknown tool '{t}'"))

    # Warn on agents with no inbound and no outbound (dead nodes are caught by
    # schema reachability, but a node that only loops to itself is suspicious).
    self_loops = [e.from_ for e in spec.graph.edges if e.from_ == e.to and e.to != END]
    for s in self_loops:
        diags.append(Diagnostic("warning", f"agent '{s}' has a self-loop; check for infinite cycles"))

    # Conditional edges should have at least one unconditional/fallback path.
    by_source: dict[str, list] = {}
    for e in spec.graph.edges:
        by_source.setdefault(e.from_, []).append(e)
    for src, edges in by_source.items():
        if any(e.condition for e in edges) and not any(not e.condition for e in edges):
            diags.append(Diagnostic(
                "warning",
                f"agent '{src}' has only conditional edges; add a fallback to avoid dead-ends",
            ))

    if not spec.success_criteria:
        diags.append(Diagnostic("warning", "no success_criteria; smoke test will be shallow"))

    # Backstop for the common architect failure: requirements describe a branch
    # (flag/route/reject/validate paths) but the graph is purely linear.
    branch_words = ("flag", "route", "reject", "invalid", "mismatch",
                    "otherwise", "branch", "depending")
    has_conditional = any(e.condition for e in spec.graph.edges)
    criteria_blob = " ".join(spec.success_criteria).lower()
    if not has_conditional and any(w in criteria_blob for w in branch_words):
        hits = sorted({w for w in branch_words if w in criteria_blob})
        diags.append(Diagnostic(
            "error",
            "success_criteria mention branching outcomes "
            f"({', '.join(hits)}) but the graph has no conditional edges — the "
            "failure/flag path is likely unreachable. Add a router agent and "
            "conditional edges.",
        ))

    # Trap-cycle detection: a cycle from which END is NOT reachable will loop
    # forever at runtime. The schema guarantees END is reachable from entry
    # globally, but a sub-cycle can still be a trap if no node in it has a path
    # out to END. Find strongly-connected components; flag any non-trivial SCC
    # with no edge leaving it.
    adj: dict[str, list[str]] = {}
    for e in spec.graph.edges:
        adj.setdefault(e.from_, []).append(e.to)

    def reaches_end(start: str) -> bool:
        seen, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n == END:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, []))
        return False

    # Nodes on a cycle that cannot reach END are traps.
    for node in list(adj):
        # node is on a cycle if it can reach itself in >=1 step
        seen, stack, on_cycle = set(), list(adj.get(node, [])), False
        while stack:
            n = stack.pop()
            if n == node:
                on_cycle = True
                break
            if n in seen or n == END:
                continue
            seen.add(n)
            stack.extend(adj.get(n, []))
        if on_cycle and not reaches_end(node):
            diags.append(Diagnostic(
                "error",
                f"agent '{node}' is on a cycle with no path to END — the graph "
                "would loop forever. Ensure the loop has a reachable exit.",
            ))

    return diags


def has_errors(diags: list[Diagnostic]) -> bool:
    return any(d.level == "error" for d in diags)
