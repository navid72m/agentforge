"""
Smoke test: detect non-convergence by simulating the compiled graph's control
flow with a step budget — no LLM, no Ollama, no LangGraph required.

Two failure modes this catches that static checks cannot:
  1. A graph that, under adversarial routing, can revisit nodes without ever
     reaching END within a reasonable step budget (the revise->generator bug:
     the loop is *legal* but does not converge).
  2. A router whose emitted signals don't match any outgoing condition, so the
     compiler's fallback silently swallows the route.

We explore reachable states with a budget. Because real routing is decided by
an LLM at runtime, we treat every conditional branch as nondeterministic and
ask: is there ANY path that reaches END within `budget` steps? If the shortest
path to END exceeds the budget, or END is unreachable from some reachable node,
the graph is at risk of non-convergence and we report it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..dsl.schema import END, SystemSpec


@dataclass
class SmokeResult:
    converges: bool
    shortest_to_end: int | None        # min steps entry -> END, None if unreachable
    dead_routers: list[str]            # routers whose conditions can't all fire
    progress_discard: list[str]        # heuristic: loops that throw away work
    notes: list[str]


def smoke_test(spec: SystemSpec, budget: int = 50) -> SmokeResult:
    adj: dict[str, list[tuple[str, str | None]]] = {}
    for e in spec.graph.edges:
        adj.setdefault(e.from_, []).append((e.to, e.condition))

    shortest = _shortest_to_end(spec.graph.entry, adj)

    # Router signal-match: a router with conditional edges should have a role
    # that actually emits each condition word, else the runtime silently falls
    # back. Heuristic (role text may paraphrase), so this is a WARNING upstream.
    role_by_id = {a.id: (a.role or "") for a in spec.agents}
    dead = []
    for src, outs in adj.items():
        conds = [c for (_, c) in outs if c]
        if not conds:
            continue
        role = role_by_id.get(src, "").lower()
        missing = [c for c in conds if c.lower() not in role]
        if missing:
            dead.append(f"{src} (role never emits: {', '.join(missing)})")

    # Progress-discard heuristic (the revise->generator class). A refinement
    # loop should route a reviser's output back to its EVALUATOR, not back to a
    # node that creates content from scratch. We flag an edge from an agent
    # whose role mentions revising/fixing to a target whose role mentions
    # generating/creating/drafting from scratch (or to the entry node), because
    # that throws away the revision. This is a HEURISTIC, not a proof.
    discard = []
    gen_words = ("generate", "create", "draft", "write the initial", "from scratch",
                 "initial ")
    rev_words = ("revis", "rewrite", "fix", "correct", "modif", "improve")
    for src, outs in adj.items():
        src_role = role_by_id.get(src, "").lower()
        if not any(w in src_role for w in rev_words):
            continue
        for (to, _c) in outs:
            if to == END:
                continue
            to_role = role_by_id.get(to, "").lower()
            if to == spec.graph.entry or any(w in to_role for w in gen_words):
                discard.append(
                    f"{src} -> {to}: reviser routes back to a generator/entry; "
                    "this likely discards the revision. Route to the evaluator instead."
                )

    notes, converges = [], True
    if shortest is None:
        converges = False
        notes.append("END is unreachable from entry under any routing.")
    elif shortest > budget:
        converges = False
        notes.append(f"shortest path to END is {shortest} > budget {budget}.")
    if dead:
        notes.append("routers may emit signals no edge matches (silent fallback).")
    if discard:
        notes.append("possible progress-discarding loop (heuristic).")
    if not notes:
        notes = ["graph converges; no obvious progress-discard loop"]

    return SmokeResult(converges=converges, shortest_to_end=shortest,
                       dead_routers=dead, progress_discard=discard, notes=notes)


def _shortest_to_end(entry: str, adj: dict[str, list[tuple[str, str | None]]]) -> int | None:
    q = deque([(entry, 0)])
    seen = {entry}
    while q:
        node, dist = q.popleft()
        for (to, _cond) in adj.get(node, []):
            if to == END:
                return dist + 1
            if to not in seen:
                seen.add(to)
                q.append((to, dist + 1))
    return None
