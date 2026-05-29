"""
Meta-agent pipeline: user prompt -> validated SystemSpec (DSL).

Autonomy level 6/10: the pipeline generates the full DSL on its own, but pauses
at defined HUMAN CHECKPOINTS where a person can review/edit before the next
stage. The checkpoints are returned as structured objects so a UI (or CLI) can
present them; nothing forces interactive blocking in code.

Each meta-agent constrains the LLM with a Pydantic schema (structured output),
so stages emit valid intermediate artifacts rather than free text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..dsl.schema import END, SystemSpec
from ..tools.registry import available_tools


# --- Intermediate artifacts between stages -------------------------------

class Requirements(BaseModel):
    goal: str
    capabilities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class Topology(str, Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"
    cyclic = "cyclic"


class Architecture(BaseModel):
    topology: Topology
    rationale: str
    agent_roles: list[str]
    patterns: list[str] = Field(default_factory=list)  # reflection, planning, ...


# --- Split-design intermediate schemas (flatter = easier for small models) ---

class _AgentDraft(BaseModel):
    id: str
    role: str
    tools: list[str] = Field(default_factory=list)


class AgentList(BaseModel):
    """Step 1 of split design: agents + the linear processing order + ownership."""
    agents: list[_AgentDraft]
    # The agent that owns loop continuation (reads the queue / decides finish),
    # or null for non-iterative systems. Identified explicitly so edge wiring
    # can enforce signal ownership structurally rather than hoping the model
    # infers it.
    iterator_id: Optional[str] = None
    # The agent(s) that make branching decisions (emit ROUTE signals).
    router_ids: list[str] = Field(default_factory=list)
    # The linear "spine" of the processing chain, in order — the sequence each
    # work item flows through before any branch (e.g. parser -> scorer -> router).
    # Spine edges are generated DETERMINISTICALLY from this order, so the model
    # cannot drop the connections (the recurring orphaned-spine failure). Each
    # consecutive pair becomes an unconditional edge unless the source is a
    # router (whose outgoing edges are conditional and wired in step 2).
    pipeline: list[str] = Field(default_factory=list)
    # Router fan-out: for each router, a mapping of {signal: handler_agent_id}.
    # E.g. {"application_router": {"reject": "reject_handler", "maybe":
    # "maybe_handler", "forward": "forward_handler"}}. Wired deterministically so
    # the branch structure can't be dropped either. Each handler then returns to
    # the iterator (or to END if there is no iterator).
    routes: dict[str, dict[str, str]] = Field(default_factory=dict)
    # The terminal agent that runs once the batch finishes (summary/report), or
    # null. The iterator's 'finish' signal routes here, and this node -> END.
    terminal_id: Optional[str] = None


class _EdgeDraft(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    condition: Optional[str] = None
    model_config = {"populate_by_name": True}


class EdgeList(BaseModel):
    """Step 2 of split design: edges over a fixed agent set."""
    entry: str
    edges: list[_EdgeDraft]


# --- Checkpoint plumbing -------------------------------------------------

class Stage(str, Enum):
    parse = "parse"
    architect = "architect"
    design = "design"
    resolve_tools = "resolve_tools"
    compile = "compile"
    validate = "validate"


# Stages that pause for human review at autonomy=6.
HUMAN_CHECKPOINTS = {Stage.parse, Stage.architect, Stage.resolve_tools}


@dataclass
class Checkpoint:
    stage: Stage
    artifact: object              # a pydantic model or dict for human review
    editable: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    spec: Optional[SystemSpec]
    checkpoints: list[Checkpoint]
    halted_at: Optional[Stage] = None


# --- LLM-backed structured generation ------------------------------------

class StructuredGenerationError(RuntimeError):
    """Raised when the model can't produce schema-valid output after retries."""


def _structured(llm, schema: type[BaseModel], system: str, user: str,
                max_attempts: int = 3) -> BaseModel:
    """
    Ask the local model for output matching `schema`. Tries langchain structured
    output first; on failure, falls back to JSON-parsing with a strict prompt and
    a validation-repair loop: each retry feeds the previous error back to the
    model so it can fix the specific missing/invalid field. Raises
    StructuredGenerationError with the last error if all attempts fail, rather
    than letting a raw ValidationError crash the pipeline.
    """
    try:
        return llm.with_structured_output(schema).invoke(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}]
        )
    except Exception:
        pass  # fall through to manual JSON + repair loop

    base_prompt = (
        f"{system}\n\nReturn ONLY valid JSON matching this schema "
        f"(no markdown, no prose):\n{json.dumps(schema.model_json_schema())}\n\n"
        f"Input:\n{user}"
    )
    last_err: Exception | None = None
    prompt = base_prompt
    for attempt in range(1, max_attempts + 1):
        resp = llm.invoke([{"role": "user", "content": prompt}])
        text = getattr(resp, "content", str(resp))
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return schema.model_validate_json(text)
        except Exception as e:  # ValidationError or JSON decode error
            last_err = e
            # Feed the exact failure back so the model repairs that field.
            prompt = (
                f"{base_prompt}\n\nYour previous response was INVALID and was "
                f"rejected with this error:\n{e}\n\nHere is what you returned:\n"
                f"{text}\n\nReturn corrected JSON that fixes the error above. "
                "Every required field — especially 'graph' with 'entry' and "
                "'edges' — must be present."
            )
    raise StructuredGenerationError(
        f"model failed to produce valid {schema.__name__} after {max_attempts} "
        f"attempts; last error: {last_err}"
    )


# --- The pipeline --------------------------------------------------------

PARSER_SYS = (
    "You extract structured requirements for an agentic AI system from a user "
    "spec. Identify the core goal, required capabilities, constraints, and "
    "measurable success criteria. List genuine ambiguities as open_questions."
)

ARCHITECT_SYS = (
    "You are a systems architect for multi-agent LangGraph applications. Given "
    "requirements, design the topology. Before choosing, reason explicitly "
    "through these questions and let the answers drive the choice:\n"
    "1. BRANCHING: Does any requirement describe a decision with different "
    "downstream paths? Look for 'if/then', 'otherwise', 'flag vs. accept', "
    "'valid vs. invalid', 'route', 'depending on'. If yes, the system is NOT "
    "purely sequential: it needs a router agent that emits a signal, and "
    "distinct paths for each outcome. Every outcome mentioned in the "
    "requirements (e.g. both the success path AND the failure/flag path) must "
    "be reachable.\n"
    "2. ITERATION: Does the task repeat over a collection (each file, every "
    "item, a batch) or retry/re-check until a condition holds? If yes, the "
    "topology is cyclic: include a loop back to the processing node and a "
    "terminal node that runs once the loop exhausts (e.g. a batch summary).\n"
    "3. AGGREGATION: Do the requirements ask for a summary, total, or report "
    "computed ACROSS items? If yes, there must be a dedicated terminal agent "
    "that collects results after iteration completes.\n"
    "Choose sequential ONLY if none of the above apply. Choose cyclic if "
    "iteration or retry is present; hierarchical if a coordinator delegates to "
    "sub-agents. In agent_roles, include every role implied above, including "
    "router agents, failure/flag handlers, and summary/aggregation agents, not "
    "just the happy-path steps. In rationale, state which branch points and "
    "loops you found (or note explicitly that there are none). Add patterns "
    "like 'reflection' or 'planning' only when they earn their keep."
)


def run_pipeline(llm, user_prompt: str, autonomy: int = 6,
                 interactive_review=None) -> PipelineResult:
    """
    Execute the meta-pipeline.

    `interactive_review`, if provided, is called as review(checkpoint) -> artifact
    at each human checkpoint and may return an edited artifact. If None, the
    pipeline records the checkpoint and continues with the unedited artifact
    (autonomy 6 still surfaces them for after-the-fact inspection).
    """
    checkpoints: list[Checkpoint] = []

    def checkpoint(stage: Stage, artifact, notes=None):
        cp = Checkpoint(stage=stage, artifact=artifact, notes=notes or [])
        checkpoints.append(cp)
        if stage in HUMAN_CHECKPOINTS and interactive_review is not None:
            edited = interactive_review(cp)
            if edited is not None:
                cp.artifact = edited
        return cp.artifact

    # Stage 1: parse
    reqs: Requirements = _structured(llm, Requirements, PARSER_SYS, user_prompt)
    reqs = checkpoint(Stage.parse, reqs,
                      notes=reqs.open_questions or ["no open questions"])

    # Stage 2: architect
    arch: Architecture = _structured(
        llm, Architecture, ARCHITECT_SYS, reqs.model_dump_json()
    )
    arch = checkpoint(Stage.architect, arch, notes=[arch.rationale])

    # Stage 3 + 4: SPLIT design — two smaller LLM calls with flatter schemas.
    # This addresses two failure modes seen with the monolithic call: (a) the
    # full nested SystemSpec is too complex for small models to emit in one shot
    # (causing missing-field crashes), and (b) signal ownership was repeatedly
    # mis-assigned. The split lets us identify the iterator/routers as DATA in
    # step 1, then enforce ownership deterministically when wiring edges.
    spec = _design_split(llm, reqs, arch)

    # Tool-resolution checkpoint: flag any unknown tools (should be none, but the
    # model can hallucinate). This is a human checkpoint.
    unknown = [t for a in spec.agents for t in a.tools if t not in available_tools()]
    checkpoint(Stage.resolve_tools, spec,
               notes=([f"unknown tools: {unknown}"] if unknown else ["all tools resolved"]))

    return PipelineResult(spec=spec, checkpoints=checkpoints)


# --- Split design implementation -----------------------------------------

_AGENTS_SYS = (
    "You design the AGENTS for a LangGraph system (not the graph yet). Given "
    "requirements and an architecture, output a flat list of agents with "
    "snake_case ids and a concrete one-sentence role each. Assign tools ONLY "
    "from the available list.\n"
    "CRITICAL — every agent you create MUST be a node in the control flow: it "
    "is either the iterator, a step in the processing pipeline, a router, a "
    "per-outcome handler, or the terminal aggregator. Do NOT create standalone "
    "agents for CROSS-CUTTING CONCERNS that apply to every step rather than "
    "occupying one point in the flow — e.g. privacy/data-protection compliance, "
    "logging, auditing, monitoring, error handling, or generic 'controllers'. "
    "These have no single place in the graph and would be unreachable. Instead, "
    "fold such a concern into the role text of the agents it applies to (e.g. "
    "add '...handling personal data in compliance with data-protection rules' to "
    "the extractor's role). Likewise do NOT create a separate 'loop controller' "
    "or 'coordinator' agent — the iterator already owns the loop.\n"
    "Then identify, as separate fields:\n"
    "- iterator_id: the single agent that reads the work queue and decides "
    "whether to continue or finish the batch (the ONLY agent that can know if "
    "work remains). Null if the system does not loop over a collection.\n"
    "- router_ids: agents that make a branching decision among 2+ outcomes.\n"
    "- pipeline: the ordered list of agent ids each work item flows through, "
    "from first processing step up to and INCLUDING the first router, in order "
    "(e.g. ['parser','skill_scorer','router']). Start with the iterator if one "
    "exists. Do NOT include the per-outcome handlers after a router or the "
    "terminal aggregator — those are wired separately. This ordering is the "
    "backbone of the graph, so list every sequential step exactly once.\n"
    "- routes: for EACH router, a mapping {signal: handler_agent_id} giving the "
    "destination agent for each outcome (e.g. {'router': {'reject': "
    "'reject_handler', 'forward': 'forward_handler'}}). Every distinct outcome "
    "MUST map to its OWN handler agent.\n"
    "- terminal_id: the agent that produces the final batch summary/report once "
    "the loop finishes, or null if none.\n"
    "Every agent in your list MUST appear in exactly one of: iterator_id, "
    "pipeline, router_ids, the routes handler values, or terminal_id. If an "
    "agent fits none of these, it does not belong — remove it.\n"
    "Every router's role text MUST instruct it to emit one line "
    "'ROUTE: <signal>' per outcome. The iterator's role MUST instruct it to "
    "emit 'ROUTE: continue' or 'ROUTE: finish'."
)


def _edges_sys(agents: "AgentList") -> str:
    ids = [a.id for a in agents.agents]
    return (
        "You wire the GRAPH edges over a FIXED set of agents. Do not invent new "
        f"agents; use only these ids: {ids}.\n"
        f"Entry point and iterator: iterator_id={agents.iterator_id!r}, "
        f"routers={agents.router_ids!r}.\n"
        "RULES (followed exactly):\n"
        "1. condition is the BARE signal word only (e.g. 'reject'), never "
        "'ROUTE: reject'.\n"
        "2. Only routers and the iterator have conditional edges (2+ outs with "
        "distinct signals). Every other agent has exactly ONE unconditional "
        "edge (condition: null).\n"
        "3. If there is an iterator, EVERY worker/handler edge returns "
        "UNCONDITIONALLY to the iterator — workers never decide continue/finish. "
        "The iterator emits 'continue' (into the processing path) and 'finish' "
        "(to the terminal aggregator).\n"
        "4. Each distinct router outcome goes to its OWN handler node; do not "
        "collapse two outcomes onto the same node.\n"
        "5. Exactly one path must reach END.\n"
        "Entry should be the iterator if one exists, else the first processing "
        "agent."
    )


def _design_split(llm, reqs, arch) -> SystemSpec:
    # Step 1: agents + ownership metadata.
    agents_input = json.dumps({
        "requirements": reqs.model_dump(),
        "architecture": arch.model_dump(),
        "available_tools": available_tools(),
    })
    al: AgentList = _structured(llm, AgentList, _AGENTS_SYS, agents_input)

    # Deterministic backstop for cross-cutting / non-flow agents (e.g. a
    # privacy_compliance_checker or loop_controller the model created despite
    # instructions). Any agent that appears in NONE of the flow roles — iterator,
    # pipeline, router, a routes handler value, or terminal — has no place in the
    # graph and would orphan. Drop it here so assembly can't fail on it. Its
    # concern is expected to be folded into other agents' roles by step 1.
    # Guarded: only applies when step 1 actually populated flow metadata
    # (pipeline/routes). Without it, agent placement comes from the model's
    # edges instead and this drop would be wrong.
    flow_ids = set(al.pipeline) | set(al.router_ids)
    if al.iterator_id:
        flow_ids.add(al.iterator_id)
    if al.terminal_id:
        flow_ids.add(al.terminal_id)
    for mapping in (al.routes or {}).values():
        flow_ids.update(mapping.values())
    if al.pipeline or al.routes:  # flow metadata present -> safe to prune orphans
        dropped = [a.id for a in al.agents if a.id not in flow_ids]
        if dropped:
            al.agents = [a for a in al.agents if a.id in flow_ids]

    agent_ids = [a.id for a in al.agents]
    owners = set(al.router_ids) | ({al.iterator_id} if al.iterator_id else set())

    # Deterministically wire the SPINE from the ordered pipeline. Consecutive
    # pairs become unconditional edges, EXCEPT a pair whose source is a router
    # (router exits are conditional, added in step 2). This guarantees the
    # processing chain is connected even when the model fails to wire it — the
    # recurring orphaned-spine failure. The iterator's entry into the spine is
    # conditional ('continue'), so we skip an edge OUT of the iterator here and
    # let step 2 / spine handle it: specifically, iterator->first_processing is
    # emitted as a 'continue' conditional edge.
    spine = [a for a in al.pipeline if a in set(agent_ids)]
    spine_edges: list[_EdgeDraft] = []
    for i in range(len(spine) - 1):
        src, dst = spine[i], spine[i + 1]
        if src in al.router_ids:
            continue  # router fan-out handled below
        cond = "continue" if src == al.iterator_id else None
        spine_edges.append(_EdgeDraft(**{"from": src, "to": dst, "condition": cond}))

    # Router fan-out + handler returns, deterministically from step-1 `routes`.
    handler_ids: set[str] = set()
    for router_id, mapping in (al.routes or {}).items():
        if router_id not in set(agent_ids):
            continue
        for signal, handler in mapping.items():
            if handler not in set(agent_ids):
                continue
            spine_edges.append(_EdgeDraft(**{"from": router_id, "to": handler,
                                             "condition": signal}))
            handler_ids.add(handler)
    # Handlers return to the iterator (loop) or to the terminal/END.
    return_target = al.iterator_id or al.terminal_id or END
    for h in handler_ids:
        if h != return_target:
            spine_edges.append(_EdgeDraft(**{"from": h, "to": return_target,
                                             "condition": None}))
    # Iterator 'finish' -> terminal -> END (or iterator 'finish' -> END).
    if al.iterator_id:
        if al.terminal_id and al.terminal_id in set(agent_ids):
            spine_edges.append(_EdgeDraft(**{"from": al.iterator_id,
                                             "to": al.terminal_id, "condition": "finish"}))
            spine_edges.append(_EdgeDraft(**{"from": al.terminal_id, "to": END,
                                             "condition": None}))
        else:
            spine_edges.append(_EdgeDraft(**{"from": al.iterator_id, "to": END,
                                             "condition": "finish"}))
    elif al.terminal_id and al.terminal_id in set(agent_ids):
        spine_edges.append(_EdgeDraft(**{"from": al.terminal_id, "to": END,
                                         "condition": None}))

    # First, try assembling from the DETERMINISTIC edges alone. For well-formed
    # step-1 metadata (pipeline + routes + terminal) this is a complete graph and
    # the model's step-2 is not needed at all — the strongest defense against the
    # orphaned-spine failure.
    def _assemble(extra_edges):
        merged_pairs = {(e.from_, e.to) for e in spine_edges}
        merged = list(spine_edges) + [e for e in extra_edges
                                      if (e.from_, e.to) not in merged_pairs]
        fixed = _enforce_ownership(merged, owners, al.iterator_id)
        entry = al.iterator_id or (spine[0] if spine else agent_ids[0])
        fixed = _patch_orphans(fixed, agent_ids, entry, al.iterator_id, owners)
        return SystemSpec.model_validate({
            "name": _slug(reqs.goal),
            "runtime": "langgraph",
            "agents": [a.model_dump() for a in al.agents],
            "graph": {"entry": entry,
                      "edges": [{"from": e.from_, "to": e.to, "condition": e.condition}
                                for e in fixed]},
            "success_criteria": reqs.success_criteria,
        })

    try:
        return _assemble([])  # deterministic-only
    except Exception:
        pass  # gaps remain; fall back to the model for the missing edges

    edges_input_base = json.dumps({
        "agents": [a.model_dump() for a in al.agents],
        "iterator_id": al.iterator_id,
        "router_ids": al.router_ids,
        "pipeline": al.pipeline,
        "already_wired": [(e.from_, e.to) for e in spine_edges],
    })

    last_err = None
    edges_sys = _edges_sys(al)
    edges_input = edges_input_base
    for attempt in range(1, 4):
        el: EdgeList = _structured(llm, EdgeList, edges_sys, edges_input)
        try:
            return _assemble(el.edges)
        except Exception as e:
            last_err = e
            edges_input = (
                f"{edges_input_base}\n\nThe 'already_wired' edges exist — do not "
                f"repeat them. Your additions produced an INVALID graph:\n{e}\n\n"
                f"Add only the missing edges so every id in {agent_ids} is "
                "reachable and exactly one path reaches END."
            )
    raise StructuredGenerationError(
        f"could not assemble a valid graph after 3 edge attempts; last error: {last_err}"
    )


def _enforce_ownership(edges, owners, iterator_id):
    """Force non-owner edges unconditional; reroute worker loop-signals to the
    iterator. Deterministic fix for the recurring signal-ownership bug."""
    out = []
    for e in edges:
        if e.from_ not in owners and e.condition:
            if iterator_id and e.condition in ("continue", "finish"):
                e = _EdgeDraft(**{"from": e.from_, "to": iterator_id, "condition": None})
            else:
                e = _EdgeDraft(**{"from": e.from_, "to": e.to, "condition": None})
        out.append(e)
    return out


def _patch_orphans(edges, agent_ids, entry, iterator_id, owners):
    """Conservatively close DANGLING EXITS only. An agent with no outgoing edge
    that is ALSO already reachable (has an incoming edge) is a leaf the model
    forgot to terminate — safe to wire back to the iterator (loop) or END. We do
    NOT touch agents with no incoming edge: that's a missing path in the main
    flow, which we can't guess correctly (it caused a worse graph when we tried).
    Those are left for the repair loop to fix with the model in the loop."""
    has_out = {e.from_ for e in edges}
    has_in = {e.to for e in edges}
    out = list(edges)
    for aid in agent_ids:
        reachable = aid == entry or aid in has_in
        if aid not in has_out and reachable:
            # A reachable leaf with no exit. Routers must keep their own
            # conditional exits (don't auto-wire an owner), so skip owners.
            if aid in owners:
                continue
            target = iterator_id if (iterator_id and aid != iterator_id) else END
            out.append(_EdgeDraft(**{"from": aid, "to": target, "condition": None}))
    return out


def _slug(text: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    s = re.sub(r"_+", "_", s) or "agent_system"
    if not s[0].isalpha():
        s = "sys_" + s
    return s[:40].rstrip("_")