"""
AgentForge CLI.

Commands:
  compile <spec.yaml> [-o out.py]   Deterministically compile a DSL to LangGraph.
  generate "<prompt>" [-o spec.yaml] Run the meta-pipeline to produce a DSL.
  validate <spec.yaml>              Static-check a DSL.

`generate` requires an OPENROUTER_API_KEY env var; `compile`/`validate` do not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .compiler.langgraph_compiler import compile_spec
from .dsl.loader import load_spec
from .meta_agents.validator import has_errors, static_check


def _cmd_compile(args):
    spec = load_spec(args.spec)
    code = compile_spec(spec)
    out = Path(args.output) if args.output else Path(f"{spec.name}_app.py")
    out.write_text(code)
    print(f"compiled -> {out}")


def _cmd_validate(args):
    spec = load_spec(args.spec)
    diags = static_check(spec)
    for d in diags:
        print(f"[{d.level}] {d.message}")
    # Convergence / progress-discard smoke test (static, no LLM needed).
    from .meta_agents.smoke import smoke_test
    sr = smoke_test(spec)
    for w in sr.dead_routers:
        print(f"[warning] router signal mismatch: {w}")
    for w in sr.progress_discard:
        print(f"[warning] {w}")
    if has_errors(diags):
        sys.exit(1)
    status = "OK" if not diags and sr.converges else "OK (with warnings)"
    print(status)


def _cmd_generate(args):
    from .meta_agents.pipeline import StructuredGenerationError, run_pipeline
    from .runtime.support import make_llm

    def review(cp):
        print(f"\n=== HUMAN CHECKPOINT: {cp.stage.value} ===")
        artifact = cp.artifact
        # mode="json" coerces enums/dates to primitives so safe_dump can serialize.
        dump = artifact.model_dump(mode="json") if hasattr(artifact, "model_dump") else artifact
        print(yaml.safe_dump(dump, sort_keys=False))
        for n in cp.notes:
            print(f"  note: {n}")
        print("Edit? paste YAML then a blank line to override, or just ENTER to accept:")
        # Read possibly-multiline YAML until a blank line (or EOF).
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        raw = "\n".join(lines).strip()
        if not raw:
            return None  # accept as-is
        try:
            overrides = yaml.safe_load(raw)
            if not isinstance(overrides, dict):
                print("  ! override ignored: expected a YAML mapping; accepting original")
                return None
            merged = {**dump, **overrides}  # shallow merge: top-level keys win
            edited = artifact.model_validate(merged) if hasattr(artifact, "model_validate") else merged
            print("  ✓ override applied")
            return edited
        except Exception as e:  # bad YAML or schema violation -> keep original, don't crash the run
            print(f"  ! override ignored ({type(e).__name__}: {e}); accepting original")
            return None

    llm = make_llm(model=args.model)
    try:
        result = run_pipeline(llm, args.prompt, autonomy=6,
                              interactive_review=review if not args.no_review else None)
    except StructuredGenerationError as e:
        print(f"\nGeneration failed: {e}")
        print("The local model couldn't produce a valid spec for this prompt. "
              "Options: retry, use a larger model (--model), or simplify the prompt "
              "into fewer decision points.")
        sys.exit(2)
    if result.spec is None:
        print(f"halted at {result.halted_at}")
        sys.exit(1)
    out = Path(args.output) if args.output else Path(f"{result.spec.name}.yaml")
    out.write_text(yaml.safe_dump(result.spec.model_dump(mode="json"), sort_keys=False))
    print(f"spec -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="agentforge")
    sub = p.add_subparsers(required=True)

    c = sub.add_parser("compile"); c.add_argument("spec"); c.add_argument("-o", "--output")
    c.set_defaults(func=_cmd_compile)

    v = sub.add_parser("validate"); v.add_argument("spec")
    v.set_defaults(func=_cmd_validate)

    g = sub.add_parser("generate"); g.add_argument("prompt")
    g.add_argument("-o", "--output"); g.add_argument("--model", default="openai/gpt-oss-120b:free")
    g.add_argument("--no-review", action="store_true")
    g.set_defaults(func=_cmd_generate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
