"""
Local tool registry.

Tools are local-first: anything that needs an external service must be
explicitly registered, and the Tool Resolver meta-agent flags such cases for the
human checkpoint. Built-ins below require no network.
"""
from __future__ import annotations

from typing import Callable, Optional

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_tool(name: str) -> Optional[Callable]:
    return _REGISTRY.get(name)


def available_tools() -> list[str]:
    return sorted(_REGISTRY)


# --- Built-in local tools -------------------------------------------------

@register("doc_loader")
def doc_loader(path: str) -> str:
    """Load a local text/markdown file."""
    from pathlib import Path
    return Path(path).read_text()


@register("calculator")
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    return str(_eval(ast.parse(expression, mode="eval").body))
