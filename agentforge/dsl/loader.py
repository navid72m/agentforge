"""Load and validate a DSL file (YAML or JSON) into a SystemSpec."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .schema import SystemSpec


def load_spec(path: str | Path) -> SystemSpec:
    path = Path(path)
    raw = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    elif path.suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(f"unsupported spec format: {path.suffix}")
    return SystemSpec.model_validate(data)


def load_spec_from_str(text: str, fmt: str = "yaml") -> SystemSpec:
    data = yaml.safe_load(text) if fmt == "yaml" else json.loads(text)
    return SystemSpec.model_validate(data)
