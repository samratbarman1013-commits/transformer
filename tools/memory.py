"""Lightweight local user-preference store.

A single JSON file (default ~/.transformer/memory.json, override with
TRANSFORMER_HOME). The agent reads it at the start of a run to personalize
responses and writes explicit user preferences only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .registry import ToolSpec

GET_SPEC = ToolSpec(
    name="memory_get",
    description="Read the user's stored preferences (e.g. preferred language, name, coding style).",
    parameters={"type": "object", "properties": {}, "required": []},
)

SET_SPEC = ToolSpec(
    name="memory_set",
    description="Store a user preference for future conversations. Only use when the user explicitly states a preference.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "short name, e.g. 'preferred_language'"},
            "value": {"type": "string", "description": "the preference value"},
        },
        "required": ["key", "value"],
    },
)


def _memory_path() -> Path:
    home = os.environ.get("TRANSFORMER_HOME")
    base = Path(home) if home else Path.home() / ".transformer"
    return Path(base) / "memory.json"


def _read() -> dict:
    path = _memory_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write(data: dict) -> None:
    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def memory_get() -> dict:
    return _read()


def memory_set(key: str, value: str) -> dict:
    data = _read()
    data[key] = value
    _write(data)
    return {"stored": {key: value}}
