"""Typed tool registry.

The model is prompted with each tool's JSON schema and emits tool calls as JSON:

    {"name": "web_search", "arguments": {"query": "..."}}

The registry validates, executes, and returns a structured result that the
agent loop feeds back as an observation.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the tool's arguments


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = (spec, fn)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name][0]

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def specs_json(self) -> str:
        return json.dumps(
            [{"name": s.name, "description": s.description, "parameters": s.parameters} for s in self.specs()],
            indent=2,
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(ok=False, error=f"unknown tool: {name}")
        spec, fn = self._tools[name]
        missing = [k for k in spec.parameters.get("required", []) if k not in (arguments or {})]
        if missing:
            return ToolResult(ok=False, error=f"{name}: missing required arguments: {', '.join(missing)}")
        try:
            return ToolResult(ok=True, data=fn(**(arguments or {})))
        except Exception as exc:  # noqa: BLE001 — a failing tool is an observation, not a crash
            return ToolResult(ok=False, error=f"{name} failed: {exc}")


def default_registry() -> ToolRegistry:
    """The registry the agent uses by default: search, code runner, memory."""
    from . import code_runner, memory, web_search

    registry = ToolRegistry()
    registry.register(web_search.SPEC, web_search.web_search)
    registry.register(code_runner.SPEC, code_runner.run_python)
    registry.register(memory.GET_SPEC, memory.memory_get)
    registry.register(memory.SET_SPEC, memory.memory_set)
    return registry
