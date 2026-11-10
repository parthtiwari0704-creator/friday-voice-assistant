"""Typed Python functions → LLM tools, with validation and a taint-based action policy.

* ``untrusted=True`` – output is third-party text. It is fenced, and it **taints** the turn.
* ``action=True``    – the tool changes something on the user's machine (opens an app or URL).

Policy: an action is refused if the current turn is tainted. A web page or email that says
"open this link" can therefore never trigger the action; the user has to ask directly in a new
turn, which clears the taint. This is a lightweight version of the taint tracking used in
defences like CaMeL (Debenedetti et al., 2025).
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from ..llm.base import ToolSpec
from ..security import wrap_untrusted

log = logging.getLogger(__name__)
MAX_RESULT_CHARS = 8000
_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    untrusted: bool = False
    action: bool = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    blocked: bool = False
    duration_ms: int = 0


@dataclass
class TurnState:
    """Per-turn security state. A new user turn starts clean."""

    tainted: bool = False
    taint_sources: list[str] | None = None


def _schema(tp: Any) -> tuple[dict[str, Any], bool]:
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        schema = _schema(inner[0])[0] if len(inner) == 1 else {"type": "string"}
        return schema, type(None) in args
    if origin is Literal:
        return {"type": _JSON.get(type(args[0]), "string"), "enum": list(args)}, False
    if origin in (list, tuple):
        return {"type": "array", "items": _schema(args[0] if args else str)[0]}, False
    return {"type": _JSON.get(tp, "string")}, False


def _docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    summary, params, current, in_args = [], {}, None, False
    for line in inspect.cleandoc(doc or "").splitlines():
        s = line.strip()
        if s in ("Args:", "Arguments:"):
            in_args = True
        elif in_args and line.startswith("    ") and ":" in s and not line.startswith("        "):
            current, _, text = s.partition(":")
            params[current] = text.strip()
        elif in_args and current and s:
            params[current] += " " + s
        elif not in_args and s:
            summary.append(s)
    return " ".join(summary), params


def build_schema(func: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    hints = get_type_hints(func)
    description, docs = _docstring(func.__doc__)
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, p in inspect.signature(func).parameters.items():
        schema, optional = _schema(hints.get(name, str))
        if name in docs:
            schema["description"] = docs[name]
        if p.default is inspect.Parameter.empty and not optional:
            required.append(name)
        elif p.default not in (inspect.Parameter.empty, None):
            schema["default"] = p.default
        props[name] = schema
    out: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return description, out


def validate(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    errors = [f"missing argument '{r}'" for r in schema.get("required", []) if r not in args]
    checks: dict[str, Callable[[Any], bool]] = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    props = schema.get("properties", {})
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            errors.append(f"unknown argument '{key}'")
        elif value is not None and spec.get("type") in checks and not checks[spec["type"]](value):
            errors.append(f"argument '{key}' should be {spec['type']}")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"argument '{key}' must be one of {spec['enum']}")
    return errors


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, func: Callable[..., Any] | None = None, *, untrusted: bool = False, action: bool = False) -> Any:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            description, params = build_schema(fn)
            self.add(Tool(fn.__name__, description, params, fn, untrusted, action))
            return fn

        return deco(func) if func is not None else deco

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' registered twice")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def execute(self, name: str, args: dict[str, Any], turn: TurnState | None = None) -> ToolResult:
        """Validate, apply the taint policy, run. Never raises."""
        turn = turn or TurnState()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(json.dumps({"error": f"unknown tool '{name}'"}), is_error=True)
        problems = validate(tool.parameters, args)
        if problems:
            return ToolResult(json.dumps({"error": "invalid arguments", "details": problems}), is_error=True)
        if tool.action and turn.tainted:
            log.warning("Blocked %s: turn tainted by %s", name, turn.taint_sources)
            reason = (
                "Blocked for safety: this turn contains third-party content "
                f"({', '.join(turn.taint_sources or [])}). Do not retry. Tell the user what you would do "
                "and ask them to request it directly."
            )
            return ToolResult(json.dumps({"status": "blocked", "reason": reason}), blocked=True)

        started = time.perf_counter()
        try:
            value = tool.func(**args)
            text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
            is_error = False
        except Exception as exc:  # noqa: BLE001 - errors go back to the model
            text, is_error = json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "…"
        if tool.untrusted and not is_error:
            text = wrap_untrusted(text, name)
            turn.tainted = True
            turn.taint_sources = [*(turn.taint_sources or []), name]
        return ToolResult(text, is_error, duration_ms=int((time.perf_counter() - started) * 1000))
