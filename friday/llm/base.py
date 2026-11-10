"""Provider-neutral chat + tool-calling types, and the streaming contract.

The assistant speaks only this vocabulary. Providers stream ``text`` deltas and then
exactly one ``final`` event carrying tool calls and token usage.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage = field(default_factory=Usage)


@dataclass
class StreamEvent:
    kind: Literal["text", "final"]
    text: str = ""
    response: LLMResponse | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def stream(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int
    ) -> Iterator[StreamEvent]: ...


class ConfigError(RuntimeError):
    """Missing configuration such as an API key."""


_TRANSIENT = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "OverloadedError"}


def is_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in _TRANSIENT or getattr(exc, "status_code", None) in {408, 429, 500, 502, 503, 529}


def retrying_stream(attempt: Callable[[], Iterator[StreamEvent]], max_attempts: int = 3) -> Iterator[StreamEvent]:
    """Retry transient failures only before the first event; afterwards the user has heard
    part of the answer, and replaying it would make FRIDAY repeat herself."""
    for n in range(1, max_attempts + 1):
        started = False
        try:
            for event in attempt():
                started = True
                yield event
            return
        except Exception as exc:
            if started or n == max_attempts or not is_transient(exc):
                raise
            time.sleep(min(2**n, 10))
