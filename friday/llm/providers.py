"""Streaming adapters: Anthropic Messages API and any OpenAI-compatible Chat API (OpenAI, Ollama…)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .base import LLMResponse, Message, StreamEvent, ToolCall, ToolSpec, Usage, retrying_stream

CACHE = {"type": "ephemeral"}


# --- Anthropic -------------------------------------------------------------------------------------
def to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
        else:  # results of parallel calls must share one user turn
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self._client, self.model = client, model

    def stream(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int
    ) -> Iterator[StreamEvent]:
        tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        if tool_defs:  # tool schemas never change: cache them (prompt caching)
            tool_defs[-1] = {**tool_defs[-1], "cache_control": CACHE}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": to_anthropic(messages),
        }
        if tool_defs:
            kwargs["tools"] = tool_defs

        def attempt() -> Iterator[StreamEvent]:
            with self._client.messages.stream(**kwargs) as s:
                for delta in s.text_stream:
                    yield StreamEvent("text", delta)
                final = s.get_final_message()
            text = "".join(b.text for b in final.content if b.type == "text").strip()
            calls = [ToolCall(b.id, b.name, dict(b.input)) for b in final.content if b.type == "tool_use"]
            usage = Usage(final.usage.input_tokens, final.usage.output_tokens)
            yield StreamEvent("final", response=LLMResponse(text, calls, usage))

        yield from retrying_stream(attempt)


# --- OpenAI-compatible ---------------------------------------------------------------------------------
def to_openai(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in m.tool_calls
                ]
            out.append(msg)
        else:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def parse_arguments(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    return value if isinstance(value, dict) else {"__invalid_json__": raw}


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str | None = None, client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, max_retries=0)
        self._client, self.model = client, model

    def stream(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int
    ) -> Iterator[StreamEvent]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": to_openai(system, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]

        def attempt() -> Iterator[StreamEvent]:
            text: list[str] = []
            calls: dict[int, dict[str, str]] = {}  # tool calls stream in fragments, keyed by index
            usage = Usage()
            for chunk in self._client.chat.completions.create(**kwargs):
                if getattr(chunk, "usage", None):
                    usage = Usage(chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text.append(delta.content)
                    yield StreamEvent("text", delta.content)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.id or slot["id"]
                    if tc.function is not None:
                        slot["name"] = tc.function.name or slot["name"]
                        slot["arguments"] += tc.function.arguments or ""
            tool_calls = [
                ToolCall(c["id"], c["name"], parse_arguments(c["arguments"])) for _, c in sorted(calls.items())
            ]
            yield StreamEvent("final", response=LLMResponse("".join(text).strip(), tool_calls, usage))

        yield from retrying_stream(attempt)
