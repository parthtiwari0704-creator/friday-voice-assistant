"""Composition root + the background services and the hands-free voice loop."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .assistant import Assistant
from .config import Settings
from .llm import LLMProvider, get_provider
from .store import Store
from .tools import ToolRegistry, build_tools
from .tools.mcp_bridge import MCPBridge
from .voice.stt import STTEngine, create_stt
from .voice.tts import TTSEngine, VoiceManager, create_engine

log = logging.getLogger(__name__)
_UNSET: Any = object()


@dataclass
class Friday:
    settings: Settings
    store: Store
    voice: VoiceManager
    registry: ToolRegistry
    stt: STTEngine | None
    bridge: MCPBridge | None = None
    bridged_tools: list[str] = field(default_factory=list)
    _provider: LLMProvider | None = None
    _assistant: Assistant | None = None

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider(self.settings)
        return self._provider

    @property
    def assistant(self) -> Assistant:
        if self._assistant is None:
            self._assistant = Assistant(self.provider, self.registry, self.settings)
        return self._assistant

    def new_assistant(self) -> Assistant:
        """A separate conversation (e.g. one per browser tab) sharing tools, voice and memory."""
        return Assistant(self.provider, self.registry, self.settings)

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
        self.store.close()


def build_app(
    settings: Settings,
    *,
    provider: LLMProvider | None = None,
    tts: TTSEngine | None = _UNSET,
    stt: STTEngine | None = _UNSET,
    store: Store | None = None,
    mcp_target: Any = None,
    **tool_kwargs: Any,
) -> Friday:
    settings.ensure_dirs()
    store = store or Store(settings.db_path)
    engine = create_engine(settings) if tts is _UNSET else tts
    voice = VoiceManager(engine, settings.voice, settings.speaking_rate)
    registry = build_tools(settings, store, voice if engine is not None else None, **tool_kwargs)
    bridge, bridged = None, []
    if mcp_target is not None or settings.jarvis_mcp_command:
        try:
            bridge = (
                MCPBridge(mcp_target) if mcp_target is not None else MCPBridge.from_command(settings.jarvis_mcp_command)
            )
            bridged = bridge.start().register_into(registry)
        except Exception as exc:  # noqa: BLE001 - FRIDAY still works without Jarvis
            log.warning("Jarvis (MCP) unavailable: %s", exc)
            bridge = None
    return Friday(
        settings, store, voice, registry, create_stt(settings) if stt is _UNSET else stt, bridge, bridged, provider
    )


class ReminderService:
    """Polls for due reminders and announces them (spoken + printed/pushed)."""

    def __init__(self, store: Store, announce: Callable[[str], None], interval: float = 1.0):
        self.store, self.announce, self.interval = store, announce, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reminders", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check()
            except sqlite3.ProgrammingError:  # the store was closed: the app is shutting down
                return

    def check(self, now: datetime | None = None) -> int:
        due = self.store.pop_due_reminders(now)
        for r in due:
            try:
                self.announce(r["text"])
            except Exception:
                log.exception("Could not announce reminder")
        return len(due)

    def start(self) -> ReminderService:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


STOP_WORDS = {"stop", "cancel", "never mind", "nevermind", "that's all", "thats all", "thank you", "thanks"}
EXIT_WORDS = {"goodbye", "good bye", "bye", "exit", "shut down", "go to sleep", "stop listening"}


def classify_command(text: str) -> str:
    """Fast-path phrases handled without the LLM: 'exit', 'stop' or 'ask'."""
    t = text.lower().strip(" .!?,")
    if t in EXIT_WORDS:
        return "exit"
    if t in STOP_WORDS:
        return "stop"
    return "ask"


def elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)
