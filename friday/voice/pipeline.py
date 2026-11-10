"""The voice pipeline: LLM text stream → sentences → TTS → audio, fully overlapped.

    LLM thread    ──deltas──► SentenceChunker ──sentences──► TTS thread ──audio──► consumer
    (consumer = local speaker thread, or the web server sending audio to the browser)

Three stages run concurrently, so while sentence 1 plays, sentence 2 is being synthesised and
sentence 3 is still being generated. The key metric is **time to first audio**: from the end
of the user's request to the first synthesised sentence being ready.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from ..assistant import AssistantEvent, TurnMetrics
from .text import SentenceChunker, clean_for_speech
from .tts import Audio, VoiceManager

log = logging.getLogger(__name__)
_DONE = object()


@dataclass
class SpeechEvent:
    kind: Literal["text", "tool_start", "tool_end", "sentence", "done", "error"]
    text: str = ""
    index: int = -1
    audio: Audio | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class SpeechStream:
    """Iterate to receive events in order; call ``cancel()`` to stop (e.g. the user interrupts)."""

    def __init__(self, events: Iterator[AssistantEvent], voice: VoiceManager | None, started: float | None = None):
        self._events, self._voice = events, voice
        self._out: queue.Queue[Any] = queue.Queue()
        self._sentences: queue.Queue[Any] = queue.Queue()
        self.cancelled = threading.Event()
        self.started = started or time.perf_counter()
        self.first_audio_ms: int | None = None
        self.tts_ms = 0
        self.metrics: TurnMetrics | None = None
        self.spoken: list[str] = []

    def _ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def _produce(self) -> None:
        chunker, index = SentenceChunker(), 0

        def emit_sentences(parts: list[str]) -> None:
            nonlocal index
            for raw in parts:
                spoken = clean_for_speech(raw)
                if spoken:
                    self._sentences.put((index, spoken))
                    index += 1

        try:
            for ev in self._events:
                if self.cancelled.is_set():
                    break
                if ev.kind == "text":
                    self._out.put(SpeechEvent("text", ev.text))
                    emit_sentences(chunker.feed(ev.text))
                elif ev.kind in ("tool_start", "tool_end"):
                    self._out.put(SpeechEvent(ev.kind, ev.tool, detail=ev.detail))
                elif ev.kind == "done":
                    self.metrics = ev.metrics
            emit_sentences(chunker.flush())
        except Exception as exc:
            log.exception("assistant failed")
            self._out.put(SpeechEvent("error", f"{type(exc).__name__}: {exc}"))
            apology = "Sorry, something went wrong on my side."
            self._sentences.put((index, apology))
        finally:
            self._sentences.put(_DONE)

    def _synthesize(self) -> None:
        while (item := self._sentences.get()) is not _DONE:
            if self.cancelled.is_set():
                continue  # drain without synthesising
            index, text = item
            self.spoken.append(text)
            audio = None
            if self._voice is not None and self._voice.enabled:
                t0 = time.perf_counter()
                try:
                    audio = self._voice.synthesize(text)
                except Exception as exc:  # noqa: BLE001 - one bad sentence must not silence the rest
                    log.warning("TTS failed for sentence %d: %s", index, exc)
                self.tts_ms += int((time.perf_counter() - t0) * 1000)
            if self.first_audio_ms is None and audio is not None:
                self.first_audio_ms = self._ms()
            self._out.put(SpeechEvent("sentence", text, index=index, audio=audio))
        self._out.put(_DONE)

    def __iter__(self) -> Iterator[SpeechEvent]:
        threading.Thread(target=self._produce, name="llm", daemon=True).start()
        threading.Thread(target=self._synthesize, name="tts", daemon=True).start()
        while (ev := self._out.get()) is not _DONE:
            yield ev
        yield SpeechEvent("done", " ".join(self.spoken), detail=self.report())

    def report(self) -> dict[str, Any]:
        out = self.metrics.as_dict() if self.metrics else {}
        return out | {"first_audio_ms": self.first_audio_ms, "tts_ms": self.tts_ms, "total_ms": self._ms()}

    def cancel(self) -> None:
        self.cancelled.set()


class Speaker:
    """Plays sentence audio in order on a background thread, so playback never blocks the stream."""

    def __init__(self, play: Callable[[Audio], bool], on_start: Callable[[], None] | None = None,
                 on_idle: Callable[[], None] | None = None):  # fmt: skip
        self._play, self._on_start, self._on_idle = play, on_start, on_idle
        self._q: queue.Queue[Any] = queue.Queue()
        self.interrupted = threading.Event()
        self.first_play_at: float | None = None
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._run, name="speaker", daemon=True).start()

    def _run(self) -> None:
        while True:
            audio = self._q.get()
            if self.interrupted.is_set():
                self._mark_idle_if_empty()
                continue
            if self.first_play_at is None:
                self.first_play_at = time.perf_counter()
            if self._on_start:
                self._on_start()
            if not self._play(audio):
                self.interrupted.set()
            self._mark_idle_if_empty()

    def _mark_idle_if_empty(self) -> None:
        if self._q.empty():
            self._idle.set()
            if self._on_idle:
                self._on_idle()

    def say(self, audio: Audio) -> None:
        self._idle.clear()
        self._q.put(audio)

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def reset(self) -> None:
        self.interrupted.clear()
        self.first_play_at = None
