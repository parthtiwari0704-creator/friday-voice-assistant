"""Hands-free conversation on this computer's microphone and speakers.

    idle ──"Hey FRIDAY, …"──► answer ──► follow-up window (no wake word needed) ──► idle
          ──"Hey FRIDAY" alone──► chime ──► listen for the command

The microphone is muted while FRIDAY speaks, so she never hears (and answers) herself.
Ctrl+C while she is speaking interrupts her; Ctrl+C while listening exits.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ..app import Friday, ReminderService, classify_command, elapsed_ms
from .audio import EnergyVAD, Microphone, Player, VADConfig, chime
from .pipeline import Speaker, SpeechStream
from .text import match_wake_word
from .tts import Audio

Printer = Callable[[str, str], None]  # (kind, text)


class VoiceLoop:
    def __init__(self, app: Friday, *, wake: bool = True, printer: Printer | None = None,
                 mic: Any = None, player: Any = None):  # fmt: skip
        if app.stt is None:
            raise RuntimeError("No speech-to-text engine. Install with: pip install -e '.[voice]'")
        self.app, self.wake = app, wake
        self.print = printer or (lambda kind, text: print(f"[{kind}] {text}"))
        self.mic = mic or Microphone()
        self.player = player or Player()
        self.speaker = Speaker(self.player.play, on_start=self.mic.muted.set, on_idle=self._unmute)
        self.stop = threading.Event()
        cfg = app.settings
        self.vad = EnergyVAD(VADConfig(silence_ms=cfg.vad_silence_ms, max_seconds=cfg.vad_max_seconds))
        self.reminders = ReminderService(app.store, self.announce)

    def _unmute(self) -> None:
        time.sleep(0.25)  # let the room echo die down
        self.mic.drain()
        self.mic.muted.clear()

    def say_text(self, text: str) -> None:
        audio = self.app.voice.synthesize(text)
        if audio is not None:
            self.speaker.say(audio)
            self.speaker.wait()

    def announce(self, text: str) -> None:
        self.print("reminder", text)
        self.say_text(f"Reminder: {text}")

    def handle(self, command: str, heard_at: float) -> bool:
        """Answer one command. Returns False when the user asked to exit."""
        kind = classify_command(command)
        if kind == "exit":
            self.say_text("Goodbye. Just say my name when you need me.")
            return False
        if kind == "stop":
            return True
        stream = SpeechStream(self.app.assistant.ask_stream(command), self.app.voice, started=heard_at)
        self.speaker.reset()
        report: dict[str, Any] = {}
        try:
            for ev in stream:
                if ev.kind == "text":
                    self.print("delta", ev.text)
                elif ev.kind == "tool_start":
                    self.print("tool", ev.text)
                elif ev.kind == "sentence" and ev.audio is not None:
                    self.speaker.say(ev.audio)
                elif ev.kind == "error":
                    self.print("error", ev.text)
                elif ev.kind == "done":
                    self.print("end", "")
                    report = ev.detail
            self.speaker.wait()
        except KeyboardInterrupt:  # barge-in from the keyboard
            stream.cancel()
            self.speaker.interrupted.set()
            self.player.stop()
            self.print("info", "(interrupted)")
            return True
        if self.speaker.first_play_at is not None:
            report["first_audio_played_ms"] = int((self.speaker.first_play_at - heard_at) * 1000)
        self.print("metrics", " · ".join(f"{k}={v}" for k, v in report.items() if v not in (None, [], 0)))
        return True

    def transcribe(self, audio: Audio) -> tuple[str, int]:
        t0 = time.perf_counter()
        assert self.app.stt is not None
        return self.app.stt.transcribe(audio), elapsed_ms(t0)

    def run(self) -> None:
        self.reminders.start()
        try:
            with self.mic:
                self.mic.muted.set()
                self.say_text(f"Online. {'Say my name' if self.wake else 'Go ahead'} whenever you need me.")
                self._loop()
        finally:
            self.reminders.stop()

    def _listen_once(self, timeout: float | None) -> tuple[Audio | None, float]:
        for utterance in self.mic.utterances(self.vad, self.stop, timeout=timeout):
            return utterance, time.perf_counter()
        return None, time.perf_counter()

    def _loop(self) -> None:
        follow_up = 0.0
        while not self.stop.is_set():
            audio, heard_at = self._listen_once(follow_up or None)
            if audio is None:  # follow-up window closed
                follow_up = 0.0
                if self.wake:
                    self.print("info", "(waiting for wake word)")
                continue
            text, stt_ms = self.transcribe(audio)
            if not text:
                continue
            command = text
            if self.wake and not follow_up:
                matched, command = match_wake_word(text, self.app.settings.wake_word_list)
                if not matched:
                    continue
                if not command:  # just the name: chime and wait for the request
                    self.mic.muted.set()
                    self.speaker.say(chime())
                    self.speaker.wait()
                    audio, heard_at = self._listen_once(self.app.settings.follow_up_seconds)
                    if audio is None:
                        continue
                    command, stt_ms = self.transcribe(audio)
                    if not command:
                        continue
            self.print("you", f"{command}  (stt {stt_ms} ms)")
            if not self.handle(command, heard_at):
                self.stop.set()
                break
            follow_up = self.app.settings.follow_up_seconds
