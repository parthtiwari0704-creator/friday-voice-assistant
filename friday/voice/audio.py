"""Microphone capture, interruptible playback, and voice-activity detection (VAD)."""

from __future__ import annotations

import collections
import contextlib
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .tts import Audio

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if len(frame) else 0.0


@dataclass
class VADConfig:
    start_frames: int = 3  # 90 ms of speech to start (ignores clicks)
    silence_ms: int = 700  # this much silence ends the utterance
    preroll_ms: int = 300  # keep audio from just before speech began (first syllable)
    max_seconds: float = 15.0
    min_rms: float = 300.0  # absolute floor (int16 scale) so a silent room never "speaks"
    ratio: float = 3.0  # speech must be this many times louder than the noise floor


class EnergyVAD:
    """Adaptive energy-based endpointing, fed one 30 ms frame at a time.

    The noise floor is an exponential moving average updated only while nobody is speaking,
    so it follows fans and traffic. Speech starts after ``start_frames`` loud frames and ends
    after ``silence_ms`` of quiet. Returns the utterance (with pre-roll) when it ends.
    """

    def __init__(self, cfg: VADConfig | None = None):
        self.cfg = cfg or VADConfig()
        self.noise = self.cfg.min_rms / self.cfg.ratio
        self.preroll: collections.deque[np.ndarray] = collections.deque(maxlen=max(1, self.cfg.preroll_ms // FRAME_MS))
        self.speech: list[np.ndarray] = []
        self.loud_run = 0
        self.quiet_run = 0
        self.active = False

    @property
    def threshold(self) -> float:
        return max(self.cfg.min_rms, self.noise * self.cfg.ratio)

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        level = rms(frame)
        loud = level > self.threshold
        if not self.active:
            if not loud:
                self.noise = 0.95 * self.noise + 0.05 * level
            self.preroll.append(frame)
            self.loud_run = self.loud_run + 1 if loud else 0
            if self.loud_run >= self.cfg.start_frames:
                self.active, self.quiet_run = True, 0
                self.speech = list(self.preroll)
            return None
        self.speech.append(frame)
        self.quiet_run = 0 if loud else self.quiet_run + 1
        too_long = len(self.speech) * FRAME_MS / 1000 >= self.cfg.max_seconds
        if self.quiet_run * FRAME_MS >= self.cfg.silence_ms or too_long:
            utterance = np.concatenate(self.speech)
            self.reset()
            return utterance
        return None

    def reset(self) -> None:
        self.active, self.loud_run, self.quiet_run, self.speech = False, 0, 0, []
        self.preroll.clear()


class Microphone:
    """16 kHz mono int16 frames from the default input device (sounddevice/PortAudio)."""

    def __init__(self, device: int | str | None = None):
        import sounddevice as sd

        self._sd, self.device = sd, device
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1000)
        self.muted = threading.Event()  # set while FRIDAY speaks, so she doesn't hear herself
        self._stream: Any = None

    def __enter__(self) -> Microphone:
        def callback(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
            if not self.muted.is_set():
                with contextlib.suppress(queue.Full):
                    self.frames.put_nowait(indata[:, 0].copy())

        stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME, device=self.device, callback=callback
        )
        stream.start()
        self._stream = stream
        return self

    def __exit__(self, *_: object) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def drain(self) -> None:
        while not self.frames.empty():
            self.frames.get_nowait()

    def utterances(self, vad: EnergyVAD, stop: threading.Event, timeout: float | None = None) -> Iterator[Audio]:
        """Yield utterances until ``stop`` is set (or ``timeout`` seconds pass without speech)."""
        deadline = time.monotonic() + timeout if timeout else None
        while not stop.is_set():
            if deadline and time.monotonic() > deadline and not vad.active:
                return
            try:
                frame = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue
            utterance = vad.feed(frame)
            if utterance is not None:
                yield Audio(utterance, SAMPLE_RATE)
                if deadline:
                    deadline = time.monotonic() + (timeout or 0)


class Player:
    """Plays Audio on the default output device; ``stop()`` interrupts immediately (barge-in)."""

    def __init__(self, device: int | str | None = None):
        import sounddevice as sd

        self._sd, self.device = sd, device
        self._stop = threading.Event()

    def play(self, audio: Audio) -> bool:
        """Block until finished. Returns False if interrupted."""
        self._stop.clear()
        if not len(audio.samples):
            return True
        self._sd.play(audio.samples, audio.sample_rate, device=self.device)
        end = time.monotonic() + audio.seconds + 0.2
        while time.monotonic() < end:
            if self._stop.wait(0.02):
                self._sd.stop()
                return False
        self._sd.wait()
        return True

    def stop(self) -> None:
        self._stop.set()


def chime(rate: int = 22050, up: bool = True) -> Audio:
    """A short two-note earcon: 'I'm listening' (up) / 'done' (down)."""
    notes = (660.0, 880.0) if up else (880.0, 660.0)
    parts = []
    for f in notes:
        t = np.arange(int(rate * 0.09)) / rate
        env = np.minimum(1, np.minimum(t / 0.01, (0.09 - t) / 0.03))
        parts.append((np.sin(2 * np.pi * f * t) * env * 9000).astype(np.int16))
    return Audio(np.concatenate(parts), rate)
