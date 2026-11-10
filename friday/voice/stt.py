"""Speech-to-text: local Whisper (faster-whisper / CTranslate2) or any OpenAI-compatible API."""

from __future__ import annotations

import io
import logging
import threading
from typing import Any, Protocol

import numpy as np

from ..config import Settings
from .tts import Audio

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000


class STTEngine(Protocol):
    name: str

    def transcribe(self, audio: Audio | bytes) -> str: ...


def to_float32_16k(audio: Audio) -> np.ndarray:
    samples = audio.samples.astype(np.float32) / 32768.0
    if audio.sample_rate != SAMPLE_RATE and len(samples):  # linear resample (mic is already 16 kHz)
        n = int(len(samples) * SAMPLE_RATE / audio.sample_rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)
    return samples


class WhisperSTT:
    """Runs Whisper locally. ``base.en`` (~150 MB) is a good speed/accuracy balance on a laptop CPU;
    ``small.en`` is more accurate; ``tiny.en`` is fastest. Models download once on first use."""

    name = "whisper"

    def __init__(self, model: str = "base.en", language: str = "en", download_root: str | None = None):
        self.model_name, self.language, self.download_root = model, language, download_root
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            log.warning("Loading Whisper model %s (first run downloads it)…", self.model_name)
            self._model = WhisperModel(
                self.model_name, device="auto", compute_type="int8", download_root=self.download_root
            )
        return self._model

    def transcribe(self, audio: Audio | bytes) -> str:
        # Encoded bytes (e.g. WebM/Opus from the browser) are decoded by faster-whisper via PyAV.
        source: Any = io.BytesIO(audio) if isinstance(audio, bytes) else to_float32_16k(audio)
        with self._lock:
            segments, _info = self._load().transcribe(
                source,
                language=self.language or None,
                beam_size=1,  # greedy: much faster, nearly as accurate for short commands
                vad_filter=True,  # skip silence before decoding
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()


class OpenAISTT:
    name = "openai"

    def __init__(self, api_key: str, base_url: str | None, model: str = "whisper-1", client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self._client, self.model = client, model

    def transcribe(self, audio: Audio | bytes) -> str:
        if isinstance(audio, Audio):
            data, filename = audio.to_wav(), "speech.wav"
        else:
            data, filename = audio, "speech.webm"
        resp = self._client.audio.transcriptions.create(model=self.model, file=(filename, data))
        return str(getattr(resp, "text", resp)).strip()


def create_stt(settings: Settings) -> STTEngine | None:
    if settings.stt_engine in ("whisper", "auto"):
        try:
            import faster_whisper  # noqa: F401

            return WhisperSTT(settings.whisper_model, settings.stt_language, str(settings.home / "whisper"))
        except ImportError:
            if settings.stt_engine == "whisper":
                raise
    if settings.stt_engine == "openai" or settings.openai_api_key.get_secret_value():
        return OpenAISTT(settings.openai_api_key.get_secret_value(), settings.openai_base_url)
    return None
