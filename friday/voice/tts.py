"""Text-to-speech engines with male and female voices, behind one small interface.

| engine   | quality          | offline | male / female voices                                   |
|----------|------------------|---------|--------------------------------------------------------|
| piper    | neural (natural) | yes     | 6 presets below; any of Piper's ~100 voices by id       |
| system   | OS voices        | yes     | Windows (David/Zira/Heera…), macOS (Daniel/Samantha…)    |
| openai   | neural (cloud)   | no      | OpenAI, or any OpenAI-compatible server (e.g. Kokoro)   |

Every engine returns ``Audio`` (mono int16 + sample rate), so the pipeline, the speaker and the
web server don't care which one is active, and the voice can be switched mid-conversation.
"""

from __future__ import annotations

import io
import logging
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from ..config import Settings

log = logging.getLogger(__name__)
Gender = Literal["female", "male"]


@dataclass
class Audio:
    samples: np.ndarray  # mono int16
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    def to_wav(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.samples.astype(np.int16).tobytes())
        return buf.getvalue()

    @classmethod
    def from_wav(cls, data: bytes) -> Audio:
        with wave.open(io.BytesIO(data), "rb") as w:
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() > 1:
                frames = frames.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
            return cls(frames, w.getframerate())

    @classmethod
    def concat(cls, parts: list[Audio], gap_ms: int = 0) -> Audio:
        if not parts:
            return cls(np.zeros(0, dtype=np.int16), 22050)
        rate = parts[0].sample_rate
        gap = np.zeros(int(rate * gap_ms / 1000), dtype=np.int16)
        chunks: list[np.ndarray] = []
        for p in parts:
            chunks += [p.samples, gap]
        return cls(np.concatenate(chunks[:-1]) if chunks else np.zeros(0, np.int16), rate)


@dataclass(frozen=True)
class VoicePreset:
    key: str
    gender: Gender
    accent: str
    piper: str
    openai: str


PRESETS: dict[str, VoicePreset] = {
    p.key: p
    for p in [
        VoicePreset("female", "female", "American", "en_US-amy-medium", "nova"),
        VoicePreset("male", "male", "American", "en_US-ryan-medium", "onyx"),
        VoicePreset("female-uk", "female", "British", "en_GB-alba-medium", "shimmer"),
        VoicePreset("male-uk", "male", "British", "en_GB-alan-medium", "fable"),
        VoicePreset("female-2", "female", "American", "en_US-lessac-medium", "coral"),
        VoicePreset("male-2", "male", "American", "en_US-joe-medium", "echo"),
    ]
}
_ALIASES = {
    "woman": "female", "girl": "female", "lady": "female", "man": "male", "boy": "male", "guy": "male",
    "british female": "female-uk", "british male": "male-uk", "uk female": "female-uk", "uk male": "male-uk",
    "female british": "female-uk", "male british": "male-uk", "english female": "female-uk", "english male": "male-uk",
}  # fmt: skip


def resolve_preset(name: str) -> VoicePreset | None:
    key = name.strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key).replace(" ", "-")
    return PRESETS.get(key)


class TTSEngine(Protocol):
    name: str

    def synthesize(self, text: str, voice: str, rate: float = 1.0) -> Audio: ...

    def voice_for(self, preset: VoicePreset) -> str: ...


# --- Piper (offline neural) ------------------------------------------------------------------------
class PiperTTS:
    """Local neural TTS (ONNX). Voices are ~60 MB each and downloaded once on first use."""

    name = "piper"

    def __init__(self, voices_dir: Path, allow_download: bool = True):
        self.voices_dir, self.allow_download = voices_dir, allow_download
        self._voices: dict[str, Any] = {}
        self._lock = threading.Lock()

    def voice_for(self, preset: VoicePreset) -> str:
        return preset.piper

    def model_path(self, voice: str) -> Path:
        return self.voices_dir / f"{voice}.onnx"

    def is_installed(self, voice: str) -> bool:
        return self.model_path(voice).exists() and self.model_path(voice).with_suffix(".onnx.json").exists()

    def download(self, voice: str) -> Path:
        from piper.download_voices import download_voice

        self.voices_dir.mkdir(parents=True, exist_ok=True)
        log.warning("Downloading Piper voice %s (one time)…", voice)
        download_voice(voice, self.voices_dir)
        return self.model_path(voice)

    def _load(self, voice: str) -> Any:
        with self._lock:
            if voice not in self._voices:
                from piper import PiperVoice

                if not self.is_installed(voice):
                    if not self.allow_download:
                        raise FileNotFoundError(
                            f"Piper voice '{voice}' is not installed: run `friday download {voice}`"
                        )
                    self.download(voice)
                self._voices[voice] = PiperVoice.load(self.model_path(voice))
            return self._voices[voice]

    def synthesize(self, text: str, voice: str, rate: float = 1.0) -> Audio:
        from piper import SynthesisConfig

        model = self._load(voice)
        # Piper's length_scale is duration: >1 is slower. Speaking rate is its inverse.
        config = SynthesisConfig(length_scale=1.0 / max(0.5, min(2.0, rate)))
        chunks = list(model.synthesize(text, syn_config=config))
        if not chunks:
            return Audio(np.zeros(0, dtype=np.int16), model.config.sample_rate)
        return Audio(np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate)


# --- System voices (SAPI5 / NSSpeech / eSpeak via pyttsx3) --------------------------------------------
_FEMALE_NAMES = ("zira", "hazel", "heera", "kalpana", "susan", "samantha", "karen", "victoria", "moira", "tessa",
                 "fiona", "veena", "allison", "ava", "serena", "catherine", "female", "woman")  # fmt: skip
_MALE_NAMES = ("david", "mark", "george", "ravi", "hemant", "james", "richard", "alex", "daniel", "fred", "tom",
               "oliver", "aaron", "arthur", "rishi", "male", "man")  # fmt: skip


def guess_gender(voice: Any) -> Gender | None:
    """pyttsx3 reports gender on some platforms only; fall back to well-known voice names."""
    gender = (getattr(voice, "gender", None) or "").lower()
    if gender.startswith(("female", "f")) and gender != "":
        return "female"
    if gender.startswith(("male", "m")):
        return "male"
    name = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
    if any(n in name for n in _FEMALE_NAMES):
        return "female"
    if any(n in name for n in _MALE_NAMES):
        return "male"
    return None


class SystemTTS:
    """The operating system's built-in voices. Lower quality, zero download."""

    name = "system"

    def __init__(self) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        self._lock = threading.Lock()
        self.voices = list(self._engine.getProperty("voices") or [])

    def voice_for(self, preset: VoicePreset) -> str:
        english = [v for v in self.voices if "en" in str(getattr(v, "languages", "") or getattr(v, "id", "")).lower()]
        for pool in (english, self.voices):
            match = next((v for v in pool if guess_gender(v) == preset.gender), None)
            if match is not None:
                return str(match.id)
        return str(self.voices[0].id) if self.voices else ""

    def synthesize(self, text: str, voice: str, rate: float = 1.0) -> Audio:
        with self._lock, tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            if voice:
                self._engine.setProperty("voice", voice)
            self._engine.setProperty("rate", int(180 * rate))
            self._engine.save_to_file(text, str(path))
            self._engine.runAndWait()
            return decode_any(path.read_bytes())


def decode_any(data: bytes, target_rate: int | None = None) -> Audio:
    """WAV via the stdlib; anything else (AIFF from macOS, MP3, WebM/Opus) via PyAV if installed."""
    try:
        audio = Audio.from_wav(data)
        if target_rate is None or audio.sample_rate == target_rate:
            return audio
    except (wave.Error, EOFError):
        pass
    import av  # ships with faster-whisper

    with av.open(io.BytesIO(data)) as container:
        rate = target_rate or container.streams.audio[0].rate
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        frames = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                frames.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            frames.append(out.to_ndarray().reshape(-1))
    samples = np.concatenate(frames).astype(np.int16) if frames else np.zeros(0, dtype=np.int16)
    return Audio(samples, rate)


# --- OpenAI-compatible cloud/local servers ---------------------------------------------------------------
class OpenAITTS:
    """OpenAI's /v1/audio/speech, or any compatible server via OPENAI_BASE_URL (e.g. Kokoro-FastAPI)."""

    name = "openai"

    def __init__(self, api_key: str, base_url: str | None, model: str = "tts-1", client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self._client, self.model = client, model

    def voice_for(self, preset: VoicePreset) -> str:
        return preset.openai

    def synthesize(self, text: str, voice: str, rate: float = 1.0) -> Audio:
        resp = self._client.audio.speech.create(
            model=self.model, voice=voice, input=text, response_format="wav", speed=max(0.25, min(4.0, rate))
        )
        return decode_any(resp.read() if hasattr(resp, "read") else resp.content)


# --- the voice manager (what tools and UIs talk to) --------------------------------------------------------
class VoiceManager:
    """Owns the active engine, voice and rate. Thread-safe; implements tools.VoiceControl."""

    def __init__(self, engine: TTSEngine | None, voice: str = "female", rate: float = 1.0):
        self.engine = engine
        self.rate = rate
        self._lock = threading.Lock()
        self.preset: VoicePreset | None = None
        self.voice_id = ""
        if engine is not None:
            self._apply(voice)

    def _apply(self, name: str) -> None:
        assert self.engine is not None
        preset = resolve_preset(name)
        self.preset = preset
        self.voice_id = self.engine.voice_for(preset) if preset else name

    @property
    def enabled(self) -> bool:
        return self.engine is not None

    def set_voice(self, voice: str, speed: float | None = None) -> dict[str, Any]:
        if self.engine is None:
            return {"error": "Speech output is disabled (FRIDAY_TTS_ENGINE=none)."}
        with self._lock:
            previous = (self.preset, self.voice_id, self.rate)
            self._apply(voice)
            if speed is not None:
                self.rate = max(0.5, min(2.0, float(speed)))
            try:  # prove the new voice works before confirming (downloads a Piper voice if needed)
                self.engine.synthesize("Okay.", self.voice_id, self.rate)
            except Exception as exc:  # noqa: BLE001
                self.preset, self.voice_id, self.rate = previous
                return {"error": f"Could not switch to '{voice}': {exc}"}
        return self.describe()

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.engine.name if self.engine else "none",
            "voice": self.preset.key if self.preset else self.voice_id,
            "gender": self.preset.gender if self.preset else None,
            "accent": self.preset.accent if self.preset else None,
            "voice_id": self.voice_id,
            "speed": self.rate,
            "presets": {k: f"{p.gender}, {p.accent}" for k, p in PRESETS.items()},
        }

    def synthesize(self, text: str) -> Audio | None:
        if self.engine is None or not text.strip():
            return None
        with self._lock:
            voice, rate = self.voice_id, self.rate
        return self.engine.synthesize(text, voice, rate)


def create_engine(settings: Settings) -> TTSEngine | None:
    choice = settings.tts_engine
    if choice == "none":
        return None
    if choice in ("piper", "auto"):
        try:
            import piper  # noqa: F401

            return PiperTTS(settings.voices_dir)
        except ImportError:
            if choice == "piper":
                raise
    if choice in ("system", "auto"):
        try:
            return SystemTTS()
        except Exception as exc:
            if choice == "system":
                raise
            log.warning("No system TTS available (%s); speech output disabled.", exc)
            return None
    return OpenAITTS(settings.openai_api_key.get_secret_value(), settings.openai_base_url, settings.openai_tts_model)
