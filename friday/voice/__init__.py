"""Speech in, speech out: TTS engines, STT engines, VAD, wake word and the streaming pipeline."""

from .pipeline import SpeechEvent, SpeechStream
from .text import SentenceChunker, clean_for_speech, match_wake_word
from .tts import PRESETS, Audio, VoiceManager, resolve_preset

__all__ = [
    "PRESETS",
    "Audio",
    "SentenceChunker",
    "SpeechEvent",
    "SpeechStream",
    "VoiceManager",
    "clean_for_speech",
    "match_wake_word",
    "resolve_preset",
]
