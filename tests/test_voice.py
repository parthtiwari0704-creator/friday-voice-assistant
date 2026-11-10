import shutil
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest

from friday.voice import tts as tts_mod
from friday.voice.audio import EnergyVAD, VADConfig, chime
from friday.voice.stt import OpenAISTT, to_float32_16k
from friday.voice.tts import PRESETS, Audio, OpenAITTS, PiperTTS, VoiceManager, decode_any, guess_gender, resolve_preset

from .conftest import FakeTTS


def test_presets_cover_male_and_female():
    genders = [p.gender for p in PRESETS.values()]
    assert genders.count("male") == 3 and genders.count("female") == 3
    assert resolve_preset("Woman").key == "female" and resolve_preset("british male").key == "male-uk"
    assert resolve_preset("MALE_UK").key == "male-uk" and resolve_preset("robot") is None


def test_voice_manager_switches_and_rolls_back():
    engine = FakeTTS(fail_on="Okay")
    vm = VoiceManager(FakeTTS(), "female")
    assert vm.describe()["voice"] == "female" and vm.voice_id == "fake-female"
    d = vm.set_voice("male", speed=1.4)
    assert d["gender"] == "male" and d["speed"] == 1.4 and vm.synthesize("Hi") is not None
    vm.engine = engine  # the next voice "fails to load"
    assert "error" in vm.set_voice("female-uk")
    assert vm.preset.key == "male" and vm.rate == 1.4  # unchanged
    assert vm.set_voice("x", speed=9)["error"]
    off = VoiceManager(None)
    assert not off.enabled and off.synthesize("hi") is None and "error" in off.set_voice("male")


def test_raw_engine_voice_ids_pass_through():
    vm = VoiceManager(FakeTTS(), "en_US-kristin-medium")
    assert vm.preset is None and vm.voice_id == "en_US-kristin-medium"


def test_audio_wav_roundtrip_and_concat():
    a = Audio(np.arange(-500, 500, dtype=np.int16), 22050)
    b = Audio.from_wav(a.to_wav())
    assert b.sample_rate == 22050 and np.array_equal(a.samples, b.samples)
    joined = Audio.concat([a, a], gap_ms=100)
    assert len(joined.samples) == 2000 + 2205


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_decode_browser_webm_opus(tmp_path):
    """The HUD records WebM/Opus; it must decode to 16 kHz mono for Whisper."""
    wav = tmp_path / "tone.wav"
    tone = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 8000).astype(np.int16)
    wav.write_bytes(Audio(tone, 48000).to_wav())
    webm = tmp_path / "tone.webm"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "libopus", str(webm)], check=True)
    audio = decode_any(webm.read_bytes(), target_rate=16000)
    assert audio.sample_rate == 16000 and 0.9 < audio.seconds < 1.1


def test_gender_guess():
    assert guess_gender(NS(name="Microsoft Zira Desktop", id="z", gender=None)) == "female"
    assert guess_gender(NS(name="Microsoft David Desktop", id="d", gender=None)) == "male"
    assert guess_gender(NS(name="Microsoft Heera - English (India)", id="h", gender=None)) == "female"
    assert guess_gender(NS(name="x", id="y", gender="Female")) == "female"
    assert guess_gender(NS(name="Robot", id="r", gender=None)) is None


def test_piper_engine_uses_rate_and_concatenates(monkeypatch, tmp_path):
    seen = {}

    class FakeVoice:
        config = NS(sample_rate=22050)

        @classmethod
        def load(cls, path):
            seen["path"] = path
            return cls()

        def synthesize(self, text, syn_config=None):
            seen["length_scale"] = syn_config.length_scale
            chunk = NS(audio_int16_array=np.ones(100, dtype=np.int16), sample_rate=22050)
            return [chunk, chunk]

    import piper

    monkeypatch.setattr(piper, "PiperVoice", FakeVoice)
    engine = PiperTTS(tmp_path, allow_download=False)
    with pytest.raises(FileNotFoundError):
        engine.synthesize("hi", "en_US-amy-medium")
    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"x")
    (tmp_path / "en_US-amy-medium.onnx.json").write_text("{}")
    audio = engine.synthesize("Hello. World.", "en_US-amy-medium", rate=2.0)
    assert len(audio.samples) == 200 and seen["length_scale"] == 0.5


def test_openai_tts_and_stt_adapters():
    wav = Audio(np.zeros(1600, dtype=np.int16), 16000).to_wav()
    sent = {}

    def create_speech(**kw):
        sent.update(kw)
        return NS(read=lambda: wav)

    client = NS(audio=NS(speech=NS(create=create_speech),
                         transcriptions=NS(create=lambda **kw: NS(text=" hello there "))))  # fmt: skip
    tts = OpenAITTS("k", None, client=client)
    assert tts.voice_for(PRESETS["male"]) == "onyx"
    assert (
        tts.synthesize("hi", "onyx", 1.2).seconds == 0.1 and sent["speed"] == 1.2 and sent["response_format"] == "wav"
    )
    assert OpenAISTT("k", None, client=client).transcribe(Audio(np.zeros(10, np.int16), 16000)) == "hello there"


def test_resample_to_16k():
    audio = Audio(np.zeros(48000, dtype=np.int16), 48000)
    assert len(to_float32_16k(audio)) == 16000


def test_create_engine_choices(settings):
    assert tts_mod.create_engine(settings.model_copy(update={"tts_engine": "none"})) is None
    assert isinstance(tts_mod.create_engine(settings.model_copy(update={"tts_engine": "piper"})), PiperTTS)


# --- VAD ---------------------------------------------------------------------------------------
def frames(signal: np.ndarray):
    for i in range(0, len(signal) - 479, 480):
        yield signal[i : i + 480]


def test_vad_finds_utterance_with_preroll():
    rng = np.random.default_rng(0)
    quiet = (rng.normal(0, 60, 16000)).astype(np.int16)  # 1 s room noise
    speech = (np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) * 6000).astype(np.int16)  # 1 s "voice"
    signal = np.concatenate([quiet, speech, quiet])
    vad = EnergyVAD(VADConfig(silence_ms=600))
    found = [u for f in frames(signal) if (u := vad.feed(f)) is not None]
    assert len(found) == 1
    seconds = len(found[0]) / 16000
    assert 1.5 < seconds < 2.1  # speech + pre-roll + trailing silence


def test_vad_ignores_clicks_and_caps_length():
    vad = EnergyVAD(VADConfig(max_seconds=1.0))
    click = np.concatenate([np.full(480, 9000, np.int16), np.zeros(480 * 20, np.int16)])
    assert all(vad.feed(f) is None for f in frames(click))  # one loud frame is not speech
    loud = np.full(16000 * 3, 9000, dtype=np.int16)
    out = [u for f in frames(loud) if (u := vad.feed(f)) is not None]
    assert out and len(out[0]) / 16000 <= 1.4


def test_chime():
    up, down = chime(), chime(up=False)
    assert up.seconds == pytest.approx(0.18, abs=0.01) and not np.array_equal(up.samples, down.samples)
