import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from friday.app import Friday, ReminderService, classify_command
from friday.voice.loop import VoiceLoop
from friday.voice.pipeline import Speaker, SpeechStream
from friday.voice.tts import Audio, VoiceManager

from .conftest import FakeSTT, FakeTTS, ScriptedProvider, call, say


def events(friday: Friday, text: str):
    return list(SpeechStream(friday.assistant.ask_stream(text), friday.voice))


def test_sentences_are_synthesised_in_order(friday: Friday, provider: ScriptedProvider, tts: FakeTTS):
    provider.responses = [say("**It's** 26°C in Pune. Rain is unlikely! Have a good day.")]
    evs = events(friday, "weather")
    sentences = [e for e in evs if e.kind == "sentence"]
    assert [s.text for s in sentences] == ["It's 26 degrees Celsius in Pune.", "Rain is unlikely!", "Have a good day."]
    assert all(s.audio is not None for s in sentences) and [s.index for s in sentences] == [0, 1, 2]
    assert [c[0] for c in tts.calls] == [s.text for s in sentences]
    done = evs[-1]
    assert done.kind == "done" and done.detail["first_audio_ms"] is not None and done.detail["tts_ms"] >= 0


def test_first_audio_arrives_before_the_answer_finishes(friday: Friday, tts: FakeTTS):
    """The whole point of streaming: speech starts while the LLM is still generating."""
    friday._assistant = None
    friday._provider = ScriptedProvider([say("First sentence here. " + "and more words " * 15 + "end.")], delay=0.02)
    stream = SpeechStream(friday.assistant.ask_stream("talk"), friday.voice)
    first_sentence_at = None
    for ev in stream:
        if ev.kind == "sentence" and first_sentence_at is None:
            first_sentence_at = time.perf_counter()
        if ev.kind == "done":
            done_at = time.perf_counter()
    assert done_at - first_sentence_at > 0.2


def test_tool_events_pass_through_and_text_only_mode(friday: Friday, provider: ScriptedProvider):
    provider.responses = [call("current_time"), say("It's evening.")]
    evs = list(SpeechStream(friday.assistant.ask_stream("time"), None))  # browser speaks: no server TTS
    kinds = [e.kind for e in evs]
    assert "tool_start" in kinds and "tool_end" in kinds
    assert [e.audio for e in evs if e.kind == "sentence"] == [None]


def test_tts_failure_skips_one_sentence_only(friday: Friday, provider: ScriptedProvider):
    friday.voice = VoiceManager(FakeTTS(fail_on="broken"), "female")
    provider.responses = [say("Fine one. This is broken. Fine two.")]
    evs = events(friday, "x")
    audio = [(e.text, e.audio is not None) for e in evs if e.kind == "sentence"]
    assert audio == [("Fine one.", True), ("This is broken.", False), ("Fine two.", True)]


def test_llm_error_becomes_an_apology(friday: Friday):
    class Broken(ScriptedProvider):
        def stream(self, **kw):
            raise RuntimeError("model down")
            yield

    friday._assistant, friday._provider = None, Broken()
    evs = events(friday, "x")
    assert any(e.kind == "error" and "model down" in e.text for e in evs)
    assert [e.text for e in evs if e.kind == "sentence"] == ["Sorry, something went wrong on my side."]


def test_cancel_stops_synthesis(friday: Friday, tts: FakeTTS):
    friday._assistant = None
    friday._provider = ScriptedProvider([say("One. Two. Three. Four. Five. Six.")], delay=0.03)
    stream = SpeechStream(friday.assistant.ask_stream("count"), friday.voice)
    got = []
    for ev in stream:
        if ev.kind == "sentence":
            got.append(ev.text)
            stream.cancel()
    assert len(got) <= 2 and len(tts.calls) <= 2


def test_speaker_plays_in_order_and_reports_idle():
    played = []
    sp = Speaker(lambda a: played.append(int(a.samples[0])) or True)
    for i in range(3):
        sp.say(Audio(np.full(10, i, np.int16), 16000))
    assert sp.wait(2) and played == [0, 1, 2] and sp.first_play_at is not None


def test_reminder_service_fires_once(friday: Friday):
    heard = []
    svc = ReminderService(friday.store, heard.append)
    friday.store.add_reminder("Stand up", datetime.now(timezone.utc) - timedelta(seconds=1))
    friday.store.add_reminder("Later", datetime.now(timezone.utc) + timedelta(hours=1))
    assert svc.check() == 1 and svc.check() == 0 and heard == ["Stand up"]


def test_classify_command():
    assert classify_command("Goodbye.") == "exit" and classify_command("never mind") == "stop"
    assert classify_command("what's the weather") == "ask"


# --- the hands-free loop, with a fake microphone and speakers ------------------------------------
class FakeMic:
    def __init__(self, n: int):
        self.muted = threading.Event()
        self.pending = [Audio(np.full(1600, 500, np.int16), 16000) for _ in range(n)]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def drain(self):
        pass

    def utterances(self, vad, stop, timeout=None):
        if self.pending:
            yield self.pending.pop(0)


class FakePlayer:
    def __init__(self):
        self.played: list[Audio] = []

    def play(self, audio):
        self.played.append(audio)
        return True

    def stop(self):
        pass


def test_voice_loop_wake_word_follow_up_and_exit(friday: Friday, provider: ScriptedProvider, tts: FakeTTS):
    friday.stt = FakeSTT([
        "random chatter",                    # ignored: no wake word
        "Hey Friday, what's the weather?",   # answered
        "and tomorrow?",                      # follow-up window: no wake word needed
        "goodbye",                            # exits
    ])  # fmt: skip
    provider.responses = [say("Sunny, 26 degrees."), say("Cooler tomorrow.")]
    log = []
    loop = VoiceLoop(friday, mic=FakeMic(4), player=FakePlayer(), printer=lambda k, t: log.append((k, t)))
    loop.run()
    asked = [m.content for c in provider.calls for m in c["messages"] if m.role == "user"]
    assert "what's the weather?" in asked and "and tomorrow?" in asked
    assert "random chatter" not in asked
    spoken = [c[0] for c in tts.calls]
    assert spoken[0].startswith("Online.") and "Sunny, 26 degrees." in spoken and "Cooler tomorrow." in spoken
    assert spoken[-1].startswith("Goodbye")
    assert any(k == "metrics" and "first_audio" in t for k, t in log)


def test_voice_loop_wake_word_alone_chimes_then_listens(friday: Friday, provider: ScriptedProvider, tts: FakeTTS):
    friday.stt = FakeSTT(["Friday", "set a timer for five minutes", ""])
    provider.responses = [call("set_timer", minutes=5), say("Timer set for five minutes.")]
    player = FakePlayer()
    loop = VoiceLoop(friday, mic=FakeMic(3), player=player, printer=lambda k, t: None)
    loop.stop.set()  # run one pass manually instead of the infinite loop
    loop.stop.clear()
    t = threading.Thread(target=loop.run, daemon=True)
    t.start()
    t.join(3)
    loop.stop.set()
    assert friday.store.pending_reminders()[0]["text"] == "Timer is done."
    assert len(player.played) >= 3  # greeting, chime, answer


def test_reminder_service_stops_when_store_closes():
    from friday.store import Store

    store = Store(":memory:")
    svc = ReminderService(store, lambda t: None, interval=0.05).start()
    store.close()
    svc._thread.join(1)
    assert not svc._thread.is_alive()
