# FRIDAY architecture

## 1. The latency budget

A voice assistant feels "alive" when it answers in under about a second. A naive pipeline adds up every stage:

```
naive:      [listen]──[STT]──[LLM: whole answer ~2–4 s]──[TTS: whole answer ~1 s]──▶ speech
FRIDAY:     [listen]──[STT]──[LLM: first sentence]──[TTS: sentence 1]──▶ speech
                                   └─ sentence 2 ─────[TTS: 2]──────────▶ (queued while 1 plays)
```

| Stage | What FRIDAY does | Typical cost |
|---|---|---|
| End of speech | Energy VAD with adaptive noise floor; 700 ms of silence ends the utterance | 0.7 s (tunable) |
| STT | faster-whisper `base.en`, int8, **greedy** (beam 1), VAD-trimmed | 0.2–0.6 s on a laptop CPU |
| LLM | Streaming; prompt-cached tool schemas | first token 0.3–1 s |
| TTS | Piper, **one sentence at a time** | ~0.1–0.3 s per sentence |
| Playback | Separate thread; next sentence synthesises while this one plays | — |

`SpeechStream` runs three threads joined by queues: **LLM → SentenceChunker → TTS → consumer**. The consumer is either the local speaker thread or the WebSocket sending WAV frames to the browser. Every turn reports `first_token_ms`, `first_audio_ms` and `tts_ms`, so improvements are measured, not guessed.

## 2. Sentence chunking

`SentenceChunker` emits a sentence once its end is certain: punctuation *followed by whitespace* (so "3.5" never splits, and "later!" waits to see if "!!" follows). Abbreviations ("Dr.", "i.e.") and initials ("J. K.") are not boundaries. A run-on clause longer than 220 characters is split at a comma so audio never stalls. Each sentence then passes through `clean_for_speech`, which strips markdown and URLs and turns "26°C" into "26 degrees Celsius" and "₹1,840" into "1,840 rupees". The persona prompt also tells the model it is *speaking*, so most of this never appears in the first place.

## 3. Listening

- **VAD** (`EnergyVAD`): RMS per 30 ms frame. The noise floor is an exponential moving average updated only during silence, so it adapts to fans and traffic. Speech starts after 3 loud frames (ignores clicks), keeps 300 ms of pre-roll (so the first syllable isn't lost), and is capped at 15 s.
- **Wake word**: matched on the *transcript* with fuzzy matching (`difflib` ratio ≥ 0.78 over the first words, allowing filler lead-ins like "um, okay", and joining split tokens like "fry day"). A dedicated keyword-spotting model (openWakeWord/Porcupine) would use less CPU when idle; the transcript approach needs no training data and works for any custom name. That trade-off is documented rather than hidden.
- **Echo**: the mic is muted while FRIDAY speaks and drained afterwards, so she never answers herself. True barge-in (talking over her) needs acoustic echo cancellation; browsers provide it (`echoCancellation: true`), which is why the HUD records with it on.
- **Conversation flow**: idle → wake word → answer → **6 s follow-up window** (no wake word needed) → idle. Saying just "Friday" plays a chime and waits for the request.

## 4. Voices

`TTSEngine` is a two-method protocol (`synthesize`, `voice_for`). Presets map a gender and accent to each engine's voice:

- **Piper**: offline neural (VITS → ONNX), ~60 MB per voice, faster than real time on CPU. Speaking rate maps to `length_scale = 1/rate`.
- **System**: pyttsx3 over SAPI5 / NSSpeechSynthesizer / eSpeak. Gender comes from voice metadata when present, otherwise from well-known voice names (Zira and Heera are female, David and Ravi male).
- **OpenAI-compatible**: `/v1/audio/speech`, which works with OpenAI or self-hosted servers such as Kokoro.

`VoiceManager.set_voice` synthesises a test word before confirming a switch. If a voice fails (for example a download error), it rolls back, so a bad request can never leave FRIDAY mute.

## 5. Security: taint tracking

FRIDAY reads third-party text (Wikipedia, and email and calendar entries through Jarvis) while holding tools that act on your computer (`open_website`, `open_app`). Defences:

1. Untrusted output is fenced in `<untrusted_data>`, which can't be escaped, and flagged when it contains injection-like phrasing.
2. **Taint policy (the guarantee):** each user turn has a `TurnState`. Any untrusted tool result taints the turn, and action tools are then **refused for the rest of that turn**, whatever the model decides. If an email says "open this link", the most FRIDAY can do is tell you about it. Asking her directly starts a clean turn. This is a deliberately simple version of the information-flow-control idea in research defences such as CaMeL.
3. Apps must be allow-listed (`FRIDAY_APPS`) and are launched without a shell. The calculator walks the AST (no `eval`) and bounds exponents.
4. The HUD binds to localhost; using it from the network requires `FRIDAY_API_TOKEN` (WebSocket `?token=`, constant-time comparison).

*Trade-off:* taint is turn-granular. If you ask "read my email and then open GitHub" in one sentence, the open is refused. That's the safe failure mode, and FRIDAY explains why.

## 6. MCP bridge (FRIDAY ↔ Jarvis)

The assistant loop is synchronous (voice threads), while MCP clients are async. `MCPBridge` owns a private event loop in a daemon thread that keeps one `Client` session open; calls cross over with `run_coroutine_threadsafe`. Tools are discovered at start-up and registered with their remote JSON Schemas as `jarvis_*`, marked untrusted. If Jarvis is missing, FRIDAY logs a warning and carries on.

## 7. Web protocol

One WebSocket per browser tab, each with its own conversation and shared tools, voice and memory. JSON control frames, with binary frames for audio:

- client → `ask` · `audio` + WebM/Opus bytes · `voice` · `stop` · `reset`
- server → `transcript` · `delta` · `tool_start`/`tool_end` · `sentence` (+ WAV) · `done` (metrics) · `reminder` (+ WAV) · `error`

Server-side Whisper decodes the browser's WebM/Opus via PyAV. In "Browser" modes the HUD uses the Web Speech API instead, so it works with zero model downloads. Reminders are pushed to every open HUD.

## 8. Testing without hardware

Everything physical sits behind an interface, so tests use fakes: `ScriptedProvider` (a streaming LLM with optional delay), `FakeTTS` and `FakeSTT`, a fake microphone and player for the full hands-free loop, `httpx.MockTransport` for web APIs, and an **in-process MCP server** standing in for Jarvis. Notable tests: first audio arrives before the LLM finishes; a TTS failure skips one sentence only; injected content can't trigger `open_website`; the WebSocket delivers ordered sentence and WAV frames; the VAD finds speech in synthetic noise. One gap that unit tests couldn't catch, a missing WebSocket dependency, was found by driving the HUD in a real headless browser, which is now part of the release checklist.

## 9. What I'd do next

- A neural wake-word model for lower idle CPU; streaming STT for partial transcripts.
- Acoustic echo cancellation for true barge-in on desktop.
- An eval set of recorded commands, measuring word error rate, intent success and p50/p95 first-audio latency per engine.
- Speaker identification, so only the owner can trigger actions.
