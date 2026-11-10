# Changelog

## 1.0.0 — 2026-11-10

First release.

- Hands-free voice loop: adaptive VAD, local Whisper, fuzzy wake word, follow-up window, chime, keyboard barge-in.
- Concurrent speech pipeline (LLM → sentence chunker → TTS → playback) with first-token and first-audio metrics.
- Six male/female voice presets across Piper (offline neural), system voices and OpenAI-compatible TTS; voice and speed switchable by command, CLI or HUD.
- Streaming tool-using assistant (Anthropic, OpenAI-compatible/Ollama) with a speech-optimised persona.
- 15 tools: time, timers, reminders (spoken, persistent), notes, calculator (AST-safe), weather, Wikipedia, open website, allow-listed apps, system status, voice control.
- Taint-tracking prompt-injection policy; untrusted-content fencing.
- MCP bridge to Jarvis for Gmail/Calendar tools.
- Web HUD: animated orb, push-to-talk and hands-free modes, server or browser voices, live tool timeline, reminder push.
- CLI: `listen`, `serve`, `chat`, `say`, `voices`, `download`, `doctor`.
- 84 tests without network, models or audio hardware; CI on Windows, macOS and Linux.
