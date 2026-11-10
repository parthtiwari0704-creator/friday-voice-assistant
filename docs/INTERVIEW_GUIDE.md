# Presenting FRIDAY in an interview

## 45-second pitch

> "FRIDAY is a voice-first AI assistant I built in Python. You say 'Hey FRIDAY', Whisper transcribes you locally, a tool-using LLM works out the answer, and she replies out loud in a male or female neural voice that runs offline.
>
> The interesting engineering is latency and safety. For latency, the LLM, the sentence splitter, the text-to-speech and the speaker all run concurrently, so she starts talking after the first sentence, and every answer reports its time to first audio. For safety, she reads web pages and emails while also being able to open apps, so I added taint tracking: once third-party content enters a turn, action tools are refused for that turn. A malicious page can't make her do anything. She can also use my other project, Jarvis, for email and calendar over MCP."

## 5-minute demo

1. `friday serve` and open the HUD. Ask *"What's the weather?"* Point out the tool chip, the orb reacting to speech, and the `first audio … ms` metric.
2. Voice tab: preview **Female**, then **Male UK**. *"Six offline neural voices; one protocol over three engines."*
3. Ask *"Remind me in one minute to stretch."* A minute later the reminder is spoken and pushed to the HUD.
4. Ask *"Tell me about Pune and then open its Wikipedia page."* The open is refused, and FRIDAY explains why. *"That's the taint policy working."*
5. In the terminal: `friday listen`, then *"Hey FRIDAY, set a timer for five minutes"*, then *"and what's 18% of 2450?"*, which needs no wake word thanks to the follow-up window.

Before interviewing, run it on your machine and quote **your own** measured first-audio latency.

## Questions and answers

**Why stream sentence by sentence?**
Perceived latency is time to first sound, not time to the whole answer. Three stages overlap through queues: while sentence 1 plays, sentence 2 is being synthesised and sentence 3 generated. I measure first-token and first-audio for every turn.

**How do you decide where a sentence ends while text is still streaming?**
Only when the punctuation is followed by whitespace, so "3.5" and "later!!" aren't split early. Abbreviations and initials are excluded, and a long run-on is split at a comma so audio never stalls.

**How does the wake word work, and what's the trade-off?**
FRIDAY transcribes short utterances and fuzzy-matches the start against the wake words, allowing filler lead-ins and misheard splits like "fry day". It needs no training and supports any name, but it costs more CPU than a dedicated keyword-spotting model, which I'd add next.

**How do you detect when the user has stopped speaking?**
An energy VAD with an adaptive noise floor: an EMA updated only in silence, a start threshold of three loud frames, 300 ms of pre-roll, and 700 ms of silence to end. It's tested on synthetic noise and speech.

**Why doesn't she hear herself?**
The mic is muted while she speaks and drained afterwards. Real barge-in needs acoustic echo cancellation, which the browser provides, so the HUD records with echo cancellation on.

**What is taint tracking and why is it needed?**
It's indirect prompt injection defence. Untrusted tool output marks the turn as tainted, and action tools then refuse in code, regardless of what the model asks for. Prompting the model to be careful helps, but it isn't a guarantee; the policy is.

**How does the Jarvis integration work?**
Over MCP. FRIDAY's loop is synchronous, so an MCPBridge runs an asyncio loop in a background thread holding one client session, and calls cross over with `run_coroutine_threadsafe`. The remote tools are registered with their schemas and marked untrusted.

**How did you test audio without a microphone?**
Every hardware boundary is an interface: a fake mic yields utterances, a fake player records audio, and fake TTS/STT and a scripted streaming LLM stand in for the models. The whole hands-free loop runs in a unit test. I also drove the HUD in a headless browser, which caught a missing WebSocket dependency that the unit tests couldn't.

**Why Piper and faster-whisper?**
Both run offline on a CPU and are fast enough for real time, so they're private and free. I kept OpenAI-compatible engines behind the same protocol for higher quality when the cloud is acceptable.

**What would you improve?**
A neural wake word, streaming STT, echo cancellation for barge-in, speaker verification before actions, and a recorded-command eval set tracking word error rate and latency percentiles.

## Resume bullets

- Built **FRIDAY**, an open-source **voice AI assistant** (Python, FastAPI, WebSockets): local Whisper speech recognition, a streaming tool-using LLM agent, and offline neural TTS with switchable **male and female voices**.
- Designed a **concurrent speech pipeline** (LLM → sentence chunker → TTS → playback) that begins speaking after the first sentence, with per-turn time-to-first-audio metrics.
- Implemented **taint-based prompt-injection defence** that blocks device actions after untrusted content, plus an **MCP bridge** to a separate email/calendar agent.
- Shipped a web HUD (animated, push-to-talk and hands-free modes) and 84 hardware-free tests with cross-platform CI.

## Be honest about

- You built it with AI assistance. Say so if asked, and make sure you can explain every design decision here.
- Latency numbers depend on the machine and model, so quote only what you measured.
- The transcript-based wake word costs CPU; you know the alternative.
