"""Text utilities for speech: sentence streaming, speech cleanup, wake-word matching."""

from __future__ import annotations

import difflib
import re

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "no", "approx", "jr", "sr", "prof", "inc"}
_BOUNDARY = re.compile(r"([.!?…]+[\"')\]]*)(\s+)|(\n+)")


class SentenceChunker:
    """Turns a stream of LLM text deltas into complete sentences, as early as possible.

    Speaking sentence 1 while the model is still writing sentence 2 is what makes a voice
    assistant feel instant: time-to-first-audio becomes *first sentence* latency instead of
    *whole answer* latency. Abbreviations ("Dr. Rao") and decimals ("3.5") are not boundaries;
    a very long clause is split at a comma so audio never waits on a run-on sentence.
    """

    def __init__(self, min_chars: int = 2, max_chars: int = 220):
        self.buffer = ""
        self.min_chars, self.max_chars = min_chars, max_chars

    def feed(self, delta: str) -> list[str]:
        self.buffer += delta
        out: list[str] = []
        pos = 0
        for m in _BOUNDARY.finditer(self.buffer):
            end = m.end(1) if m.group(1) else m.start()
            candidate = self.buffer[pos:end].strip()
            if m.group(1):
                last_word = re.split(r"\s+", candidate)[-1].rstrip(".!?…\"')]").lower() if candidate else ""
                if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]", last_word):
                    continue  # "Dr." or an initial like "J." is not the end of a sentence
            if len(candidate) >= self.min_chars:
                out.append(candidate)
                pos = m.end()
        self.buffer = self.buffer[pos:]
        if len(self.buffer) > self.max_chars:  # run-on: split at the last comma/semicolon
            cut = max(self.buffer.rfind(", ", 0, self.max_chars), self.buffer.rfind("; ", 0, self.max_chars))
            if cut > self.max_chars // 3:
                out.append(self.buffer[: cut + 1].strip())
                self.buffer = self.buffer[cut + 2 :]
        return out

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest else []


_MD = [
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"https?://\S+"), "a link"),
    (re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1"), r"\2"),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),
    (re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.MULTILINE), ""),
    (re.compile(r"<[^>]+>"), " "),
]
_SPOKEN = [
    (re.compile(r"(\d)\s*°\s*C\b"), r"\1 degrees Celsius"),
    (re.compile(r"(\d)\s*°\s*F\b"), r"\1 degrees Fahrenheit"),
    (re.compile(r"(\d)\s*°"), r"\1 degrees"),
    (re.compile(r"(\d)\s*%"), r"\1 percent"),
    (re.compile(r"\s&\s"), " and "),
    (re.compile(r"₹\s?(\d[\d,]*)"), r"\1 rupees"),
    (re.compile(r"\$\s?(\d[\d,.]*)"), r"\1 dollars"),
]
_EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]")


def clean_for_speech(text: str) -> str:
    """Remove what a TTS engine would read out literally (markdown, URLs, emoji)."""
    for pattern, repl in _MD:
        text = pattern.sub(repl, text)
    for pattern, repl in _SPOKEN:
        text = pattern.sub(repl, text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


_FILLERS = {"um", "uh", "erm", "hmm", "okay", "ok", "so", "and", "oh", "yo", "hey", "hi", "hello", "please"}


def _normalise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def match_wake_word(transcript: str, wake_words: list[str], threshold: float = 0.78) -> tuple[bool, str]:
    """Does the utterance start with the wake word? Returns (matched, the command after it).

    Speech recognisers mishear names ("fry day", "Friday's", "hey Freddie"), so tokens in the
    first few words are compared fuzzily (difflib ratio), and a two-token split is joined.
    """
    words = _normalise(transcript)
    if not words:
        return False, ""
    for phrase in sorted(wake_words, key=len, reverse=True):
        target = _normalise(phrase)
        n = len(target)
        for start in range(min(3, len(words))):  # allow filler lead-in only ("um", "okay so")
            if start and not all(w in _FILLERS for w in words[:start]):
                break
            window = words[start : start + n]
            joined = words[start : start + n + 1]
            candidates = [
                " ".join(window),
                " ".join([*window[:-1], "".join(joined[n - 1 :])]) if len(joined) > n else "",
            ]
            for cand in candidates:
                if cand and difflib.SequenceMatcher(None, cand, " ".join(target)).ratio() >= threshold:
                    used = n + (1 if cand == candidates[1] and cand != candidates[0] else 0)
                    rest = " ".join(transcript.split()[start + used :])
                    return True, rest.lstrip(",.!? ").strip()
    return False, ""
