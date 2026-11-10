import pytest

from friday.voice.text import SentenceChunker, clean_for_speech, match_wake_word

WAKE = ["friday", "hey friday", "ok friday"]


def chunk(deltas: list[str]) -> list[str]:
    c = SentenceChunker()
    out = [s for d in deltas for s in c.feed(d)]
    return out + c.flush()


def test_sentences_are_emitted_as_soon_as_they_complete():
    c = SentenceChunker()
    assert c.feed("It's 26 degrees") == []
    assert c.feed(" in Pune. Light") == ["It's 26 degrees in Pune."]
    assert c.feed(" rain later!") == []  # no trailing space yet: might be "later!!"
    assert c.feed(" Bring") == ["Light rain later!"]
    assert c.flush() == ["Bring"]


def test_abbreviations_initials_and_decimals_do_not_split():
    out = chunk(["Dr. Rao and J. K. Singh paid 3.5 lakh, ", "i.e. a lot. Next sentence."])
    assert out == ["Dr. Rao and J. K. Singh paid 3.5 lakh, i.e. a lot.", "Next sentence."]


def test_run_on_text_is_split_at_a_comma():
    c = SentenceChunker(max_chars=60)
    out = c.feed("word " * 8 + ", and then " + "more " * 12)
    assert out and out[0].endswith(",") and len(out[0]) <= 60


def test_newlines_split():
    assert chunk(["First line\nSecond line"]) == ["First line", "Second line"]


def test_clean_for_speech():
    text = "## Plan\n- **Gym** at 6:30 → `fun` [link](https://x.y) 😊\n1. It's 26°C, 20% rain & ₹1,840 due https://pay.example"
    assert clean_for_speech(text) == (
        "Plan Gym at 6:30 → fun link It's 26 degrees Celsius, 20 percent rain and 1,840 rupees due a link"
    )


@pytest.mark.parametrize(
    "heard, command",
    [
        ("Hey, Friday, what's the time?", "what's the time?"),
        ("friday set a timer", "set a timer"),
        ("Fry day, open Spotify.", "open Spotify."),
        ("um okay friday weather", "weather"),
        ("Friday's weather please", "weather please"),
        ("OK Friday.", ""),
        ("Friday", ""),
    ],
)
def test_wake_word_variants(heard, command):
    assert match_wake_word(heard, WAKE) == (True, command)


@pytest.mark.parametrize("heard", ["what's the time", "I love Fridays", "see you on friday", "", "fried eggs please"])
def test_not_the_wake_word(heard):
    assert match_wake_word(heard, WAKE)[0] is False
