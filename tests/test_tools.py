import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from friday.app import Friday
from friday.tools import TurnState, parse_when, safe_eval
from friday.tools.registry import ToolRegistry, build_schema, validate

TZ = ZoneInfo("Asia/Kolkata")


def run(f: Friday, _tool: str, _turn: TurnState | None = None, **args):
    result = f.registry.execute(_tool, args, _turn)
    return result, (json.loads(result.content) if result.content.startswith(("{", "[")) else result.content)


@pytest.mark.parametrize(
    "expr, value",
    [
        ("15% of 2400", 360.0),
        ("(3.5 * 12) / 7", 6.0),
        ("2^10", 1024),
        ("sqrt(16) + abs(-2)", 6.0),
        ("round(pi, 2)", 3.14),
    ],
)
def test_calculator(expr, value):
    assert safe_eval(expr) == value


@pytest.mark.parametrize(
    "expr", ["__import__('os').system('ls')", "open('x')", "9**9**9", "(lambda: 1)()", "a.b", "[1,2]", "x" * 300]
)
def test_calculator_rejects_code_and_bombs(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_parse_when():
    now = datetime(2026, 11, 10, 17, 0, tzinfo=TZ)
    assert parse_when(None, 20, now) == now + timedelta(minutes=20)
    assert parse_when("18:30", None, now).hour == 18
    assert parse_when("6:30 pm", None, now) == now.replace(hour=18, minute=30)
    assert parse_when("7 am", None, now) == (now + timedelta(days=1)).replace(hour=7, minute=0)  # passed → tomorrow
    assert parse_when("12 am", None, now).hour == 0
    with pytest.raises(ValueError):
        parse_when("25:00", None, now)
    with pytest.raises(ValueError):
        parse_when(None, None, now)
    with pytest.raises(ValueError):
        parse_when("2020-01-01T10:00:00", None, now)


def test_reminders_and_timers(friday: Friday):
    _, timer = run(friday, "set_timer", minutes=5, label="Tea")
    _, rem = run(friday, "set_reminder", text="Call mom", at="9 pm")
    _, listed = run(friday, "list_reminders")
    assert {r["text"] for r in listed} == {"Tea is done.", "Call mom"}
    assert run(friday, "cancel_reminder", reminder_id=timer["id"])[1] == {"cancelled": True}
    assert [r["id"] for r in run(friday, "list_reminders")[1]] == [rem["id"]]


def test_notes(friday: Friday):
    run(friday, "take_note", text="My manager is Anil")
    run(friday, "take_note", text="Parking is on level B2")
    assert run(friday, "search_notes", query="parking level")[1][0]["text"] == "Parking is on level B2"
    assert len(run(friday, "search_notes")[1]) == 2


def test_weather_and_wikipedia(friday: Friday):
    _, wx = run(friday, "get_weather")
    assert wx["city"] == "Pune" and wx["high_c"] == 31
    _, text = run(friday, "wikipedia", topic="Pune")
    assert text.startswith('<untrusted_data source="wikipedia">') and "Maharashtra" in text
    assert "error" in run(friday, "wikipedia", topic="Nowhere")[1]


def test_open_website_and_apps(friday: Friday, opened: list[str]):
    assert run(friday, "open_website", target="github.com")[1] == {"opened": "https://github.com"}
    assert run(friday, "open_website", target="best biryani in pune")[1]["opened"].startswith(
        "https://duckduckgo.com/?q=best+biryani"
    )
    assert run(friday, "open_app", name="Spotify")[1] == {"opened": "Spotify"}
    _, refused = run(friday, "open_app", name="terminal")
    assert "not allow-listed" in refused["error"] and refused["allowed"] == ["calculator", "spotify"]
    assert opened == ["https://github.com", opened[1], "app:spotify"]


def test_taint_blocks_actions_after_untrusted_content(friday: Friday, opened: list[str]):
    turn = TurnState()
    _, page = run(friday, "wikipedia", turn, topic="Evil")
    assert 'injection_suspected="true"' in page and turn.tainted
    result, body = run(friday, "open_website", turn, target="https://evil.example")
    assert result.blocked and body["status"] == "blocked" and "wikipedia" in body["reason"]
    assert opened == []
    # a fresh turn (the user asking directly) is allowed
    assert run(friday, "open_website", TurnState(), target="https://example.com")[0].blocked is False


def test_reads_do_not_taint(friday: Friday):
    turn = TurnState()
    run(friday, "get_weather", turn)
    run(friday, "calculate", turn, expression="1+1")
    assert not turn.tainted


def test_voice_tools(friday: Friday, tts):
    assert run(friday, "set_voice", voice_name="male")[1]["gender"] == "male"
    assert run(friday, "current_voice")[1]["voice_id"] == "fake-male"


def test_system_status(friday: Friday):
    status = run(friday, "system_status")[1]
    assert "cpu_percent" in status and "memory_percent" in status


def test_schema_and_validation():
    def f(text: str, count: int = 2, mode: str | None = None) -> None:
        """Do it.

        Args:
            text: The text,
                continued here.
            count: How many.
        """

    desc, schema = build_schema(f)
    assert desc == "Do it." and schema["required"] == ["text"]
    assert schema["properties"]["text"]["description"] == "The text, continued here."
    assert schema["properties"]["count"] == {"type": "integer", "description": "How many.", "default": 2}
    assert validate(schema, {"count": True, "x": 1}) == [
        "missing argument 'text'", "argument 'count' should be integer", "unknown argument 'x'",
    ]  # fmt: skip


def test_registry_errors_never_raise():
    reg = ToolRegistry()

    @reg.register
    def boom() -> None:
        """Fails."""
        raise RuntimeError("nope")

    assert reg.execute("boom", {}).is_error
    assert reg.execute("missing", {}).is_error
    with pytest.raises(ValueError):
        reg.register(boom)
