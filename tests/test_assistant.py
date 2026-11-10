from friday.app import Friday
from friday.assistant import STEP_LIMIT
from friday.llm.base import LLMResponse, ToolCall, Usage

from .conftest import ScriptedProvider, call, say


def test_answers_with_persona(friday: Friday, provider: ScriptedProvider):
    provider.responses = [say("Hello Parth.")]
    text, metrics = friday.assistant.ask("hi")
    assert text == "Hello Parth." and metrics.steps == 1 and metrics.first_token_ms is not None
    system = provider.calls[0]["system"]
    assert "Parth" in system and "SPEAKING" in system and "untrusted_data" in system


def test_tool_loop_events(friday: Friday, provider: ScriptedProvider):
    provider.responses = [call("get_weather"), say("It's about 26 degrees.")]
    events = list(friday.assistant.ask_stream("weather?"))
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["tool_start", "tool_end"] and kinds[-1] == "done"
    assert events[-1].metrics.tools == ["get_weather"] and events[-1].metrics.steps == 2
    tool_msg = provider.calls[1]["messages"][-1]
    assert tool_msg.role == "tool" and '"city": "Pune"' in tool_msg.content


def test_parallel_reads_and_ordering(friday: Friday, provider: ScriptedProvider):
    both = LLMResponse("", [ToolCall("a", "get_weather", {}), ToolCall("b", "current_time", {})], Usage())
    provider.responses = [both, say("ok")]
    friday.assistant.ask("weather and time")
    ids = [m.tool_call_id for m in provider.calls[1]["messages"] if m.role == "tool"]
    assert ids == ["a", "b"]


def test_injection_then_action_is_blocked_within_the_turn(friday: Friday, provider: ScriptedProvider, opened):
    provider.responses = [
        call("wikipedia", topic="Evil"),
        call("open_website", call_id="c2", target="https://evil.example"),
        say("That page tried to make me open a link, so I didn't."),
    ]
    events = list(friday.assistant.ask_stream("tell me about Evil"))
    ends = [e for e in events if e.kind == "tool_end"]
    assert ends[1].tool == "open_website" and ends[1].detail["blocked"] is True
    assert opened == []
    # next turn, the user asks directly → allowed
    provider.responses = [call("open_website", target="example.com"), say("Opened.")]
    friday.assistant.ask("open example.com")
    assert opened == ["https://example.com"]


def test_history_window_starts_at_a_user_turn(friday: Friday, provider: ScriptedProvider):
    friday.settings.history_turns = 1
    provider.responses = [call("current_time"), say("It's five."), say("Sure.")]
    a = friday.assistant
    a.ask("time?")
    a.ask("thanks")
    sent = provider.calls[-1]["messages"]
    assert sent[0].role == "user" and sent[0].content == "thanks" and len(sent) == 1


def test_step_limit(friday: Friday, provider: ScriptedProvider):
    friday.settings.max_steps = 2
    provider.responses = [call("current_time"), call("current_time", call_id="c2")]
    text, metrics = friday.assistant.ask("loop")
    assert text == STEP_LIMIT and metrics.steps == 2
