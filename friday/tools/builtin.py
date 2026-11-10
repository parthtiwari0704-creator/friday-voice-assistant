"""FRIDAY's built-in tools. Each is a plain typed function; see registry.py."""

from __future__ import annotations

import ast
import math
import operator
import os
import platform
import re
import shlex
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Protocol
from urllib.parse import quote, quote_plus

import httpx

from ..config import Settings
from ..store import Store
from .registry import ToolRegistry

TIMEOUT = 10.0
UA = {"User-Agent": "FRIDAY-assistant/1.0 (open-source voice assistant)"}


class VoiceControl(Protocol):
    def set_voice(self, voice: str, speed: float | None = None) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...


# --- safe calculator -----------------------------------------------------------------------------
_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}  # fmt: skip
_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
    "exp": math.exp, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil, "pi": math.pi, "e": math.e,
}  # fmt: skip


def safe_eval(expression: str) -> float | int:
    """Evaluate arithmetic without ``eval``: walk the AST and allow only numbers, operators
    and a whitelist of math functions. Exponents are bounded to stop CPU-exhaustion inputs."""
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"\1/100*", expression, flags=re.IGNORECASE)

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _FUNCS and not callable(_FUNCS[node.id]):
            return _FUNCS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and callable(_FUNCS.get(node.func.id)):
            if node.keywords or len(node.args) > 2:
                raise ValueError("unsupported call")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression: {ast.dump(node)[:40]}")

    if len(expression) > 200:
        raise ValueError("expression too long")
    result = ev(ast.parse(expression, mode="eval"))
    return round(result, 10) if isinstance(result, float) else result


# --- time parsing ----------------------------------------------------------------------------------
def parse_when(at: str | None, minutes: float | None, now: datetime) -> datetime:
    """'in N minutes', '18:30', '6:30 pm', '7 am' or an ISO datetime → aware datetime (future)."""
    if minutes is not None:
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        return now + timedelta(minutes=minutes)
    if not at:
        raise ValueError("give either minutes_from_now or at")
    text = at.strip().lower().replace(".", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if m:
        hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if half == "pm" and hour < 12:
            hour += 12
        if half == "am" and hour == 12:
            hour = 0
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(f"invalid time '{at}'")
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return due if due > now else due + timedelta(days=1)
    due = datetime.fromisoformat(at)
    due = due if due.tzinfo else due.replace(tzinfo=now.tzinfo)
    if due <= now:
        raise ValueError("that time is in the past")
    return due


def spoken_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


# --- launching ---------------------------------------------------------------------------------------
def launch(command: str) -> None:
    """Start an allow-listed app without a shell (no injection via the command string)."""
    if sys.platform.startswith("win"):
        os.startfile(command)  # type: ignore[attr-defined]
    elif sys.platform == "darwin" and not command.startswith("/"):
        subprocess.Popen(["open", "-a", command])
    else:
        subprocess.Popen(shlex.split(command))


def build_tools(
    settings: Settings,
    store: Store,
    voice: VoiceControl | None = None,
    *,
    open_url: Callable[[str], Any] = webbrowser.open,
    open_app_fn: Callable[[str], Any] = launch,
    http: httpx.Client | None = None,
) -> ToolRegistry:
    reg = ToolRegistry()
    tz = settings.tz
    client = http or httpx.Client(timeout=TIMEOUT, headers=UA, follow_redirects=True)

    # --- time, timers, reminders ---------------------------------------------------------------
    @reg.register
    def current_time() -> str:
        """The current local date and time."""
        return datetime.now(tz).strftime("%A, %d %B %Y, %I:%M %p")

    @reg.register
    def set_timer(minutes: float, label: str = "Timer") -> dict[str, Any]:
        """Start a countdown timer; FRIDAY announces it when it ends.

        Args:
            minutes: Length in minutes (fractions allowed, e.g. 0.5 for 30 seconds).
            label: What the timer is for.
        """
        due = parse_when(None, minutes, datetime.now(tz))
        rid = store.add_reminder(f"{label} is done.", due)
        return {"id": rid, "ends_at": spoken_time(due)}

    @reg.register
    def set_reminder(text: str, minutes_from_now: float | None = None, at: str | None = None) -> dict[str, Any]:
        """Remind the user of something later. Give minutes_from_now OR at.

        Args:
            text: What to remind them about.
            minutes_from_now: e.g. 20.
            at: A clock time like '18:30' or '6:30 pm' (today, or tomorrow if passed), or ISO datetime.
        """
        due = parse_when(at, minutes_from_now, datetime.now(tz))
        rid = store.add_reminder(text, due)
        return {"id": rid, "due": due.strftime("%A %I:%M %p")}

    @reg.register
    def list_reminders() -> list[dict[str, Any]]:
        """Upcoming reminders and timers."""
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "due": datetime.fromisoformat(r["due_at"]).astimezone(tz).strftime("%a %I:%M %p"),
            }
            for r in store.pending_reminders()
        ]

    @reg.register
    def cancel_reminder(reminder_id: int) -> dict[str, Any]:
        """Cancel a reminder or timer.

        Args:
            reminder_id: The id from list_reminders.
        """
        return {"cancelled": store.cancel_reminder(reminder_id)}

    # --- memory ------------------------------------------------------------------------------------
    @reg.register
    def take_note(text: str) -> dict[str, Any]:
        """Save a note or a fact the user wants remembered.

        Args:
            text: The note, as a full sentence.
        """
        return {"saved": True, "id": store.add_note(text)}

    @reg.register
    def search_notes(query: str = "") -> list[dict[str, Any]]:
        """Find saved notes (most recent first when no query).

        Args:
            query: Keywords; empty for the latest notes.
        """
        return [{"id": n["id"], "text": n["text"]} for n in store.notes(query)]

    # --- knowledge ---------------------------------------------------------------------------------
    @reg.register
    def calculate(expression: str) -> dict[str, Any]:
        """Exact arithmetic. Use for any calculation instead of mental math.

        Args:
            expression: e.g. '15% of 2400', '(3.5 * 12) / 7', 'sqrt(2)'.
        """
        return {"expression": expression, "result": safe_eval(expression)}

    @reg.register
    def get_weather(city: str | None = None) -> dict[str, Any]:
        """Current weather and today's forecast.

        Args:
            city: City name; defaults to the user's city.
        """
        name = city or settings.city
        if not name:
            return {"error": "No city given and FRIDAY_CITY is not set."}
        geo = client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": name, "count": 1}).json()
        if not geo.get("results"):
            return {"error": f"City '{name}' not found"}
        loc = geo["results"][0]
        wx = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": settings.timezone,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "forecast_days": 1,
            },
        ).json()  # fmt: skip
        cur, day = wx["current"], wx["daily"]
        return {
            "city": loc["name"], "now_c": cur["temperature_2m"], "feels_like_c": cur["apparent_temperature"],
            "conditions_code": cur["weather_code"], "wind_kmh": cur["wind_speed_10m"],
            "high_c": day["temperature_2m_max"][0], "low_c": day["temperature_2m_min"][0],
            "rain_chance_pct": day["precipitation_probability_max"][0],
        }  # fmt: skip

    @reg.register(untrusted=True)
    def wikipedia(topic: str) -> dict[str, Any]:
        """A short encyclopedia summary of a person, place or thing.

        Args:
            topic: The article title or subject.
        """
        r = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(topic.replace(' ', '_'))}")
        if r.status_code == 404:
            return {"error": f"No Wikipedia article for '{topic}'"}
        r.raise_for_status()
        data = r.json()
        return {"title": data.get("title"), "summary": (data.get("extract") or "")[:1500]}

    # --- the machine ------------------------------------------------------------------------------
    @reg.register
    def system_status() -> dict[str, Any]:
        """CPU, memory and battery status of this computer."""
        try:
            import psutil
        except ImportError:
            return {"os": platform.platform(), "note": "install psutil for CPU/memory/battery details"}
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        return {
            "os": platform.system(),
            "cpu_percent": psutil.cpu_percent(interval=0.3),
            "memory_percent": psutil.virtual_memory().percent,
            "battery_percent": round(battery.percent) if battery else None,
            "plugged_in": battery.power_plugged if battery else None,
        }

    @reg.register(action=True)
    def open_website(target: str) -> dict[str, Any]:
        """Open a website, or search the web, in the user's browser.

        Args:
            target: A URL or domain (e.g. 'github.com') or a search query.
        """
        target = target.strip()
        if re.fullmatch(r"(https?://)?[\w.-]+\.[a-z]{2,}(/\S*)?", target, flags=re.IGNORECASE):
            url = target if target.lower().startswith(("http://", "https://")) else f"https://{target}"
        else:
            url = f"https://duckduckgo.com/?q={quote_plus(target)}"
        open_url(url)
        return {"opened": url}

    @reg.register(action=True)
    def open_app(name: str) -> dict[str, Any]:
        """Open an application the user has allow-listed in FRIDAY_APPS.

        Args:
            name: The app's spoken name, e.g. 'spotify'.
        """
        apps = settings.app_map
        key = name.strip().lower()
        match = apps.get(key) or next((cmd for k, cmd in apps.items() if key in k or k in key), None)
        if match is None:
            return {"error": f"'{name}' is not allow-listed.", "allowed": sorted(apps)}
        open_app_fn(str(match))
        return {"opened": name}

    # --- FRIDAY herself ------------------------------------------------------------------------------
    if voice is not None:

        @reg.register
        def set_voice(voice_name: str, speed: float | None = None) -> dict[str, Any]:
            """Change how FRIDAY sounds: 'male', 'female', 'male-uk', 'female-uk', 'male-2', 'female-2'.

            Args:
                voice_name: A voice preset, or an engine-specific voice id.
                speed: Speaking rate from 0.5 (slow) to 2.0 (fast); 1.0 is normal.
            """
            return voice.set_voice(voice_name, speed)

        @reg.register
        def current_voice() -> dict[str, Any]:
            """Which voice and speaking rate FRIDAY is using, and which voices are available."""
            return voice.describe()

    return reg
