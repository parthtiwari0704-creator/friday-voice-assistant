"""FRIDAY's command line.

friday listen     hands-free voice conversation ("Hey FRIDAY, …")
friday serve      web HUD at http://localhost:8765
friday chat       type to FRIDAY (add --speak to hear her)
friday say        speak a line in any voice (or save it to a WAV file)
friday voices     list male/female voices · friday download: fetch voice models
friday doctor     check your setup
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Settings, get_settings
from .llm import ConfigError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="F.R.I.D.A.Y. — your open-source voice assistant.")
console = Console()
err = Console(stderr=True)
_state: dict[str, Any] = {}


@app.callback()
def main(
    voice: str = typer.Option(
        None, "--voice", help="Voice preset: female, male, female-uk, male-uk, female-2, male-2."
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    if voice:
        os.environ["FRIDAY_VOICE"] = voice
    get_settings.cache_clear()
    settings = get_settings()
    logging.basicConfig(level="INFO" if verbose else settings.log_level, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")  # fmt: skip
    _state["settings"] = settings


def _settings() -> Settings:
    return _state.get("settings") or get_settings()


def _app(**kwargs: Any) -> Any:
    from .app import build_app

    if "friday" not in _state:
        _state["friday"] = build_app(_settings(), **kwargs)
    return _state["friday"]


def _fail(message: str) -> None:
    err.print(f"[red]✗[/] {message}")
    raise typer.Exit(1)


def _printer() -> Any:
    def show(kind: str, text: str) -> None:
        if kind == "delta":
            console.print(text, end="", markup=False, highlight=False)
        elif kind == "end":
            console.print()
        elif kind == "you":
            console.print(f"\n[bold #ff7a6b]You ›[/] {text}")
            console.print("[bold #48c6ff]FRIDAY ›[/] ", end="")
        elif kind == "tool":
            console.print(f"[dim]⟨{text}⟩[/] ", end="")
        elif kind == "metrics":
            console.print(f"[dim]{text}[/]")
        elif kind == "reminder":
            console.print(f"\n[bold yellow]⏰ {text}[/]")
        elif kind == "error":
            console.print(f"[red]{text}[/]")
        else:
            console.print(f"[dim]{text}[/]")

    return show


@app.command()
def listen(no_wake: bool = typer.Option(False, "--no-wake", help="Answer everything you say (no wake word).")) -> None:
    """Hands-free voice conversation through your microphone and speakers."""
    from .voice.loop import VoiceLoop

    friday = _app()
    try:
        _ = friday.assistant
        loop = VoiceLoop(friday, wake=not no_wake, printer=_printer())
    except (ConfigError, RuntimeError, ImportError, OSError) as exc:
        _fail(str(exc))
    v = friday.voice.describe()
    console.print(f"[bold #48c6ff]F.R.I.D.A.Y. v{__version__}[/] · voice [b]{v['voice']}[/] ({v['engine']}) · "
                  f"ears [b]{friday.stt.name}[/] · say [b]“{friday.settings.wake_word_list[0]}”[/] · Ctrl+C to quit")  # fmt: skip
    try:
        loop.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Offline.[/]")


@app.command()
def chat(speak: bool = typer.Option(False, "--speak", help="Also speak the answers aloud.")) -> None:
    """Type to FRIDAY in the terminal (/voice <name>, /reset, /exit)."""
    from .app import ReminderService
    from .voice.pipeline import Speaker, SpeechStream

    friday = _app()
    try:
        assistant = friday.assistant
    except ConfigError as exc:
        _fail(str(exc))
    speaker = None
    if speak and friday.voice.enabled:
        from .voice.audio import Player

        speaker = Speaker(Player().play)
    show = _printer()
    ReminderService(friday.store, lambda t: show("reminder", t)).start()
    console.print("[bold #48c6ff]F.R.I.D.A.Y.[/] ready. [dim]/voice male · /reset · /exit[/]")
    while True:
        try:
            text = console.input("\n[bold #ff7a6b]You ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/exit", "exit", "quit"):
            break
        if text == "/reset":
            assistant.reset()
            continue
        if text.startswith("/voice"):
            console.print(friday.voice.set_voice(text.split(maxsplit=1)[1] if " " in text else "female"))
            continue
        console.print("[bold #48c6ff]FRIDAY ›[/] ", end="")
        stream = SpeechStream(assistant.ask_stream(text), friday.voice if speaker else None)
        for ev in stream:
            if ev.kind == "text":
                show("delta", ev.text)
            elif ev.kind == "tool_start":
                show("tool", ev.text)
            elif ev.kind == "sentence" and speaker and ev.audio is not None:
                speaker.say(ev.audio)
            elif ev.kind == "error":
                show("error", ev.text)
            elif ev.kind == "done":
                console.print()
                m = ev.detail
                show("metrics", f"first token {m.get('first_token_ms')} ms"
                     + (f" · first audio {m['first_audio_ms']} ms" if m.get("first_audio_ms") else "")
                     + (f" · tools: {', '.join(m['tools'])}" if m.get("tools") else ""))  # fmt: skip
        if speaker:
            speaker.wait()


@app.command()
def say(
    text: str,
    out: Path = typer.Option(None, "--out", "-o", help="Save to a WAV file instead of playing."),
    rate: float = typer.Option(None, help="Speaking rate, 0.5–2.0."),
) -> None:
    """Speak a line in the current voice (choose with --voice before the command)."""
    friday = _app(stt=None)
    if not friday.voice.enabled:
        _fail("No text-to-speech engine. Install: pip install -e '.[voice]'")
    if rate:
        friday.voice.rate = rate
    with console.status(f"Synthesising with {friday.voice.describe()['voice']}…"):
        try:
            audio = friday.voice.synthesize(text)
        except Exception as exc:  # noqa: BLE001
            _fail(f"Speech failed: {exc}")
    assert audio is not None
    if out:
        out.write_bytes(audio.to_wav())
        console.print(f"[green]✓[/] {out} ({audio.seconds:.1f} s, {audio.sample_rate} Hz)")
        return
    from .voice.audio import Player

    Player().play(audio)


@app.command()
def voices(all_voices: bool = typer.Option(False, "--all", help="List Piper's full online voice catalogue.")) -> None:
    """Male and female voice presets and whether they are installed."""
    from .voice.tts import PRESETS

    s = _settings()
    if all_voices:
        from piper.download_voices import list_voices

        list_voices()
        return
    table = Table("preset", "gender", "accent", "piper voice", "installed", "openai voice", title="FRIDAY voices")
    for p in PRESETS.values():
        installed = (s.voices_dir / f"{p.piper}.onnx").exists()
        table.add_row(("➤ " if p.key == s.voice else "  ") + p.key, p.gender, p.accent, p.piper,
                      "[green]yes[/]" if installed else "[dim]no[/]", p.openai)  # fmt: skip
    console.print(table)
    console.print(
        '[dim]Try one: friday --voice male say "Hello, I\'m FRIDAY." · download: friday download male female[/]'
    )


@app.command()
def download(
    names: list[str] = typer.Argument(None, help="Presets or Piper voice ids (default: female and male)."),
    whisper: bool = typer.Option(True, help="Also download the Whisper speech-recognition model."),
) -> None:
    """Download voice models (and Whisper) so FRIDAY works offline."""
    from .voice.tts import PiperTTS, resolve_preset

    s = _settings()
    s.ensure_dirs()
    tts = PiperTTS(s.voices_dir)
    for name in names or ["female", "male"]:
        preset = resolve_preset(name)
        voice = preset.piper if preset else name
        if tts.is_installed(voice):
            console.print(f"[green]✓[/] {voice} already installed")
            continue
        with console.status(f"Downloading {voice}…"):
            try:
                tts.download(voice)
            except Exception as exc:  # noqa: BLE001
                _fail(f"{voice}: {exc}")
        console.print(f"[green]✓[/] {voice}")
    if whisper:
        from .voice.stt import WhisperSTT

        with console.status(f"Downloading Whisper {s.whisper_model}…"):
            WhisperSTT(s.whisper_model, s.stt_language, str(s.home / "whisper"))._load()
        console.print(f"[green]✓[/] whisper {s.whisper_model}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
    insecure: bool = typer.Option(False, "--insecure", help="Allow a network bind without FRIDAY_API_TOKEN."),
) -> None:
    """Run the web HUD (voice in the browser) at http://localhost:8765."""
    import uvicorn

    from .server import create_app

    s = _settings()
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not s.api_token.get_secret_value() and not insecure:
        _fail("Refusing to expose FRIDAY on a network without FRIDAY_API_TOKEN (or pass --insecure).")
    friday = _app()
    console.print(f"[bold #48c6ff]F.R.I.D.A.Y.[/] HUD → http://{'localhost' if loopback else host}:{port}")
    uvicorn.run(create_app(friday), host=host, port=port, log_level="warning")


@app.command()
def doctor() -> None:
    """Check brain, voice, ears and integrations."""
    s = _settings()

    def has(module: str) -> bool:
        try:
            __import__(module)
            return True
        except Exception:  # noqa: BLE001
            return False

    key = (
        s.anthropic_api_key.get_secret_value()
        if s.provider == "anthropic"
        else (s.openai_api_key.get_secret_value() or s.openai_base_url)
    )
    installed = [p.name.removesuffix(".onnx") for p in s.voices_dir.glob("*.onnx")] if s.voices_dir.exists() else []
    rows = [
        ("Brain (LLM key)", bool(key), f"{s.provider} · {s.model}"),
        ("Neural voices (piper)", has("piper"), ", ".join(installed) or "none downloaded yet → friday download"),
        ("System voices (pyttsx3)", has("pyttsx3"), "fallback voice engine"),
        ("Speech recognition (faster-whisper)", has("faster_whisper"), s.whisper_model),
        ("Microphone/speakers (sounddevice)", has("sounddevice"), "needed for `friday listen`"),
        ("Active voice", True, f"{s.voice} · rate {s.speaking_rate}"),
        ("Wake words", True, ", ".join(s.wake_word_list)),
        ("Jarvis link (MCP)", True, s.jarvis_mcp_command or "off — set FRIDAY_JARVIS_MCP_COMMAND"),
        ("Allow-listed apps", True, ", ".join(s.app_map) or "none — set FRIDAY_APPS"),
    ]
    table = Table("check", "", "detail")
    for name, ok, detail in rows:
        table.add_row(name, "[green]✓[/]" if ok else "[yellow]–[/]", detail)
    console.print(table)


@app.command()
def version() -> None:
    """Show the version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
