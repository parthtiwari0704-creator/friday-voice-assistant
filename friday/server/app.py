"""Web HUD + WebSocket voice protocol (FastAPI).

Client → server (JSON text frames; audio as one binary frame right after an ``audio`` message):
    {"type": "ask", "text": "...", "speak": "server" | "browser" | "none"}
    {"type": "audio", "mime": "audio/webm", "speak": ...}  + <binary frame>
    {"type": "voice", "voice": "male", "speed": 1.1}
    {"type": "stop"} | {"type": "reset"}

Server → client:
    transcript · delta · tool_start · tool_end · sentence (+ binary WAV when has_audio) · done ·
    voice · reminder (+ binary WAV) · error

Each socket gets its own conversation; tools, voice settings and memory are shared.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .. import __version__
from ..app import Friday, ReminderService, elapsed_ms
from ..llm import ConfigError
from ..voice.pipeline import SpeechStream
from ..voice.tts import PRESETS, resolve_preset

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class SayRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    voice: str | None = None


class Hub:
    """Tracks open sockets so background events (reminders) reach every HUD."""

    def __init__(self) -> None:
        self.clients: set[tuple[WebSocket, asyncio.AbstractEventLoop]] = set()
        self.lock = threading.Lock()

    def broadcast(self, message: dict[str, Any], audio: bytes | None = None) -> None:
        with self.lock:
            targets = list(self.clients)
        for ws, loop in targets:
            asyncio.run_coroutine_threadsafe(_send(ws, message, audio), loop)


async def _send(ws: WebSocket, message: dict[str, Any], audio: bytes | None = None) -> None:
    with suppress(Exception):
        await ws.send_text(json.dumps(message, default=str))
        if audio is not None:
            await ws.send_bytes(audio)


def create_app(friday: Friday) -> FastAPI:
    f = friday
    hub = Hub()
    bearer = HTTPBearer(auto_error=False)

    def token_ok(token: str | None) -> bool:
        expected = f.settings.api_token.get_secret_value()
        return not expected or (token is not None and secrets.compare_digest(token, expected))

    def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if not token_ok(creds.credentials if creds else None):
            raise HTTPException(401, "Invalid or missing API token")

    def announce(text: str) -> None:
        audio = f.voice.synthesize(f"Reminder: {text}") if f.voice.enabled else None
        hub.broadcast({"type": "reminder", "text": text, "has_audio": audio is not None},
                      audio.to_wav() if audio else None)  # fmt: skip

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service = ReminderService(f.store, announce).start()
        yield
        service.stop()

    app = FastAPI(title="FRIDAY", version=__version__, lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/capabilities", dependencies=[Depends(require_auth)])
    def capabilities() -> dict[str, Any]:
        return {
            "user_name": f.settings.user_name,
            "wake_words": f.settings.wake_word_list,
            "tts": f.voice.describe() | {"enabled": f.voice.enabled},
            "stt": f.stt.name if f.stt else None,
            "llm": {"provider": f.settings.provider, "model": f.settings.model},
            "tools": f.registry.names(),
            "jarvis_tools": len(f.bridged_tools),
            "presets": {k: {"gender": p.gender, "accent": p.accent} for k, p in PRESETS.items()},
        }

    @app.post("/api/say", dependencies=[Depends(require_auth)])
    def say(req: SayRequest) -> Response:
        """Synthesise text with any voice preset without changing the active voice (voice preview)."""
        if f.voice.engine is None:
            raise HTTPException(503, "Server speech is disabled")
        preset = resolve_preset(req.voice) if req.voice else f.voice.preset
        voice_id = f.voice.engine.voice_for(preset) if preset else (req.voice or f.voice.voice_id)
        try:
            audio = f.voice.engine.synthesize(req.text, voice_id, f.voice.rate)
        except Exception as exc:
            raise HTTPException(503, f"Speech failed: {exc}") from exc
        return Response(audio.to_wav(), media_type="audio/wav")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, token: str | None = Query(default=None)) -> None:
        if not token_ok(token):
            await ws.close(code=4401)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        entry = (ws, loop)
        with hub.lock:
            hub.clients.add(entry)
        try:
            assistant = f.new_assistant()
        except ConfigError as exc:
            await _send(ws, {"type": "error", "detail": str(exc)})
            await ws.close()
            return
        current: dict[str, Any] = {"stream": None, "task": None}
        outbox: asyncio.Queue[tuple[dict[str, Any], bytes | None] | None] = asyncio.Queue()

        def emit(message: dict[str, Any], audio: bytes | None = None) -> None:
            loop.call_soon_threadsafe(outbox.put_nowait, (message, audio))

        def run_turn(text: str, speak: str, heard_at: float) -> None:
            stream = SpeechStream(assistant.ask_stream(text), f.voice if speak == "server" else None, started=heard_at)
            current["stream"] = stream
            for ev in stream:
                if ev.kind == "text":
                    emit({"type": "delta", "text": ev.text})
                elif ev.kind in ("tool_start", "tool_end"):
                    emit({"type": ev.kind, "tool": ev.text, **ev.detail})
                elif ev.kind == "sentence":
                    audio = ev.audio.to_wav() if ev.audio is not None else None
                    emit(
                        {"type": "sentence", "index": ev.index, "text": ev.text, "has_audio": audio is not None}, audio
                    )
                elif ev.kind == "error":
                    emit({"type": "error", "detail": ev.text})
                elif ev.kind == "done":
                    emit({"type": "done", "text": ev.text, "metrics": ev.detail})

        async def start_turn(text: str, speak: str, heard_at: float) -> None:
            if current["stream"] is not None:
                current["stream"].cancel()
            if current["task"] is not None:
                with suppress(Exception):
                    await current["task"]
            current["task"] = loop.run_in_executor(None, run_turn, text, speak, heard_at)

        async def sender() -> None:
            while (item := await outbox.get()) is not None:
                await _send(ws, *item)

        send_task = asyncio.create_task(sender())
        await _send(ws, {"type": "voice", **f.voice.describe()})
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                kind = msg.get("type")
                speak = msg.get("speak", "server")
                if kind == "ask" and str(msg.get("text", "")).strip():
                    await start_turn(str(msg["text"])[:2000], speak, time.perf_counter())
                elif kind == "audio":
                    data = await ws.receive_bytes()
                    heard_at = time.perf_counter()
                    if f.stt is None:
                        emit({"type": "error", "detail": "Server speech recognition is not installed."})
                        continue
                    if len(data) > MAX_AUDIO_BYTES:
                        emit({"type": "error", "detail": "Recording too long."})
                        continue
                    stt = f.stt
                    try:
                        text = await loop.run_in_executor(None, stt.transcribe, data)
                    except Exception as exc:  # noqa: BLE001
                        emit({"type": "error", "detail": f"Could not transcribe: {exc}"})
                        continue
                    emit({"type": "transcript", "text": text, "stt_ms": elapsed_ms(heard_at)})
                    if text:
                        await start_turn(text, speak, heard_at)
                elif kind == "voice":
                    result = await loop.run_in_executor(
                        None, f.voice.set_voice, str(msg.get("voice", "")), msg.get("speed")
                    )
                    emit({"type": "voice", **result})
                elif kind == "stop" and current["stream"] is not None:
                    current["stream"].cancel()
                elif kind == "reset":
                    assistant.reset()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if current["stream"] is not None:
                current["stream"].cancel()
            with hub.lock:
                hub.clients.discard(entry)
            outbox.put_nowait(None)
            with suppress(Exception):
                await send_task

    return app
