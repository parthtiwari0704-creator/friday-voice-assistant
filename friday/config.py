"""Typed settings from environment variables / ``.env`` (prefix ``FRIDAY_``)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRIDAY_", env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    # --- brain ----------------------------------------------------------------------------
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-sonnet-5-5"
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    max_tokens: int = 800  # spoken answers are short
    max_steps: int = 6
    history_turns: int = 20

    # --- you ----------------------------------------------------------------------------------
    user_name: str = "Boss"
    timezone: str = "Asia/Kolkata"
    city: str = ""

    # --- voice out (text-to-speech) -------------------------------------------------------------
    tts_engine: Literal["auto", "piper", "system", "openai", "none"] = "auto"
    voice: str = "female"  # a preset (female, male, female-uk, male-uk, female-2, male-2) or an engine voice id
    speaking_rate: float = 1.0
    openai_tts_model: str = "tts-1"

    # --- voice in (speech-to-text) ----------------------------------------------------------------
    stt_engine: Literal["auto", "whisper", "openai"] = "auto"
    whisper_model: str = "base.en"
    stt_language: str = "en"
    wake_words: str = "friday,hey friday,ok friday"
    follow_up_seconds: float = 6.0  # keep listening this long after answering, without the wake word
    vad_silence_ms: int = 700
    vad_max_seconds: float = 15.0

    # --- integrations -----------------------------------------------------------------------------
    jarvis_mcp_command: str = ""  # e.g. "jarvis mcp" — gives FRIDAY Jarvis's email/calendar tools
    apps: str = "{}"  # JSON {"spoken name": "command or path"} — the only apps open_app may launch

    # --- runtime -------------------------------------------------------------------------------------
    home: Path = Field(default_factory=lambda: Path.home() / ".friday")
    api_token: SecretStr = SecretStr("")
    log_level: str = "WARNING"

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone '{v}'") from exc
        return v

    @field_validator("speaking_rate")
    @classmethod
    def _rate(cls, v: float) -> float:
        if not 0.5 <= v <= 2.0:
            raise ValueError("speaking_rate must be between 0.5 and 2.0")
        return v

    @field_validator("apps")
    @classmethod
    def _apps(cls, v: str) -> str:
        if not isinstance(json.loads(v or "{}"), dict):
            raise ValueError('FRIDAY_APPS must be a JSON object like {"notepad": "notepad.exe"}')
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def wake_word_list(self) -> list[str]:
        return [w.strip().lower() for w in self.wake_words.split(",") if w.strip()]

    @property
    def app_map(self) -> dict[str, Any]:
        return {k.lower(): v for k, v in json.loads(self.apps or "{}").items()}

    @property
    def voices_dir(self) -> Path:
        return self.home / "voices"

    @property
    def db_path(self) -> Path:
        return self.home / "friday.db"

    def ensure_dirs(self) -> None:
        self.voices_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
