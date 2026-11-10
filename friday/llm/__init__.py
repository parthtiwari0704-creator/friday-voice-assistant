from ..config import Settings
from .base import ConfigError, LLMProvider, LLMResponse, Message, StreamEvent, ToolCall, ToolSpec, Usage

__all__ = [
    "ConfigError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "StreamEvent",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "get_provider",
]


def get_provider(settings: Settings) -> LLMProvider:
    from .providers import AnthropicProvider, OpenAIProvider

    if settings.provider == "anthropic":
        key = settings.anthropic_api_key.get_secret_value()
        if not key:
            raise ConfigError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return AnthropicProvider(key, settings.model)
    key = settings.openai_api_key.get_secret_value()
    if not key and not settings.openai_base_url:
        raise ConfigError("Set OPENAI_API_KEY, or OPENAI_BASE_URL for a local server such as Ollama.")
    return OpenAIProvider(key, settings.model, settings.openai_base_url)
