"""Prompt-injection containment for third-party text (web pages, emails, calendar invites)."""

from __future__ import annotations

import re

_INJECTION = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
        r"disregard\s+(all\s+|the\s+)?(previous|prior|above)",
        r"\byou\s+are\s+now\b",
        r"\b(reveal|print|show)\s+(your\s+)?system\s+prompt",
        r"\b(hey|hi|dear)\s+(friday|jarvis|assistant|ai)\b.*\b(must|should|need to|please)\b",
        r"\b(open|visit|go to)\s+(this|the following)\s+(link|url|site|website)",
        r"</?\s*untrusted_data",
    )
]


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def wrap_untrusted(content: str, source: str) -> str:
    """Fence third-party content so the model treats it as data. Content that is already
    fenced (e.g. by Jarvis over MCP) is passed through unchanged."""
    if content.lstrip().startswith("<untrusted_data"):
        return content
    safe = re.sub(r"<(/?)\s*untrusted_data", r"&lt;\1untrusted_data", content, flags=re.IGNORECASE)
    source = re.sub(r"[^A-Za-z0-9_\-]", "", source)
    flag = ' injection_suspected="true"' if detect_injection(content) else ""
    return f'<untrusted_data source="{source}"{flag}>\n{safe}\n</untrusted_data>'
