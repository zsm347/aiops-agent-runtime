from __future__ import annotations

import json
import re
from typing import Any


SENSITIVE_KEYWORDS = {
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "credential",
}

_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|token|password|passwd|secret|authorization)"
    r"\b(\s*[:=]\s*)(?:bearer\s+)?[^,\s;]+"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.])(?:/[A-Za-z0-9._@%+=:, -]+){2,}")
_WINDOWS_PATH_RE = re.compile(r"(?i)\b[a-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)+[^\\/:*?\"<>|\r\n]*")
_TRACEBACK_RE = re.compile(r"(?is)traceback \(most recent call last\):.*")


def sanitize_error_message(message: str, *, max_length: int = 500) -> str:
    if not message:
        return "Tool execution failed."
    sanitized = _TRACEBACK_RE.sub("traceback redacted", message)
    sanitized = redact_text(
        sanitized,
        replacement="<redacted>",
        redact_paths=True,
        redact_traceback=False,
    )
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > max_length:
        sanitized = f"{sanitized[: max_length - 3]}..."
    return sanitized or "Tool execution failed."


def sanitize_tool_result(content: str) -> Any:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return redact_text(content, replacement="[REDACTED]")
    return redact_value(payload)


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_key(key_text):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = redact_value(item)
        return redacted
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value, replacement="[REDACTED]")
    return value


def redact_text(
    text: str,
    *,
    replacement: str = "[REDACTED]",
    redact_paths: bool = False,
    redact_traceback: bool = False,
) -> str:
    redacted = str(text)
    if redact_traceback:
        redacted = _TRACEBACK_RE.sub("traceback redacted", redacted)
    redacted = _SENSITIVE_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{replacement}",
        redacted,
    )
    redacted = _BEARER_TOKEN_RE.sub(f"Bearer {replacement}", redacted)
    if redact_paths:
        redacted = _WINDOWS_PATH_RE.sub("<path redacted>", redacted)
        redacted = _ABSOLUTE_PATH_RE.sub("<path redacted>", redacted)
    return redacted


def is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(keyword.replace("_", "") in normalized for keyword in SENSITIVE_KEYWORDS)
