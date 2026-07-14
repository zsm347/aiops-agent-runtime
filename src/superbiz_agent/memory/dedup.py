from __future__ import annotations

import hashlib
import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_TERMINATOR_RE = re.compile(r"[.!?。！？…]+$")


def canonicalize_archival_content(content: str) -> str:
    """Return the conservative exact-dedupe representation for archival content."""

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return _TRAILING_TERMINATOR_RE.sub("", normalized).rstrip()


def canonical_content_hash(content: str) -> str:
    canonical = canonicalize_archival_content(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
