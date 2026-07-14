from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable


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


def stable_tag_union(existing: Iterable[str], incoming: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw_tag in (*existing, *incoming):
        tag = raw_tag.strip()
        if not tag or tag in seen:
            continue
        merged.append(tag)
        seen.add(tag)
    return merged
