from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


@dataclass(frozen=True)
class ContentCompressionResult:
    content: str
    backend: str
    compressed: bool
    warnings: list[str] = field(default_factory=list)


class ContentCompressionBackend(Protocol):
    backend_name: str

    def compress(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContentCompressionResult:
        ...


class NoopContentCompressionBackend:
    backend_name = "noop"

    def compress(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContentCompressionResult:
        return ContentCompressionResult(
            content=content,
            backend=self.backend_name,
            compressed=False,
        )


class DeterministicContentCompressionBackend:
    backend_name = "deterministic"

    def __init__(self, *, max_chars: int = 3_200) -> None:
        self.max_chars = max(400, max_chars)

    def compress(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContentCompressionResult:
        summary = _summarize_json(content, self.max_chars) or _head_tail(content, self.max_chars)
        return ContentCompressionResult(
            content=summary,
            backend=self.backend_name,
            compressed=summary != content,
        )


class HeadroomContentCompressionBackend:
    """Optional adapter. Missing or incompatible headroom falls back deterministically."""

    backend_name = "headroom"

    def __init__(self, fallback: ContentCompressionBackend | None = None) -> None:
        self.fallback = fallback or DeterministicContentCompressionBackend()
        self._headroom_module: Any | None = None
        self._import_error: str | None = None
        try:
            import headroom  # type: ignore[import-not-found]

            self._headroom_module = headroom
        except Exception as exc:  # pragma: no cover - depends on optional package.
            self._import_error = str(exc)

    def compress(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContentCompressionResult:
        if self._headroom_module is None:
            fallback_result = self.fallback.compress(content, metadata=metadata)
            return replace(
                fallback_result,
                backend=f"{self.backend_name}:fallback:{fallback_result.backend}",
                warnings=[
                    f"headroom unavailable: {self._import_error or 'import failed'}",
                    *fallback_result.warnings,
                ],
            )

        try:
            compressor = getattr(self._headroom_module, "compress", None)
            if callable(compressor):
                compressed = compressor(content)
                if not isinstance(compressed, str):
                    compressed = str(compressed)
                return ContentCompressionResult(
                    content=compressed,
                    backend=self.backend_name,
                    compressed=compressed != content,
                )
        except Exception as exc:  # pragma: no cover - defensive optional adapter path.
            fallback_result = self.fallback.compress(content, metadata=metadata)
            return replace(
                fallback_result,
                backend=f"{self.backend_name}:fallback:{fallback_result.backend}",
                warnings=[f"headroom compression failed: {exc}", *fallback_result.warnings],
            )

        fallback_result = self.fallback.compress(content, metadata=metadata)
        return replace(
            fallback_result,
            backend=f"{self.backend_name}:fallback:{fallback_result.backend}",
            warnings=[
                "headroom module has no supported compress() adapter",
                *fallback_result.warnings,
            ],
        )


def build_content_compression_backend(name: str | None) -> ContentCompressionBackend:
    normalized = (name or "headroom").strip().lower()
    if normalized in {"noop", "none", "off", "disabled"}:
        return NoopContentCompressionBackend()
    if normalized in {"fake", "deterministic"}:
        return DeterministicContentCompressionBackend()
    if normalized == "headroom":
        return HeadroomContentCompressionBackend()
    return DeterministicContentCompressionBackend()


def _summarize_json(content: str, max_chars: int) -> str | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, (dict, list)):
        return _head_tail(json.dumps(payload, ensure_ascii=False, sort_keys=True), max_chars)

    lines = ["<compressed_tool_result>"]
    if isinstance(payload, dict):
        lines.append(f"type=dict keys={', '.join(str(key) for key in list(payload)[:20])}")
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"- {key}: list count={len(value)} sample={_json_sample(value[:2])}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: dict keys={', '.join(str(item) for item in list(value)[:12])}")
            else:
                lines.append(f"- {key}: {_short_scalar(value)}")
    else:
        lines.append(f"type=list count={len(payload)} sample={_json_sample(payload[:3])}")
    lines.append("</compressed_tool_result>")
    return _head_tail("\n".join(lines), max_chars)


def _head_tail(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    head_size = max_chars // 2
    tail_size = max_chars - head_size
    return (
        content[:head_size].rstrip()
        + "\n...[content compressed deterministically]...\n"
        + content[-tail_size:].lstrip()
    )


def _json_sample(value: Any) -> str:
    return _head_tail(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str), 800)


def _short_scalar(value: Any) -> str:
    return _head_tail(str(value), 240)
