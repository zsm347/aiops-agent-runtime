from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-/]+|[\u4e00-\u9fff]+")


class DeterministicEmbeddingService:
    """Small local embedding used only for deterministic tests and evals."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            weight = 2.0 if _looks_like_slug_token(token) else 1.0
            vector[index] += weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        left_values = list(left)
        right_values = list(right)
        if not left_values or not right_values:
            return 0.0
        dot = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = math.sqrt(sum(value * value for value in left_values))
        right_norm = math.sqrt(sum(value * value for value in right_values))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _looks_like_slug_token(token: str) -> bool:
    return any(ch.isascii() and (ch.isalnum() or ch in "-_/") for ch in token)
