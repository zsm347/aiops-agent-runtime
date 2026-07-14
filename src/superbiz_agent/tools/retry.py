from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable


RetryBackoff = Callable[[int], float]
RetrySleep = Callable[[float], Awaitable[None]]


def default_backoff_seconds(retry_number: int) -> float:
    retry_number = max(1, retry_number)
    base = min(0.1 * (2 ** (retry_number - 1)), 2.0)
    jitter = random.uniform(0.0, base * 0.5)
    return base + jitter


async def default_retry_sleep(delay_seconds: float) -> None:
    await asyncio.sleep(max(0.0, delay_seconds))


def delay_to_ms(delay_seconds: float) -> int:
    return int(round(max(0.0, delay_seconds) * 1000))
