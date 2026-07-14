from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from superbiz_agent.harness.context import AgentRequestContext


class ConversationLockManager:
    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    async def lock_for(self, context: AgentRequestContext) -> threading.Lock:
        key = context.conversation_key
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    @asynccontextmanager
    async def acquire(self, context: AgentRequestContext) -> AsyncIterator[None]:
        lock = await self.lock_for(context)
        await asyncio.to_thread(lock.acquire)
        try:
            yield
        finally:
            lock.release()
