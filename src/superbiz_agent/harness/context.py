from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


def _default(value: Optional[str], fallback: str) -> str:
    return value if value is not None and value.strip() else fallback


@dataclass(frozen=True)
class AgentRequestContext:
    tenant_id: Optional[str]
    user_id: Optional[str]
    agent_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str] = None
    roles: frozenset[str] = field(default_factory=frozenset)
    permissions: frozenset[str] = field(default_factory=frozenset)
    auth_mode: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _default(self.tenant_id, "default-tenant"))
        object.__setattr__(self, "user_id", _default(self.user_id, "default-user"))
        object.__setattr__(self, "agent_id", _default(self.agent_id, "ops-agent"))
        object.__setattr__(self, "session_id", _default(self.session_id, str(uuid4())))
        object.__setattr__(self, "request_id", _default(self.request_id, str(uuid4())))
        object.__setattr__(self, "roles", frozenset(self.roles or ()))
        object.__setattr__(self, "permissions", frozenset(self.permissions or ()))
        object.__setattr__(
            self,
            "auth_mode",
            self.auth_mode.strip() if self.auth_mode and self.auth_mode.strip() else None,
        )

    @property
    def conversation_key(self) -> str:
        return f"{self.tenant_id}:{self.user_id}:{self.agent_id}:{self.session_id}"


@dataclass(frozen=True)
class RunContext:
    request_context: AgentRequestContext
    run_id: str
    prompt_version: str
    tool_schema_version: str
    model_provider: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tool_call_started: bool = False
