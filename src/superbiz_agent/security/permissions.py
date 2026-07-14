from __future__ import annotations

from collections.abc import Iterable

from superbiz_agent.config import Settings, get_settings
from superbiz_agent.harness.context import AgentRequestContext


CHAT_INVOKE = "chat:invoke"
SESSION_READ = "session:read"
SESSION_CLEAR = "session:clear"
MEMORY_READ = "memory:read"
MEMORY_WRITE = "memory:write"
TOOL_EXECUTE = "tool:execute"
RAW_REF_READ = "raw_ref:read"
ADMIN_ALL = "admin:*"

LOCAL_DEFAULT_PERMISSIONS = frozenset(
    {
        CHAT_INVOKE,
        SESSION_READ,
        SESSION_CLEAR,
        MEMORY_READ,
        MEMORY_WRITE,
        TOOL_EXECUTE,
        RAW_REF_READ,
    }
)

MEMORY_READ_TOOLS = frozenset({"searchMemory", "listMemoryTopics"})
MEMORY_WRITE_TOOLS = frozenset({"updateCoreMemory", "saveArchivalMemory"})
PERMISSION_BYPASS_ENVS = frozenset({"local", "test"})


class PermissionDenied(Exception):
    def __init__(self, message: str, *, required_permission: str | None = None) -> None:
        super().__init__(message)
        self.required_permission = required_permission


def require_permission(
    context: AgentRequestContext,
    permission: str,
    *,
    settings: Settings | None = None,
) -> None:
    active_settings = settings or get_settings()
    if not active_settings.auth_permission_enforcement:
        _ensure_permission_bypass_allowed(active_settings)
        return

    if not has_permission(context.permissions, permission):
        raise PermissionDenied(
            f"Missing required permission: {permission}",
            required_permission=permission,
        )


def require_tool_permission(
    context: AgentRequestContext,
    tool_name: str,
    *,
    settings: Settings | None = None,
) -> None:
    if tool_name in MEMORY_READ_TOOLS:
        require_permission(context, MEMORY_READ, settings=settings)
        return
    if tool_name in MEMORY_WRITE_TOOLS:
        require_permission(context, MEMORY_WRITE, settings=settings)
        return

    active_settings = settings or get_settings()
    if not active_settings.auth_permission_enforcement:
        _ensure_permission_bypass_allowed(active_settings)
        return

    allowed = has_permission(context.permissions, TOOL_EXECUTE) or has_permission(
        context.permissions,
        tool_execute_permission(tool_name),
    )
    if not allowed:
        raise PermissionDenied(
            f"Missing required permission: {TOOL_EXECUTE} or {tool_execute_permission(tool_name)}",
            required_permission=TOOL_EXECUTE,
        )


def has_permission(permissions: Iterable[str], permission: str) -> bool:
    permission_set = frozenset(item for item in permissions if item)
    return ADMIN_ALL in permission_set or permission in permission_set


def tool_execute_permission(tool_name: str) -> str:
    return f"{TOOL_EXECUTE}:{tool_name}"


def validate_permission_enforcement_settings(settings: Settings) -> None:
    if settings.auth_permission_enforcement:
        return
    _ensure_permission_bypass_allowed(settings)


def _ensure_permission_bypass_allowed(settings: Settings) -> None:
    app_env = settings.app_env.strip().lower()
    if app_env not in PERMISSION_BYPASS_ENVS:
        raise PermissionDenied(
            "auth_permission_enforcement=false is only allowed in local/test environments."
        )
