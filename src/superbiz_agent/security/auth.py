from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from superbiz_agent.config import Settings, get_settings
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.security.permissions import (
    LOCAL_DEFAULT_PERMISSIONS,
    validate_permission_enforcement_settings,
)


DEV_HEADER_ENVS = frozenset({"local", "dev", "test"})
SUPPORTED_AUTH_MODES = frozenset({"dev_headers", "trusted_gateway", "jwt"})


class AuthError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    tenant_id: str
    user_id: str
    agent_id: str
    roles: frozenset[str] = field(default_factory=frozenset)
    permissions: frozenset[str] = field(default_factory=frozenset)
    auth_mode: str = "dev_headers"
    subject: str | None = None
    issuer: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        tenant_id = _required(self.tenant_id, "tenant_id")
        user_id = _required(self.user_id, "user_id")
        agent_id = _required(self.agent_id, "agent_id")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "roles", frozenset(self.roles or ()))
        object.__setattr__(self, "permissions", frozenset(self.permissions or ()))
        object.__setattr__(self, "auth_mode", _required(self.auth_mode, "auth_mode"))
        object.__setattr__(self, "request_id", self.request_id or str(uuid4()))

    def to_request_context(self, *, session_id: str | None) -> AgentRequestContext:
        return AgentRequestContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=session_id,
            request_id=self.request_id,
            roles=self.roles,
            permissions=self.permissions,
            auth_mode=self.auth_mode,
        )


class AuthContextResolver:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def resolve(self, request_or_headers: Any) -> AuthenticatedPrincipal:
        self._validate_auth_settings()
        headers = _headers(request_or_headers)
        mode = self.settings.auth_mode.strip().lower()

        if mode == "dev_headers":
            return self._resolve_dev_headers(headers)
        if mode == "trusted_gateway":
            return self._resolve_trusted_gateway(headers)
        if mode == "jwt":
            raise AuthError("JWT auth not implemented in P0.")
        raise AuthError(f"Unsupported auth_mode: {self.settings.auth_mode}")

    def _validate_auth_settings(self) -> None:
        mode = self.settings.auth_mode.strip().lower()
        if mode not in SUPPORTED_AUTH_MODES:
            raise AuthError(f"Unsupported auth_mode: {self.settings.auth_mode}")
        try:
            validate_permission_enforcement_settings(self.settings)
        except Exception as exc:
            raise AuthError(str(exc)) from exc

    def _resolve_dev_headers(self, headers: Mapping[str, str]) -> AuthenticatedPrincipal:
        env = self.settings.app_env.strip().lower()
        if env not in DEV_HEADER_ENVS:
            raise AuthError("dev_headers auth mode is only allowed in local/dev/test environments.")
        if not self.settings.auth_dev_headers_enabled:
            raise AuthError("dev_headers auth mode is disabled.")

        permissions = _parse_csv(headers.get("X-Permissions"))
        if not permissions:
            permissions = LOCAL_DEFAULT_PERMISSIONS
        return AuthenticatedPrincipal(
            tenant_id=_default_header(headers.get("X-Tenant-Id"), self.settings.tenant_default),
            user_id=_default_header(headers.get("X-User-Id"), self.settings.user_default),
            agent_id=_default_header(headers.get("X-Agent-Id"), self.settings.agent_default),
            roles=_parse_csv(headers.get("X-Roles")),
            permissions=permissions,
            auth_mode="dev_headers",
            subject=_default_header(headers.get("X-User-Id"), self.settings.user_default),
            issuer="dev_headers",
            request_id=_default_header(headers.get("X-Request-Id"), str(uuid4())),
        )

    def _resolve_trusted_gateway(self, headers: Mapping[str, str]) -> AuthenticatedPrincipal:
        configured_secret = self.settings.auth_trusted_gateway_secret
        if not configured_secret:
            raise AuthError("trusted_gateway auth requires auth_trusted_gateway_secret.")

        provided_secret = headers.get("X-Auth-Gateway-Secret")
        if not provided_secret or not secrets.compare_digest(provided_secret, configured_secret):
            raise AuthError("Invalid trusted gateway secret.")

        return AuthenticatedPrincipal(
            tenant_id=_required_header(headers, "X-Auth-Tenant-Id"),
            user_id=_required_header(headers, "X-Auth-User-Id"),
            agent_id=_required_header(headers, "X-Auth-Agent-Id"),
            roles=_parse_csv(headers.get("X-Auth-Roles")),
            permissions=_parse_csv(headers.get("X-Auth-Permissions")),
            auth_mode="trusted_gateway",
            subject=headers.get("X-Auth-Subject") or headers.get("X-Auth-User-Id"),
            issuer=headers.get("X-Auth-Issuer") or "trusted_gateway",
            request_id=_default_header(
                headers.get("X-Auth-Request-Id") or headers.get("X-Request-Id"),
                str(uuid4()),
            ),
        )


def _headers(request_or_headers: Any) -> Mapping[str, str]:
    headers = getattr(request_or_headers, "headers", request_or_headers)
    if not isinstance(headers, Mapping):
        raise AuthError("Request headers are required for authentication.")
    return headers


def _parse_csv(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def _default_header(value: str | None, fallback: str) -> str:
    return value.strip() if value and value.strip() else fallback


def _required_header(headers: Mapping[str, str], name: str) -> str:
    return _required(headers.get(name), name)


def _required(value: str | None, name: str) -> str:
    if value is None or not str(value).strip():
        raise AuthError(f"Missing required authentication field: {name}")
    return str(value).strip()
