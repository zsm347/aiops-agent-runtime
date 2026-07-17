from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from superbiz_agent.config import Settings, get_settings


EXPLICIT_DATABASE_URL_ATTRIBUTE = "superbiz_agent.explicit_database_url"
CONNECTION_VALIDATOR_ATTRIBUTE = "superbiz_agent.connection_validator"


class AlembicConfig(Protocol):
    attributes: dict[str, Any]


def resolve_alembic_database_url(
    config: AlembicConfig,
    *,
    settings_factory: Callable[[], Settings] = get_settings,
) -> str:
    """Prefer a trusted programmatic URL without consulting Settings or ``.env``."""

    if EXPLICIT_DATABASE_URL_ATTRIBUTE in config.attributes:
        explicit_url = config.attributes[EXPLICIT_DATABASE_URL_ATTRIBUTE]
        if not isinstance(explicit_url, str) or not explicit_url.strip():
            raise RuntimeError("Explicit Alembic database URL is invalid.")
        return explicit_url
    return settings_factory().database_url
