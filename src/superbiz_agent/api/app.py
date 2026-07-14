from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from superbiz_agent.api.routes_chat import router as chat_router
from superbiz_agent.config import Settings, get_settings
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.security.auth import AuthContextResolver


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            service = getattr(app.state, "harness_service", None)
            if isinstance(service, AgentHarnessService):
                await service.aclose()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.auth_resolver = AuthContextResolver(settings)
    app.state.harness_service = AgentHarnessService.build_default(settings)
    app.include_router(chat_router)
    return app
