from __future__ import annotations

from superbiz_agent.config import Settings
from superbiz_agent.model_gateway.base import ModelGateway
from superbiz_agent.model_gateway.openai_compatible import OpenAICompatibleModelGateway
from superbiz_agent.model_gateway.stub import StubModelGateway


OPENAI_COMPATIBLE_PROVIDERS = {
    "openai-compatible",
    "qwen-openai-compatible",
    "dashscope-openai-compatible",
}


def build_model_gateway(settings: Settings) -> ModelGateway:
    provider = settings.model_provider.strip().lower()
    if provider == "stub":
        return StubModelGateway()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        if not settings.model_api_key or not settings.model_api_key.strip():
            raise ValueError("model_api_key is required for OpenAI-compatible model providers")
        return OpenAICompatibleModelGateway(
            model_name=settings.model_name,
            api_key=settings.model_api_key,
            base_url=settings.model_base_url,
            timeout_ms=settings.model_timeout_ms,
            max_retries=settings.model_max_retries,
            provider_name=provider,
        )
    raise ValueError(f"Unsupported model_provider: {settings.model_provider}")
