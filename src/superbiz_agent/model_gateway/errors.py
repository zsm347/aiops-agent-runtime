from __future__ import annotations


class ModelGatewayError(Exception):
    """Base class for normalized model provider failures."""

    error_type = "provider-error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class ModelGatewayTimeoutError(ModelGatewayError):
    error_type = "timeout"
    retryable = True


class ModelGatewayAuthError(ModelGatewayError):
    error_type = "auth"


class ModelGatewayRateLimitError(ModelGatewayError):
    error_type = "rate-limit"
    retryable = True


class ModelGatewayContextOverflowError(ModelGatewayError):
    error_type = "context-overflow"


class ModelGatewayProviderError(ModelGatewayError):
    error_type = "provider-error"

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code)
        self.retryable = retryable
