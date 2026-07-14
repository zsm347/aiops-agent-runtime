from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolErrorType(str, Enum):
    PARAM_VALIDATION_FAILED = "PARAM_VALIDATION_FAILED"
    TOOL_BLOCKED = "TOOL_BLOCKED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_RETRY_EXHAUSTED = "TOOL_RETRY_EXHAUSTED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TOOL_ERROR = "TOOL_ERROR"


class ToolViolation(BaseModel):
    field: str
    problem: str
    expected: Optional[str] = None
    received: Any = None
    fix_hint: Optional[str] = None


class ToolRetryBudget(BaseModel):
    runtime_used: int = 0
    runtime_max: int = 0
    model_used: int = 0
    model_max: int = 0


class ToolErrorResult(BaseModel):
    success: bool = False
    error_type: ToolErrorType
    message: str
    suggestion: Optional[str] = None
    reason: Optional[str] = None
    violations: list[ToolViolation] = Field(default_factory=list)
    retryable_by_runtime: bool = False
    retryable_by_model: bool = False
    allowed_next_actions: list[str] = Field(default_factory=list)
    disallowed_next_actions: list[str] = Field(default_factory=list)
    retry_budget: Optional[ToolRetryBudget] = None
