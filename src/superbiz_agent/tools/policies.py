from dataclasses import dataclass
from enum import Enum


class ToolAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class DangerLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolPolicy:
    tool_name: str
    danger_level: DangerLevel = DangerLevel.MEDIUM
    action: ToolAction = ToolAction.ALLOW
    timeout_seconds: int = 10
    max_retries: int = 0
    idempotent: bool = False
