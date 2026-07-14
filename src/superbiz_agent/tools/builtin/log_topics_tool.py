from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from superbiz_agent.tools.fixtures import get_available_log_topics_result
from superbiz_agent.tools.policies import DangerLevel, ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition


class GetAvailableLogTopicsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def get_available_log_topics(_: GetAvailableLogTopicsArgs) -> dict[str, Any]:
    return get_available_log_topics_result()


def build_log_topics_tool() -> ToolDefinition:
    return ToolDefinition(
        name="getAvailableLogTopics",
        description=(
            "获取所有可用的日志主题及其描述、示例查询和关联告警。"
            "用于用户想查日志但不确定有哪些日志类型或主题。"
        ),
        args_model=GetAvailableLogTopicsArgs,
        handler=get_available_log_topics,
        policy=ToolPolicy(
            tool_name="getAvailableLogTopics",
            danger_level=DangerLevel.LOW,
            timeout_seconds=2,
            max_retries=0,
            idempotent=True,
        ),
    )
