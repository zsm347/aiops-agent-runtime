from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.tools.fixtures import query_logs_result
from superbiz_agent.tools.policies import DangerLevel, ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition


LogRegion = Literal["ap-guangzhou", "ap-shanghai", "ap-beijing", "ap-chengdu"]


class QueryLogsArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    region: LogRegion
    log_topic: str = Field(..., alias="logTopic", min_length=1)
    query: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


async def query_logs(args: QueryLogsArgs) -> dict[str, Any]:
    return query_logs_result(
        region=args.region,
        log_topic=args.log_topic,
        query=args.query,
        limit=args.limit,
    )


def build_log_query_tool() -> ToolDefinition:
    return ToolDefinition(
        name="queryLogs",
        description=(
            "从云日志服务查询日志。用于用户需要查看应用日志、系统指标、"
            "慢查询或系统事件。"
        ),
        args_model=QueryLogsArgs,
        handler=query_logs,
        policy=ToolPolicy(
            tool_name="queryLogs",
            danger_level=DangerLevel.MEDIUM,
            timeout_seconds=8,
            max_retries=1,
            idempotent=True,
        ),
    )
