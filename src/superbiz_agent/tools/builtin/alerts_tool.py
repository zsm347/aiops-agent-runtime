from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from superbiz_agent.tools.fixtures import query_prometheus_alerts_result
from superbiz_agent.tools.policies import DangerLevel, ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition


class QueryPrometheusAlertsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def query_prometheus_alerts(_: QueryPrometheusAlertsArgs) -> dict[str, Any]:
    return query_prometheus_alerts_result()


def build_prometheus_alerts_tool() -> ToolDefinition:
    return ToolDefinition(
        name="queryPrometheusAlerts",
        description=(
            "查询 Prometheus 中当前活跃的告警。用于用户询问当前告警、"
            "有哪些告警在触发或系统告警状态。"
        ),
        args_model=QueryPrometheusAlertsArgs,
        handler=query_prometheus_alerts,
        policy=ToolPolicy(
            tool_name="queryPrometheusAlerts",
            danger_level=DangerLevel.MEDIUM,
            timeout_seconds=5,
            max_retries=1,
            idempotent=True,
        ),
    )
