from __future__ import annotations

from superbiz_agent.tools.builtin.alerts_tool import build_prometheus_alerts_tool
from superbiz_agent.tools.builtin.datetime_tool import build_datetime_tool
from superbiz_agent.tools.builtin.internal_docs_tool import build_internal_docs_tool
from superbiz_agent.tools.builtin.log_query_tool import build_log_query_tool
from superbiz_agent.tools.builtin.log_topics_tool import build_log_topics_tool
from superbiz_agent.tools.registry import ToolDefinition
from superbiz_agent.rag.retrieval import RagRetrievalService


def build_builtin_tools(
    extra_tools: list[ToolDefinition] | None = None,
    *,
    rag_retrieval_service: RagRetrievalService | None = None,
) -> list[ToolDefinition]:
    tools = [
        build_datetime_tool(),
        build_log_topics_tool(),
        build_log_query_tool(),
        build_prometheus_alerts_tool(),
        build_internal_docs_tool(rag_retrieval_service),
    ]
    if extra_tools:
        tools.extend(extra_tools)
    return tools


__all__ = [
    "build_builtin_tools",
    "build_datetime_tool",
    "build_internal_docs_tool",
    "build_log_query_tool",
    "build_log_topics_tool",
    "build_prometheus_alerts_tool",
]
