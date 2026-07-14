from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

from superbiz_agent.model_gateway.base import (
    ModelMessage,
    ModelResponse,
    ModelStreamChunk,
    ModelToolCall,
)
from superbiz_agent.tools.fixtures import DEFAULT_LOG_REGION, LOG_TOPICS
from superbiz_agent.tools.registry import ToolDefinition


def chunk_text(text: str, chunk_size: int = 16) -> list[str]:
    if not text:
        return []
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


class StubModelGateway:
    provider_name = "stub"
    _business_tools = (
        "queryPrometheusAlerts",
        "getAvailableLogTopics",
        "queryLogs",
        "queryInternalDocs",
        "getCurrentDateTime",
        "listMemoryTopics",
        "searchMemory",
        "updateCoreMemory",
        "saveArchivalMemory",
    )

    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        latest_user_index = self._latest_message_index(messages, "user")
        latest_user_message = messages[latest_user_index].content if latest_user_index >= 0 else ""
        tool_result = self._latest_any_tool_result_after_index(messages, latest_user_index)

        if tool_result is not None:
            tool_name, result_content = tool_result
            answer = self._format_tool_answer(tool_name, result_content)
            return ModelResponse(
                content=answer,
                raw={"provider": self.provider_name, "mode": "final_from_tool"},
            )

        tool_call = self._build_business_tool_call(latest_user_message)
        if tool_call is not None:
            return ModelResponse(
                content="",
                tool_calls=[tool_call],
                raw={"provider": self.provider_name, "mode": "tool_call"},
            )

        if self._looks_like_datetime_question(latest_user_message):
            tool_call = ModelToolCall(
                id=f"toolcall-{uuid4()}",
                name="getCurrentDateTime",
                arguments={"timezone": "Asia/Shanghai"},
            )
            return ModelResponse(
                content="",
                tool_calls=[tool_call],
                raw={"provider": self.provider_name, "mode": "tool_call"},
            )

        memory_tool_call = self._build_memory_tool_call(latest_user_message)
        if memory_tool_call is not None:
            return ModelResponse(
                content="",
                tool_calls=[memory_tool_call],
                raw={"provider": self.provider_name, "mode": "memory_tool_call"},
            )

        return ModelResponse(
            content=f"[stub] {latest_user_message}",
            raw={"provider": self.provider_name, "mode": "deterministic_answer"},
        )

    async def stream(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        response = await self.complete(messages, tools=tools)
        if response.tool_calls:
            yield ModelStreamChunk(
                tool_calls=response.tool_calls,
                usage=response.usage,
                finish_reason=response.finish_reason,
                raw=response.raw,
            )
            return
        chunks = chunk_text(response.content)
        for chunk in chunks:
            yield ModelStreamChunk(content_delta=chunk, raw={"provider": self.provider_name})
        yield ModelStreamChunk(
            usage=response.usage,
            finish_reason=response.finish_reason,
            raw=response.raw,
        )

    @staticmethod
    def _latest_message_index(messages: list[ModelMessage], role: str) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == role:
                return index
        return -1

    @staticmethod
    def _latest_any_tool_result_after_index(
        messages: list[ModelMessage],
        after_index: int,
    ) -> tuple[str, str] | None:
        for message in reversed(messages[after_index + 1 :]):
            if message.role == "tool" and message.name in StubModelGateway._business_tools:
                return message.name, message.content
        return None

    @staticmethod
    def _looks_like_datetime_question(message: str) -> bool:
        lowered = message.lower()
        return any(keyword in lowered for keyword in ("时间", "日期", "几点", "time", "date"))

    def _build_business_tool_call(self, message: str) -> ModelToolCall | None:
        if self._looks_like_alerts_question(message):
            return self._tool_call("queryPrometheusAlerts", {})
        if self._looks_like_log_topics_question(message):
            return self._tool_call("getAvailableLogTopics", {})
        if self._looks_like_query_logs_question(message):
            return self._tool_call("queryLogs", self._infer_query_logs_args(message))
        if self._looks_like_internal_docs_question(message):
            return self._tool_call("queryInternalDocs", {"query": message.strip()})
        return None

    def _build_memory_tool_call(self, message: str) -> ModelToolCall | None:
        if self._looks_like_memory_topics_question(message):
            return self._tool_call("listMemoryTopics", {"type": "all"})
        if self._looks_like_memory_search_question(message):
            return self._tool_call("searchMemory", {"query": message.strip(), "type": "all"})
        if self._looks_like_archival_save_question(message):
            return self._tool_call("saveArchivalMemory", self._infer_save_archival_args(message))
        if self._looks_like_core_memory_update_question(message):
            return self._tool_call("updateCoreMemory", self._infer_core_memory_args(message))
        return None

    @staticmethod
    def _tool_call(name: str, arguments: dict) -> ModelToolCall:
        return ModelToolCall(
            id=f"toolcall-{uuid4()}",
            name=name,
            arguments=arguments,
        )

    @staticmethod
    def _looks_like_alerts_question(message: str) -> bool:
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in (
                "当前告警",
                "活跃告警",
                "告警状态",
                "哪些告警",
                "告警在触发",
                "active alert",
                "alerts",
            )
        )

    @staticmethod
    def _looks_like_log_topics_question(message: str) -> bool:
        lowered = message.lower()
        return (
            any(keyword in lowered for keyword in ("日志主题", "日志类型", "log topics", "log topic"))
            or ("哪些" in lowered and "日志" in lowered and "查" in lowered)
        )

    @staticmethod
    def _looks_like_query_logs_question(message: str) -> bool:
        lowered = message.lower()
        explicit_log_query = "日志" in lowered and any(
            keyword in lowered for keyword in ("查", "查询", "看", "查看")
        )
        return explicit_log_query or any(
            keyword in lowered
            for keyword in (
                "查日志",
                "查询日志",
                "看日志",
                "查看日志",
                "日志里",
                "日志中",
                "logs",
            )
        )

    @staticmethod
    def _looks_like_internal_docs_question(message: str) -> bool:
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in (
                "内部流程",
                "最佳实践",
                "操作步骤",
                "排查指南",
                "排查流程",
                "处理手册",
                "runbook",
                "runbooks",
                "playbook",
            )
        )

    @staticmethod
    def _looks_like_core_memory_update_question(message: str) -> bool:
        lowered = message.lower()
        return any(keyword in lowered for keyword in ("记住", "以后都", "以后回答按", "please remember"))

    @staticmethod
    def _looks_like_archival_save_question(message: str) -> bool:
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in (
                "根因确认",
                "保存经验",
                "写入长期记忆",
                "长期经验",
                "这次经验",
                "save this memory",
            )
        )

    @staticmethod
    def _looks_like_memory_search_question(message: str) -> bool:
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in (
                "参考历史经验",
                "以前有没有类似",
                "查一下长期记忆",
                "搜索长期记忆",
                "search memory",
            )
        )

    @staticmethod
    def _looks_like_memory_topics_question(message: str) -> bool:
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in (
                "有哪些长期记忆主题",
                "长期记忆 topic",
                "长期记忆主题",
                "memory topics",
            )
        )

    @staticmethod
    def _infer_core_memory_args(message: str) -> dict:
        content = message.strip()
        for prefix in ("请记住，", "请记住,", "记住，", "记住,"):
            if content.startswith(prefix):
                content = content[len(prefix) :].strip()
                break
        return {
            "blockKey": "user_rules",
            "newContent": content,
            "changeReason": "user requested durable rule update",
        }

    @staticmethod
    def _infer_save_archival_args(message: str) -> dict:
        lowered = message.lower()
        service = next(
            (
                candidate
                for candidate in (
                    "order-service",
                    "payment-service",
                    "inventory-service",
                    "user-service",
                )
                if candidate in lowered
            ),
            "general",
        )
        issue = "incident"
        if "5xx" in lowered:
            issue = "5xx"
        elif "timeout" in lowered or "超时" in lowered:
            issue = "timeout"
        elif "oom" in lowered or "内存" in lowered:
            issue = "oom"
        return {
            "topic": f"{service}/{issue}",
            "content": message.strip(),
            "evidenceSummary": "user stated the root cause was confirmed",
            "scopeService": service if service != "general" else None,
            "scopeEnv": "production" if "生产" in lowered or "prod" in lowered else None,
            "tags": ",".join(tag for tag in (service, issue) if tag != "general"),
        }

    @staticmethod
    def _infer_query_logs_args(message: str) -> dict:
        lowered = message.lower()
        region = next(
            (
                candidate
                for candidate in (
                    "ap-guangzhou",
                    "ap-shanghai",
                    "ap-beijing",
                    "ap-chengdu",
                )
                if candidate in lowered
            ),
            DEFAULT_LOG_REGION,
        )
        log_topic = next(
            (
                topic["topicName"]
                for topic in LOG_TOPICS
                if str(topic["topicName"]).lower() in lowered
            ),
            "application-logs",
        )
        query = StubModelGateway._infer_query_clause(lowered)
        return {
            "region": region,
            "logTopic": log_topic,
            "query": query,
            "limit": 20,
        }

    @staticmethod
    def _infer_query_clause(lowered_message: str) -> str:
        if "error" in lowered_message or "错误" in lowered_message:
            return "level:ERROR"
        if "fatal" in lowered_message:
            return "level:FATAL"
        if "warn" in lowered_message or "告警" in lowered_message or "警告" in lowered_message:
            return "level:WARN"
        if "cpu" in lowered_message:
            return "cpu_usage:>80"
        if "memory" in lowered_message or "内存" in lowered_message:
            return "memory_usage:>85"
        if "慢" in lowered_message or "slow" in lowered_message:
            return "slow"
        if "重启" in lowered_message or "restart" in lowered_message:
            return "restart OR crash"
        return ""

    def _format_tool_answer(self, tool_name: str, tool_result: str) -> str:
        if tool_name == "getCurrentDateTime":
            return self._format_datetime_answer(tool_result)
        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            return f"{tool_name} 工具返回结果：{tool_result}"

        if tool_name == "queryPrometheusAlerts":
            return self._format_alerts_answer(payload)
        if tool_name == "getAvailableLogTopics":
            return self._format_log_topics_answer(payload)
        if tool_name == "queryLogs":
            return self._format_logs_answer(payload)
        if tool_name == "queryInternalDocs":
            return self._format_internal_docs_answer(payload)
        if tool_name == "updateCoreMemory":
            return self._format_update_core_memory_answer(payload)
        if tool_name == "saveArchivalMemory":
            return self._format_save_archival_memory_answer(payload)
        if tool_name == "searchMemory":
            return self._format_search_memory_answer(payload)
        if tool_name == "listMemoryTopics":
            return self._format_memory_topics_answer(payload)
        return f"{tool_name} 工具返回结果：{payload}"

    @staticmethod
    def _format_datetime_answer(tool_result: str) -> str:
        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            return f"当前时间工具返回结果：{tool_result}"

        if not payload.get("success"):
            message = payload.get("message") or "工具调用失败"
            return f"当前时间工具未能成功返回结果：{message}"

        data = payload.get("data") or {}
        timezone = data.get("timezone", "Asia/Shanghai")
        iso_time = data.get("isoTime", "")
        return f"当前时间（{timezone}）是 {iso_time}。"

    @staticmethod
    def _format_alerts_answer(payload: dict) -> str:
        if not payload.get("success"):
            return f"当前告警查询失败：{payload.get('message', '未知错误')}"
        alerts = payload.get("alerts") or []
        alert_names = [
            str(alert.get("alertName"))
            for alert in alerts
            if isinstance(alert, dict) and alert.get("alertName")
        ]
        return f"当前有 {len(alerts)} 个活跃告警：{', '.join(alert_names)}。"

    @staticmethod
    def _format_log_topics_answer(payload: dict) -> str:
        if not payload.get("success"):
            return f"日志主题查询失败：{payload.get('message', '未知错误')}"
        topics = payload.get("topics") or []
        topic_names = [
            str(topic.get("topicName"))
            for topic in topics
            if isinstance(topic, dict) and topic.get("topicName")
        ]
        default_region = payload.get("defaultRegion", DEFAULT_LOG_REGION)
        return (
            f"共有 {len(topics)} 个可用日志主题：{', '.join(topic_names)}。"
            f"默认地域是 {default_region}。"
        )

    @staticmethod
    def _format_logs_answer(payload: dict) -> str:
        if not payload.get("success"):
            return f"日志查询失败：{payload.get('message', '未知错误')}"
        logs = payload.get("logs") or []
        details = []
        for log in logs[:3]:
            if isinstance(log, dict):
                details.append(
                    "/".join(
                        item
                        for item in (
                            str(log.get("level", "")),
                            str(log.get("service", "")),
                        )
                        if item
                    )
                )
        summary = f"，关键日志：{', '.join(details)}" if details else ""
        return (
            f"查询 {payload.get('logTopic')} 得到 {payload.get('total', len(logs))} 条日志"
            f"{summary}。"
        )

    @staticmethod
    def _format_internal_docs_answer(payload: dict) -> str:
        status = payload.get("status")
        if status == "no_results":
            return f"内部文档没有找到相关结果：{payload.get('message', '')}"
        if status != "ok":
            return f"内部文档查询失败：{payload.get('message', '未知错误')}"
        chunks = payload.get("chunks") or []
        refs = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                refs.append(f"[{chunk.get('ref')}] {chunk.get('source')}")
        return f"内部文档命中 {payload.get('count', len(chunks))} 个片段：{'; '.join(refs)}。"

    @staticmethod
    def _format_update_core_memory_answer(payload: dict) -> str:
        if payload.get("success"):
            return f"已更新核心记忆 {payload.get('blockKey')}，版本 {payload.get('version')}。"
        return f"核心记忆更新被拒绝：{payload.get('message', '未知原因')}"

    @staticmethod
    def _format_save_archival_memory_answer(payload: dict) -> str:
        status = payload.get("status")
        if payload.get("success") and status == "written":
            return f"已保存长期经验，memoryId={payload.get('memoryId')}。"
        if payload.get("success") and status == "duplicate_skipped":
            return "长期经验已存在，已跳过去重写入。"
        return f"长期经验保存失败：{payload.get('message', '未知原因')}"

    @staticmethod
    def _format_search_memory_answer(payload: dict) -> str:
        if not payload.get("success"):
            return f"长期记忆检索失败：{payload.get('message', '未知错误')}"
        memories = payload.get("memories") or []
        if not memories:
            return "长期记忆没有找到相关结果。"
        parts = []
        for memory in memories:
            if isinstance(memory, dict):
                parts.append(
                    f"{memory.get('topic')}[{memory.get('confidenceLabel')}]: {memory.get('content')}"
                )
        return f"长期记忆命中 {payload.get('count', len(memories))} 条：{'; '.join(parts)}"

    @staticmethod
    def _format_memory_topics_answer(payload: dict) -> str:
        if not payload.get("success"):
            return f"长期记忆主题查询失败：{payload.get('message', '未知错误')}"
        topics = payload.get("topics") or []
        names = []
        for topic in topics:
            if isinstance(topic, dict):
                names.append(f"{topic.get('type')}:{topic.get('topic')}({topic.get('count')})")
        return f"共有 {payload.get('count', len(topics))} 个长期记忆主题：{', '.join(names)}。"
