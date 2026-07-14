from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    """Deterministic eval fixture for the local Skeleton P0 gate."""

    model_config = ConfigDict(protected_namespaces=())

    case_id: str
    name: str
    description: str = ""
    session_id: str
    tenant_id: str = "eval-tenant"
    user_id: str = "eval-user"
    agent_id: str = "ops-agent"
    user_input: str
    expected_answer_contains: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    required_events: list[str] = Field(default_factory=list)
    forbidden_events: list[str] = Field(default_factory=list)
    expected_success: bool = True
    scoring_rules: dict[str, Any] = Field(default_factory=dict)
    initial_context: dict[str, Any] = Field(default_factory=dict)
    available_tools: Optional[list[str]] = None
    mock_tool_returns: dict[str, Any] = Field(default_factory=dict)
    required_tool_argument_keys: dict[str, list[str]] = Field(default_factory=dict)
    min_history_item_count: Optional[int] = None


class EvalSuite(BaseModel):
    """A named group of deterministic eval cases."""

    model_config = ConfigDict(protected_namespaces=())

    suite_id: str
    name: str
    description: str = ""
    cases: list[EvalCase]


def skeleton_p0_smoke_suite() -> EvalSuite:
    """Built-in smoke suite for the current Skeleton P0 harness."""

    return EvalSuite(
        suite_id="skeleton_p0_smoke",
        name="Skeleton P0 smoke",
        description="Deterministic smoke checks for the minimal Agent Harness loop.",
        cases=[
            EvalCase(
                case_id="datetime_requires_tool",
                name="Datetime question calls tool",
                description="A datetime question must call getCurrentDateTime.",
                session_id="eval-skeleton-p0-datetime",
                user_input="现在几点？",
                expected_answer_contains=["Asia/Shanghai"],
                required_tools=["getCurrentDateTime"],
                required_events=[
                    "RUN_STARTED",
                    "USER_MESSAGE_APPENDED",
                    "CONTEXT_ASSEMBLED",
                    "MODEL_CALL_STARTED",
                    "MODEL_CALL_COMPLETED",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "ASSISTANT_MESSAGE_APPENDED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                required_tool_argument_keys={"getCurrentDateTime": ["timezone"]},
            ),
            EvalCase(
                case_id="alerts_tool_required",
                name="Current alerts call Prometheus alerts tool",
                description="A current-alerts question must call queryPrometheusAlerts.",
                session_id="eval-business-tools-alerts",
                user_input="现在有哪些活跃告警？",
                expected_answer_contains=["HighCPUUsage"],
                required_tools=["queryPrometheusAlerts"],
                required_events=[
                    "RUN_STARTED",
                    "USER_MESSAGE_APPENDED",
                    "CONTEXT_ASSEMBLED",
                    "MODEL_CALL_STARTED",
                    "MODEL_CALL_COMPLETED",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "ASSISTANT_MESSAGE_APPENDED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
            ),
            EvalCase(
                case_id="log_topics_required",
                name="Log topics question calls topics tool",
                description="A log topic discovery question must call getAvailableLogTopics.",
                session_id="eval-business-tools-log-topics",
                user_input="有哪些日志主题可以查？",
                expected_answer_contains=["application-logs"],
                required_tools=["getAvailableLogTopics"],
                required_events=[
                    "RUN_STARTED",
                    "USER_MESSAGE_APPENDED",
                    "CONTEXT_ASSEMBLED",
                    "MODEL_CALL_STARTED",
                    "MODEL_CALL_COMPLETED",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "ASSISTANT_MESSAGE_APPENDED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
            ),
            EvalCase(
                case_id="query_logs_required",
                name="Explicit log query calls logs tool",
                description="An explicit log query must call queryLogs with required arguments.",
                session_id="eval-business-tools-query-logs",
                user_input="查一下 application-logs 里的 ERROR 日志",
                expected_answer_contains=["application-logs"],
                required_tools=["queryLogs"],
                required_events=[
                    "RUN_STARTED",
                    "USER_MESSAGE_APPENDED",
                    "CONTEXT_ASSEMBLED",
                    "MODEL_CALL_STARTED",
                    "MODEL_CALL_COMPLETED",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "ASSISTANT_MESSAGE_APPENDED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                required_tool_argument_keys={
                    "queryLogs": ["region", "logTopic", "query"],
                },
            ),
            EvalCase(
                case_id="internal_docs_required",
                name="Runbook question calls internal docs tool",
                description="A runbook or troubleshooting question must call queryInternalDocs.",
                session_id="eval-business-tools-internal-docs",
                user_input="Pod 重启排查流程是什么？",
                expected_answer_contains=["runbooks"],
                required_tools=["queryInternalDocs"],
                required_events=[
                    "RUN_STARTED",
                    "USER_MESSAGE_APPENDED",
                    "CONTEXT_ASSEMBLED",
                    "MODEL_CALL_STARTED",
                    "MODEL_CALL_COMPLETED",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "ASSISTANT_MESSAGE_APPENDED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                required_tool_argument_keys={"queryInternalDocs": ["query"]},
            ),
            EvalCase(
                case_id="plain_chat_forbids_tools",
                name="Plain chat does not call tools",
                description="A plain chat request should stay in the model path only.",
                session_id="eval-skeleton-p0-plain-chat",
                user_input="hello",
                expected_answer_contains=["[stub] hello"],
                forbidden_tools=[
                    "getCurrentDateTime",
                    "getAvailableLogTopics",
                    "queryLogs",
                    "queryPrometheusAlerts",
                    "queryInternalDocs",
                ],
                required_events=[
                    "RUN_STARTED",
                    "USER_MESSAGE_APPENDED",
                    "CONTEXT_ASSEMBLED",
                    "MODEL_CALL_STARTED",
                    "MODEL_CALL_COMPLETED",
                    "ASSISTANT_MESSAGE_APPENDED",
                    "RUN_COMPLETED",
                ],
                forbidden_events=[
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "TOOL_CALL_FAILED",
                    "TOOL_CALL_BLOCKED",
                ],
                expected_success=True,
            ),
            EvalCase(
                case_id="empty_question_no_runtime",
                name="Empty question fails before runtime",
                description="An empty question should not enter model or tool execution.",
                session_id="eval-skeleton-p0-empty-question",
                user_input="  ",
                forbidden_events=[
                    "RUN_STARTED",
                    "USER_MESSAGE_APPENDED",
                    "CONTEXT_ASSEMBLED",
                    "MODEL_CALL_STARTED",
                    "MODEL_CALL_COMPLETED",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "TOOL_CALL_FAILED",
                    "TOOL_CALL_BLOCKED",
                    "ASSISTANT_MESSAGE_APPENDED",
                    "RUN_COMPLETED",
                ],
                expected_success=False,
            ),
            EvalCase(
                case_id="session_replay_recovers_history",
                name="Session replay recovers previous turn",
                description="A second request in the same session should recover prior messages.",
                session_id="eval-migration-p0-replay",
                user_input="第二轮 hello",
                expected_answer_contains=["[stub] 第二轮 hello"],
                required_events=[
                    "THREAD_RECOVERED",
                    "CONTEXT_ASSEMBLED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                initial_context={"prime_user_input": "第一轮 hello"},
                min_history_item_count=2,
            ),
            EvalCase(
                case_id="core_memory_update_required",
                name="Core memory update calls memory tool",
                description="A durable response rule should update user_rules core memory.",
                session_id="eval-memory-core-update",
                user_input="请记住，以后排障回答按现象、证据、判断、建议、未确认项组织。",
                expected_answer_contains=["已更新核心记忆", "user_rules"],
                required_tools=["updateCoreMemory"],
                required_events=[
                    "MEMORY_INJECTED",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "CORE_MEMORY_UPDATED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                required_tool_argument_keys={
                    "updateCoreMemory": ["blockKey", "newContent", "changeReason"],
                },
                scoring_rules={
                    "requires_core_memory": True,
                    "requires_memory_index": True,
                    "min_core_memory_block_count": 3,
                },
            ),
            EvalCase(
                case_id="core_memory_injected_on_next_turn",
                name="Core memory is injected after update",
                description="A later turn should see core memory injection in trace payload.",
                session_id="eval-memory-core-next-turn",
                user_input="hello",
                expected_answer_contains=["[stub] hello"],
                required_events=[
                    "THREAD_RECOVERED",
                    "MEMORY_INJECTED",
                    "CONTEXT_ASSEMBLED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                initial_context={
                    "prime_user_input": "请记住，以后排障回答按现象、证据、判断、建议、未确认项组织。"
                },
                scoring_rules={
                    "requires_core_memory": True,
                    "requires_memory_index": True,
                    "min_core_memory_block_count": 3,
                },
            ),
            EvalCase(
                case_id="archival_memory_save_required",
                name="Archival memory save calls memory tool",
                description="A confirmed root cause should be saved as archival memory.",
                session_id="eval-memory-archival-save",
                user_input="这次 order-service 5xx 根因确认是 payment-service 连接池耗尽，请保存为长期经验。",
                expected_answer_contains=["已保存长期经验"],
                required_tools=["saveArchivalMemory"],
                required_events=[
                    "ARCHIVAL_MEMORY_WRITTEN",
                    "TOOL_CALL_STARTED",
                    "TOOL_CALL_COMPLETED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                required_tool_argument_keys={
                    "saveArchivalMemory": ["topic", "content", "evidenceSummary"]
                },
            ),
            EvalCase(
                case_id="memory_search_required",
                name="Memory search retrieves saved experience",
                description="A historical-experience question should call searchMemory.",
                session_id="eval-memory-search",
                user_input="参考历史经验，order-service 5xx 以前有没有类似原因？",
                expected_answer_contains=["payment-service"],
                required_tools=["searchMemory"],
                required_events=[
                    "ARCHIVAL_MEMORY_WRITTEN",
                    "MEMORY_SEARCHED",
                    "TOOL_CALL_COMPLETED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                initial_context={
                    "prime_user_input": "这次 order-service 5xx 根因确认是 payment-service 连接池耗尽，请保存为长期经验。"
                },
            ),
            EvalCase(
                case_id="memory_topics_required",
                name="Memory topics lists saved experience",
                description="Topic discovery should call listMemoryTopics.",
                session_id="eval-memory-topics",
                user_input="有哪些长期记忆主题？",
                expected_answer_contains=["order-service"],
                required_tools=["listMemoryTopics"],
                required_events=[
                    "ARCHIVAL_MEMORY_WRITTEN",
                    "TOOL_CALL_COMPLETED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
                initial_context={
                    "prime_user_input": "这次 order-service 5xx 根因确认是 payment-service 连接池耗尽，请保存为长期经验。"
                },
                scoring_rules={"min_memory_index_topic_count": 1},
            ),
            EvalCase(
                case_id="memory_policy_rejects_secret",
                name="Memory policy rejects secrets",
                description="Secret-like content should be rejected by deterministic policy.",
                session_id="eval-memory-policy-secret",
                user_input="请记住 api_key=abc123secret456",
                expected_answer_contains=["核心记忆更新被拒绝"],
                required_tools=["updateCoreMemory"],
                required_events=[
                    "CORE_MEMORY_UPDATE_REJECTED",
                    "TOOL_CALL_COMPLETED",
                    "RUN_COMPLETED",
                ],
                expected_success=True,
            ),
        ],
    )
