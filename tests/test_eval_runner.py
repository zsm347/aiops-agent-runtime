import pytest

from superbiz_agent.evals import (
    EvalCase,
    EvalReport,
    EvalRunner,
    EvalSuite,
    RuleJudge,
    TraceArtifact,
    skeleton_p0_smoke_suite,
)


def test_eval_public_objects_are_importable() -> None:
    assert EvalCase
    assert EvalSuite
    assert EvalRunner
    assert RuleJudge
    assert TraceArtifact
    assert EvalReport


def test_builtin_skeleton_p0_smoke_suite_shape() -> None:
    suite = skeleton_p0_smoke_suite()

    assert suite.suite_id == "skeleton_p0_smoke"
    assert [case.case_id for case in suite.cases[:8]] == [
        "datetime_requires_tool",
        "alerts_tool_required",
        "log_topics_required",
        "query_logs_required",
        "internal_docs_required",
        "plain_chat_forbids_tools",
        "empty_question_no_runtime",
        "session_replay_recovers_history",
    ]
    assert [case.case_id for case in suite.cases[8:]] == [
        "core_memory_update_required",
        "core_memory_injected_on_next_turn",
        "archival_memory_save_required",
        "memory_search_required",
        "memory_topics_required",
        "memory_policy_rejects_secret",
    ]
    assert suite.cases[0].required_tools == ["getCurrentDateTime"]
    assert suite.cases[0].required_tool_argument_keys == {
        "getCurrentDateTime": ["timezone"]
    }
    assert suite.cases[1].required_tools == ["queryPrometheusAlerts"]
    assert suite.cases[2].required_tools == ["getAvailableLogTopics"]
    assert suite.cases[3].required_tools == ["queryLogs"]
    assert suite.cases[3].required_tool_argument_keys == {
        "queryLogs": ["region", "logTopic", "query"]
    }
    assert suite.cases[4].required_tools == ["queryInternalDocs"]
    assert suite.cases[4].required_tool_argument_keys == {"queryInternalDocs": ["query"]}
    assert "queryLogs" in suite.cases[5].forbidden_tools
    assert suite.cases[6].expected_success is False
    assert suite.cases[7].min_history_item_count == 2
    assert suite.cases[8].required_tools == ["updateCoreMemory"]
    assert suite.cases[10].required_tools == ["saveArchivalMemory"]
    assert suite.cases[11].required_tools == ["searchMemory"]


@pytest.mark.asyncio
async def test_eval_runner_builtin_suite_passes_and_serializes_report() -> None:
    report = await EvalRunner().run_suite_async(skeleton_p0_smoke_suite())

    assert isinstance(report, EvalReport)
    assert report.suite_id == "skeleton_p0_smoke"
    assert report.case_count == 14
    assert report.passed_count == 14
    assert report.failed_count == 0
    assert report.pass_rate == 1.0
    assert all(result.passed for result in report.case_results)

    dumped = report.model_dump(mode="json")
    assert dumped["suite_id"] == "skeleton_p0_smoke"
    assert dumped["case_count"] == 14
    assert len(dumped["case_results"]) == 14


@pytest.mark.asyncio
async def test_eval_runner_collects_datetime_trace_artifact() -> None:
    report = await EvalRunner().run_suite_async(skeleton_p0_smoke_suite())
    result = _case_result(report, "datetime_requires_tool")
    artifact = result.artifact

    assert artifact.success is True
    assert artifact.run_id
    assert artifact.session_id == "eval-skeleton-p0-datetime"
    assert artifact.prompt_version == "ops-agent-system-v3"
    assert artifact.model_provider == "stub"
    assert artifact.tool_schema_version == "ops-tools-v3"
    assert artifact.tool_calls == ["getCurrentDateTime"]
    assert artifact.tool_arguments["getCurrentDateTime"][0]["timezone"] == "Asia/Shanghai"
    assert artifact.tool_results["getCurrentDateTime"][0]["success"] is True
    assert artifact.final_answer and "Asia/Shanghai" in artifact.final_answer
    assert artifact.raw_event_count > 0
    assert artifact.model_event_count == 4
    assert artifact.saw_core_memory is True
    assert artifact.max_context_core_memory_block_count == 3
    assert artifact.event_sequences == sorted(artifact.event_sequences)


@pytest.mark.asyncio
async def test_eval_runner_business_tool_cases_observe_results_and_traces() -> None:
    report = await EvalRunner().run_suite_async(skeleton_p0_smoke_suite())

    alerts = _case_result(report, "alerts_tool_required").artifact
    assert alerts.tool_calls == ["queryPrometheusAlerts"]
    assert alerts.tool_results["queryPrometheusAlerts"][0]["alerts"][0]["alertName"] == (
        "HighCPUUsage"
    )
    assert alerts.final_answer and "HighCPUUsage" in alerts.final_answer
    assert "TOOL_CALL_STARTED" in alerts.event_types
    assert "TOOL_CALL_COMPLETED" in alerts.event_types

    topics = _case_result(report, "log_topics_required").artifact
    assert topics.tool_calls == ["getAvailableLogTopics"]
    assert topics.tool_results["getAvailableLogTopics"][0]["defaultRegion"] == "ap-guangzhou"
    assert "application-logs" in (topics.final_answer or "")

    logs = _case_result(report, "query_logs_required").artifact
    assert logs.tool_calls == ["queryLogs"]
    assert logs.tool_arguments["queryLogs"][0] == {
        "region": "ap-guangzhou",
        "logTopic": "application-logs",
        "query": "level:ERROR",
        "limit": 20,
    }
    assert logs.tool_results["queryLogs"][0]["logs"][0]["fields"]
    assert "application-logs" in (logs.final_answer or "")

    docs = _case_result(report, "internal_docs_required").artifact
    assert docs.tool_calls == ["queryInternalDocs"]
    assert docs.tool_arguments["queryInternalDocs"][0]["query"] == "Pod 重启排查流程是什么？"
    assert docs.tool_results["queryInternalDocs"][0]["chunks"][0]["source"] == (
        "runbooks/pod-restart.md"
    )
    assert "runbooks" in (docs.final_answer or "")


@pytest.mark.asyncio
async def test_eval_runner_plain_chat_has_no_tool_trace() -> None:
    report = await EvalRunner().run_suite_async(skeleton_p0_smoke_suite())
    artifact = _case_result(report, "plain_chat_forbids_tools").artifact

    assert artifact.success is True
    assert artifact.run_id
    assert artifact.tool_calls == []
    assert artifact.tool_arguments == {}
    assert artifact.tool_results == {}
    assert "TOOL_CALL_STARTED" not in artifact.event_types
    assert artifact.final_answer == "[stub] hello"
    assert artifact.saw_core_memory is True
    assert artifact.saw_memory_index is True
    assert artifact.saw_memory_metadata is True


@pytest.mark.asyncio
async def test_eval_runner_empty_question_stays_before_runtime() -> None:
    report = await EvalRunner().run_suite_async(skeleton_p0_smoke_suite())
    artifact = _case_result(report, "empty_question_no_runtime").artifact

    assert artifact.success is False
    assert artifact.error_message == "问题内容不能为空"
    assert artifact.run_id is None
    assert artifact.raw_event_count == 0
    assert artifact.event_types == []
    assert artifact.tool_calls == []
    assert artifact.model_event_count == 0


@pytest.mark.asyncio
async def test_eval_runner_replay_case_observes_recovered_history() -> None:
    report = await EvalRunner().run_suite_async(skeleton_p0_smoke_suite())
    artifact = _case_result(report, "session_replay_recovers_history").artifact

    assert artifact.success is True
    assert "THREAD_RECOVERED" in artifact.event_types
    assert artifact.max_context_history_item_count >= 2
    assert artifact.prompt_version == "ops-agent-system-v3"
    assert artifact.tool_schema_version == "ops-tools-v3"


@pytest.mark.asyncio
async def test_eval_runner_memory_cases_observe_tools_and_traces() -> None:
    report = await EvalRunner().run_suite_async(skeleton_p0_smoke_suite())

    core = _case_result(report, "core_memory_update_required").artifact
    assert core.tool_calls == ["updateCoreMemory"]
    assert "CORE_MEMORY_UPDATED" in core.event_types
    assert core.max_context_core_memory_block_count == 3

    search = _case_result(report, "memory_search_required").artifact
    assert "saveArchivalMemory" in search.tool_calls
    assert "searchMemory" in search.tool_calls
    assert "payment-service" in (search.final_answer or "")
    assert "MEMORY_SEARCHED" in search.event_types

    topics = _case_result(report, "memory_topics_required").artifact
    assert "listMemoryTopics" in topics.tool_calls
    assert "order-service" in (topics.final_answer or "")
    assert topics.max_context_memory_index_topic_count >= 1

    rejected = _case_result(report, "memory_policy_rejects_secret").artifact
    assert rejected.tool_calls == ["updateCoreMemory"]
    assert "CORE_MEMORY_UPDATE_REJECTED" in rejected.event_types
    assert "核心记忆更新被拒绝" in (rejected.final_answer or "")


def test_rule_judge_reports_deterministic_rule_failures() -> None:
    case = EvalCase(
        case_id="failure-demo",
        name="Failure demo",
        session_id="s-failure",
        user_input="hello",
        expected_answer_contains=["expected"],
        required_tools=["getCurrentDateTime"],
        forbidden_tools=["forbiddenTool"],
        required_events=["RUN_STARTED"],
        forbidden_events=["RUN_FAILED"],
        expected_success=True,
        required_tool_argument_keys={"getCurrentDateTime": ["timezone", "nested.key"]},
    )
    artifact = TraceArtifact(
        run_id=None,
        session_id="s-failure",
        event_types=["RUN_FAILED"],
        event_sequences=[2, 1],
        tool_calls=["forbiddenTool", "getCurrentDateTime"],
        tool_arguments={"getCurrentDateTime": [{}]},
        final_answer="actual",
        success=False,
        raw_event_count=2,
    )

    result = RuleJudge().judge(case, artifact)

    assert result.passed is False
    assert result.score < 1.0
    assert any("expected_success mismatch" in failure for failure in result.failures)
    assert any("missing expected fragment" in failure for failure in result.failures)
    assert any("forbidden tool was called" in failure for failure in result.failures)
    assert any("required event was not emitted" in failure for failure in result.failures)
    assert any("forbidden event was emitted" in failure for failure in result.failures)
    assert any("timezone" in failure for failure in result.failures)
    assert any("run_id" in failure for failure in result.failures)
    assert any("sequence" in failure for failure in result.failures)


def _case_result(report: EvalReport, case_id: str):
    return next(result for result in report.case_results if result.case_id == case_id)
