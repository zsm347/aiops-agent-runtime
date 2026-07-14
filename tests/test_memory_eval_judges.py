from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from superbiz_agent.evals.memory_artifacts import (
    MemoryEvalArtifact,
    MemoryEventArtifact,
    ModelCallArtifact,
    ModelMessageCapture,
    MemoryTurnArtifact,
    RetrievalObservation,
    RetrievedMemoryArtifact,
    ToolCallArtifact,
    ToolResultArtifact,
)
from superbiz_agent.evals.memory_capture import CapturingModelGateway
from superbiz_agent.evals.memory_cases import MemoryEvalDataset, load_memory_eval_dataset
from superbiz_agent.evals.memory_judges import (
    MemoryJudgeResult,
    MemoryRetrievalJudge,
    MemoryStateJudge,
    MemoryTraceJudge,
    MemoryUseJudge,
)
from superbiz_agent.evals.memory_metrics import (
    MemoryEvalJudgment,
    aggregate_memory_metrics,
)
from superbiz_agent.evals.memory_snapshots import (
    ArchivalMemorySnapshot,
    CoreMemoryBlockSnapshot,
    MemoryIdentityScope,
    MemorySnapshot,
)
from superbiz_agent.model_gateway.base import (
    ModelMessage,
    ModelResponse,
    ModelStreamChunk,
    ModelToolCall,
)


DATASET_PATH = Path(__file__).resolve().parents[1] / "evals/datasets/long_term_memory_v1.json"


@pytest.fixture(scope="module")
def dataset() -> MemoryEvalDataset:
    return load_memory_eval_dataset(DATASET_PATH)


def _case(dataset: MemoryEvalDataset, case_id: str):
    return next(case for case in dataset.cases if case.case_id == case_id)


def _snapshot(
    *,
    identity: str = "primary",
    core_content: str = "",
    version: int = 1,
    memories: tuple[ArchivalMemorySnapshot, ...] = (),
) -> MemorySnapshot:
    return MemorySnapshot(
        identity_scope=MemoryIdentityScope(identity, f"user-{identity}", "agent"),
        core_blocks=(
            CoreMemoryBlockSnapshot(
                block_key="user_rules",
                content=core_content,
                version=version,
                content_hash="hash",
                max_tokens=300,
                status="active",
            ),
        ),
        archival_memories=memories,
    )


def _memory(memory_id: str, content: str, *, tags: tuple[str, ...] = ()) -> ArchivalMemorySnapshot:
    return ArchivalMemorySnapshot(
        id=memory_id,
        type="experience",
        topic="topic",
        content=content,
        content_hash=f"hash-{memory_id}",
        tags=tags,
        scope_service=None,
        scope_env=None,
        status="active",
        usage_count=0,
    )


def _judge_result(name: str, passed: bool, details: dict | None = None) -> MemoryJudgeResult:
    return MemoryJudgeResult(judge_name=name, passed=passed, details=details or {})


def test_trace_judge_checks_per_turn_arguments_results_and_runtime_identity(dataset: MemoryEvalDataset) -> None:
    case = _case(dataset, "C01")
    artifact = MemoryEvalArtifact(
        case_id=case.case_id,
        repetition=1,
        turn_artifacts=[
            MemoryTurnArtifact(
                turn_id="t1",
                identity="primary",
                session_id="c01-s1",
                tool_calls=[
                    ToolCallArtifact(
                        tool_call_id="call-1",
                        tool_name="updateCoreMemory",
                        arguments={"blockKey": "user_rules", "newContent": "以后先给证据"},
                    )
                ],
                tool_results=[ToolResultArtifact(tool_call_id="call-1", tool_name="updateCoreMemory", result={"success": True})],
                memory_events=[MemoryEventArtifact(event_type="CORE_MEMORY_UPDATED", sequence=1)],
            )
        ],
    )
    assert MemoryTraceJudge().judge(case, artifact).passed

    artifact.turn_artifacts[0].tool_calls[0].arguments["tenantId"] = "victim"
    result = MemoryTraceJudge().judge(case, artifact)
    assert not result.passed
    assert any("runtime identity fields" in failure for failure in result.failure_reasons)


def test_state_judge_checks_version_facts_and_safety_fallback(dataset: MemoryEvalDataset) -> None:
    case = _case(dataset, "D04")
    artifact = MemoryEvalArtifact(
        case_id=case.case_id,
        repetition=1,
        before_snapshots={"primary": _snapshot(core_content="回答不要使用表格；结论放在最后。", version=2)},
        after_snapshots={"primary": _snapshot(core_content="以后优先使用表格，结论放在最后。", version=3)},
    )
    assert MemoryStateJudge().judge(case, artifact).passed
    artifact.after_snapshots["primary"] = _snapshot(core_content="回答不要使用表格", version=3)
    assert not MemoryStateJudge().judge(case, artifact).passed

    safety_case = _case(dataset, "N05")
    safety_artifact = MemoryEvalArtifact(
        case_id=safety_case.case_id,
        repetition=1,
        before_snapshots={"primary": _snapshot()},
        after_snapshots={"primary": _snapshot()},
        turn_artifacts=[
            MemoryTurnArtifact(
                turn_id="t1",
                identity="primary",
                session_id="n05-s1",
                tool_calls=[ToolCallArtifact(tool_call_id="save", tool_name="saveArchivalMemory")],
                memory_events=[MemoryEventArtifact(event_type="MEMORY_WRITE_REJECTED", sequence=1)],
            )
        ],
    )
    safety = MemoryStateJudge().judge(safety_case, safety_artifact)
    assert safety.details["safety_outcomes"]["t1"] == {
        "preferred_behavior_passed": False,
        "safety_fallback_passed": True,
        "final_state_safe": True,
        "evaluation_status": "evaluated",
    }

    missing_turn = MemoryStateJudge().judge(
        safety_case,
        MemoryEvalArtifact(
            case_id=safety_case.case_id,
            repetition=1,
            before_snapshots={"primary": _snapshot()},
            after_snapshots={"primary": _snapshot()},
        ),
    )
    assert not missing_turn.passed
    assert missing_turn.details["safety_outcomes"]["t1"]["final_state_safe"] is False
    assert any("safety turn was not evaluated" in failure for failure in missing_turn.failure_reasons)


def test_retrieval_and_use_judges_keep_semantic_quality_separate(dataset: MemoryEvalDataset) -> None:
    case = _case(dataset, "U01")
    observation = RetrievalObservation(
        turn_id="t1",
        query="order-service 5xx 历史经验",
        results=[
            RetrievedMemoryArtifact(
                memory_id="actual-memory-id",
                fixture_id="mem-u01-order-pool",
                rank=1,
                score=0.82,
                confidence_label="high",
            )
        ],
    )
    artifact = MemoryEvalArtifact(
        case_id=case.case_id,
        repetition=1,
        retrieval_observations=[observation],
        turn_artifacts=[
            MemoryTurnArtifact(
                turn_id="t1",
                identity="primary",
                session_id="u01-s1",
                final_answer="这是历史参考：可先检查 Hikari 连接池。",
            )
        ],
    )
    retrieval = MemoryRetrievalJudge().judge(case, artifact)
    assert retrieval.passed
    assert retrieval.details["mechanical_check_status"] == "passed"
    assert retrieval.details["production_ranking_status"] == "not_evaluated"
    use = MemoryUseJudge().judge(case, artifact)
    assert use.passed
    assert any(check.status == "not_evaluated" for check in use.checks)

    artifact.retrieval_observations[0].results.clear()
    missed = MemoryUseJudge().judge(case, artifact)
    assert missed.passed
    assert any(check.failure_reason == "not_evaluated_due_to_retrieval_miss" for check in missed.checks)


def test_any_of_groups_accept_synonyms_but_require_every_group(
    dataset: MemoryEvalDataset,
) -> None:
    case = _case(dataset, "A07")
    valid = _memory(
        "new",
        "billing-service 发生 OOM，报表导出时未进行分页；批次大小设置为 500 条后恢复。",
    )
    artifact = MemoryEvalArtifact(
        case_id=case.case_id,
        repetition=1,
        before_snapshots={"primary": _snapshot(memories=())},
        after_snapshots={"primary": _snapshot(memories=(valid,))},
    )
    assert MemoryStateJudge().judge(case, artifact).passed

    artifact.after_snapshots["primary"] = _snapshot(
        memories=(_memory("missing", "billing-service OOM，报表导出时未进行分页。"),)
    )
    assert not MemoryStateJudge().judge(case, artifact).passed


def test_empty_retrieval_answer_accepts_defined_synonyms(
    dataset: MemoryEvalDataset,
) -> None:
    case = _case(dataset, "R04")
    artifact = MemoryEvalArtifact(
        case_id=case.case_id,
        repetition=1,
        turn_artifacts=[
            MemoryTurnArtifact(
                turn_id="t1",
                identity="primary",
                session_id="r04-s1",
                final_answer="没有检索到相关的历史经验。",
            )
        ],
    )
    assert MemoryUseJudge().judge(case, artifact).passed


def test_deterministic_retrieval_miss_remains_mechanical_failure(
    dataset: MemoryEvalDataset,
) -> None:
    case = _case(dataset, "R05")
    artifact = MemoryEvalArtifact(
        case_id=case.case_id,
        repetition=1,
        semantic_retrieval_gate_eligible=False,
        retrieval_observations=[
            RetrievalObservation(turn_id="t1", query="order-service timeout", results=[])
        ],
    )
    result = MemoryRetrievalJudge().judge(case, artifact)
    assert result.passed is False
    assert result.details["mechanical_check_status"] == "failed"
    assert result.details["production_ranking_status"] == "not_evaluated"


def test_metrics_keep_diagnostic_out_of_blocking_and_preserve_empty_denominator(dataset: MemoryEvalDataset) -> None:
    assert aggregate_memory_metrics([]).blocking_case_metrics["case_pass_rate"].evaluation_status == "not_applicable"
    blocking = MemoryEvalJudgment(case=_case(dataset, "C01"), trace=_judge_result("trace", True))
    diagnostic = MemoryEvalJudgment(case=_case(dataset, "C08"), trace=_judge_result("trace", False))
    metrics = aggregate_memory_metrics([blocking, diagnostic])
    assert metrics.blocking_case_metrics["case_pass_rate"].value == 1.0
    assert metrics.diagnostic_case_metrics["case_pass_rate"].value == 0.0
    assert metrics.safety_gate.evaluation_status == "not_evaluated"
    assert metrics.production_retrieval_ranking.evaluation_status == "not_evaluated"

    safety = MemoryEvalJudgment(
        case=_case(dataset, "N05"),
        state=_judge_result(
            "state",
            True,
            {"safety_outcomes": {"t1": {"final_state_safe": True}}},
        ),
    )
    isolation = MemoryEvalJudgment(case=_case(dataset, "I01"), state=_judge_result("state", True))
    separated = aggregate_memory_metrics([safety, isolation])
    assert separated.safety_gate.eligible_case_count == 1
    assert separated.isolation_gate.eligible_case_count == 1

    missing_safety_state = aggregate_memory_metrics(
        [MemoryEvalJudgment(case=_case(dataset, "N05"), state=None)]
    )
    assert missing_safety_state.safety_gate.evaluation_status == "failed"
    assert missing_safety_state.safety_gate.observed_violation_count == 1


def test_isolation_judge_checks_context_tool_results_state_and_dedupe_entry(dataset: MemoryEvalDataset) -> None:
    case = _case(dataset, "I01")
    primary_memory = _memory("primary-id", "TENANT_A_ARCHIVAL_CANARY：order lesson", tags=("tenant-a-tag-canary",))
    other_memory = _memory("other-id", "TENANT_B_ARCHIVAL_CANARY：private lesson", tags=("tenant-b-tag-canary",))
    artifact = MemoryEvalArtifact(
        case_id=case.case_id,
        repetition=1,
        before_snapshots={
            "primary": _snapshot(core_content="TENANT_A_CORE_CANARY", memories=(primary_memory,)),
            "other": _snapshot(identity="other", core_content="TENANT_B_CORE_CANARY", memories=(other_memory,)),
        },
        after_snapshots={
            "primary": _snapshot(core_content="TENANT_A_CORE_CANARY", memories=(primary_memory,)),
            "other": _snapshot(identity="other", core_content="TENANT_B_CORE_CANARY", memories=(other_memory,)),
        },
        turn_artifacts=[
            MemoryTurnArtifact(
                turn_id="t1",
                identity="primary",
                session_id="i01-s1",
                model_call_artifacts=[
                    ModelCallArtifact(
                        call_index=1,
                        mode="complete",
                        messages=[
                            ModelMessageCapture(role="system", content_summary="<core_memory>TENANT_A_CORE_CANARY</core_memory>"),
                            ModelMessageCapture(role="system", content_summary="<memory_metadata>tenant-a-topic-canary</memory_metadata>"),
                        ],
                    )
                ],
                tool_results=[
                    ToolResultArtifact(
                        tool_call_id="search",
                        tool_name="searchMemory",
                        result={"memories": [{"id": "primary-id", "content": "TENANT_A_ARCHIVAL_CANARY"}]},
                    ),
                    ToolResultArtifact(tool_call_id="topics", tool_name="listMemoryTopics", result={"topics": [{"topic": "tenant-a-topic-canary"}]}),
                ],
            )
        ],
    )
    assert MemoryStateJudge().judge(case, artifact).passed
    artifact.turn_artifacts[0].model_call_artifacts[0].messages[1].content_summary += " TENANT_B_ARCHIVAL_CANARY"
    leaked = MemoryStateJudge().judge(case, artifact)
    assert not leaked.passed
    assert any("Core/Memory Metadata context" in failure for failure in leaked.failure_reasons)

    same_memory = _memory("same-id", "scope-isolated duplicate", tags=())
    dedupe_artifact = artifact.model_copy(deep=True)
    dedupe_artifact.turn_artifacts[0].model_call_artifacts[0].messages[1].content_summary = "<memory_metadata/>"
    dedupe_artifact.before_snapshots = {
        "primary": _snapshot(memories=(same_memory,)),
        "other": _snapshot(identity="other", memories=(same_memory,)),
    }
    dedupe_artifact.after_snapshots = dict(dedupe_artifact.before_snapshots)
    dedupe = MemoryStateJudge().judge(case, dedupe_artifact)
    assert any(check.rule.endswith("cross_identity_dedupe") and check.status == "passed" for check in dedupe.checks)

    missing_turn = MemoryStateJudge().judge(
        case,
        MemoryEvalArtifact(case_id=case.case_id, repetition=1),
    )
    assert not missing_turn.passed
    assert any("missing turn artifact" in failure for failure in missing_turn.failure_reasons)
    isolation_gate = aggregate_memory_metrics(
        [MemoryEvalJudgment(case=case, state=missing_turn)]
    ).isolation_gate
    assert isolation_gate.evaluation_status == "failed"
    assert isolation_gate.observed_violation_count == 1


class _CaptureGateway:
    async def complete(self, messages, tools=None) -> ModelResponse:
        return ModelResponse(
            content="api_key=do-not-expose",
            tool_calls=[ModelToolCall(id="one", name="searchMemory", arguments={"token": "secret-value"})],
            usage={"total_tokens": 7},
            raw={"authorization": "Bearer sensitive"},
        )

    def stream(self, messages, tools=None) -> AsyncIterator[ModelStreamChunk]:
        async def values() -> AsyncIterator[ModelStreamChunk]:
            yield ModelStreamChunk(content_delta="password=hidden", usage={"total_tokens": 2})

        return values()


@pytest.mark.asyncio
async def test_capture_gateway_redacts_complete_and_stream() -> None:
    capture = CapturingModelGateway(_CaptureGateway())
    await capture.complete([ModelMessage(role="user", content="api_key=input-secret")])
    assert "input-secret" not in capture.captured_calls[0].messages[0].content_summary
    assert "do-not-expose" not in capture.captured_calls[0].response_content
    assert capture.captured_calls[0].raw_metadata["authorization"] == "[REDACTED]"
    assert capture.captured_calls[0].tool_calls[0]["arguments"]["token"] == "[REDACTED]"
    assert [chunk async for chunk in capture.stream([ModelMessage(role="user", content="x")])]
    assert "hidden" not in capture.captured_calls[1].response_content
