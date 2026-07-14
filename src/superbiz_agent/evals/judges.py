from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.evals.cases import EvalCase
from superbiz_agent.evals.traces import TraceArtifact


class RuleJudgeResult(BaseModel):
    """Result of deterministic rule evaluation for one case."""

    model_config = ConfigDict(protected_namespaces=())

    passed: bool
    score: float
    failures: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class RuleJudge:
    """Deterministic evaluator for local migration smoke tests."""

    def judge(self, case: EvalCase, artifact: TraceArtifact) -> RuleJudgeResult:
        failures: list[str] = []
        passed_checks = 0
        total_checks = 0

        def check(ok: bool, failure: str) -> None:
            nonlocal passed_checks, total_checks
            total_checks += 1
            if ok:
                passed_checks += 1
            else:
                failures.append(failure)

        check(
            artifact.success == case.expected_success,
            (
                "expected_success mismatch: "
                f"expected={case.expected_success} actual={artifact.success}"
            ),
        )

        final_answer = artifact.final_answer or ""
        for fragment in case.expected_answer_contains:
            check(
                fragment in final_answer,
                f"final answer missing expected fragment: {fragment!r}",
            )

        observed_tools = set(artifact.tool_calls)
        for tool_name in case.required_tools:
            check(tool_name in observed_tools, f"required tool was not called: {tool_name}")

        for tool_name in case.forbidden_tools:
            check(tool_name not in observed_tools, f"forbidden tool was called: {tool_name}")

        observed_events = set(artifact.event_types)
        for event_type in case.required_events:
            check(event_type in observed_events, f"required event was not emitted: {event_type}")

        for event_type in case.forbidden_events:
            check(event_type not in observed_events, f"forbidden event was emitted: {event_type}")

        for tool_name, required_keys in case.required_tool_argument_keys.items():
            calls = artifact.tool_arguments.get(tool_name, [])
            missing_keys = [
                key
                for key in required_keys
                if not any(_has_key_path(arguments, key) for arguments in calls)
            ]
            check(
                not missing_keys,
                (
                    f"tool arguments missing required keys for {tool_name}: "
                    f"{', '.join(missing_keys)}"
                ),
            )

        if case.expected_success and artifact.raw_event_count > 0:
            check(bool(artifact.run_id), "expected successful traced case to include run_id")

        if case.min_history_item_count is not None:
            check(
                artifact.max_context_history_item_count >= case.min_history_item_count,
                (
                    "context history item count too low: "
                    f"expected>={case.min_history_item_count} "
                    f"actual={artifact.max_context_history_item_count}"
                ),
            )

        min_core_blocks = case.scoring_rules.get("min_core_memory_block_count")
        if isinstance(min_core_blocks, int):
            check(
                artifact.max_context_core_memory_block_count >= min_core_blocks,
                (
                    "core memory block count too low: "
                    f"expected>={min_core_blocks} "
                    f"actual={artifact.max_context_core_memory_block_count}"
                ),
            )

        min_memory_topics = case.scoring_rules.get("min_memory_index_topic_count")
        if isinstance(min_memory_topics, int):
            check(
                artifact.max_context_memory_index_topic_count >= min_memory_topics,
                (
                    "memory index topic count too low: "
                    f"expected>={min_memory_topics} "
                    f"actual={artifact.max_context_memory_index_topic_count}"
                ),
            )

        if case.scoring_rules.get("requires_core_memory"):
            check(artifact.saw_core_memory, "context did not report hasCoreMemory=true")

        if case.scoring_rules.get("requires_memory_index"):
            check(artifact.saw_memory_index, "context did not report hasMemoryIndex=true")

        check(
            _is_strictly_increasing(artifact.event_sequences),
            "trace event sequence is not strictly increasing",
        )

        score = 1.0 if total_checks == 0 else passed_checks / total_checks
        return RuleJudgeResult(
            passed=not failures,
            score=score,
            failures=failures,
            details={
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "observed_tools": artifact.tool_calls,
                "observed_events": artifact.event_types,
                "raw_event_count": artifact.raw_event_count,
                "run_id": artifact.run_id,
            },
        )


def _has_key_path(payload: dict[str, Any], key_path: str) -> bool:
    current: Any = payload
    for part in key_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _is_strictly_increasing(values: list[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))
