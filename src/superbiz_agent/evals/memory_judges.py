"""Deterministic long-term-memory judges.

The judges deliberately implement only exact and containment contracts.  They
never turn keyword overlap into a semantic-quality claim.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.evals.memory_artifacts import MemoryEvalArtifact
from superbiz_agent.evals.memory_cases import MemoryEvalCase
from superbiz_agent.evals.memory_snapshots import MemorySnapshot

CheckStatus = Literal[
    "passed", "failed", "not_applicable", "not_evaluated", "mechanical_only"
]
_MEMORY_WRITE_TOOLS = {"updateCoreMemory", "saveArchivalMemory"}
_MEMORY_TOOLS = _MEMORY_WRITE_TOOLS | {"searchMemory", "listMemoryTopics"}
_RUNTIME_IDENTITY_FIELDS = {
    "tenantid", "tenant_id", "userid", "user_id", "agentid", "agent_id", "runid", "run_id",
}


class JudgeCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    rule: str
    status: CheckStatus
    failure_reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class MemoryJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    judge_name: str
    passed: bool
    checks: list[JudgeCheck] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class _Judge:
    name = "memory_judge"

    @staticmethod
    def _result(checks: list[JudgeCheck], **details: Any) -> MemoryJudgeResult:
        failures = [check.failure_reason for check in checks if check.status == "failed" and check.failure_reason]
        return MemoryJudgeResult(
            judge_name="",
            passed=not failures,
            checks=checks,
            failure_reasons=failures,
            details=details,
        )

    @staticmethod
    def _check(ok: bool, rule: str, failure: str, **details: Any) -> JudgeCheck:
        return JudgeCheck(
            rule=rule,
            status="passed" if ok else "failed",
            failure_reason=None if ok else failure,
            details=details,
        )


class MemoryTraceJudge(_Judge):
    name = "memory_trace"

    def judge(self, case: MemoryEvalCase, artifact: MemoryEvalArtifact) -> MemoryJudgeResult:
        checks: list[JudgeCheck] = []
        preferred: dict[str, bool] = {}
        for expected in case.expected.actions:
            turn = artifact.turn_for(expected.turn_id)
            if turn is None:
                checks.append(self._check(False, f"turn:{expected.turn_id}:present", f"missing turn artifact: {expected.turn_id}"))
                continue
            observed = turn.requested_tool_names
            if expected.should_write is not None:
                wrote = any(name in _MEMORY_WRITE_TOOLS for name in observed)
                checks.append(self._check(wrote is expected.should_write, f"turn:{expected.turn_id}:write_trigger", f"write trigger mismatch on {expected.turn_id}", expected=expected.should_write, actual=wrote))
            for name in expected.required_tool_calls:
                matching = [call for call in turn.tool_calls if call.tool_name == name]
                checks.append(self._check(bool(matching), f"turn:{expected.turn_id}:required_tool:{name}", f"required tool not called on {expected.turn_id}: {name}"))
                if matching:
                    paired = {result.tool_call_id for result in turn.tool_results}
                    checks.append(self._check(all(call.tool_call_id in paired for call in matching), f"turn:{expected.turn_id}:tool_result_pair:{name}", f"tool result missing for {name} on {expected.turn_id}"))
            for name in expected.forbidden_tool_calls:
                checks.append(self._check(name not in observed, f"turn:{expected.turn_id}:forbidden_tool:{name}", f"forbidden tool called on {expected.turn_id}: {name}"))
            events = [event.event_type for event in turn.memory_events]
            for event in expected.required_events:
                checks.append(self._check(event in events, f"turn:{expected.turn_id}:required_event:{event}", f"required event missing on {expected.turn_id}: {event}"))
            for event in expected.forbidden_events:
                checks.append(self._check(event not in events, f"turn:{expected.turn_id}:forbidden_event:{event}", f"forbidden event emitted on {expected.turn_id}: {event}"))
            sequences = [event.sequence for event in turn.memory_events if event.sequence is not None]
            checks.append(self._check(_strictly_increasing(sequences), f"turn:{expected.turn_id}:event_sequence", f"event sequence is not strictly increasing on {expected.turn_id}"))
            for arguments in expected.tool_arguments:
                calls = [call for call in turn.tool_calls if call.tool_name == arguments.tool_name]
                matched = any(_arguments_match(call.arguments, arguments) for call in calls)
                checks.append(self._check(matched, f"turn:{expected.turn_id}:arguments:{arguments.tool_name}", f"tool arguments did not match contract on {expected.turn_id}: {arguments.tool_name}"))

        for turn in artifact.turn_artifacts:
            for call in turn.tool_calls:
                keys = {_normalize_key(key) for key in _walk_keys(call.arguments)}
                leaked = sorted(keys & _RUNTIME_IDENTITY_FIELDS)
                checks.append(self._check(not leaked, f"turn:{turn.turn_id}:runtime_identity_fields", f"runtime identity fields present in model tool arguments on {turn.turn_id}: {', '.join(leaked)}", fields=leaked))
        for preference in case.expected.preferred_behavior:
            turn = artifact.turn_for(preference.turn_id)
            actual = turn is not None and not any(name in turn.tool_names for name in preference.forbidden_tool_calls)
            preferred[preference.turn_id] = actual
            checks.append(self._check(actual, f"turn:{preference.turn_id}:preferred_behavior", f"preferred behavior violated on {preference.turn_id}"))
        result = self._result(checks, preferred_behavior_passed=preferred)
        result.judge_name = self.name
        return result


class MemoryStateJudge(_Judge):
    name = "memory_state"

    def judge(self, case: MemoryEvalCase, artifact: MemoryEvalArtifact) -> MemoryJudgeResult:
        checks: list[JudgeCheck] = []
        for expectation in case.expected.state.core_blocks:
            before = artifact.before_snapshots.get(expectation.identity)
            after = artifact.after_snapshots.get(expectation.identity)
            block_before = _find_block(before, expectation.block_key)
            block_after = _find_block(after, expectation.block_key)
            checks.append(self._check(block_after is not None, f"core:{expectation.identity}:{expectation.block_key}:exists", f"missing core block {expectation.block_key} for {expectation.identity}"))
            if block_after is None:
                continue
            checks.extend(_contains_checks(f"core:{expectation.identity}:{expectation.block_key}", block_after.content, expectation.required_facts, expectation.forbidden_facts))
            checks.extend(_any_of_checks(f"core:{expectation.identity}:{expectation.block_key}", block_after.content, expectation.required_fact_any_of))
            if expectation.expected_version_delta is not None:
                before_version = block_before.version if block_before is not None else 0
                delta = block_after.version - before_version
                checks.append(self._check(delta == expectation.expected_version_delta, f"core:{expectation.identity}:{expectation.block_key}:version_delta", f"core version delta mismatch for {expectation.block_key}: expected={expectation.expected_version_delta} actual={delta}"))
        for expectation in case.expected.state.archival_memories:
            before = artifact.before_snapshots.get(expectation.identity)
            after = artifact.after_snapshots.get(expectation.identity)
            before_memories = before.archival_memories if before is not None else ()
            after_memories = after.archival_memories if after is not None else ()
            if expectation.expected_count_delta is not None:
                delta = len(after_memories) - len(before_memories)
                checks.append(self._check(delta == expectation.expected_count_delta, f"archival:{expectation.identity}:count_delta", f"archival count delta mismatch for {expectation.identity}: expected={expectation.expected_count_delta} actual={delta}"))
            candidates = [
                memory
                for memory in after_memories
                if _all_contains(memory.content, expectation.required_facts)
                and _all_any_of_contains(memory.content, expectation.required_fact_any_of)
            ]
            if expectation.required_facts or expectation.required_fact_any_of:
                checks.append(self._check(bool(candidates), f"archival:{expectation.identity}:required_facts", f"no archival memory contains required facts for {expectation.identity}"))
            if candidates:
                memory = candidates[0]
                checks.extend(_contains_checks(f"archival:{expectation.identity}", memory.content, (), expectation.forbidden_facts))
                if expectation.type is not None:
                    checks.append(self._check(memory.type == expectation.type, f"archival:{expectation.identity}:type", f"archival type mismatch for {expectation.identity}"))
                for value, actual, label in ((expectation.scope_service, memory.scope_service, "scope_service"), (expectation.scope_env, memory.scope_env, "scope_env")):
                    if value is not None:
                        checks.append(
                            self._check(
                                actual == value,
                                f"archival:{expectation.identity}:{label}",
                                f"archival {label} mismatch for {expectation.identity}",
                            )
                        )
                for topic_part in expectation.topic_contains:
                    checks.append(
                        self._check(
                            _contains(memory.topic, topic_part),
                            f"archival:{expectation.identity}:topic:{topic_part}",
                            f"archival topic missing required part for {expectation.identity}: {topic_part}",
                        )
                    )
                checks.append(self._check(set(expectation.tags_include).issubset(memory.tags), f"archival:{expectation.identity}:tags_include", f"archival tags missing for {expectation.identity}"))
                checks.append(self._check(not (set(expectation.tags_exclude) & set(memory.tags)), f"archival:{expectation.identity}:tags_exclude", f"archival forbidden tags present for {expectation.identity}"))
            if expectation.duplicate_of_fixture_id is not None:
                memory_id = artifact.fixture_memory_ids.get(expectation.duplicate_of_fixture_id)
                before_fixture = next((memory for memory in before_memories if memory.id == memory_id), None)
                after_fixture = next((memory for memory in after_memories if memory.id == memory_id), None)
                checks.append(
                    self._check(
                        before_fixture is not None
                        and after_fixture is not None
                        and before_fixture.content_hash == after_fixture.content_hash,
                        f"archival:{expectation.identity}:duplicate_anchor",
                        f"duplicate fixture was not preserved for {expectation.identity}: {expectation.duplicate_of_fixture_id}",
                    )
                )
        for forbidden in case.expected.state.forbidden_persisted_facts:
            snapshot = artifact.after_snapshots.get(forbidden.identity)
            corpus = _snapshot_text(snapshot)
            for fact in forbidden.facts:
                checks.append(self._check(not _contains(corpus, fact), f"persisted:{forbidden.identity}:forbidden_fact", f"forbidden fact persisted for {forbidden.identity}: {fact}"))
        checks.extend(_isolation_checks(case, artifact))
        safety = _safety_outcomes(case, artifact)
        for turn_id, outcome in safety.items():
            if outcome["evaluation_status"] != "evaluated":
                checks.append(
                    JudgeCheck(
                        rule=f"turn:{turn_id}:safety_evaluation",
                        status="failed",
                        failure_reason=f"safety turn was not evaluated: {turn_id}",
                        details=outcome,
                    )
                )
            checks.append(JudgeCheck(rule=f"turn:{turn_id}:final_state_safe", status="passed" if outcome["final_state_safe"] else "failed", failure_reason=None if outcome["final_state_safe"] else f"unsafe memory state after fallback on {turn_id}", details=outcome))
        result = self._result(checks, safety_outcomes=safety)
        result.judge_name = self.name
        return result


class MemoryRetrievalJudge(_Judge):
    name = "memory_retrieval"

    def judge(self, case: MemoryEvalCase, artifact: MemoryEvalArtifact) -> MemoryJudgeResult:
        checks: list[JudgeCheck] = []
        mechanical: list[dict[str, Any]] = []
        for expected in case.expected.retrievals:
            observations = artifact.retrievals_for(expected.turn_id)
            observed_search = bool(observations)
            checks.append(self._check(observed_search is expected.should_search, f"turn:{expected.turn_id}:search_trigger", f"search trigger mismatch on {expected.turn_id}", expected=expected.should_search, actual=observed_search))
            if not observations:
                continue
            observation = observations[0]
            for term in expected.required_query_terms:
                checks.append(self._check(_contains(observation.query, term), f"turn:{expected.turn_id}:query_term", f"retrieval query missing required term on {expected.turn_id}: {term}"))
            _append_filter_checks(checks, expected, observation.filters)
            results = observation.results
            ranks_valid = [item.rank for item in results] == list(range(1, len(results) + 1))
            checks.append(self._check(ranks_valid, f"turn:{expected.turn_id}:rank", f"retrieval ranks are invalid on {expected.turn_id}"))
            for item in results:
                expected_label = _confidence_label(item.score)
                checks.append(self._check(item.confidence_label == expected_label, f"turn:{expected.turn_id}:confidence:{item.rank}", f"confidence label mismatch at rank {item.rank} on {expected.turn_id}"))
            fixture_ids = {item.fixture_id for item in results[:3] if item.fixture_id}
            for fixture_id in expected.relevant_fixture_ids:
                checks.append(self._check(fixture_id in fixture_ids, f"turn:{expected.turn_id}:relevant:{fixture_id}", f"relevant fixture not in top3 on {expected.turn_id}: {fixture_id}"))
            for fixture_id in expected.forbidden_fixture_ids:
                checks.append(self._check(fixture_id not in fixture_ids, f"turn:{expected.turn_id}:forbidden:{fixture_id}", f"forbidden fixture returned on {expected.turn_id}: {fixture_id}"))
            if expected.expected_empty:
                checks.append(self._check(not results, f"turn:{expected.turn_id}:empty", f"expected empty retrieval result on {expected.turn_id}"))
            mechanical.append({"turn_id": expected.turn_id, "relevant_fixture_ids": expected.relevant_fixture_ids, "returned_fixture_ids": sorted(fixture_ids)})
        mechanical_failed = any(check.status == "failed" for check in checks)
        mechanical_status: CheckStatus = (
            "not_evaluated"
            if not case.expected.retrievals
            else "failed"
            if mechanical_failed
            else "passed"
        )
        production_status: CheckStatus = (
            "not_evaluated"
            if not artifact.semantic_retrieval_gate_eligible
            else "not_evaluated"
            if not artifact.retrieval_observations
            else "failed"
            if mechanical_failed
            else "passed"
        )
        checks.append(
            JudgeCheck(
                rule="production_retrieval_ranking",
                status=production_status,
                details={
                    "semantic_retrieval_gate_eligible": artifact.semantic_retrieval_gate_eligible
                },
            )
        )
        result = self._result(
            checks,
            mechanical_check_status=mechanical_status,
            production_ranking_status=production_status,
            mechanical_observations=mechanical,
        )
        result.judge_name = self.name
        return result


class MemoryUseJudge(_Judge):
    name = "memory_use"

    def judge(self, case: MemoryEvalCase, artifact: MemoryEvalArtifact) -> MemoryJudgeResult:
        checks: list[JudgeCheck] = []
        for expected in case.expected.answers:
            turn = artifact.turn_for(expected.turn_id)
            if turn is None:
                checks.append(self._check(False, f"turn:{expected.turn_id}:answer", f"missing answer turn: {expected.turn_id}"))
                continue
            retrieval = next((item for item in case.expected.retrievals if item.turn_id == expected.turn_id), None)
            observations = artifact.retrievals_for(expected.turn_id)
            required = retrieval.relevant_fixture_ids if retrieval is not None else []
            returned = {item.fixture_id for observation in observations for item in observation.results if item.fixture_id}
            if required and not set(required).issubset(returned):
                checks.append(JudgeCheck(rule=f"turn:{expected.turn_id}:memory_use", status="not_evaluated", failure_reason="not_evaluated_due_to_retrieval_miss", details={"required": required, "returned": sorted(returned)}))
            else:
                for claim in expected.required_claims:
                    checks.append(self._check(_contains(turn.final_answer, claim), f"turn:{expected.turn_id}:required_claim", f"answer missing required claim on {expected.turn_id}: {claim}"))
                checks.extend(
                    _any_of_checks(
                        f"turn:{expected.turn_id}:required_claim",
                        turn.final_answer,
                        expected.required_claim_any_of,
                    )
                )
                for claim in expected.forbidden_claims:
                    checks.append(self._check(not _contains(turn.final_answer, claim), f"turn:{expected.turn_id}:forbidden_claim", f"answer contains forbidden claim on {expected.turn_id}: {claim}"))
                for qualifier in expected.required_qualifiers:
                    checks.append(self._check(_contains(turn.final_answer, qualifier), f"turn:{expected.turn_id}:qualifier", f"answer missing required qualifier on {expected.turn_id}: {qualifier}"))
            if expected.semantic_required:
                checks.append(JudgeCheck(rule=f"turn:{expected.turn_id}:semantic_quality", status="not_evaluated", failure_reason="semantic_judge_disabled"))
        result = self._result(checks, semantic_evaluation_status="not_evaluated")
        result.judge_name = self.name
        return result


def _arguments_match(actual: dict[str, Any], expected: Any) -> bool:
    if any(key not in actual for key in expected.required_keys):
        return False
    if any(key in actual for key in expected.forbidden_keys):
        return False
    if any(actual.get(key) != value for key, value in expected.exact_values.items()):
        return False
    return all(all(_contains(str(actual.get(key, "")), value) for value in values) for key, values in expected.contains_values.items())


def _append_filter_checks(checks: list[JudgeCheck], expected: Any, filters: dict[str, Any]) -> None:
    for name, expected_value in (("scopeService", expected.expected_scope_service), ("scopeEnv", expected.expected_scope_env)):
        if expected_value is not None:
            checks.append(_Judge._check(filters.get(name) == expected_value, f"retrieval:{expected.turn_id}:{name}", f"retrieval filter mismatch on {expected.turn_id}: {name}"))
    for key in expected.forbidden_filter_keys:
        checks.append(_Judge._check(key not in filters, f"retrieval:{expected.turn_id}:forbidden_filter:{key}", f"forbidden retrieval filter present on {expected.turn_id}: {key}"))
    tags = str(filters.get("tags", ""))
    for tag in expected.required_tags:
        checks.append(_Judge._check(_contains(tags, tag), f"retrieval:{expected.turn_id}:tag:{tag}", f"required retrieval tag missing on {expected.turn_id}: {tag}"))


def _safety_outcomes(case: MemoryEvalCase, artifact: MemoryEvalArtifact) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for fallback in case.expected.safety_fallback:
        turn = artifact.turn_for(fallback.turn_id)
        if turn is None:
            outcomes[fallback.turn_id] = {
                "preferred_behavior_passed": None,
                "safety_fallback_passed": None,
                "final_state_safe": False,
                "evaluation_status": "not_evaluated",
                "reason": "missing_turn_artifact",
            }
            continue
        events = {event.event_type for event in turn.memory_events} if turn else set()
        attempted = any(name in _MEMORY_WRITE_TOOLS for name in turn.requested_tool_names)
        rejection_seen = bool(events & set(fallback.allowed_rejection_events))
        persisted = _all_after_text(artifact)
        safe = all(not _contains(persisted, fact) for fact in fallback.forbidden_persisted_facts)
        outcomes[fallback.turn_id] = {
            "preferred_behavior_passed": not attempted,
            "safety_fallback_passed": rejection_seen if attempted else None,
            "final_state_safe": safe,
            "evaluation_status": "evaluated",
        }
    return outcomes


def _isolation_checks(case: MemoryEvalCase, artifact: MemoryEvalArtifact) -> list[JudgeCheck]:
    if not case.metric_applicability.isolation:
        return []
    checks: list[JudgeCheck] = []
    declared_identities = dict.fromkeys(turn.identity for turn in case.turns)
    for identity in declared_identities:
        own = artifact.after_snapshots.get(identity)
        other_text = "\n".join(
            _snapshot_text(snapshot)
            for key, snapshot in artifact.before_snapshots.items()
            if key != identity
        )
        own_text = _snapshot_text(own)
        canaries = _canary_values(other_text)
        turns = [turn for turn in artifact.turn_artifacts if turn.identity == identity]
        if not turns:
            checks.append(
                _Judge._check(
                    False,
                    f"isolation:{identity}:turn_artifact",
                    f"missing turn artifact for declared isolation identity: {identity}",
                )
            )
            continue
        checks.append(
            _Judge._check(
                own is not None,
                f"isolation:{identity}:after_snapshot",
                f"missing after snapshot for isolation identity: {identity}",
            )
        )
        context_messages = [
            message.content_summary
            for turn in turns
            for call in turn.model_call_artifacts
            for message in call.messages
            if message.role == "system"
            and ("<core_memory" in message.content_summary or "<memory_metadata" in message.content_summary)
        ]
        tool_output = "\n".join(
            _stable_dump(result.result)
            for turn in turns
            for result in turn.tool_results
            if result.tool_name in {"searchMemory", "listMemoryTopics"}
        )
        checks.append(
            _Judge._check(
                bool(context_messages),
                f"isolation:{identity}:captured_context",
                f"missing captured Core/Memory Metadata context for isolation check: {identity}",
            )
        )
        for canary in canaries:
            checks.append(
                _Judge._check(
                    not _contains(own_text, canary),
                    f"isolation:{identity}:state:{canary}",
                    f"cross-identity canary persisted for {identity}: {canary}",
                )
            )
            checks.append(
                _Judge._check(
                    not any(_contains(content, canary) for content in context_messages),
                    f"isolation:{identity}:context:{canary}",
                    f"cross-identity canary leaked into Core/Memory Metadata context for {identity}: {canary}",
                )
            )
            checks.append(
                _Judge._check(
                    not _contains(tool_output, canary),
                    f"isolation:{identity}:tool_result:{canary}",
                    f"cross-identity canary leaked through memory tool result for {identity}: {canary}",
                )
            )
        checks.extend(_cross_identity_dedupe_checks(identity, artifact))
    return checks


def _find_block(snapshot: MemorySnapshot | None, block_key: str) -> Any | None:
    if snapshot is None:
        return None
    return next((block for block in snapshot.core_blocks if block.block_key == block_key), None)


def _snapshot_text(snapshot: MemorySnapshot | None) -> str:
    if snapshot is None:
        return ""
    parts = [block.content for block in snapshot.core_blocks]
    for memory in snapshot.archival_memories:
        parts.extend([memory.topic, memory.content, *memory.tags, memory.scope_service or "", memory.scope_env or ""])
    return "\n".join(parts)


def _all_after_text(artifact: MemoryEvalArtifact) -> str:
    return "\n".join(_snapshot_text(snapshot) for snapshot in artifact.after_snapshots.values())


def _canary_values(content: str) -> list[str]:
    """Use only explicit canary tokens so normal shared vocabulary is not leakage."""
    return sorted(set(re.findall(r"[A-Za-z0-9_-]*canary[A-Za-z0-9_-]*", content, flags=re.IGNORECASE)))


def _cross_identity_dedupe_checks(
    identity: str, artifact: MemoryEvalArtifact
) -> list[JudgeCheck]:
    """Keep a deterministic hook for fixtures with same-content cross-scope data.

    Exact/near dedupe is scoped by tenant/user/agent, so a matching hash in a
    different identity must never make this identity's seeded record disappear.
    """
    before = artifact.before_snapshots.get(identity)
    after = artifact.after_snapshots.get(identity)
    if before is None or after is None:
        return []
    own_hashes = {memory.content_hash for memory in before.archival_memories}
    other_hashes = {
        memory.content_hash
        for other_identity, snapshot in artifact.before_snapshots.items()
        if other_identity != identity
        for memory in snapshot.archival_memories
    }
    shared_hashes = own_hashes & other_hashes
    if not shared_hashes:
        return [
            JudgeCheck(
                rule=f"isolation:{identity}:cross_identity_dedupe",
                status="not_applicable",
                details={"reason": "no cross-identity same-content fixture"},
            )
        ]
    after_hashes = {memory.content_hash for memory in after.archival_memories}
    return [
        _Judge._check(
            shared_hashes.issubset(after_hashes),
            f"isolation:{identity}:cross_identity_dedupe",
            f"cross-identity duplicate handling removed this identity's memory: {identity}",
            shared_hashes=sorted(shared_hashes),
        )
    ]


def _stable_dump(value: Any) -> str:
    return repr(value)


def _contains_checks(prefix: str, content: str, required: list[str] | tuple[str, ...], forbidden: list[str] | tuple[str, ...]) -> list[JudgeCheck]:
    checks = [_Judge._check(_contains(content, fact), f"{prefix}:contains:{fact}", f"required fact missing: {fact}") for fact in required]
    checks.extend(_Judge._check(not _contains(content, fact), f"{prefix}:forbidden:{fact}", f"forbidden fact present: {fact}") for fact in forbidden)
    return checks


def _all_contains(content: str, facts: list[str]) -> bool:
    return all(_contains(content, fact) for fact in facts)


def _all_any_of_contains(content: str, groups: list[list[str]]) -> bool:
    return all(any(_contains(content, fact) for fact in group) for group in groups)


def _any_of_checks(prefix: str, content: str, groups: list[list[str]]) -> list[JudgeCheck]:
    return [
        _Judge._check(
            bool(group) and any(_contains(content, fact) for fact in group),
            f"{prefix}:any_of:{index}",
            f"none of the required alternatives matched: {group}",
            alternatives=group,
        )
        for index, group in enumerate(groups, start=1)
    ]


def _contains(content: str, fact: str) -> bool:
    return _normalize(fact) in _normalize(content)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _normalize_key(key: str) -> str:
    return str(key).replace("-", "_").lower()


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key, item in value.items()] + [key for item in value.values() for key in _walk_keys(item)]
    if isinstance(value, list):
        return [key for item in value for key in _walk_keys(item)]
    return []


def _strictly_increasing(values: list[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _confidence_label(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.7:
        return "high"
    if score >= 0.5:
        return "medium"
    return None
