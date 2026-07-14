"""Execution runner and durable reports for long-term-memory evaluation.

Track A is evaluator conformance only.  Track B is the only mode that may
claim model quality, and it is skipped rather than silently falling back to
the deterministic stub when a real provider is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_artifacts import (
    MemoryEvalArtifact,
    MemoryEventArtifact,
    MemoryTurnArtifact,
    ToolCallArtifact,
    ToolResultArtifact,
    extract_retrieval_observations,
)
from superbiz_agent.evals.memory_capture import CapturingModelGateway
from superbiz_agent.evals.memory_cases import (
    InitialArchivalMemory,
    InitialCoreMemory,
    MemoryEvalCase,
    MemoryEvalDataset,
    load_memory_eval_dataset,
)
from superbiz_agent.evals.memory_fixtures import (
    ArchivalMemoryFixture,
    CoreMemoryFixture,
    MemoryFixtureSeeder,
)
from superbiz_agent.evals.memory_judges import (
    MemoryJudgeResult,
    MemoryRetrievalJudge,
    MemoryStateJudge,
    MemoryTraceJudge,
    MemoryUseJudge,
)
from superbiz_agent.evals.memory_metrics import (
    MemoryEvalJudgment,
    MemoryEvalMetrics,
    aggregate_memory_metrics,
)
from superbiz_agent.evals.memory_scripted import ScriptedModelGateway
from superbiz_agent.evals.memory_snapshots import MemorySnapshotProvider
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.events import RolloutEvent, RolloutEventType
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.memory.runtime import build_memory_runtime
from superbiz_agent.model_gateway.base import ModelGateway
from superbiz_agent.model_gateway.factory import build_model_gateway
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS
from superbiz_agent.security.redaction import redact_text, redact_value, sanitize_error_message
from superbiz_agent.tools.builtin import build_builtin_tools


EvalMode = Literal["track_a", "track_b"]
EVALUATOR_VERSION = "1.2.0"
_REDACTED_USAGE_SENTINEL = "[REDACTED]"
_REDACTABLE_USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
}
CaseRunStatus = Literal[
    "completed", "failed", "timeout", "infrastructure_failed", "skipped"
]
ReportStatus = Literal["completed", "partial", "budget_exhausted", "skipped"]
ServiceFactory = Callable[[Settings, ModelGateway], AgentHarnessService]
ScriptedGatewayFactory = Callable[[MemoryEvalCase, int], ModelGateway]


class MemoryEvalRunConfig(BaseModel):
    """Bounded, resumable execution settings for a memory evaluation run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    dataset_path: Path
    mode: EvalMode
    case_ids: list[str] | None = None
    split: Literal["dev", "holdout"] | None = None
    repetitions: int | None = Field(default=None, ge=1)
    output_dir: Path = Path("artifacts/evals/memory")
    max_case_runs: int | None = Field(default=None, ge=1)
    max_agent_model_calls: int | None = Field(default=None, ge=1)
    max_judge_calls: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_estimated_cost: float | None = Field(default=None, ge=0)
    max_concurrency: int = Field(default=1, ge=1)
    case_timeout_seconds: float = Field(default=30.0, gt=0)
    resume_from_checkpoint: bool | Path = False

    @property
    def resolved_repetitions(self) -> int:
        return self.repetitions if self.repetitions is not None else (1 if self.mode == "track_a" else 3)


class MemoryEvalCaseResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    case_id: str
    repetition: int
    gate_mode: Literal["blocking", "diagnostic"]
    status: CaseRunStatus
    executed: bool
    passed: bool | None = None
    artifact: MemoryEvalArtifact | None = None
    trace_judge: MemoryJudgeResult | None = None
    state_judge: MemoryJudgeResult | None = None
    retrieval_judge: MemoryJudgeResult | None = None
    use_judge: MemoryJudgeResult | None = None
    error_reason: str | None = None


class MemoryEvalReport(BaseModel):
    """Redacted persisted result; quality is claimed only after real execution."""

    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    dataset_id: str
    dataset_version: str
    dataset_hash: str
    mode: EvalMode
    quality_claim: bool
    quality_evaluation_performed: bool = False
    overall_quality_gate_status: Literal["not_defined", "not_evaluated"] = (
        "not_evaluated"
    )
    evaluator_version: str = "legacy-unknown"
    report_derivation: Literal["execution", "offline_rejudge"] = "execution"
    parent_report_path: str | None = None
    execution_scope: Literal["static_catalog", "scripted_conformance", "real_model"]
    status: ReportStatus
    skip_reason: str | None = None
    requested_case_count: int
    requested_repetition_count: int
    planned_case_runs: int
    executed_case_runs: int
    skipped_case_runs: int
    model_provider: str
    model_name: str | None = None
    resolved_model_name: str | None = None
    prompt_version: str
    prompt_hash: str | None = None
    tool_schema_version: str
    tool_schema_hash: str | None = None
    tool_schema_fingerprint: str
    provider_endpoint_fingerprint: str
    embedding_provider: str = "local-deterministic"
    embedding_model: str = "local-deterministic"
    embedding_dimension: int | None = None
    semantic_retrieval_gate_eligible: bool = False
    metrics: MemoryEvalMetrics | None = None
    case_results: list[MemoryEvalCaseResult] = Field(default_factory=list)
    total_latency_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    estimated_cost_status: Literal["not_configured", "not_enforced_not_configured"] = "not_configured"
    actual_cost_status: Literal["not_configured"] = "not_configured"
    agent_model_calls: int = 0
    judge_model_calls: int = 0
    offline_rejudge_model_calls: int = 0
    infrastructure_failure_count: int = 0
    budget: dict[str, Any] = Field(default_factory=dict)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    report_path: str | None = None


class MemoryEvalRunner:
    """Run isolated multi-turn memory cases and write a redacted JSON report."""

    def __init__(
        self,
        settings: Settings,
        *,
        service_factory: ServiceFactory | None = None,
        scripted_gateway_factory: ScriptedGatewayFactory | None = None,
        trace_judge: MemoryTraceJudge | None = None,
        state_judge: MemoryStateJudge | None = None,
        retrieval_judge: MemoryRetrievalJudge | None = None,
        use_judge: MemoryUseJudge | None = None,
    ) -> None:
        self.settings = settings
        self.service_factory = service_factory or _default_service_factory
        self.scripted_gateway_factory = scripted_gateway_factory
        self.trace_judge = trace_judge or MemoryTraceJudge()
        self.state_judge = state_judge or MemoryStateJudge()
        self.retrieval_judge = retrieval_judge or MemoryRetrievalJudge()
        self.use_judge = use_judge or MemoryUseJudge()

    def run(self, config: MemoryEvalRunConfig) -> MemoryEvalReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(config))
        raise RuntimeError("MemoryEvalRunner.run cannot run inside an active event loop; use run_async.")

    async def run_async(self, config: MemoryEvalRunConfig) -> MemoryEvalReport:
        if config.max_concurrency != 1:
            raise ValueError("MemoryEvalRunner currently supports max_concurrency=1 only")
        dataset = load_memory_eval_dataset(config.dataset_path)
        cases = _select_cases(dataset, config)
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        checkpoint_path = _checkpoint_path(config, dataset)
        identity = _checkpoint_identity(dataset, config, self.settings)
        previous = _load_checkpoint(config.resume_from_checkpoint, checkpoint_path, identity)
        completed = {
            item_key: MemoryEvalCaseResult.model_validate(payload)
            for item_key, payload in previous.get("completed", {}).items()
        }

        if config.mode == "track_a" and self.scripted_gateway_factory is None:
            report = self._build_report(
                dataset=dataset,
                config=config,
                cases=cases,
                results=[],
                status="completed",
                skip_reason="track_a_static_catalog_only_no_conformance_script",
                started_at=started_at,
                duration_ms=int((perf_counter() - started) * 1000),
                checkpoint_path=checkpoint_path,
                identity=identity,
                resumed=bool(previous),
                planned_case_runs=0,
            )
            return self._write_report(config, report)
        if config.mode == "track_a" and config.case_ids is None:
            raise ValueError(
                "track_a conformance requires explicit case_ids; it must not script the full catalog"
            )

        if config.mode == "track_b" and not _track_b_available(self.settings):
            report = self._build_report(
                dataset=dataset,
                config=config,
                cases=cases,
                results=[],
                status="skipped",
                skip_reason="track_b_requires_non_stub_provider_and_model_api_key",
                started_at=started_at,
                duration_ms=int((perf_counter() - started) * 1000),
                checkpoint_path=checkpoint_path,
                identity=identity,
                resumed=bool(previous),
            )
            return self._write_report(config, report)

        results: list[MemoryEvalCaseResult] = list(completed.values())
        budget_exhausted = False
        for case in cases:
            for repetition in range(1, config.resolved_repetitions + 1):
                key = _case_run_key(case, repetition)
                if key in completed:
                    continue
                if _budget_reached(config, results):
                    budget_exhausted = True
                    break
                result = await self._run_case_with_timeout(case, repetition, config)
                results.append(result)
                completed[key] = result
                _write_checkpoint(checkpoint_path, identity, completed)
                if _budget_reached(config, results):
                    budget_exhausted = True
                    break
            if budget_exhausted:
                break

        planned = len(cases) * config.resolved_repetitions
        status: ReportStatus
        if budget_exhausted and len(results) < planned:
            status = "budget_exhausted"
        elif len(results) < planned:
            status = "partial"
        else:
            status = "completed"
        report = self._build_report(
            dataset=dataset,
            config=config,
            cases=cases,
            results=results,
            status=status,
            skip_reason=None,
            started_at=started_at,
            duration_ms=int((perf_counter() - started) * 1000),
            checkpoint_path=checkpoint_path,
            identity=identity,
            resumed=bool(previous),
        )
        return self._write_report(config, report)

    async def _run_case_with_timeout(
        self,
        case: MemoryEvalCase,
        repetition: int,
        config: MemoryEvalRunConfig,
    ) -> MemoryEvalCaseResult:
        try:
            return await asyncio.wait_for(
                self._run_case(case, repetition, config),
                timeout=config.case_timeout_seconds,
            )
        except TimeoutError:
            return MemoryEvalCaseResult(
                case_id=case.case_id,
                repetition=repetition,
                gate_mode=case.gate_mode,
                status="timeout",
                executed=True,
                passed=False,
                error_reason="case_timeout",
            )
        except Exception as exc:  # Keep a provider/infrastructure failure isolated to one case.
            return MemoryEvalCaseResult(
                case_id=case.case_id,
                repetition=repetition,
                gate_mode=case.gate_mode,
                status="infrastructure_failed",
                executed=True,
                passed=False,
                error_reason=sanitize_error_message(str(exc)),
            )

    async def _run_case(
        self,
        case: MemoryEvalCase,
        repetition: int,
        config: MemoryEvalRunConfig,
    ) -> MemoryEvalCaseResult:
        gateway = self._gateway_for(case, repetition, config.mode)
        capturing_gateway = CapturingModelGateway(gateway)
        service = self.service_factory(self.settings, capturing_gateway)
        try:
            return await self._execute_case(case, repetition, service, capturing_gateway)
        finally:
            await service.aclose()

    async def _execute_case(
        self,
        case: MemoryEvalCase,
        repetition: int,
        service: AgentHarnessService,
        capturing_gateway: CapturingModelGateway,
    ) -> MemoryEvalCaseResult:
        if service.memory_runtime is None:
            raise RuntimeError("memory evaluation requires an enabled memory runtime")
        seeder = MemoryFixtureSeeder(service.memory_runtime)
        snapshots = MemorySnapshotProvider(service.memory_runtime.inspection_repository)
        fixture_ids = await self._seed_case(case, seeder)
        before = {}
        for key, identity in case.identities.items():
            before[key] = await snapshots.capture(
                identity.tenant_id,
                identity.user_id,
                identity.agent_id,
            )
        artifact = MemoryEvalArtifact(
            case_id=case.case_id,
            repetition=repetition,
            model_provider=self.settings.model_provider,
            model_name=self.settings.model_name,
            prompt_version=self.settings.prompt_version,
            prompt_hash=_file_hash(self.settings.resolve_prompt_dir() / f"{self.settings.prompt_version}.md"),
            tool_schema_version=self.settings.tool_schema_version,
            embedding_dimension=service.memory_runtime.embedding_service.dimension,
            before_snapshots=before,
            fixture_memory_ids=fixture_ids,
        )
        all_usage: defaultdict[str, int] = defaultdict(int)
        case_started = perf_counter()
        for turn in case.turns:
            identity = case.identities[turn.identity]
            context = AgentRequestContext(
                tenant_id=identity.tenant_id,
                user_id=identity.user_id,
                agent_id=identity.agent_id,
                session_id=turn.session_id,
                permissions=LOCAL_DEFAULT_PERMISSIONS,
                auth_mode="dev_headers",
            )
            turn_started = perf_counter()
            chat_result = await service.chat(context, turn.user_input)
            events = await service.trace_store.list_by_run(
                identity.tenant_id, chat_result.run_id or ""
            ) if chat_result.run_id else []
            calls = capturing_gateway.drain()
            turn_artifact = _turn_artifact(
                turn_id=turn.turn_id,
                identity=turn.identity,
                session_id=turn.session_id,
                run_id=chat_result.run_id,
                user_input=turn.user_input,
                final_answer=chat_result.answer or "",
                events=events,
                model_calls=calls,
                latency_ms=int((perf_counter() - turn_started) * 1000),
                error=chat_result.error_message,
            )
            artifact.turn_artifacts.append(turn_artifact)
            artifact.retrieval_observations.extend(
                extract_retrieval_observations(turn_artifact, fixture_memory_ids=fixture_ids)
            )
            for name, count in turn_artifact.token_usage.items():
                all_usage[name] += count
        artifact.after_snapshots = {}
        for key, identity in case.identities.items():
            artifact.after_snapshots[key] = await snapshots.capture(
                identity.tenant_id,
                identity.user_id,
                identity.agent_id,
            )
        artifact.total_latency_ms = int((perf_counter() - case_started) * 1000)
        artifact.token_usage = dict(all_usage)
        artifact.tool_schema_hash = _tool_schema_hash(artifact)
        artifact.resolved_model_name = _resolved_model_name(artifact)
        artifact.embedding_provider = "local-deterministic"
        artifact.embedding_model = "local-deterministic"
        artifact.semantic_retrieval_gate_eligible = False

        trace = self.trace_judge.judge(case, artifact)
        state = self.state_judge.judge(case, artifact)
        retrieval = self.retrieval_judge.judge(case, artifact)
        use = self.use_judge.judge(case, artifact)
        judgment = MemoryEvalJudgment(
            case=case,
            trace=trace,
            state=state,
            retrieval=retrieval,
            use=use,
            semantic_retrieval_gate_eligible=False,
            retrieval_ranking_observed=bool(artifact.retrieval_observations),
            retrieval_ranking_passed=None,
        )
        return MemoryEvalCaseResult(
            case_id=case.case_id,
            repetition=repetition,
            gate_mode=case.gate_mode,
            status="completed",
            executed=True,
            passed=judgment.passed,
            artifact=artifact,
            trace_judge=trace,
            state_judge=state,
            retrieval_judge=retrieval,
            use_judge=use,
        )

    def _gateway_for(
        self, case: MemoryEvalCase, repetition: int, mode: EvalMode
    ) -> ModelGateway:
        if mode == "track_a":
            if self.scripted_gateway_factory is not None:
                return self.scripted_gateway_factory(case, repetition)
            return ScriptedModelGateway(())
        return build_model_gateway(self.settings)

    @staticmethod
    async def _seed_case(
        case: MemoryEvalCase,
        seeder: MemoryFixtureSeeder,
    ) -> dict[str, str]:
        core_by_identity: dict[str, dict[str, CoreMemoryFixture]] = defaultdict(dict)
        for block in case.initial_memory.core_blocks:
            core_by_identity[block.identity][block.block_key] = _core_fixture(block)
        archival_by_identity: dict[str, list[ArchivalMemoryFixture]] = defaultdict(list)
        for memory in case.initial_memory.archival_memories:
            archival_by_identity[memory.identity].append(_archival_fixture(memory))
        fixture_ids: dict[str, str] = {}
        for key, identity in case.identities.items():
            seeded = await seeder.seed(
                identity.tenant_id,
                identity.user_id,
                identity.agent_id,
                core_blocks=core_by_identity[key],
                archival_memories=archival_by_identity[key],
            )
            fixture_ids.update(seeded.fixture_memory_ids)
        return fixture_ids

    def _build_report(
        self,
        *,
        dataset: MemoryEvalDataset,
        config: MemoryEvalRunConfig,
        cases: list[MemoryEvalCase],
        results: list[MemoryEvalCaseResult],
        status: ReportStatus,
        skip_reason: str | None,
        started_at: datetime,
        duration_ms: int,
        checkpoint_path: Path,
        identity: str,
        resumed: bool,
        planned_case_runs: int | None = None,
    ) -> MemoryEvalReport:
        judgments = [_judgment_from_result(result, cases) for result in results]
        judgments = [item for item in judgments if item is not None]
        total_usage: defaultdict[str, int] = defaultdict(int)
        for result in results:
            if result.artifact is not None:
                for name, count in result.artifact.token_usage.items():
                    total_usage[name] += count
        planned = (
            len(cases) * config.resolved_repetitions
            if planned_case_runs is None
            else planned_case_runs
        )
        model_calls = sum(
            len(turn.model_call_artifacts)
            for result in results
            if result.artifact is not None
            for turn in result.artifact.turn_artifacts
        )
        prompt_hash = _required_file_hash(
            self.settings.resolve_prompt_dir() / f"{self.settings.prompt_version}.md"
        )
        tool_schema_fingerprint = _configured_tool_schema_fingerprint(self.settings)
        quality_claim = config.mode == "track_b" and any(
            item.status == "completed" and item.executed for item in results
        )
        # No price table or provider billing callback is configured in 10G.2A.
        # A caller may declare a ceiling, but it cannot be enforced truthfully.
        cost_status: Literal["not_configured", "not_enforced_not_configured"] = (
            "not_enforced_not_configured"
            if config.max_estimated_cost is not None
            else "not_configured"
        )
        return MemoryEvalReport(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            dataset_hash=dataset.dataset_hash,
            mode=config.mode,
            quality_claim=quality_claim,
            quality_evaluation_performed=quality_claim,
            overall_quality_gate_status=(
                "not_defined" if quality_claim else "not_evaluated"
            ),
            evaluator_version=EVALUATOR_VERSION,
            execution_scope=(
                "real_model"
                if config.mode == "track_b"
                else "scripted_conformance"
                if self.scripted_gateway_factory is not None
                else "static_catalog"
            ),
            status=status,
            skip_reason=skip_reason,
            requested_case_count=len(cases),
            requested_repetition_count=config.resolved_repetitions,
            planned_case_runs=planned,
            executed_case_runs=sum(result.executed for result in results),
            skipped_case_runs=max(0, planned - len(results)),
            model_provider=self.settings.model_provider,
            model_name=self.settings.model_name,
            resolved_model_name=next(
                (result.artifact.resolved_model_name for result in results if result.artifact and result.artifact.resolved_model_name),
                None,
            ),
            prompt_version=self.settings.prompt_version,
            prompt_hash=prompt_hash,
            tool_schema_version=self.settings.tool_schema_version,
            tool_schema_hash=next(
                (result.artifact.tool_schema_hash for result in results if result.artifact),
                None,
            ),
            tool_schema_fingerprint=tool_schema_fingerprint,
            provider_endpoint_fingerprint=_provider_endpoint_fingerprint(self.settings),
            embedding_dimension=next(
                (result.artifact.embedding_dimension for result in results if result.artifact),
                self.settings.memory_embedding_dimension,
            ),
            metrics=aggregate_memory_metrics(judgments),
            case_results=results,
            total_latency_ms=sum(
                result.artifact.total_latency_ms or 0 for result in results if result.artifact
            ),
            token_usage=dict(total_usage),
            estimated_cost_status=cost_status,
            agent_model_calls=model_calls,
            infrastructure_failure_count=sum(
                item.status == "infrastructure_failed" for item in results
            ),
            budget={
                "max_case_runs": config.max_case_runs,
                "max_agent_model_calls": config.max_agent_model_calls,
                "max_judge_calls": config.max_judge_calls,
                "max_total_tokens": config.max_total_tokens,
                "max_estimated_cost": config.max_estimated_cost,
                "estimated_cost_enforcement": cost_status,
                "estimated_cost_actual": "not_available_no_cost_estimator",
                "max_concurrency": config.max_concurrency,
                "case_timeout_seconds": config.case_timeout_seconds,
                "actual_judge_model_calls": 0,
                "judge_call_enforcement": "not_applicable_no_semantic_judge",
            },
            checkpoint={
                "path": str(checkpoint_path),
                "identity": identity,
                "resumed": resumed,
                "completed_case_runs": len(results),
            },
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
        )

    @staticmethod
    def _write_report(config: MemoryEvalRunConfig, report: MemoryEvalReport) -> MemoryEvalReport:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report.started_at.strftime("%Y%m%dT%H%M%SZ")
        path = config.output_dir / f"{timestamp}-{report.dataset_version}-{report.mode}.json"
        completed_report = report.model_copy(update={"report_path": str(path)})
        safe_payload = _redact_report_payload(completed_report.model_dump(mode="json"))
        path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return completed_report


def rejudge_memory_eval_report(
    report_path: str | Path,
    dataset_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    trace_judge: MemoryTraceJudge | None = None,
    state_judge: MemoryStateJudge | None = None,
    retrieval_judge: MemoryRetrievalJudge | None = None,
    use_judge: MemoryUseJudge | None = None,
) -> MemoryEvalReport:
    """Rejudge persisted artifacts without constructing or calling a model gateway."""
    source_path = Path(report_path).resolve()
    source_bytes = source_path.read_bytes()
    parent = MemoryEvalReport.model_validate(
        _prepare_persisted_report_for_rejudge(json.loads(source_bytes))
    )
    dataset = load_memory_eval_dataset(dataset_path)
    cases = {case.case_id: case for case in dataset.cases}
    missing = sorted({result.case_id for result in parent.case_results} - set(cases))
    if missing:
        raise ValueError(f"report contains case IDs absent from current dataset: {missing}")

    trace = trace_judge or MemoryTraceJudge()
    state = state_judge or MemoryStateJudge()
    retrieval = retrieval_judge or MemoryRetrievalJudge()
    use = use_judge or MemoryUseJudge()
    derived_results: list[MemoryEvalCaseResult] = []
    judgments: list[MemoryEvalJudgment] = []
    for source_result in parent.case_results:
        result = source_result.model_copy(deep=True)
        artifact = result.artifact
        if artifact is not None:
            _resolve_legacy_tool_call_execution_status(artifact)
            case = cases[result.case_id]
            trace_result = trace.judge(case, artifact)
            state_result = state.judge(case, artifact)
            retrieval_result = retrieval.judge(case, artifact)
            use_result = use.judge(case, artifact)
            production_status = retrieval_result.details.get(
                "production_ranking_status"
            )
            judgment = MemoryEvalJudgment(
                case=case,
                executed=result.executed,
                trace=trace_result,
                state=state_result,
                retrieval=retrieval_result,
                use=use_result,
                semantic_retrieval_gate_eligible=artifact.semantic_retrieval_gate_eligible,
                retrieval_ranking_observed=bool(artifact.retrieval_observations),
                retrieval_ranking_passed=(
                    production_status == "passed"
                    if artifact.semantic_retrieval_gate_eligible
                    else None
                ),
            )
            judgments.append(judgment)
            result = result.model_copy(
                update={
                    "passed": judgment.passed,
                    "trace_judge": trace_result,
                    "state_judge": state_result,
                    "retrieval_judge": retrieval_result,
                    "use_judge": use_result,
                }
            )
        derived_results.append(result)

    quality_performed = parent.mode == "track_b" and any(
        result.status == "completed" and result.executed
        for result in derived_results
    )
    derived = parent.model_copy(
        update={
            "dataset_id": dataset.dataset_id,
            "dataset_version": dataset.version,
            "dataset_hash": dataset.dataset_hash,
            "quality_claim": quality_performed,
            "quality_evaluation_performed": quality_performed,
            "overall_quality_gate_status": (
                "not_defined" if quality_performed else "not_evaluated"
            ),
            "evaluator_version": EVALUATOR_VERSION,
            "report_derivation": "offline_rejudge",
            "parent_report_path": str(source_path),
            "metrics": aggregate_memory_metrics(judgments),
            "case_results": derived_results,
            "judge_model_calls": 0,
            "offline_rejudge_model_calls": 0,
            "report_path": None,
        }
    )
    destination_dir = Path(output_dir) if output_dir is not None else source_path.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = destination_dir / f"{source_path.stem}-rejudged-{timestamp}.json"
    completed = derived.model_copy(update={"report_path": str(destination)})
    safe_payload = _redact_report_payload(completed.model_dump(mode="json"))
    destination.write_text(
        json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if source_path.read_bytes() != source_bytes:
        raise RuntimeError("source report changed during offline rejudge")
    return completed


def _resolve_legacy_tool_call_execution_status(artifact: MemoryEvalArtifact) -> None:
    for turn in artifact.turn_artifacts:
        executed = {
            (result.tool_call_id, result.tool_name) for result in turn.tool_results
        }
        for call in turn.tool_calls:
            if (
                call.execution_status == "unknown"
                and (call.tool_call_id, call.tool_name) in executed
            ):
                call.execution_status = "executed"


def _prepare_persisted_report_for_rejudge(payload: Any) -> Any:
    """Remove only known token counters destroyed by persisted-report redaction."""
    if not isinstance(payload, dict):
        return payload
    case_results = payload.get("case_results")
    if not isinstance(case_results, list):
        return payload
    for case_result in case_results:
        if not isinstance(case_result, dict):
            continue
        artifact = case_result.get("artifact")
        if not isinstance(artifact, dict):
            continue
        turns = artifact.get("turn_artifacts")
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            model_calls = turn.get("model_call_artifacts")
            if not isinstance(model_calls, list):
                continue
            for model_call in model_calls:
                if not isinstance(model_call, dict):
                    continue
                usage = model_call.get("usage")
                if not isinstance(usage, dict):
                    continue
                for field, value in list(usage.items()):
                    if value != _REDACTED_USAGE_SENTINEL:
                        continue
                    if field not in _REDACTABLE_USAGE_FIELDS:
                        raise ValueError(
                            "unexpected redacted model usage field in persisted report: "
                            f"{field}"
                        )
                    del usage[field]
    return payload


def _default_service_factory(settings: Settings, gateway: ModelGateway) -> AgentHarnessService:
    trace_store = InMemoryRolloutEventStore()
    memory_runtime = build_memory_runtime(settings, trace_store=trace_store)
    return AgentHarnessService.build_default(
        settings,
        model_gateway=gateway,
        trace_store=trace_store,
        memory_runtime=memory_runtime,
    )


def _select_cases(dataset: MemoryEvalDataset, config: MemoryEvalRunConfig) -> list[MemoryEvalCase]:
    selected = dataset.cases
    if config.case_ids is not None:
        requested = set(config.case_ids)
        found = {case.case_id for case in selected}
        unknown = sorted(requested - found)
        if unknown:
            raise ValueError(f"unknown memory eval case IDs: {unknown}")
        selected = [case for case in selected if case.case_id in requested]
    if config.split is not None:
        selected = [case for case in selected if case.split == config.split]
    if not selected:
        raise ValueError("memory eval selection is empty")
    return selected


def _track_b_available(settings: Settings) -> bool:
    return settings.model_provider.strip().lower() != "stub" and bool(
        settings.model_api_key and settings.model_api_key.strip()
    )


def _core_fixture(block: InitialCoreMemory) -> CoreMemoryFixture:
    return CoreMemoryFixture(content=block.content, version=block.version)


def _archival_fixture(memory: InitialArchivalMemory) -> ArchivalMemoryFixture:
    return ArchivalMemoryFixture(
        fixture_id=memory.fixture_id,
        topic=memory.topic,
        content=memory.content,
        type=memory.type,
        tags=tuple(memory.tags),
        scope_service=memory.scope_service,
        scope_env=memory.scope_env,
    )


def _turn_artifact(
    *,
    turn_id: str,
    identity: str,
    session_id: str,
    run_id: str | None,
    user_input: str,
    final_answer: str,
    events: Iterable[RolloutEvent],
    model_calls: list[Any],
    latency_ms: int,
    error: str | None,
) -> MemoryTurnArtifact:
    tool_calls: list[ToolCallArtifact] = []
    tool_results: list[ToolResultArtifact] = []
    memory_events: list[MemoryEventArtifact] = []
    usage: defaultdict[str, int] = defaultdict(int)
    errors: list[str] = []
    memory_event_types = {
        RolloutEventType.MEMORY_WRITTEN,
        RolloutEventType.MEMORY_DUPLICATE_SKIPPED,
        RolloutEventType.MEMORY_WRITE_FAILED,
        RolloutEventType.MEMORY_WRITE_REJECTED,
        RolloutEventType.MEMORY_SEARCHED,
        RolloutEventType.MEMORY_INJECTED,
        RolloutEventType.CORE_MEMORY_UPDATED,
        RolloutEventType.CORE_MEMORY_UNCHANGED,
        RolloutEventType.CORE_MEMORY_UPDATE_REJECTED,
        RolloutEventType.ARCHIVAL_MEMORY_WRITTEN,
        RolloutEventType.ARCHIVAL_MEMORY_DUPLICATE_SKIPPED,
        RolloutEventType.ARCHIVAL_MEMORY_WRITE_FAILED,
    }
    for event in events:
        payload = redact_value(event.payload)
        tool_name = payload.get("toolName") if isinstance(payload.get("toolName"), str) else None
        if event.event_type == RolloutEventType.TOOL_CALL_STARTED and tool_name:
            arguments = payload.get("arguments")
            tool_calls.append(
                ToolCallArtifact(
                    tool_call_id=event.tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    execution_status="executed",
                )
            )
        if event.event_type in {
            RolloutEventType.TOOL_CALL_COMPLETED,
            RolloutEventType.TOOL_CALL_FAILED,
            RolloutEventType.TOOL_CALL_BLOCKED,
        } and tool_name:
            result = payload.get("result")
            tool_results.append(
                ToolResultArtifact(
                    tool_call_id=event.tool_call_id,
                    tool_name=tool_name,
                    result=result if isinstance(result, dict) else {},
                    status=payload.get("status") if isinstance(payload.get("status"), str) else None,
                )
            )
        if event.event_type in memory_event_types:
            memory_events.append(
                MemoryEventArtifact(
                    event_type=event.event_type.value,
                    sequence=event.sequence,
                    tool_call_id=event.tool_call_id,
                    payload=payload,
                )
            )
        if event.event_type == RolloutEventType.TOKEN_USAGE:
            event_usage = payload.get("usage")
            if isinstance(event_usage, dict):
                for name, count in event_usage.items():
                    if isinstance(name, str) and isinstance(count, int):
                        usage[name] += count
        if event.event_type in {RolloutEventType.MODEL_CALL_FAILED, RolloutEventType.RUN_FAILED}:
            message = payload.get("errorMessage")
            if isinstance(message, str):
                errors.append(sanitize_error_message(message))
    observed_tool_calls = {
        (tool_call.tool_call_id, tool_call.tool_name) for tool_call in tool_calls
    }
    for model_call in model_calls:
        for captured_call in model_call.tool_calls:
            tool_call_id = captured_call.get("id")
            tool_name = captured_call.get("name")
            arguments = captured_call.get("arguments")
            if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                continue
            key = (tool_call_id, tool_name)
            if key in observed_tool_calls:
                continue
            tool_calls.append(
                ToolCallArtifact(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    execution_status="requested",
                )
            )
            observed_tool_calls.add(key)
    if error:
        errors.append(sanitize_error_message(error))
    return MemoryTurnArtifact(
        turn_id=turn_id,
        identity=identity,
        session_id=session_id,
        run_id=run_id,
        user_input_summary=redact_text(user_input, replacement="[REDACTED]"),
        tool_calls=tool_calls,
        tool_results=tool_results,
        memory_events=memory_events,
        final_answer=redact_text(final_answer, replacement="[REDACTED]"),
        latency_ms=latency_ms,
        token_usage=dict(usage),
        model_call_artifacts=model_calls,
        errors=errors,
    )


def _tool_schema_hash(artifact: MemoryEvalArtifact) -> str | None:
    schemas = [
        schema.model_dump(mode="json")
        for turn in artifact.turn_artifacts
        for call in turn.model_call_artifacts
        for schema in call.tool_schemas
    ]
    if not schemas:
        return None
    payload = json.dumps(sorted(schemas, key=lambda item: item["name"]), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolved_model_name(artifact: MemoryEvalArtifact) -> str | None:
    for turn in artifact.turn_artifacts:
        for call in turn.model_call_artifacts:
            for key in ("model", "model_name", "resolved_model"):
                value = call.raw_metadata.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_file_hash(path: Path) -> str:
    digest = _file_hash(path)
    if digest is None:
        raise ValueError(f"prompt file is required for memory eval checkpointing: {path}")
    return digest


def _configured_tool_schema_fingerprint(settings: Settings) -> str:
    """Hash the active tool contract without constructing a provider client."""
    trace_store = InMemoryRolloutEventStore()
    memory_runtime = build_memory_runtime(settings, trace_store=trace_store)
    definitions = build_builtin_tools(list(memory_runtime.tools))
    payload = [
        {
            "name": definition.name,
            "description": definition.description,
            "args_schema": definition.args_model.model_json_schema(by_alias=True),
        }
        for definition in sorted(definitions, key=lambda item: item.name)
    ]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _provider_endpoint_fingerprint(settings: Settings) -> str:
    """Return a non-reversible endpoint identity with no query, credential, or key."""
    raw = (settings.model_base_url or "").strip()
    provider = settings.model_provider.strip().lower()
    if not raw:
        canonical = f"provider={provider};endpoint=provider-default"
    else:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        port = f":{parsed.port}" if parsed.port is not None else ""
        path = parsed.path.rstrip("/") or "/"
        canonical = f"provider={provider};scheme={parsed.scheme.lower()};host={host}{port};path={path}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _judgment_from_result(
    result: MemoryEvalCaseResult, cases: list[MemoryEvalCase]
) -> MemoryEvalJudgment | None:
    case = next((item for item in cases if item.case_id == result.case_id), None)
    if case is None or not result.executed:
        return None
    return MemoryEvalJudgment(
        case=case,
        executed=result.executed,
        trace=result.trace_judge,
        state=result.state_judge,
        retrieval=result.retrieval_judge,
        use=result.use_judge,
        semantic_retrieval_gate_eligible=False,
        retrieval_ranking_observed=bool(result.artifact and result.artifact.retrieval_observations),
        retrieval_ranking_passed=None,
    )


def _budget_reached(config: MemoryEvalRunConfig, results: list[MemoryEvalCaseResult]) -> bool:
    executed = sum(item.executed for item in results)
    if config.max_case_runs is not None and executed >= config.max_case_runs:
        return True
    model_calls = sum(
        len(turn.model_call_artifacts)
        for item in results
        if item.artifact is not None
        for turn in item.artifact.turn_artifacts
    )
    if config.max_agent_model_calls is not None and model_calls >= config.max_agent_model_calls:
        return True
    tokens = sum(
        count
        for item in results
        if item.artifact is not None
        for count in item.artifact.token_usage.values()
    )
    return config.max_total_tokens is not None and tokens >= config.max_total_tokens


def _case_run_key(case: MemoryEvalCase, repetition: int) -> str:
    return f"{case.case_id}:{repetition}"


def _checkpoint_identity(
    dataset: MemoryEvalDataset, config: MemoryEvalRunConfig, settings: Settings
) -> str:
    payload = {
        "dataset_hash": dataset.dataset_hash,
        "mode": config.mode,
        "case_ids": sorted(config.case_ids or []),
        "split": config.split,
        "repetitions": config.resolved_repetitions,
        "model_provider": settings.model_provider,
        "model_name": settings.model_name,
        "prompt_version": settings.prompt_version,
        "prompt_hash": _required_file_hash(
            settings.resolve_prompt_dir() / f"{settings.prompt_version}.md"
        ),
        "tool_schema_version": settings.tool_schema_version,
        "tool_schema_fingerprint": _configured_tool_schema_fingerprint(settings),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(config: MemoryEvalRunConfig, dataset: MemoryEvalDataset) -> Path:
    if isinstance(config.resume_from_checkpoint, Path):
        return config.resume_from_checkpoint
    return config.output_dir / f"checkpoint-{dataset.version}-{config.mode}.json"


def _load_checkpoint(value: bool | Path, path: Path, identity: str) -> dict[str, Any]:
    if not value:
        return {}
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != identity:
        raise ValueError("checkpoint identity does not match this dataset/mode/model configuration")
    completed = payload.get("completed")
    if not isinstance(completed, dict):
        raise ValueError("checkpoint completed entries must be an object")
    return payload


def _write_checkpoint(
    path: Path, identity: str, completed: dict[str, MemoryEvalCaseResult]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity,
        "completed": {
            key: _redact_report_payload(result.model_dump(mode="json"))
            for key, result in completed.items()
        },
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# The shared value redactor treats every key containing ``token`` as secret.
# That is correct for credentials but would corrupt typed, non-secret metric
# fields such as ``token_usage`` and ``max_tokens`` in a resumable checkpoint.
_NON_SECRET_METRIC_KEYS = {
    "token_usage",
    "max_tokens",
    "max_total_tokens",
    "max_output_tokens",
}


def _redact_report_payload(value: Any, *, parent_key: str | None = None) -> Any:
    """Preserve typed metric values while applying the shared text redaction."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in _NON_SECRET_METRIC_KEYS:
                redacted[key] = _redact_report_payload(item, parent_key=key)
            elif _is_secret_field(key):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_report_payload(item, parent_key=key)
        return redacted
    if isinstance(value, list):
        return [_redact_report_payload(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        return redact_text(value, replacement="[REDACTED]")
    return value


def _is_secret_field(key: str) -> bool:
    normalized = key.lower().replace("_", "")
    return any(
        marker in normalized
        for marker in ("apikey", "accesskey", "password", "passwd", "authorization", "cookie", "credential")
    ) or ("secret" in normalized) or ("token" in normalized and key not in _NON_SECRET_METRIC_KEYS)
