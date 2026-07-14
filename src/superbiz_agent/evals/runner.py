from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.evals.cases import EvalCase, EvalSuite, skeleton_p0_smoke_suite
from superbiz_agent.evals.judges import RuleJudge, RuleJudgeResult
from superbiz_agent.evals.traces import TraceArtifact
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS


class EvalCaseResult(BaseModel):
    """Runner output for one eval case."""

    model_config = ConfigDict(protected_namespaces=())

    case_id: str
    name: str
    passed: bool
    score: float
    artifact: TraceArtifact
    judge_result: RuleJudgeResult


class EvalReport(BaseModel):
    """Aggregated report for a deterministic eval suite run."""

    model_config = ConfigDict(protected_namespaces=())

    suite_id: str
    case_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    case_results: list[EvalCaseResult] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    duration_ms: int


class EvalRunner:
    """Local deterministic eval runner for the Skeleton P0 harness."""

    def __init__(self, *, judge: RuleJudge | None = None) -> None:
        self.judge = judge or RuleJudge()

    def run_suite(self, suite: EvalSuite | None = None) -> EvalReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_suite_async(suite))
        raise RuntimeError("EvalRunner.run_suite cannot run inside an active event loop; use await run_suite_async().")

    async def run_suite_async(self, suite: EvalSuite | None = None) -> EvalReport:
        selected_suite = suite or skeleton_p0_smoke_suite()
        started_at = datetime.now(timezone.utc)
        suite_start = perf_counter()
        case_results: list[EvalCaseResult] = []

        for case in selected_suite.cases:
            case_results.append(await self._run_case(case))

        finished_at = datetime.now(timezone.utc)
        duration_ms = int((perf_counter() - suite_start) * 1000)
        passed_count = sum(1 for result in case_results if result.passed)
        case_count = len(case_results)
        failed_count = case_count - passed_count
        pass_rate = passed_count / case_count if case_count else 1.0
        return EvalReport(
            suite_id=selected_suite.suite_id,
            case_count=case_count,
            passed_count=passed_count,
            failed_count=failed_count,
            pass_rate=pass_rate,
            case_results=case_results,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

    async def _run_case(self, case: EvalCase) -> EvalCaseResult:
        service = AgentHarnessService.placeholder()
        context = AgentRequestContext(
            tenant_id=case.tenant_id,
            user_id=case.user_id,
            agent_id=case.agent_id,
            session_id=case.session_id,
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
        prime_user_input = case.initial_context.get("prime_user_input")
        if isinstance(prime_user_input, str) and prime_user_input.strip():
            await service.chat(context, prime_user_input)
        prime_user_inputs = case.initial_context.get("prime_user_inputs")
        if isinstance(prime_user_inputs, list):
            for user_input in prime_user_inputs:
                if isinstance(user_input, str) and user_input.strip():
                    await service.chat(context, user_input)

        case_start = perf_counter()
        result = await service.chat(context, case.user_input)
        latency_ms = int((perf_counter() - case_start) * 1000)
        events = await service.trace_store.list_by_session(context)
        artifact = TraceArtifact.from_events(
            context=context,
            chat_result=result,
            events=events,
            latency_ms=latency_ms,
        )
        judge_result = self.judge.judge(case, artifact)
        return EvalCaseResult(
            case_id=case.case_id,
            name=case.name,
            passed=judge_result.passed,
            score=judge_result.score,
            artifact=artifact,
            judge_result=judge_result,
        )


def main() -> None:
    report = EvalRunner().run_suite()
    print(_format_summary(report))


def _format_summary(report: EvalReport) -> str:
    return (
        f"suite={report.suite_id} passed={report.passed_count} "
        f"failed={report.failed_count} pass_rate={report.pass_rate:.2f}"
    )


def _jsonable_report(report: EvalReport) -> dict[str, Any]:
    return report.model_dump(mode="json")


if __name__ == "__main__":
    main()
