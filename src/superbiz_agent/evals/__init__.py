"""Local deterministic eval runner package."""

from typing import Any

__all__ = [
    "EvalCase",
    "EvalCaseResult",
    "EvalReport",
    "EvalRunner",
    "EvalSuite",
    "RuleJudge",
    "RuleJudgeResult",
    "TraceArtifact",
    "skeleton_p0_smoke_suite",
]


def __getattr__(name: str) -> Any:
    if name in {"EvalCase", "EvalSuite", "skeleton_p0_smoke_suite"}:
        from superbiz_agent.evals import cases

        return getattr(cases, name)
    if name in {"RuleJudge", "RuleJudgeResult"}:
        from superbiz_agent.evals import judges

        return getattr(judges, name)
    if name in {"EvalCaseResult", "EvalReport", "EvalRunner"}:
        from superbiz_agent.evals import runner

        return getattr(runner, name)
    if name == "TraceArtifact":
        from superbiz_agent.evals import traces

        return getattr(traces, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
