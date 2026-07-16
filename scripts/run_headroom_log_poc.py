#!/usr/bin/env python3
"""Run the CTX-P0A Headroom compression PoC on pinned Loghub cases."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from superbiz_agent.config import Settings  # noqa: E402
from superbiz_agent.harness.token_estimator import ApproxTokenEstimator  # noqa: E402
from superbiz_agent.model_gateway.base import ModelMessage, ModelToolCall  # noqa: E402
from superbiz_agent.model_gateway.factory import build_model_gateway  # noqa: E402


HEADROOM_VERSION = "0.31.0"
COMPRESSION_THRESHOLD_TOKENS = 1_000
DEFAULT_DATASET = PROJECT_ROOT / "evals" / "datasets" / "headroom_log_poc_v1.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "evals" / "context" / "headroom_log_poc"
HARD_FACT_KINDS = {"error_code", "trace_id", "instance", "negative_fact"}
HEADROOM_TOKENIZER_BACKEND = "headroom.tokenizers.estimator.EstimatingTokenCounter"


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def configure_headroom_tokenizer(model_name: str) -> str:
    """Use Headroom's no-ML estimator for qwen without changing the model name."""
    from headroom.tokenizers import register_tokenizer
    from headroom.tokenizers.estimator import EstimatingTokenCounter

    register_tokenizer(model_name, tokenizer=EstimatingTokenCounter())
    return HEADROOM_TOKENIZER_BACKEND


def validate_dataset(dataset: dict[str, Any]) -> None:
    if dataset.get("dataset_id") != "headroom_log_poc_v1":
        raise ValueError("unexpected dataset_id")
    if dataset.get("headroom_version") != HEADROOM_VERSION:
        raise ValueError("dataset Headroom version does not match the PoC constraint")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not 8 <= len(cases) <= 12:
        raise ValueError("dataset must contain 8-12 cases")

    controls = [case for case in cases if case.get("negative_control")]
    candidates = [case for case in cases if not case.get("negative_control")]
    if len(controls) != 2:
        raise ValueError("dataset must contain exactly two below-threshold controls")
    if len(candidates) < 6:
        raise ValueError("dataset must contain at least six compression candidates")

    required_fields = {
        "source",
        "user_query",
        "tool_args",
        "raw_tool_result",
        "expected_facts",
        "size",
        "critical_event_position",
    }
    estimator = ApproxTokenEstimator()
    case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("case_id must be non-empty and unique")
        case_ids.add(case_id)
        missing = required_fields - set(case)
        if missing:
            raise ValueError(f"{case_id} is missing fields: {sorted(missing)}")

        raw_result = case["raw_tool_result"]
        raw_text = json.dumps(raw_result, ensure_ascii=False, sort_keys=True)
        estimated_tokens = estimator.estimate_text(raw_text)
        if estimated_tokens != case["size"]["estimated_tokens"]:
            raise ValueError(f"{case_id} size estimate is stale")
        if case["negative_control"] and estimated_tokens >= COMPRESSION_THRESHOLD_TOKENS:
            raise ValueError(f"{case_id} control is not below threshold")
        if not case["negative_control"] and estimated_tokens <= COMPRESSION_THRESHOLD_TOKENS:
            raise ValueError(f"{case_id} compression candidate is not above threshold")
        for fact in case["expected_facts"]:
            if fact["value"] not in raw_text:
                raise ValueError(f"{case_id} expected fact is absent from the raw tool result")
        expected_hashes = {
            item["lineNumber"]: item["sha256"] for item in case["source_message_sha256"]
        }
        for row in raw_result.get("logs", []):
            actual = hashlib.sha256(row["message"].encode("utf-8")).hexdigest()
            if actual != expected_hashes.get(row["lineNumber"]):
                raise ValueError(f"{case_id} contains a modified source message")


def build_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    tool_call_id = f"call-{case['case_id']}"
    return [
        {"role": "user", "content": case["user_query"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "queryLogs",
                        "arguments": json.dumps(
                            case["tool_args"], ensure_ascii=False, sort_keys=True
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "queryLogs",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                case["raw_tool_result"], ensure_ascii=False, sort_keys=True
            ),
        },
    ]


def to_project_messages(messages: list[dict[str, Any]]) -> list[ModelMessage]:
    converted: list[ModelMessage] = []
    for message in messages:
        tool_calls: list[ModelToolCall] = []
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            arguments = function.get("arguments", "{}")
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            tool_calls.append(
                ModelToolCall(
                    id=tool_call.get("id", ""),
                    name=function.get("name", ""),
                    arguments=parsed_arguments if isinstance(parsed_arguments, dict) else {},
                )
            )
        content = message.get("content", "")
        converted.append(
            ModelMessage(
                role=message.get("role", ""),
                content=content if isinstance(content, str) else json.dumps(content),
                name=message.get("name"),
                tool_call_id=message.get("tool_call_id"),
                tool_calls=tool_calls,
            )
        )
    return converted


def evaluate_contract(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, bool]:
    before_tool_messages = [message for message in before if message.get("role") == "tool"]
    after_tool_messages = [message for message in after if message.get("role") == "tool"]
    checks = {
        "message_count_preserved": len(before) == len(after),
        "roles_preserved": [message.get("role") for message in before]
        == [message.get("role") for message in after],
        "tool_call_ids_preserved": [message.get("tool_call_id") for message in before_tool_messages]
        == [message.get("tool_call_id") for message in after_tool_messages],
        "tool_names_preserved": [message.get("name") for message in before_tool_messages]
        == [message.get("name") for message in after_tool_messages],
        "assistant_tool_calls_preserved": _assistant_tool_calls(before)
        == _assistant_tool_calls(after),
        "tool_json_parseable": False,
        "tool_json_root_keys_preserved": False,
    }
    if len(before_tool_messages) == len(after_tool_messages) == 1:
        try:
            before_json = json.loads(before_tool_messages[0]["content"])
            after_json = json.loads(after_tool_messages[0]["content"])
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
        else:
            checks["tool_json_parseable"] = isinstance(after_json, dict)
            checks["tool_json_root_keys_preserved"] = (
                isinstance(before_json, dict)
                and isinstance(after_json, dict)
                and set(before_json).issubset(after_json)
            )
    return checks


def _assistant_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        tool_call
        for message in messages
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls", [])
    ]


def _tool_content(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        message.get("content", "")
        for message in messages
        if message.get("role") == "tool" and isinstance(message.get("content"), str)
    )


def _fact_checks(case: dict[str, Any], content: str) -> list[dict[str, Any]]:
    return [
        {
            "kind": fact["kind"],
            "value": fact["value"],
            "source_line": fact.get("source_line"),
            "preserved": fact["value"] in content,
            "hard_gate": fact["kind"] in HARD_FACT_KINDS,
        }
        for fact in case["expected_facts"]
    ]


def _new_fault_conclusions(case: dict[str, Any], before: str, after: str) -> list[str]:
    return [
        conclusion
        for conclusion in case.get("forbidden_fault_conclusions", [])
        if conclusion.casefold() not in before.casefold()
        and conclusion.casefold() in after.casefold()
    ]


def compress_case(
    case: dict[str, Any],
    *,
    model_name: str,
    compress_fn: Callable[..., Any] | None = None,
    config_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if compress_fn is None or config_factory is None:
        from headroom import CompressConfig, compress

        compress_fn = compress_fn or compress
        config_factory = config_factory or CompressConfig

    messages = build_messages(case)
    estimator = ApproxTokenEstimator()
    project_before = estimator.estimate_messages(to_project_messages(messages))
    config = config_factory(
        kompress_model="disabled",
        protect_recent=0,
        min_tokens_to_compress=COMPRESSION_THRESHOLD_TOKENS,
        compress_user_messages=False,
        compress_system_messages=False,
    )

    started = time.perf_counter()
    result = compress_fn(messages, model=model_name, config=config)
    elapsed_ms = (time.perf_counter() - started) * 1_000

    # CompressResult is intentionally consumed field-by-field. Never stringify it.
    compressed_messages = result.messages
    tokens_before = result.tokens_before
    tokens_after = result.tokens_after
    compression_ratio = result.compression_ratio
    transforms_applied = result.transforms_applied
    if not isinstance(compressed_messages, list):
        raise TypeError("CompressResult.messages must be a list")
    if not all(isinstance(message, dict) for message in compressed_messages):
        raise TypeError("CompressResult.messages must contain dictionaries")
    if not isinstance(tokens_before, int) or not isinstance(tokens_after, int):
        raise TypeError("CompressResult token metrics must be integers")
    if not isinstance(compression_ratio, (int, float)):
        raise TypeError("CompressResult.compression_ratio must be numeric")
    if not isinstance(transforms_applied, list):
        raise TypeError("CompressResult.transforms_applied must be a list")

    project_after = estimator.estimate_messages(to_project_messages(compressed_messages))
    before_tool = _tool_content(messages)
    after_tool = _tool_content(compressed_messages)
    fact_checks = _fact_checks(case, after_tool)
    baseline_fact_checks = _fact_checks(case, before_tool)
    contract_checks = evaluate_contract(messages, compressed_messages)
    new_faults = _new_fault_conclusions(case, before_tool, after_tool)
    headroom_metrics_valid = tokens_before > 0 and 0 <= tokens_after <= tokens_before
    no_project_inflation = project_after <= project_before
    control_unchanged = not case["negative_control"] or compressed_messages == messages

    failures: list[str] = []
    if not all(check["preserved"] for check in baseline_fact_checks):
        failures.append("invalid_baseline_expected_fact")
    if not all(check["preserved"] for check in fact_checks):
        failures.append("expected_fact_lost")
    if not all(contract_checks.values()):
        failures.append("message_or_json_contract_broken")
    if not headroom_metrics_valid:
        failures.append("headroom_metrics_invalid_or_inflated")
    if not no_project_inflation:
        failures.append("project_estimator_token_inflation")
    if new_faults:
        failures.append("new_fault_conclusion")
    if not control_unchanged:
        failures.append("below_threshold_control_changed")

    return {
        "case_id": case["case_id"],
        "source_name": Path(case["source"]["path"]).parts[0],
        "status": "passed" if not failures else "failed",
        "hard_gate_failures": failures,
        "negative_control": case["negative_control"],
        "critical_event_position": case["critical_event_position"],
        "dataset_estimated_tokens": case["size"]["estimated_tokens"],
        "project_estimated_tokens_before": project_before,
        "project_estimated_tokens_after": project_after,
        "headroom_tokens_before": tokens_before,
        "headroom_tokens_after": tokens_after,
        "headroom_compression_ratio": float(compression_ratio),
        "latency_ms": round(elapsed_ms, 3),
        "transforms_applied": list(transforms_applied),
        "headroom_metrics_valid_no_inflation": headroom_metrics_valid,
        "project_estimator_no_inflation": no_project_inflation,
        "control_unchanged": control_unchanged,
        "baseline_fact_checks": baseline_fact_checks,
        "fact_checks": fact_checks,
        "facts_preserved": all(check["preserved"] for check in fact_checks),
        "contract_checks": contract_checks,
        "contract_preserved": all(contract_checks.values()),
        "new_fault_conclusions": new_faults,
        "compressed_messages": compressed_messages,
    }


def _failed_case_record(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "source_name": Path(case["source"]["path"]).parts[0],
        "status": "failed",
        "hard_gate_failures": ["compression_execution_failed"],
        "failure_type": type(exc).__name__,
        "negative_control": case["negative_control"],
        "critical_event_position": case["critical_event_position"],
        "dataset_estimated_tokens": case["size"]["estimated_tokens"],
        "fact_checks": [],
        "facts_preserved": False,
        "contract_checks": {},
        "contract_preserved": False,
        "new_fault_conclusions": [],
        "compressed_messages": [],
    }


def _answer_checks(case: dict[str, Any], content: str) -> dict[str, Any]:
    fact_checks = _fact_checks(case, content)
    raw_content = json.dumps(case["raw_tool_result"], ensure_ascii=False, sort_keys=True)
    new_faults = _new_fault_conclusions(case, raw_content, content)
    return {
        "fact_checks": fact_checks,
        "all_expected_facts_present": all(check["preserved"] for check in fact_checks),
        "new_fault_conclusions": new_faults,
        "passed": all(check["preserved"] for check in fact_checks) and not new_faults,
    }


async def run_model_ab(
    dataset_cases: list[dict[str, Any]],
    compression_results: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    result_by_id = {result["case_id"]: result for result in compression_results}
    preferred = [
        case
        for case in dataset_cases
        if case.get("ab_candidate") and result_by_id.get(case["case_id"], {}).get("status") == "passed"
    ]
    fallback = [
        case
        for case in dataset_cases
        if not case.get("negative_control")
        and not case.get("ab_candidate")
        and result_by_id.get(case["case_id"], {}).get("status") == "passed"
    ]
    selected = (preferred + fallback)[:4]
    selected_ids = [case["case_id"] for case in selected]
    base = {
        "provider": settings.model_provider,
        "model": settings.model_name,
        "selected_case_ids": selected_ids,
        "attempted_calls": 0,
        "successful_calls": 0,
        "expected_calls": 8,
        "cases": [],
    }
    if len(selected) != 4:
        return {
            **base,
            "status": "pending",
            "reason": "fewer_than_four_mechanically_passing_ab_candidates",
        }
    if settings.model_provider.strip().lower() == "stub":
        return {**base, "status": "pending", "reason": "configured_model_provider_is_stub"}

    try:
        gateway = build_model_gateway(settings)
    except Exception as exc:
        return {
            **base,
            "status": "pending",
            "reason": "configured_model_gateway_unavailable",
            "failure_type": type(exc).__name__,
        }

    attempted = 0
    successful = 0
    ab_cases: list[dict[str, Any]] = []
    for case in selected:
        compression = result_by_id[case["case_id"]]
        entry: dict[str, Any] = {"case_id": case["case_id"], "status": "pending"}
        try:
            attempted += 1
            raw_response = await gateway.complete(to_project_messages(build_messages(case)))
            successful += 1
            raw_checks = _answer_checks(case, raw_response.content)

            attempted += 1
            compressed_response = await gateway.complete(
                to_project_messages(compression["compressed_messages"])
            )
            successful += 1
            compressed_checks = _answer_checks(case, compressed_response.content)
        except Exception as exc:
            entry.update(
                {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "reason": "configured_model_call_failed",
                }
            )
            ab_cases.append(entry)
            break

        baseline_eligible = raw_checks["passed"]
        entry.update(
            {
                "status": "completed",
                "user_query": case["user_query"],
                "raw": {
                    "answer": raw_response.content,
                    "usage": raw_response.usage,
                    "finish_reason": raw_response.finish_reason,
                    "checks": raw_checks,
                },
                "compressed": {
                    "answer": compressed_response.content,
                    "usage": compressed_response.usage,
                    "finish_reason": compressed_response.finish_reason,
                    "checks": compressed_checks,
                },
                "baseline_eligible_for_regression": baseline_eligible,
                "headroom_regression": baseline_eligible and not compressed_checks["passed"],
            }
        )
        ab_cases.append(entry)

    completed = len(ab_cases) == 4 and all(case["status"] == "completed" for case in ab_cases)
    return {
        **base,
        "status": "completed" if completed and successful == 8 else "failed",
        "reason": None if completed and successful == 8 else "model_ab_incomplete",
        "attempted_calls": attempted,
        "successful_calls": successful,
        "cases": ab_cases,
    }


def aggregate_results(
    case_results: list[dict[str, Any]], model_ab: dict[str, Any]
) -> dict[str, Any]:
    completed_metrics = [
        result for result in case_results if "headroom_tokens_before" in result
    ]
    candidates = [result for result in completed_metrics if not result["negative_control"]]
    fact_losses = [
        result["case_id"] for result in candidates if not result.get("facts_preserved", False)
    ]
    contract_failures = [
        result["case_id"] for result in candidates if not result.get("contract_preserved", False)
    ]
    new_faults = [
        result["case_id"] for result in candidates if result.get("new_fault_conclusions")
    ]
    execution_failures = [
        result["case_id"]
        for result in case_results
        if "compression_execution_failed" in result.get("hard_gate_failures", [])
    ]
    regressions = [
        item["case_id"]
        for item in model_ab.get("cases", [])
        if item.get("headroom_regression")
    ]
    reductions = [result["headroom_compression_ratio"] for result in candidates]
    latencies = [result["latency_ms"] for result in candidates]
    any_reduction = any(
        result["headroom_tokens_after"] < result["headroom_tokens_before"]
        for result in candidates
    )

    if fact_losses or contract_failures or new_faults or execution_failures:
        verdict = "No-Go"
        reason = "mechanical correctness or contract hard gate failed"
    elif model_ab.get("status") == "completed" and regressions:
        verdict = "No-Go"
        reason = "qwen A/B found a regression on a baseline-correct case"
    elif not any_reduction:
        verdict = "No-Go"
        reason = "no compression benefit was observed"
    elif model_ab.get("status") != "completed":
        verdict = "Conditional Go"
        reason = "mechanical gates passed but qwen A/B remains pending or failed"
    else:
        verdict = "Go"
        reason = "mechanical gates and qwen A/B passed with observed token reduction"

    source_verdicts: dict[str, str] = {}
    for source in sorted({result["source_name"] for result in case_results}):
        source_results = [result for result in case_results if result["source_name"] == source]
        source_verdicts[source] = (
            "No-Go" if any(result["status"] != "passed" for result in source_results) else "passed"
        )

    return {
        "verdict": verdict,
        "verdict_reason": reason,
        "case_count": len(case_results),
        "mechanical_pass_count": sum(result["status"] == "passed" for result in case_results),
        "mechanical_fail_count": sum(result["status"] != "passed" for result in case_results),
        "compression_candidate_count": len(candidates),
        "below_threshold_control_count": sum(
            result.get("negative_control", False) for result in case_results
        ),
        "fact_loss_cases": fact_losses,
        "contract_failure_cases": contract_failures,
        "new_fault_conclusion_cases": new_faults,
        "execution_failure_cases": execution_failures,
        "model_regression_cases": regressions,
        "headroom_tokens_before_total": sum(
            result["headroom_tokens_before"] for result in candidates
        ),
        "headroom_tokens_after_total": sum(
            result["headroom_tokens_after"] for result in candidates
        ),
        "median_headroom_compression_ratio": (
            statistics.median(reductions) if reductions else None
        ),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "model_ab_status": model_ab.get("status"),
        "model_calls_attempted": model_ab.get("attempted_calls", 0),
        "model_calls_successful": model_ab.get("successful_calls", 0),
        "source_verdicts": source_verdicts,
    }


def write_artifacts(
    *,
    output_dir: Path,
    dataset_path: Path,
    dataset: dict[str, Any],
    case_results: list[dict[str, Any]],
    model_ab: dict[str, Any],
    metrics: dict[str, Any],
    run_id: str,
    started_at: str,
    ended_at: str,
    headroom_version: str,
    headroom_tokenizer_backend: str,
    model_name: str,
    event_log: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha = _sha256_path(dataset_path)
    _write_json(output_dir / "case_results.json", {"cases": case_results})
    _write_json(output_dir / "model_ab.json", model_ab)
    _write_json(output_dir / "metrics.json", metrics)

    run_manifest = {
        "run_id": run_id,
        "experiment": "CTX-P0A Headroom log compression lightweight PoC",
        "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "baseline_commit": _git_value("rev-parse", "origin/main"),
        "head_commit_at_run": _git_value("rev-parse", "HEAD"),
        "dataset_id": dataset["dataset_id"],
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_sha,
        "headroom_version": headroom_version,
        "compression_config": {
            "kompress_model": "disabled",
            "protect_recent": 0,
            "min_tokens_to_compress": COMPRESSION_THRESHOLD_TOKENS,
            "compress_user_messages": False,
            "compress_system_messages": False,
        },
        "headroom_token_model": model_name,
        "headroom_tokenizer_backend": headroom_tokenizer_backend,
        "command": [sys.executable, *sys.argv],
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "completed",
        "model_ab_status": model_ab.get("status"),
        "model_call_count": model_ab.get("attempted_calls", 0),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_constraint": "headroom-ai==0.31.0",
        },
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    (output_dir / "metrics.md").write_text(
        _render_metrics_markdown(case_results, model_ab, metrics), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _render_summary(metrics, model_ab), encoding="utf-8"
    )
    (output_dir / "claim_validation.md").write_text(
        _render_claim_validation(metrics), encoding="utf-8"
    )
    (output_dir / "runlog.summary.md").write_text(
        "# Run Log Summary\n\n" + "\n".join(f"- {event}" for event in event_log) + "\n",
        encoding="utf-8",
    )
    (output_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\nheadroom-ai={headroom_version}\n",
        encoding="utf-8",
    )

    manifest_entries = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        manifest_entries.append(
            {"path": path.name, "sha256": _sha256_path(path), "bytes": path.stat().st_size}
        )
    _write_json(
        output_dir / "artifact_manifest.json",
        {"run_id": run_id, "files": manifest_entries},
    )


def _render_metrics_markdown(
    case_results: list[dict[str, Any]], model_ab: dict[str, Any], metrics: dict[str, Any]
) -> str:
    lines = [
        "# CTX-P0A Metrics",
        "",
        "Headroom `compression_ratio` is reported as the fraction of input tokens removed.",
        "",
        "| Case | Project est. | Headroom before | after | ratio | ms | transforms | facts | contract | status |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for result in case_results:
        transforms = ", ".join(result.get("transforms_applied", [])) or "none"
        lines.append(
            "| {case} | {project} | {before} | {after} | {ratio} | {latency} | {transforms} | {facts} | {contract} | {status} |".format(
                case=result["case_id"],
                project=result.get("project_estimated_tokens_before", "failed"),
                before=result.get("headroom_tokens_before", "failed"),
                after=result.get("headroom_tokens_after", "failed"),
                ratio=(
                    f"{result['headroom_compression_ratio']:.4f}"
                    if "headroom_compression_ratio" in result
                    else "failed"
                ),
                latency=result.get("latency_ms", "failed"),
                transforms=transforms.replace("|", "\\|"),
                facts="pass" if result.get("facts_preserved") else "fail",
                contract="pass" if result.get("contract_preserved") else "fail",
                status=result["status"],
            )
        )
    lines.extend(
        [
            "",
            f"Model A/B: `{model_ab.get('status')}`; calls attempted "
            f"{model_ab.get('attempted_calls', 0)}/8, successful "
            f"{model_ab.get('successful_calls', 0)}/8.",
            "",
            f"Decision: **{metrics['verdict']}** - {metrics['verdict_reason']}.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_summary(metrics: dict[str, Any], model_ab: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CTX-P0A Summary",
            "",
            f"Decision: **{metrics['verdict']}**.",
            "",
            metrics["verdict_reason"].capitalize() + ".",
            "",
            f"Mechanical cases: {metrics['mechanical_pass_count']}/{metrics['case_count']} passed.",
            f"Model A/B: {model_ab.get('status')} with "
            f"{model_ab.get('successful_calls', 0)} successful calls.",
            "",
        ]
    )


def _render_claim_validation(metrics: dict[str, Any]) -> str:
    reduction_supported = (
        metrics["headroom_tokens_after_total"] < metrics["headroom_tokens_before_total"]
    )
    correctness_supported = not (
        metrics["fact_loss_cases"]
        or metrics["contract_failure_cases"]
        or metrics["new_fault_conclusion_cases"]
    )
    model_verdict = (
        "supported"
        if metrics["model_ab_status"] == "completed" and not metrics["model_regression_cases"]
        else "inconclusive"
    )
    return "\n".join(
        [
            "# Claim Validation",
            "",
            "| Claim | Metric | Observed | Verdict |",
            "|---|---|---|---|",
            f"| Headroom reduces tokens | aggregate Headroom tokens | "
            f"{metrics['headroom_tokens_before_total']} -> {metrics['headroom_tokens_after_total']} | "
            f"{'supported' if reduction_supported else 'refuted'} |",
            f"| Critical facts and contracts survive | fact/contract failures | "
            f"{len(metrics['fact_loss_cases'])}/{len(metrics['contract_failure_cases'])} | "
            f"{'supported' if correctness_supported else 'refuted'} |",
            f"| qwen answers do not regress | eligible A/B regressions | "
            f"{len(metrics['model_regression_cases'])} | {model_verdict} |",
            "",
        ]
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--case-id", default=None, help="Run one case as a smoke test")
    parser.add_argument("--skip-model-ab", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = datetime.now(timezone.utc)
    run_id = args.run_id or started.strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / run_id
    event_log = [f"{started.isoformat()} run started"]

    dataset = load_dataset(args.dataset)
    cases = dataset["cases"]
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
        if not cases:
            raise ValueError(f"unknown case id: {args.case_id}")

    installed_version = importlib.metadata.version("headroom-ai")
    if installed_version != HEADROOM_VERSION:
        raise RuntimeError(
            f"headroom-ai=={HEADROOM_VERSION} is required, found {installed_version}"
        )
    settings = Settings()
    headroom_tokenizer_backend = configure_headroom_tokenizer(settings.model_name)
    event_log.append(f"validated headroom-ai=={installed_version}")
    event_log.append(
        f"registered {headroom_tokenizer_backend} for model={settings.model_name}"
    )

    case_results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        try:
            result = compress_case(case, model_name=settings.model_name)
        except Exception as exc:
            result = _failed_case_record(case, exc)
        case_results.append(result)
        print(
            json.dumps(
                {
                    "case": case["case_id"],
                    "index": index,
                    "total": len(cases),
                    "status": result["status"],
                    "before": result.get("headroom_tokens_before"),
                    "after": result.get("headroom_tokens_after"),
                },
                sort_keys=True,
            )
        )
    event_log.append(f"completed mechanical compression for {len(case_results)} cases")

    if args.skip_model_ab or args.case_id:
        model_ab = {
            "status": "pending",
            "reason": "explicitly_skipped_for_smoke_or_cli",
            "provider": settings.model_provider,
            "model": settings.model_name,
            "selected_case_ids": [],
            "expected_calls": 8,
            "attempted_calls": 0,
            "successful_calls": 0,
            "cases": [],
        }
    else:
        model_ab = asyncio.run(run_model_ab(dataset["cases"], case_results, settings))
    event_log.append(
        f"model A/B status={model_ab['status']} calls={model_ab.get('attempted_calls', 0)}"
    )

    metrics = aggregate_results(case_results, model_ab)
    ended = datetime.now(timezone.utc)
    event_log.append(f"{ended.isoformat()} run completed verdict={metrics['verdict']}")
    write_artifacts(
        output_dir=output_dir,
        dataset_path=args.dataset,
        dataset=dataset,
        case_results=case_results,
        model_ab=model_ab,
        metrics=metrics,
        run_id=run_id,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
        headroom_version=installed_version,
        headroom_tokenizer_backend=headroom_tokenizer_backend,
        model_name=settings.model_name,
        event_log=event_log,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "verdict": metrics["verdict"],
                "model_calls": model_ab.get("attempted_calls", 0),
                "artifact_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
