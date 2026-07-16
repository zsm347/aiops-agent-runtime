from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.fetch_headroom_poc_logs import (
    CASE_SPECS,
    COMPRESSION_THRESHOLD_TOKENS,
    LOGHUB_COMMIT,
    LOG_SOURCES,
    DataIntegrityError,
    LogSource,
    fetch_source,
)
from scripts.run_headroom_log_poc import (
    DEFAULT_DATASET,
    build_messages,
    compress_case,
    configure_headroom_tokenizer,
    evaluate_contract,
    load_dataset,
    run_model_ab,
)
from superbiz_agent.config import Settings


def _dataset() -> dict:
    return load_dataset(DEFAULT_DATASET)


def _case(case_id: str) -> dict:
    return next(case for case in _dataset()["cases"] if case["case_id"] == case_id)


def test_source_manifest_is_pinned_to_five_loghub_2k_files() -> None:
    assert LOGHUB_COMMIT == "dd61d0952749ee7963bde24220d1be5ede023033"
    assert [source.path for source in LOG_SOURCES] == [
        "Apache/Apache_2k.log",
        "OpenStack/OpenStack_2k.log",
        "HDFS/HDFS_2k.log",
        "Zookeeper/Zookeeper_2k.log",
        "Linux/Linux_2k.log",
    ]
    assert all(len(source.sha256) == 64 for source in LOG_SOURCES)
    assert all(source.line_count == 2_000 for source in LOG_SOURCES)


def test_dataset_has_required_cases_thresholds_and_unmodified_messages() -> None:
    dataset = _dataset()
    cases = dataset["cases"]
    controls = [case for case in cases if case["negative_control"]]
    candidates = [case for case in cases if not case["negative_control"]]

    assert len(cases) == len(CASE_SPECS) == 10
    assert len(candidates) == 8
    assert len(controls) == 2
    assert {case["critical_event_position"] for case in candidates} == {
        "head",
        "middle",
        "tail",
    }
    assert all(
        case["size"]["estimated_tokens"] > COMPRESSION_THRESHOLD_TOKENS
        for case in candidates
    )
    assert all(
        case["size"]["estimated_tokens"] < COMPRESSION_THRESHOLD_TOKENS
        for case in controls
    )

    for case in cases:
        assert {
            "source",
            "user_query",
            "tool_args",
            "raw_tool_result",
            "expected_facts",
            "size",
            "critical_event_position",
        }.issubset(case)
        rendered = json.dumps(case["raw_tool_result"], ensure_ascii=False, sort_keys=True)
        assert all(fact["value"] in rendered for fact in case["expected_facts"])
        expected_hashes = {
            item["lineNumber"]: item["sha256"] for item in case["source_message_sha256"]
        }
        for row in case["raw_tool_result"]["logs"]:
            assert expected_hashes[row["lineNumber"]] == hashlib.sha256(
                row["message"].encode("utf-8")
            ).hexdigest()


def test_fetch_source_downloads_and_reuses_only_matching_content(tmp_path: Path) -> None:
    payload = b"line one\nline two"
    source = LogSource(
        name="tiny",
        path="Tiny/Tiny_2k.log",
        sha256=hashlib.sha256(payload).hexdigest(),
        line_count=2,
        byte_count=len(payload),
    )
    calls: list[str] = []

    def opener(url: str) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(payload)

    downloaded = fetch_source(source, tmp_path, opener=opener, attempts=1)
    reused = fetch_source(source, tmp_path, opener=opener, attempts=1)

    assert downloaded == reused
    assert downloaded.read_bytes() == payload
    assert calls == [source.url]


def test_fetch_source_rejects_corrupt_cached_content(tmp_path: Path) -> None:
    source = LogSource(
        name="tiny",
        path="Tiny/Tiny_2k.log",
        sha256=hashlib.sha256(b"expected").hexdigest(),
        line_count=1,
        byte_count=len(b"expected"),
    )
    destination = tmp_path / source.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupt")

    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        fetch_source(source, tmp_path, attempts=1)


def test_compress_case_reads_result_fields_without_stringifying() -> None:
    case = _case("hdfs_block_exception_head")
    captured_config: dict = {}

    class Config:
        def __init__(self, **kwargs: object) -> None:
            captured_config.update(kwargs)

    class Result:
        def __init__(self, messages: list[dict]) -> None:
            self.messages = messages
            self.tokens_before = 100
            self.tokens_after = 100
            self.compression_ratio = 0.0
            self.transforms_applied: list[str] = []

        def __str__(self) -> str:
            raise AssertionError("CompressResult must not be stringified")

    def fake_compress(messages: list[dict], **_: object) -> Result:
        return Result(messages)

    result = compress_case(
        case,
        model_name="qwen-plus",
        compress_fn=fake_compress,
        config_factory=Config,
    )

    assert result["status"] == "passed"
    assert captured_config["kompress_model"] == "disabled"
    assert captured_config["protect_recent"] == 0
    assert captured_config["min_tokens_to_compress"] == 1_000


def test_contract_checker_detects_tool_call_and_json_breakage() -> None:
    messages = build_messages(_case("hdfs_normal_below_threshold_control"))
    broken = json.loads(json.dumps(messages))
    broken[-1]["tool_call_id"] = "wrong-call"
    broken[-1]["content"] = "not-json"

    checks = evaluate_contract(messages, broken)

    assert checks["message_count_preserved"] is True
    assert checks["tool_call_ids_preserved"] is False
    assert checks["tool_json_parseable"] is False
    assert checks["tool_json_root_keys_preserved"] is False


def test_real_headroom_preserves_hdfs_case_without_ml_model() -> None:
    configure_headroom_tokenizer("qwen-plus")

    result = compress_case(_case("hdfs_block_exception_head"), model_name="qwen-plus")

    assert result["status"] == "passed"
    assert result["facts_preserved"] is True
    assert result["contract_preserved"] is True
    assert result["headroom_tokens_after"] < result["headroom_tokens_before"]
    assert not any("kompress" in transform.lower() for transform in result["transforms_applied"])


def test_real_headroom_exposes_openstack_trace_loss_hard_gate() -> None:
    configure_headroom_tokenizer("qwen-plus")
    case = _case("openstack_404_trace_instance_middle")

    result = compress_case(case, model_name="qwen-plus")
    lost = {check["kind"] for check in result["fact_checks"] if not check["preserved"]}

    assert result["status"] == "failed"
    assert "expected_fact_lost" in result["hard_gate_failures"]
    assert "trace_id" in lost
    assert result["contract_preserved"] is True
    assert result["headroom_tokens_after"] < result["headroom_tokens_before"]


def test_model_ab_is_pending_with_zero_calls_for_stub_settings() -> None:
    dataset = _dataset()
    passing_results = [
        {
            "case_id": case["case_id"],
            "status": "passed",
            "compressed_messages": build_messages(case),
        }
        for case in dataset["cases"]
        if not case["negative_control"]
    ]
    settings = Settings(model_provider="stub", model_name="qwen-plus")

    result = asyncio.run(run_model_ab(dataset["cases"], passing_results, settings))

    assert result["status"] == "pending"
    assert result["reason"] == "configured_model_provider_is_stub"
    assert len(result["selected_case_ids"]) == 4
    assert result["attempted_calls"] == 0
    assert result["successful_calls"] == 0
