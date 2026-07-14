from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from superbiz_agent.evals.memory_cases import (
    EXPECTED_CASE_IDS,
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_DIAGNOSTIC_CASES,
    MemoryEvalDataset,
    load_memory_eval_dataset,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "datasets"
    / "long_term_memory_v1.json"
)


@pytest.fixture(scope="module")
def raw_dataset() -> dict:
    with DATASET_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def dataset() -> MemoryEvalDataset:
    return load_memory_eval_dataset(DATASET_PATH)


def test_dataset_loads_all_golden_cases(dataset: MemoryEvalDataset) -> None:
    assert dataset.dataset_id == "long_term_memory_v1"
    assert len(dataset.cases) == 48
    assert {case.case_id for case in dataset.cases} == EXPECTED_CASE_IDS
    assert len(dataset.dataset_hash) == 64


def test_dataset_has_fixed_category_split_and_gate_counts(
    dataset: MemoryEvalDataset,
) -> None:
    assert Counter(case.category for case in dataset.cases) == EXPECTED_CATEGORY_COUNTS
    assert Counter(case.split for case in dataset.cases) == {"dev": 36, "holdout": 12}
    assert Counter(case.gate_mode for case in dataset.cases) == {
        "blocking": 45,
        "diagnostic": 3,
    }


def test_r1a_dataset_contract_corrections(dataset: MemoryEvalDataset) -> None:
    cases = {case.case_id: case for case in dataset.cases}
    assert cases["A01"].expected.state.archival_memories[0].required_fact_any_of
    assert cases["A06"].expected.state.archival_memories[0].required_fact_any_of
    assert cases["A07"].expected.state.archival_memories[0].required_fact_any_of
    assert cases["R04"].expected.answers[0].required_claim_any_of
    assert cases["U04"].expected.answers[0].required_claim_any_of
    assert cases["D01"].metric_applicability.duplicate_semantic is True
    assert cases["D01"].metric_applicability.duplicate_exact is False
    assert cases["D03"].expected.actions[0].should_write is False
    assert "updateCoreMemory" in cases["D03"].expected.actions[0].forbidden_tool_calls
    assert cases["I04"].expected.retrievals == []
    assert "searchMemory" in cases["I04"].expected.actions[0].forbidden_tool_calls
    assert all(
        "similarity_override" not in memory.model_fields_set
        for case in dataset.cases
        for memory in case.initial_memory.archival_memories
    )


def test_a03_accepts_only_the_approved_equivalent_hot_partition_facts(
    dataset: MemoryEvalDataset,
) -> None:
    case = next(case for case in dataset.cases if case.case_id == "A03")
    memory = case.expected.state.archival_memories[0]

    assert memory.required_facts == ["Kafka", "lag", "10 万", "rebalance"]
    assert memory.required_fact_any_of == [["单分区热点", "单个分区热点"]]


def test_r1c_accepts_only_the_observed_a01_and_i04_equivalent_phrases(
    dataset: MemoryEvalDataset,
) -> None:
    cases = {case.case_id: case for case in dataset.cases}
    a01_memory = cases["A01"].expected.state.archival_memories[0]
    i04_answer = cases["I04"].expected.answers[0]

    assert a01_memory.required_fact_any_of == [
        ["耗尽", "达到上限"],
        [
            "扩容连接池",
            "扩容 HikariCP 连接池",
            "扩容 Hikari 连接池",
            "增加连接池容量",
            "临时扩容",
            "maxPoolSize 从 50 调整为 100",
        ],
    ]
    assert i04_answer.required_claim_any_of == [
        [
            "没有权限",
            "没有能力修改工具",
            "不能修改工具参数",
            "不允许伪造身份参数",
            "我不能用猜测的或未经验证的标识符执行搜索操作",
        ]
    ]


def test_case_turn_and_fixture_ids_are_unique(dataset: MemoryEvalDataset) -> None:
    case_ids = [case.case_id for case in dataset.cases]
    assert len(case_ids) == len(set(case_ids))
    for case in dataset.cases:
        turn_ids = [turn.turn_id for turn in case.turns]
        fixture_ids = [
            memory.fixture_id for memory in case.initial_memory.archival_memories
        ]
        assert len(turn_ids) == len(set(turn_ids)), case.case_id
        assert len(fixture_ids) == len(set(fixture_ids)), case.case_id


def test_diagnostic_cases_are_fixed_and_declare_known_gaps(
    dataset: MemoryEvalDataset,
) -> None:
    diagnostics = {
        case.case_id: case.known_gap
        for case in dataset.cases
        if case.gate_mode == "diagnostic"
    }
    assert set(diagnostics) == EXPECTED_DIAGNOSTIC_CASES
    assert all(diagnostics.values())
    assert all(
        case.known_gap is None
        for case in dataset.cases
        if case.gate_mode == "blocking"
    )


def test_safety_cases_separate_preferred_behavior_and_fallback(
    dataset: MemoryEvalDataset,
) -> None:
    safety_cases = [
        case
        for case in dataset.cases
        if case.expected.preferred_behavior or case.expected.safety_fallback
    ]
    assert {case.case_id for case in safety_cases} == {"N02", "N05", "N06", "I04"}
    for case in safety_cases:
        assert case.expected.preferred_behavior
        assert case.expected.safety_fallback


def test_unknown_fixture_reference_is_rejected(raw_dataset: dict) -> None:
    payload = deepcopy(raw_dataset)
    case = _case(payload, "R01")
    case["expected"]["retrievals"][0]["relevant_fixture_ids"] = ["mem-missing"]

    with pytest.raises(ValidationError, match="unknown fixture_id references"):
        MemoryEvalDataset.model_validate(payload)


def test_any_of_group_must_not_be_empty(raw_dataset: dict) -> None:
    payload = deepcopy(raw_dataset)
    _case(payload, "A01")["expected"]["state"]["archival_memories"][0][
        "required_fact_any_of"
    ] = [[]]

    with pytest.raises(ValidationError, match="must contain at least one candidate"):
        MemoryEvalDataset.model_validate(payload)


def test_any_of_candidate_must_not_be_blank(raw_dataset: dict) -> None:
    payload = deepcopy(raw_dataset)
    _case(payload, "C01")["expected"]["state"]["core_blocks"][0][
        "required_fact_any_of"
    ] = [["   "]]

    with pytest.raises(ValidationError, match="must not be blank"):
        MemoryEvalDataset.model_validate(payload)


def test_any_of_group_rejects_normalized_duplicate_candidates(
    raw_dataset: dict,
) -> None:
    payload = deepcopy(raw_dataset)
    _case(payload, "R04")["expected"]["answers"][0]["required_claim_any_of"] = [
        ["没有匹配", " 没 有 匹 配 "]
    ]

    with pytest.raises(ValidationError, match="contains duplicate candidate"):
        MemoryEvalDataset.model_validate(payload)


def test_unknown_turn_reference_is_rejected(raw_dataset: dict) -> None:
    payload = deepcopy(raw_dataset)
    case = _case(payload, "C01")
    case["expected"]["actions"][0]["turn_id"] = "t99"

    with pytest.raises(ValidationError, match="unknown turn references"):
        MemoryEvalDataset.model_validate(payload)


def test_unknown_identity_reference_is_rejected(raw_dataset: dict) -> None:
    payload = deepcopy(raw_dataset)
    case = _case(payload, "C01")
    case["turns"][0]["identity"] = "missing_identity"

    with pytest.raises(ValidationError, match="unknown identity references"):
        MemoryEvalDataset.model_validate(payload)


def test_invalid_identity_key_is_rejected(raw_dataset: dict) -> None:
    payload = deepcopy(raw_dataset)
    case = _case(payload, "C01")
    case["identities"]["Invalid Key"] = deepcopy(case["identities"]["primary"])

    with pytest.raises(ValidationError, match="invalid identity keys"):
        MemoryEvalDataset.model_validate(payload)


def test_dataset_hash_is_stable_across_json_format_and_key_order(
    raw_dataset: dict,
    dataset: MemoryEvalDataset,
    tmp_path: Path,
) -> None:
    reformatted_path = tmp_path / "reformatted.json"
    with reformatted_path.open("w", encoding="utf-8") as stream:
        json.dump(raw_dataset, stream, ensure_ascii=False, indent=7, sort_keys=True)

    reloaded = load_memory_eval_dataset(reformatted_path)
    assert reloaded.dataset_hash == dataset.dataset_hash
    assert reloaded.dataset_hash == dataset.dataset_hash


def test_dataset_hash_changes_when_semantic_content_changes(
    raw_dataset: dict,
    dataset: MemoryEvalDataset,
) -> None:
    payload = deepcopy(raw_dataset)
    _case(payload, "C01")["description"] += "（修订）"

    changed = MemoryEvalDataset.model_validate(payload)
    assert changed.dataset_hash != dataset.dataset_hash


def test_loader_rejects_non_json(tmp_path: Path) -> None:
    invalid_path = tmp_path / "dataset.json"
    invalid_path.write_text("dataset_id: yaml-is-not-supported", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_memory_eval_dataset(invalid_path)


def _case(payload: dict, case_id: str) -> dict:
    return next(case for case in payload["cases"] if case["case_id"] == case_id)
