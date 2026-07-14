from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


MemoryEvalCategory = Literal[
    "core_write",
    "archival_write",
    "no_write_security",
    "dedupe_conflict",
    "retrieval",
    "memory_use",
    "isolation",
]
MemoryEvalSplit = Literal["dev", "holdout"]
MemoryEvalGateMode = Literal["blocking", "diagnostic"]
MemoryType = Literal["experience", "knowledge"]
CoreBlockKey = Literal["user_rules", "user_ops_profile", "service_notes"]


def _validate_any_of_groups(groups: list[list[str]]) -> list[list[str]]:
    for group_index, group in enumerate(groups):
        if not group:
            raise ValueError(f"any-of group {group_index} must contain at least one candidate")
        normalized: set[str] = set()
        for candidate_index, candidate in enumerate(group):
            if not candidate.strip():
                raise ValueError(
                    f"any-of group {group_index} candidate {candidate_index} must not be blank"
                )
            key = re.sub(r"\s+", "", candidate).casefold()
            if key in normalized:
                raise ValueError(
                    f"any-of group {group_index} contains duplicate candidate: {candidate}"
                )
            normalized.add(key)
    return groups


AnyOfGroups = Annotated[list[list[str]], AfterValidator(_validate_any_of_groups)]


class MemoryEvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MemoryEvalIdentity(MemoryEvalModel):
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)


class MemoryEvalTurn(MemoryEvalModel):
    turn_id: str = Field(pattern=r"^t[1-9][0-9]*$")
    identity: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    session_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)


class InitialCoreMemory(MemoryEvalModel):
    identity: str
    block_key: CoreBlockKey
    content: str = ""
    version: int = Field(default=1, ge=1)


class InitialArchivalMemory(MemoryEvalModel):
    fixture_id: str = Field(pattern=r"^mem-[a-z0-9][a-z0-9-]*$")
    identity: str
    type: MemoryType = "experience"
    topic: str = Field(min_length=1)
    content: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    scope_service: str | None = None
    scope_env: str | None = None


class InitialMemoryState(MemoryEvalModel):
    core_blocks: list[InitialCoreMemory] = Field(default_factory=list)
    archival_memories: list[InitialArchivalMemory] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_fixtures(self) -> "InitialMemoryState":
        fixture_ids = [memory.fixture_id for memory in self.archival_memories]
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("initial archival fixture_id values must be unique within a case")
        core_keys = [(block.identity, block.block_key) for block in self.core_blocks]
        if len(core_keys) != len(set(core_keys)):
            raise ValueError("initial core block identity/block_key pairs must be unique")
        return self


class ExpectedToolArguments(MemoryEvalModel):
    tool_name: str = Field(min_length=1)
    required_keys: list[str] = Field(default_factory=list)
    forbidden_keys: list[str] = Field(default_factory=list)
    exact_values: dict[str, Any] = Field(default_factory=dict)
    contains_values: dict[str, list[str]] = Field(default_factory=dict)


class ExpectedMemoryAction(MemoryEvalModel):
    turn_id: str
    should_write: bool | None = None
    required_tool_calls: list[str] = Field(default_factory=list)
    forbidden_tool_calls: list[str] = Field(default_factory=list)
    required_events: list[str] = Field(default_factory=list)
    forbidden_events: list[str] = Field(default_factory=list)
    tool_arguments: list[ExpectedToolArguments] = Field(default_factory=list)
    duplicate_of_fixture_id: str | None = None


class ExpectedCoreBlock(MemoryEvalModel):
    identity: str
    block_key: CoreBlockKey
    required_facts: list[str] = Field(default_factory=list)
    required_fact_any_of: AnyOfGroups = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    expected_version_delta: int | None = Field(default=None, ge=0)


class ExpectedArchivalMemory(MemoryEvalModel):
    identity: str
    required_facts: list[str] = Field(default_factory=list)
    required_fact_any_of: AnyOfGroups = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    type: MemoryType | None = None
    topic_contains: list[str] = Field(default_factory=list)
    tags_include: list[str] = Field(default_factory=list)
    tags_exclude: list[str] = Field(default_factory=list)
    scope_service: str | None = None
    scope_env: str | None = None
    expected_count_delta: int | None = None
    duplicate_of_fixture_id: str | None = None


class ForbiddenPersistedFact(MemoryEvalModel):
    identity: str
    facts: list[str] = Field(min_length=1)


class ExpectedLaterTurnMemory(MemoryEvalModel):
    turn_id: str
    identity: str
    required_facts: list[str] = Field(min_length=1)


class ExpectedMemoryState(MemoryEvalModel):
    core_blocks: list[ExpectedCoreBlock] = Field(default_factory=list)
    archival_memories: list[ExpectedArchivalMemory] = Field(default_factory=list)
    forbidden_persisted_facts: list[ForbiddenPersistedFact] = Field(default_factory=list)
    required_in_later_turn: list[ExpectedLaterTurnMemory] = Field(default_factory=list)


class ExpectedRetrieval(MemoryEvalModel):
    turn_id: str
    should_search: bool
    relevant_fixture_ids: list[str] = Field(default_factory=list)
    forbidden_fixture_ids: list[str] = Field(default_factory=list)
    required_query_terms: list[str] = Field(default_factory=list)
    expected_empty: bool = False
    expected_scope_service: str | None = None
    expected_scope_env: str | None = None
    required_tags: list[str] = Field(default_factory=list)
    forbidden_filter_keys: list[str] = Field(default_factory=list)


class ExpectedAnswerBehavior(MemoryEvalModel):
    turn_id: str
    required_claims: list[str] = Field(default_factory=list)
    required_claim_any_of: AnyOfGroups = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    required_qualifiers: list[str] = Field(default_factory=list)
    semantic_required: bool = False


class PreferredBehavior(MemoryEvalModel):
    turn_id: str
    forbidden_tool_calls: list[str] = Field(min_length=1)


class SafetyFallback(MemoryEvalModel):
    turn_id: str
    allowed_rejection_events: list[str] = Field(default_factory=list)
    forbidden_persisted_facts: list[str] = Field(min_length=1)


class ExpectedMemoryBehavior(MemoryEvalModel):
    actions: list[ExpectedMemoryAction] = Field(default_factory=list)
    state: ExpectedMemoryState = Field(default_factory=ExpectedMemoryState)
    retrievals: list[ExpectedRetrieval] = Field(default_factory=list)
    answers: list[ExpectedAnswerBehavior] = Field(default_factory=list)
    preferred_behavior: list[PreferredBehavior] = Field(default_factory=list)
    safety_fallback: list[SafetyFallback] = Field(default_factory=list)


class MetricApplicability(MemoryEvalModel):
    write_trigger: bool = False
    route: bool = False
    state_quality: bool = False
    duplicate_exact: bool = False
    duplicate_semantic: bool = False
    search_trigger: bool = False
    retrieval_ranking: bool = False
    memory_use: bool = False
    live_evidence: bool = False
    isolation: bool = False


class MemoryEvalCase(MemoryEvalModel):
    case_id: str = Field(pattern=r"^[A-Z][0-9]{2}$")
    category: MemoryEvalCategory
    split: MemoryEvalSplit
    gate_mode: MemoryEvalGateMode = "blocking"
    known_gap: str | None = None
    description: str = Field(min_length=1)
    capability_tags: list[str] = Field(min_length=1)
    metric_applicability: MetricApplicability
    identities: dict[str, MemoryEvalIdentity]
    initial_memory: InitialMemoryState = Field(default_factory=InitialMemoryState)
    turns: list[MemoryEvalTurn] = Field(min_length=1)
    expected: ExpectedMemoryBehavior

    @model_validator(mode="after")
    def validate_case_references(self) -> "MemoryEvalCase":
        if not self.identities:
            raise ValueError("identities must not be empty")
        invalid_identity_keys = [
            key
            for key in self.identities
            if re.fullmatch(r"[a-z][a-z0-9_-]*", key) is None
        ]
        if invalid_identity_keys:
            raise ValueError(f"invalid identity keys: {invalid_identity_keys}")

        turn_ids = [turn.turn_id for turn in self.turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("turn_id values must be unique within a case")
        known_turns = set(turn_ids)
        known_identities = set(self.identities)
        fixture_ids = {
            memory.fixture_id for memory in self.initial_memory.archival_memories
        }

        identity_references = [turn.identity for turn in self.turns]
        identity_references.extend(block.identity for block in self.initial_memory.core_blocks)
        identity_references.extend(
            memory.identity for memory in self.initial_memory.archival_memories
        )
        identity_references.extend(block.identity for block in self.expected.state.core_blocks)
        identity_references.extend(
            memory.identity for memory in self.expected.state.archival_memories
        )
        identity_references.extend(
            item.identity for item in self.expected.state.forbidden_persisted_facts
        )
        identity_references.extend(
            item.identity for item in self.expected.state.required_in_later_turn
        )
        unknown_identities = sorted(set(identity_references) - known_identities)
        if unknown_identities:
            raise ValueError(f"unknown identity references: {unknown_identities}")

        turn_references = [action.turn_id for action in self.expected.actions]
        turn_references.extend(item.turn_id for item in self.expected.retrievals)
        turn_references.extend(item.turn_id for item in self.expected.answers)
        turn_references.extend(item.turn_id for item in self.expected.preferred_behavior)
        turn_references.extend(item.turn_id for item in self.expected.safety_fallback)
        turn_references.extend(
            item.turn_id for item in self.expected.state.required_in_later_turn
        )
        unknown_turns = sorted(set(turn_references) - known_turns)
        if unknown_turns:
            raise ValueError(f"unknown turn references: {unknown_turns}")

        fixture_references: list[str] = []
        fixture_references.extend(
            action.duplicate_of_fixture_id
            for action in self.expected.actions
            if action.duplicate_of_fixture_id
        )
        for retrieval in self.expected.retrievals:
            fixture_references.extend(retrieval.relevant_fixture_ids)
            fixture_references.extend(retrieval.forbidden_fixture_ids)
        fixture_references.extend(
            memory.duplicate_of_fixture_id
            for memory in self.expected.state.archival_memories
            if memory.duplicate_of_fixture_id
        )
        unknown_fixtures = sorted(set(fixture_references) - fixture_ids)
        if unknown_fixtures:
            raise ValueError(f"unknown fixture_id references: {unknown_fixtures}")

        if self.gate_mode == "diagnostic" and not self.known_gap:
            raise ValueError("diagnostic cases must declare known_gap")
        if self.gate_mode == "blocking" and self.known_gap is not None:
            raise ValueError("blocking cases must not declare known_gap")
        if bool(self.expected.preferred_behavior) != bool(self.expected.safety_fallback):
            raise ValueError(
                "preferred_behavior and safety_fallback must be declared together"
            )
        return self


EXPECTED_CASE_IDS = frozenset(
    [*(f"C{i:02d}" for i in range(1, 9))]
    + [*(f"A{i:02d}" for i in range(1, 9))]
    + [*(f"N{i:02d}" for i in range(1, 9))]
    + [*(f"D{i:02d}" for i in range(1, 7))]
    + [*(f"R{i:02d}" for i in range(1, 9))]
    + [*(f"U{i:02d}" for i in range(1, 7))]
    + [*(f"I{i:02d}" for i in range(1, 5))]
)
EXPECTED_CATEGORY_COUNTS = {
    "core_write": 8,
    "archival_write": 8,
    "no_write_security": 8,
    "dedupe_conflict": 6,
    "retrieval": 8,
    "memory_use": 6,
    "isolation": 4,
}
EXPECTED_DIAGNOSTIC_CASES = frozenset({"C08", "U02", "U03"})


class MemoryEvalDataset(MemoryEvalModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = ""
    cases: list[MemoryEvalCase]

    @model_validator(mode="after")
    def validate_fixed_catalog(self) -> "MemoryEvalDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be globally unique")
        if set(case_ids) != EXPECTED_CASE_IDS:
            missing = sorted(EXPECTED_CASE_IDS - set(case_ids))
            extra = sorted(set(case_ids) - EXPECTED_CASE_IDS)
            raise ValueError(f"invalid case catalog; missing={missing}, extra={extra}")

        category_counts = {
            category: sum(case.category == category for case in self.cases)
            for category in EXPECTED_CATEGORY_COUNTS
        }
        if category_counts != EXPECTED_CATEGORY_COUNTS:
            raise ValueError(f"invalid category counts: {category_counts}")
        split_counts = {
            split: sum(case.split == split for case in self.cases)
            for split in ("dev", "holdout")
        }
        if split_counts != {"dev": 36, "holdout": 12}:
            raise ValueError(f"invalid split counts: {split_counts}")
        diagnostic_cases = {
            case.case_id for case in self.cases if case.gate_mode == "diagnostic"
        }
        if diagnostic_cases != EXPECTED_DIAGNOSTIC_CASES:
            raise ValueError(f"invalid diagnostic cases: {sorted(diagnostic_cases)}")
        return self

    @property
    def dataset_hash(self) -> str:
        return stable_dataset_hash(self)


def stable_dataset_hash(dataset: MemoryEvalDataset) -> str:
    canonical = json.dumps(
        dataset.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_memory_eval_dataset(path: str | Path) -> MemoryEvalDataset:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return MemoryEvalDataset.model_validate(payload)
