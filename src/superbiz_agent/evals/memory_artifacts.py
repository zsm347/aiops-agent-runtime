"""Eval-only artifacts for long-term-memory evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.evals.memory_snapshots import MemorySnapshot


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", protected_namespaces=(), arbitrary_types_allowed=True
    )


class ModelMessageCapture(_ArtifactModel):
    role: str
    content_summary: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_call_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolSchemaCapture(_ArtifactModel):
    name: str
    schema_hash: str
    description_summary: str = ""
    argument_keys: list[str] = Field(default_factory=list)


class ModelCallArtifact(_ArtifactModel):
    call_index: int
    mode: Literal["complete", "stream"]
    messages: list[ModelMessageCapture] = Field(default_factory=list)
    tool_schemas: list[ToolSchemaCapture] = Field(default_factory=list)
    response_content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolCallArtifact(_ArtifactModel):
    tool_call_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    execution_status: Literal["requested", "executed", "unknown"] = "unknown"


class ToolResultArtifact(_ArtifactModel):
    tool_call_id: str | None = None
    tool_name: str
    result: dict[str, Any] = Field(default_factory=dict)
    status: str | None = None


class MemoryEventArtifact(_ArtifactModel):
    event_type: str
    sequence: int | None = None
    tool_call_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RetrievedMemoryArtifact(_ArtifactModel):
    memory_id: str
    fixture_id: str | None = None
    rank: int = Field(ge=1)
    score: float | None = None
    confidence_label: str | None = None


class RetrievalObservation(_ArtifactModel):
    turn_id: str
    run_id: str | None = None
    tool_call_id: str | None = None
    query: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    results: list[RetrievedMemoryArtifact] = Field(default_factory=list)
    success: bool | None = None


class MemoryTurnArtifact(_ArtifactModel):
    turn_id: str
    identity: str
    session_id: str
    run_id: str | None = None
    user_input_summary: str = ""
    tool_calls: list[ToolCallArtifact] = Field(default_factory=list)
    tool_results: list[ToolResultArtifact] = Field(default_factory=list)
    memory_events: list[MemoryEventArtifact] = Field(default_factory=list)
    retrieved_memories: list[RetrievedMemoryArtifact] = Field(default_factory=list)
    final_answer: str = ""
    latency_ms: int | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    model_call_artifacts: list[ModelCallArtifact] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [call.tool_name for call in self.tool_calls]

    @property
    def requested_tool_names(self) -> list[str]:
        return [call.tool_name for call in self.tool_calls]

    @property
    def executed_tool_names(self) -> list[str]:
        return [
            call.tool_name
            for call in self.tool_calls
            if call.execution_status == "executed"
        ]


class MemoryEvalArtifact(_ArtifactModel):
    case_id: str
    repetition: int = Field(ge=1)
    model_provider: str | None = None
    model_name: str | None = None
    resolved_model_name: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    tool_schema_version: str | None = None
    tool_schema_hash: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    semantic_retrieval_gate_eligible: bool = False
    before_snapshots: dict[str, MemorySnapshot] = Field(default_factory=dict)
    turn_artifacts: list[MemoryTurnArtifact] = Field(default_factory=list)
    after_snapshots: dict[str, MemorySnapshot] = Field(default_factory=dict)
    retrieval_observations: list[RetrievalObservation] = Field(default_factory=list)
    fixture_memory_ids: dict[str, str] = Field(default_factory=dict)
    total_latency_ms: int | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    def turn_for(self, turn_id: str) -> MemoryTurnArtifact | None:
        return next((turn for turn in self.turn_artifacts if turn.turn_id == turn_id), None)

    def retrievals_for(self, turn_id: str) -> list[RetrievalObservation]:
        return [item for item in self.retrieval_observations if item.turn_id == turn_id]


def extract_retrieval_observations(
    turn: MemoryTurnArtifact,
    *,
    fixture_memory_ids: dict[str, str] | None = None,
) -> list[RetrievalObservation]:
    """Build observations from paired ``searchMemory`` results without guessing."""
    inverse_fixture_ids = {
        memory_id: fixture_id for fixture_id, memory_id in (fixture_memory_ids or {}).items()
    }
    calls = {
        call.tool_call_id: call
        for call in turn.tool_calls
        if call.tool_name == "searchMemory" and call.tool_call_id
    }
    observations: list[RetrievalObservation] = []
    for result in turn.tool_results:
        if result.tool_name != "searchMemory":
            continue
        call = calls.get(result.tool_call_id)
        arguments = call.arguments if call is not None else {}
        retrieved: list[RetrievedMemoryArtifact] = []
        memories = result.result.get("memories")
        if isinstance(memories, list):
            for rank, memory in enumerate(memories, start=1):
                if not isinstance(memory, dict) or not isinstance(memory.get("id"), str):
                    continue
                score = memory.get("similarity")
                retrieved.append(
                    RetrievedMemoryArtifact(
                        memory_id=memory["id"],
                        fixture_id=inverse_fixture_ids.get(memory["id"]),
                        rank=rank,
                        score=float(score) if isinstance(score, int | float) else None,
                        confidence_label=(
                            memory.get("confidenceLabel")
                            if isinstance(memory.get("confidenceLabel"), str)
                            else None
                        ),
                    )
                )
        filters = {
            key: value
            for key, value in arguments.items()
            if key in {"type", "scopeService", "scopeEnv", "tags"}
        }
        observations.append(
            RetrievalObservation(
                turn_id=turn.turn_id,
                run_id=turn.run_id,
                tool_call_id=result.tool_call_id,
                query=arguments.get("query", "") if isinstance(arguments.get("query", ""), str) else "",
                filters=filters,
                results=retrieved,
                success=(
                    result.result.get("success")
                    if isinstance(result.result.get("success"), bool)
                    else None
                ),
            )
        )
    return observations
