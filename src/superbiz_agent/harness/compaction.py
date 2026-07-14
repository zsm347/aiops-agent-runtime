from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from superbiz_agent.harness.context_budget import ContextBudget
from superbiz_agent.harness.token_estimator import TokenEstimator
from superbiz_agent.harness.tool_result_reducer import redact_text
from superbiz_agent.model_gateway.base import ModelMessage


@dataclass(frozen=True)
class CompactionResult:
    compacted: bool
    summary_message: ModelMessage | None
    recent_messages: list[ModelMessage]
    source_messages: list[ModelMessage]
    estimated_tokens_before: int
    estimated_tokens_after: int


class DeterministicHistoryCompactor:
    def compact(
        self,
        messages: list[ModelMessage],
        *,
        budget: ContextBudget,
        estimator: TokenEstimator,
    ) -> CompactionResult:
        old_messages, recent_messages = split_recent_turns(
            messages,
            turns_to_keep=budget.recent_turns_to_keep,
        )
        before_tokens = estimator.estimate_messages(messages)
        if not old_messages:
            return CompactionResult(
                compacted=False,
                summary_message=None,
                recent_messages=messages,
                source_messages=[],
                estimated_tokens_before=before_tokens,
                estimated_tokens_after=before_tokens,
            )

        summary_content = deterministic_summary(old_messages)
        summary_message = ModelMessage(
            role="system",
            content=summary_content,
            metadata={
                "context_summary": True,
                "compaction_prompt_version": budget.compaction_prompt_version,
            },
        )
        after_tokens = estimator.estimate_messages([summary_message, *recent_messages])
        return CompactionResult(
            compacted=True,
            summary_message=summary_message,
            recent_messages=recent_messages,
            source_messages=old_messages,
            estimated_tokens_before=before_tokens,
            estimated_tokens_after=after_tokens,
        )


def split_recent_turns(
    messages: list[ModelMessage],
    *,
    turns_to_keep: int,
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    if turns_to_keep <= 0:
        return messages, []
    user_indices = [index for index, message in enumerate(messages) if message.role == "user"]
    if len(user_indices) <= turns_to_keep:
        return [], messages
    recent_start = user_indices[-turns_to_keep]
    return messages[:recent_start], messages[recent_start:]


def deterministic_summary(messages: list[ModelMessage]) -> str:
    user_constraints = []
    confirmed_facts = []
    tool_evidence = []
    decisions = []
    open_questions = []
    rejected = []
    lookup_hints = []

    for message in messages:
        content = _trim(redact_text(message.content), 420)
        if message.role == "system" and "<conversation_summary>" in message.content:
            confirmed_facts.append(f"prior summary retained: {_trim(content, 520)}")
        elif message.role == "user":
            if _looks_like_constraint(content):
                user_constraints.append(content)
            else:
                open_questions.append(content)
        elif message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                lookup_hints.append(
                    f"assistant requested tool {tool_call.name} toolCallId={tool_call.id} "
                    f"args={_trim(_json(tool_call.arguments), 220)}"
                )
        elif message.role == "assistant":
            decisions.append(content)
        elif message.role == "tool":
            tool_evidence.append(_tool_summary(message))

    lines = [
        "<conversation_summary>",
        "  <session_goal>",
        "  - Continue the current troubleshooting conversation using this compacted context.",
        "  </session_goal>",
        "",
        "  <confirmed_facts>",
        *_xml_items(confirmed_facts or decisions[:6]),
        "  </confirmed_facts>",
        "",
        "  <user_constraints>",
        *_xml_items(user_constraints),
        "  </user_constraints>",
        "",
        "  <tool_evidence>",
        *_xml_items(tool_evidence),
        "  </tool_evidence>",
        "",
        "  <decisions_or_progress>",
        *_xml_items(decisions[:8]),
        "  </decisions_or_progress>",
        "",
        "  <open_questions>",
        *_xml_items(open_questions[-8:]),
        "  </open_questions>",
        "",
        "  <rejected_or_failed_paths>",
        *_xml_items(rejected),
        "  </rejected_or_failed_paths>",
        "",
        "  <do_not_assume>",
        "  - Do not treat compacted historical guesses as current evidence.",
        "  - Re-check realtime tools when the user asks about current system state.",
        "  </do_not_assume>",
        "",
        "  <lookup_hints>",
        *_xml_items(lookup_hints[-8:]),
        "  </lookup_hints>",
        "</conversation_summary>",
    ]
    return "\n".join(lines)


def _tool_summary(message: ModelMessage) -> str:
    raw_ref = ""
    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        raw_ref_value = payload.get("raw_ref")
        raw_ref = f" raw_ref={raw_ref_value}" if raw_ref_value else ""
        status = payload.get("status") or payload.get("success")
        count = payload.get("resultCount")
        if count is None and isinstance(payload.get("result"), dict):
            count = _nested_count(payload["result"])
        return (
            f"toolName={message.name} toolCallId={message.tool_call_id} "
            f"status={status} resultCount={count}{raw_ref} "
            f"content={_trim(json.dumps(payload, ensure_ascii=False, sort_keys=True), 520)}"
        )
    return (
        f"toolName={message.name} toolCallId={message.tool_call_id}{raw_ref} "
        f"content={_trim(message.content, 520)}"
    )


def _nested_count(payload: dict[str, Any]) -> int | None:
    for key in ("count", "total", "resultCount"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    for key in ("alerts", "logs", "chunks", "topics", "memories", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _looks_like_constraint(content: str) -> bool:
    lowered = content.lower()
    return any(
        keyword in lowered
        for keyword in (
            "必须",
            "不要",
            "禁止",
            "请记住",
            "以后",
            "must",
            "never",
            "do not",
            "remember",
        )
    )


def _xml_items(items: list[str]) -> list[str]:
    if not items:
        return ["  - none"]
    return [f"  - {_escape_xml(_trim(item, 700))}" for item in items if item.strip()]


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit // 2].rstrip()} ... {text[-limit // 2 :].lstrip()}"


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
