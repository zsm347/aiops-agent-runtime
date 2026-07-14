# 10D 上下文窗口与压缩实施计划

> 状态：已完成前置调研后的正式实施计划。实现前置依据见 `docs/10D-context-management-research.md`。本阶段仍未开始代码实现，下一步才能按本文交给 subAgent 执行。

## 1. 阶段定位

本文是第 10 步高级能力中的第四个代码子阶段：`10D 上下文窗口与压缩`。

本阶段目标是建立一套可控的上下文窗口治理机制，避免真实模型调用时因为上下文过长失败，同时尽量保留排障链路中的关键证据、用户约束、当前任务状态和工具调用上下文。

调研后修正一个关键点：

```text
10D 不应该只是“把历史总结一下”；
它应该是模型调用前的 ContextManager / Reducer 层。
```

也就是说，`ContextAssembler` 不再承担所有预算、裁剪、压缩判断；它只负责把已经治理好的上下文组件组装成模型消息。

## 2. 调研依据

前置调研结论见：

```text
docs/10D-context-management-research.md
```

关键参考：

- LangChain / LangGraph：模型调用前 middleware、trim messages、SummarizationMiddleware、ContextEditingMiddleware。
- Letta：sliding-window compaction、summary message、完整事实源 + 可压缩上下文视图。
- OpenAI Agents SDK：Session、turn-based trimming、summary session。
- Anthropic：server-side compaction、context editing、tool result clearing、token counting。
- LlamaIndex：token limit、chat history ratio、memory block priority。
- Semantic Kernel：ChatHistoryReducer、target + threshold 滞回。
- AutoGen：ChatCompletionContext 作为模型可见上下文视图。
- Headroom：作为内容级上下文压缩层，适合大工具输出、日志、JSON、搜索结果和 RAG chunk 压缩；不作为会话级 ContextManager。
- LLMLingua：可作为未来 RAG 长文本压缩实验方向，但不作为 10D P0 大工具输出主路径。

本阶段采用的原则：

```text
借鉴成熟框架的上下文治理策略；
不直接替换本项目事件流、trace、tenant context、ToolGateway、Memory/RAG 契约。
```

## 3. 当前基线

当前 Python 版本已经具备：

- `ConversationRuntime.start_run()` 在请求开始时从事件流恢复 `active_history`。
- `recover_active_history()` 从 rollout events 恢复：
  - user message
  - assistant message
  - tool call started
  - tool result / failed / blocked
- `ConversationRuntime` 在一次 run 内用 `_active_history_by_run` 维护短期历史。
- `ContextAssembler` 组装：
  - system prompt
  - core memory
  - memory index
  - active history
  - current user message
- `AssembledContext.trace_payload` 已记录：
  - prompt version
  - message count
  - estimated chars
  - history item count
  - memory 注入状态
- `RolloutEventType` 已有：
  - `HISTORY_TRIMMED`
  - `MODEL_CALL_CONTEXT_OVERFLOW`

当前不足：

- 只有字符数估算，没有 token 估算。
- 没有上下文预算配置。
- `active_history` 会随事件流持续增长。
- 工具结果会被完整恢复进历史，大日志、大 RAG 结果、大告警列表会快速撑爆上下文。
- 没有模型调用前的统一 ContextManager。
- 没有 tool call / tool result 配对保护。
- 没有 context component usage trace。
- 没有压缩摘要 prompt、事件格式和恢复逻辑。
- 没有压缩质量评测。

## 4. 设计原则

本阶段必须遵守：

- 不引入 `agent_session_state` 作为必选表。
- 不改变“请求开始从 PostgreSQL 事件流恢复历史，run 内内存维护，run 结束释放”的短期记忆设计。
- 不把压缩摘要当作长期记忆。
- 不做后台异步长期记忆抽取。
- 不改长期记忆的 Core / Archival 设计。
- 不要求真实模型 API 才能跑本地测试。
- 不破坏当前 `/api/chat`、`/api/chat_stream`、eval 和 trace 契约。
- 不直接接入 Letta / LlamaIndex Memory / OpenAI Responses compaction session 替代本项目 Harness。

核心取舍：

```text
10D P0 先实现项目内轻量 ContextManager；
token 估算、工具结果清理、历史压缩、trace 记录形成闭环；
真实 LLM compactor 作为 adapter，测试默认使用 deterministic/fake compactor。
```

## 5. 总体架构

10D 后的模型调用前流程：

```text
start_run
  -> 从事件流恢复 active_history
  -> ContextManager.prepare(...)
       -> TokenEstimator 估算各组件 token
       -> ContextBudgeter 计算预算和触发条件
       -> ToolResultReducer 脱敏、持久化原文、压缩大工具结果
       -> HistoryCompactor 判断是否压缩旧历史
       -> 生成 PreparedContext 和 ContextComponentUsage
       -> 如发生压缩，写入 HISTORY_TRIMMED
  -> ContextAssembler.assemble(prepared_context)
  -> CONTEXT_ASSEMBLED trace 记录预算和组件使用情况
  -> model/tool loop
```

核心边界：

- `ContextManager`：做预算、清理、压缩和决策。
- `ContextAssembler`：只做消息顺序组装。
- `ConversationRuntime`：负责调用 ContextManager、写 trace、维护 run 内 active history。
- `recover_active_history()`：负责从事件流恢复 summary + recent history。

## 6. 核心对象设计

建议新增：

```text
src/superbiz_agent/harness/token_estimator.py
src/superbiz_agent/harness/context_budget.py
src/superbiz_agent/harness/context_manager.py
src/superbiz_agent/harness/tool_result_reducer.py
src/superbiz_agent/harness/content_compression.py
src/superbiz_agent/harness/compaction.py
prompts/context-compaction-v1.md
tests/test_context_window_compaction.py
```

### 6.1 ContextBudget

职责：

- 表示模型输入预算。
- 表示保留输出 token。
- 表示触发压缩阈值。
- 表示压缩目标水位。

建议字段：

```text
max_input_tokens
reserved_output_tokens
effective_input_budget
trigger_tokens
target_tokens_after_compaction
tool_result_compress_threshold_tokens
content_compression_backend
recent_turns_to_keep
recent_keep_ratio
```

### 6.2 ContextComponentUsage

用于 trace 和调试。

建议字段：

```text
name
required
priority
estimated_tokens_before
estimated_tokens_after
included
action
dropped_reason
```

示例组件：

- system_prompt
- core_memory
- memory_index
- active_history
- compacted_summary
- recent_history
- current_user_message
- tool_results

### 6.3 PreparedContext

`ContextManager` 的输出。

建议字段：

```text
summary_messages
history_messages
current_user_message
component_usages
estimated_input_tokens
budget
compaction_triggered
tool_results_reduced
overflow
```

`ContextAssembler` 使用 `PreparedContext` 组装最终 `ModelMessage`。

## 7. 配置项

建议新增配置：

```text
context_max_tokens
context_reserved_output_tokens
context_compaction_trigger_ratio
context_compaction_target_ratio
context_recent_turns_to_keep
context_recent_keep_ratio
context_tool_result_compress_threshold_tokens
context_content_compression_backend
context_tool_results_to_keep
context_compaction_prompt_version
```

建议默认值：

```text
context_max_tokens = 32000
context_reserved_output_tokens = 4000
context_compaction_trigger_ratio = 0.85
context_compaction_target_ratio = 0.65
context_recent_turns_to_keep = 4
context_recent_keep_ratio = 0.70
context_tool_result_compress_threshold_tokens = 1000
context_content_compression_backend = headroom
context_tool_results_to_keep = 3
context_compaction_prompt_version = context-compaction-v1
```

计算：

```text
effective_input_budget = context_max_tokens - context_reserved_output_tokens
trigger_tokens = effective_input_budget * context_compaction_trigger_ratio
target_tokens_after_compaction = effective_input_budget * context_compaction_target_ratio
```

说明：

- P0 可优先使用 `recent_turns_to_keep` 简化实现。
- 设计上必须保留 `recent_keep_ratio` 扩展点，方便后续接近 Letta sliding-window。
- 工具结果内容级压缩使用单一阈值：单个工具输出估算超过 `1000 tokens` 才触发。
- 本地测试可以把 `context_content_compression_backend` 切到 `noop` 或 fake backend，避免依赖真实 Headroom 调用。
- 测试可以使用很小的预算触发压缩。

## 8. Token Estimator

### 8.1 接口

```python
class TokenEstimator(Protocol):
    def estimate_text(self, text: str) -> int: ...
    def estimate_messages(self, messages: list[ModelMessage]) -> int: ...
```

P0 实现：

```text
ApproxTokenEstimator
```

建议规则：

- ASCII / 英文：约 4 chars = 1 token。
- 中文：约 1.5 到 2 chars = 1 token。
- 每条 message 加固定 overhead。
- tool call JSON 需要额外 overhead。

后续增强：

- DashScope/Qwen tokenizer。
- provider token counting API。
- 使用真实模型返回 usage 对 estimator 做校准。

### 8.2 trace 字段

`CONTEXT_ASSEMBLED` payload 建议新增：

```json
{
  "estimatedInputTokens": 1234,
  "effectiveInputBudgetTokens": 28000,
  "compactionTriggerTokens": 23800,
  "compactionTargetTokens": 18200,
  "compactionTriggered": false,
  "toolResultsReduced": false,
  "componentUsages": []
}
```

保留现有 `estimatedChars`，避免破坏已有断言。

## 9. 上下文预算优先级

上下文组件优先级：

| 内容 | 处理策略 |
|---|---|
| system prompt | 必须保留 |
| current user message | 必须保留 |
| tool schema | 必须保留，但当前项目未显式放入 messages，可在 token 估算中预留 |
| core memory | 默认必须保留；如果超限，应该由长期记忆模块治理，不在 10D 随意删除 |
| memory index | 默认保留；如果过大，后续由长期记忆模块治理 |
| 最近对话 | 优先原文保留 |
| 历史摘要 | 压缩后保留 |
| 旧对话原文 | 超预算时压缩 |
| 旧工具结果 | 优先清理、截断、placeholder |

不可通过压缩解决的情况：

- system prompt + current user message 已超预算。
- core memory / memory index 本身过大。
- 当前用户输入过大。

这些情况应记录 `MODEL_CALL_CONTEXT_OVERFLOW`，并返回明确错误或降级提示。

## 10. 工具结果治理

调研结论显示，工具结果治理应独立于历史摘要。

本阶段新增 `ToolResultReducer`，在会话级历史压缩前先处理工具结果。

核心定位：

```text
Headroom = 内容级压缩后端，解决单个大工具输出太大的问题。
ContextManager / HistoryCompactor = 会话级上下文治理，解决长会话持续增长的问题。
```

Headroom 不负责删除旧 message、不负责 session summary、不负责长期记忆，也不替代本项目的 `ContextManager`。

P0 处理流程：

```text
工具结果 JSON
  -> 结构化解析
  -> 敏感信息脱敏
  -> 原始结果先写入 rollout event / artifact store
  -> 生成 tenant/run/session/tool_call 受控的 raw_ref
  -> 估算单个工具结果 tokens
  -> 如果 <= 1000 tokens，脱敏后的短结果直接进入上下文
  -> 如果 > 1000 tokens，调用 ContentCompressionBackend
  -> 压缩结果 + raw_ref + compression metadata 进入上下文
  -> 旧工具结果可替换为 placeholder
```

P0 规则：

- 单个工具结果估算 `<= 1000 tokens`：不压缩，但仍持久化原文并记录 `raw_ref`。
- 单个工具结果估算 `> 1000 tokens`：触发内容级压缩，P0 默认 backend 为 `HeadroomContentCompressionBackend`。
- `NoopContentCompressionBackend` 只用于测试、禁用压缩或 Headroom 不可用时的降级路径。
- Headroom 压缩失败时，不阻断主链路；记录 trace warning，退回确定性脱敏、截断和 placeholder 策略。
- 最近 `context_tool_results_to_keep` 个工具结果尽量保留压缩摘要或短结果。
- 旧工具结果可替换为：

```text
[tool result cleared: toolName=..., toolCallId=..., originalLength=..., raw_ref=..., raw event retained in rollout store]
```

- 保留：
  - toolName
  - toolCallId
  - status / success
  - errorType / errorMessage
  - query/window/filter 等关键查询参数摘要
  - resultCount
  - top evidence summary
  - raw_ref 或 rollout event sequence
  - compression backend / before tokens / after tokens

`raw_ref` 不是让模型直接访问数据库的裸 id，而是后端可校验的原文引用。后续如果模型需要原文，只能通过受控的原文取回工具读取；该工具必须校验 tenant_id、user_id、run_id、session_id、tool_call_id 和权限，不能让模型拼接任意 DB key。

不依赖 Headroom CCR 本地存储作为事实源。可以借鉴 CCR 的可逆压缩思想，但本项目的原文事实源必须是自己的事件流或 artifact store。

禁止：

- 把密钥、token、密码等敏感字段返回给模型。
- 把大段日志全文、指标点全文、RAG 文档全文无限塞进上下文。
- 无痕删除工具结果，导致模型以为没查过。

## 11. 压缩触发条件

触发条件：

```text
estimated_input_tokens > trigger_tokens
```

附加条件：

```text
active_history 至少超过 recent keep 范围；
可压缩区域 token > 0；
超限不是由 system prompt / current user message 单独导致。
```

滞回策略：

```text
超过 trigger_tokens 才触发；
触发后目标是压到 target_tokens_after_compaction 以下。
```

这样可以避免每次只超过一点就压缩。

## 12. 压缩对象选择

### 12.1 P0 实现策略

P0 可使用 recent turns 简化：

```text
保留最近 context_recent_turns_to_keep 轮原文；
压缩更旧历史。
```

默认保留最近 4 轮 user-assistant turn。

### 12.2 设计扩展点

设计上要保留 sliding-window 扩展：

```text
默认保留最近 70% 可见历史；
总结较旧 30%；
如果压缩后仍超预算，再扩大被压缩比例。
```

这来自 Letta 的 sliding-window compaction 思路。

### 12.3 边界保护

压缩不能拆散：

- assistant tool call 与对应 tool result。
- 同一次用户问题下的 assistant -> tool -> assistant 小链路。
- 当前 run 内尚未完成的工具调用。

如果 cutoff 落在 tool result 上，应向前或向后调整到安全边界。

## 13. 压缩摘要格式

压缩摘要不是自由总结，必须结构化。

建议摘要格式：

```text
<conversation_summary>
  <session_goal>
  ...
  </session_goal>

  <confirmed_facts>
  - ...
  </confirmed_facts>

  <user_constraints>
  - ...
  </user_constraints>

  <tool_evidence>
  - 工具名 / toolCallId / 时间窗口 / 关键结果 / 证据限制
  </tool_evidence>

  <decisions_or_progress>
  - ...
  </decisions_or_progress>

  <open_questions>
  - ...
  </open_questions>

  <rejected_or_failed_paths>
  - ...
  </rejected_or_failed_paths>

  <do_not_assume>
  - ...
  </do_not_assume>

  <lookup_hints>
  - ...
  </lookup_hints>
</conversation_summary>
```

必须保留：

- 用户明确要求。
- 已确认事实。
- 工具证据、证据来源和证据限制。
- 未解决问题。
- 仍需验证的假设。
- 已经排除的方向。
- 关键 ID：runId、toolCallId、traceId、service、env、time window、file path、doc title。

禁止：

- 把猜测写成事实。
- 把工具失败写成工具成功。
- 把历史经验写成当前实时证据。
- 保留密钥、token、密码等敏感信息。
- 大段复制日志原文。

## 14. 压缩 Prompt

新增：

```text
prompts/context-compaction-v1.md
```

职责：

- 输入旧历史窗口、已有摘要、工具结果摘要。
- 输出结构化 `<conversation_summary>`。
- 区分事实、证据、猜测、未确认项。
- 保留用户约束、当前进展、未解决问题和关键 ID。
- 不编造。
- 不把压缩摘要当长期记忆。

P0 测试：

- 使用 deterministic/fake compactor，保证本地测试不依赖真实模型。

真实 LLM compactor：

- 通过 `ModelGateway.complete()` 调用。
- 使用独立 prompt version。
- trace 记录 compaction prompt version、模型 provider、token 估算、状态。

## 15. HISTORY_TRIMMED 事件

使用事件流，不引入必选 `agent_session_state`。

payload 建议：

```json
{
  "summary": "<conversation_summary>...</conversation_summary>",
  "compactionMode": "recent_turns",
  "compactionPromptVersion": "context-compaction-v1",
  "triggerReason": "estimated_input_tokens_exceeded",
  "fromSequence": 12,
  "toSequence": 48,
  "sourceMessageCount": 24,
  "recentMessagesKept": 8,
  "estimatedTokensBefore": 24000,
  "estimatedTokensAfter": 9000,
  "tokenReductionRatio": 0.62,
  "toolResultsReduced": true,
  "componentUsages": [],
  "status": "success"
}
```

说明：

- `fromSequence/toSequence` 表示压缩覆盖的事件范围。
- `compactionMode` P0 可为 `recent_turns`，后续可扩展为 `sliding_window`。
- `componentUsages` 记录各组件是否被压缩、保留、截断。
- P0 必须至少使用最新 `HISTORY_TRIMMED` 作为恢复边界。

## 16. 恢复逻辑

当前恢复逻辑是从事件流逐条重建完整 history。

10D 后：

```text
如果事件流里存在 HISTORY_TRIMMED：
  取最新 status=success 的 HISTORY_TRIMMED
  将 summary 作为 summary/system message 放到 active_history 前部
  只恢复该 HISTORY_TRIMMED 之后的事件
否则：
  按当前逻辑恢复完整 active_history
```

推荐消息形式：

```python
ModelMessage(role="system", content="<conversation_summary>...</conversation_summary>")
```

注意：

- summary 不应伪装成用户或助手的新发言。
- summary 应在最终组装中位于 system prompt / memory 之后、recent history 之前。
- 不应重复恢复已被 summary 覆盖的旧消息。

## 17. 与长期记忆的关系

上下文压缩摘要不是长期记忆。

| 项 | 上下文压缩摘要 | 长期记忆 |
|---|---|---|
| 目的 | 让当前会话继续可用 | 跨会话复用 |
| 来源 | 会话历史压缩 | 用户偏好、稳定背景、历史经验 |
| 生命周期 | 会话级 | 用户/租户级 |
| 使用方式 | 自动进入上下文 | Core 常驻或 Archival 按需检索 |
| 内容要求 | 保留当前任务状态 | 只保存稳定、可复用信息 |

10D 不做后台记忆抽取。

如果未来要在压缩前提醒模型保存记忆，应放到后续单独子阶段。

## 18. 需要修改的文件

建议新增：

```text
src/superbiz_agent/harness/token_estimator.py
src/superbiz_agent/harness/context_budget.py
src/superbiz_agent/harness/context_manager.py
src/superbiz_agent/harness/tool_result_reducer.py
src/superbiz_agent/harness/compaction.py
prompts/context-compaction-v1.md
tests/test_context_window_compaction.py
```

建议修改：

```text
src/superbiz_agent/harness/context_assembler.py
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/harness/history.py
src/superbiz_agent/harness/events.py
src/superbiz_agent/config.py
tests/test_skeleton.py
tests/test_migration_p0_runtime.py
docs/04-python-migration-roadmap.md
```

原则上不改：

```text
src/superbiz_agent/tools/
src/superbiz_agent/model_gateway/openai_compatible.py
src/superbiz_agent/memory/
src/superbiz_agent/rag/
src/superbiz_agent/api/
```

除非测试发现必须小幅适配。

## 19. 测试计划

### 19.1 Token estimator

覆盖：

- 英文文本估算。
- 中文文本估算。
- 多 message 估算。
- tool call / tool result message overhead。
- 空文本。

### 19.2 Context budget

覆盖：

- effective input budget 计算。
- trigger tokens 计算。
- target tokens 计算。
- 小预算触发压缩。
- 未超预算不触发。

### 19.3 Tool result reducer

构造包含大工具结果的历史。

断言：

- 大数组被截断。
- 敏感字段被脱敏。
- 旧工具结果可被 placeholder 替代。
- `<= 1000 tokens` 的工具结果不调用内容压缩 backend。
- `> 1000 tokens` 的工具结果调用内容压缩 backend。
- 工具原文先持久化，再生成压缩结果。
- 压缩结果带 `raw_ref`、backend、before tokens、after tokens。
- Headroom backend 失败时可降级为确定性 reducer，并记录 trace warning。
- 保留 toolName、toolCallId、status、errorType、关键摘要。
- 不产生孤立 tool result。

### 19.4 不触发压缩

构造短历史。

断言：

- 不产生 `HISTORY_TRIMMED`。
- `CONTEXT_ASSEMBLED` 中 `compactionTriggered=false`。
- 上下文消息顺序保持不变。

### 19.5 触发压缩

构造长历史和小预算。

断言：

- 产生 `HISTORY_TRIMMED`。
- summary 被注入上下文。
- 最近 N 轮原文保留。
- 旧历史原文不再全部进入上下文。
- estimated tokens 降低。
- component usage 记录完整。

### 19.6 tool call / tool result 配对保护

构造：

```text
assistant tool_call
tool result
assistant answer
```

断言：

- 压缩 cutoff 不会让 tool result 孤立。
- 不会让 assistant tool_call 没有对应 tool result。
- 最近未完成工具调用不被压缩。

### 19.7 恢复逻辑

构造事件流：

```text
旧消息
HISTORY_TRIMMED
新消息
```

断言：

- `recover_active_history()` 恢复 summary + 新消息。
- 不重复恢复已被 summary 覆盖的旧消息。
- summary 顺序在 recent history 之前。

### 19.8 压缩质量专项 eval

最少设计：

1. 用户偏好保留。
2. 已确认根因保留。
3. 工具证据保留。
4. 未确认假设仍标记为未确认。
5. 已排除方向保留。
6. 敏感信息不进入 summary。
7. 错误工具调用不能被写成成功。

P0 用 deterministic rule judge；后续可加 LLM-as-judge。

## 20. 验收命令

本阶段完成后至少运行：

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
PYTHONPATH=src python3 -m compileall -q src tests
```

如果安装了 ruff：

```bash
python3 -m ruff check src tests
```

## 21. 不做清单

本阶段不做：

- 不引入必选 `agent_session_state`。
- 不做后台异步长期记忆提取。
- 不改变 Core Memory / Archival Memory 设计。
- 不把压缩摘要写入长期记忆。
- 不做 Recall Memory。
- 不做 RAG 管线改造。
- 不改 ToolGateway 可靠性逻辑。
- 不依赖真实模型 API 通过本地测试。
- 不引入 LangSmith 作为必需依赖。
- 不接入 Letta 作为运行时。
- 不接入 LlamaIndex Memory 替代事件流。
- 不接入 OpenAI Responses compaction session 作为必需能力。
- 不把 LLMLingua 作为主 history compactor。
- 不把 Headroom 作为完整会话级 ContextManager。
- 不依赖 Headroom CCR 本地存储作为原文事实源。
- 不做前端展示。

## 22. subAgent 实施边界

如果交给 subAgent，实现边界必须明确。

允许写：

```text
src/superbiz_agent/harness/token_estimator.py
src/superbiz_agent/harness/context_budget.py
src/superbiz_agent/harness/context_manager.py
src/superbiz_agent/harness/tool_result_reducer.py
src/superbiz_agent/harness/content_compression.py
src/superbiz_agent/harness/compaction.py
src/superbiz_agent/harness/context_assembler.py
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/harness/history.py
src/superbiz_agent/harness/events.py
src/superbiz_agent/config.py
prompts/context-compaction-v1.md
tests/test_context_window_compaction.py
tests/test_skeleton.py
tests/test_migration_p0_runtime.py
docs/04-python-migration-roadmap.md
```

原则上不写：

```text
src/superbiz_agent/tools/
src/superbiz_agent/memory/
src/superbiz_agent/rag/
src/superbiz_agent/model_gateway/openai_compatible.py
src/superbiz_agent/api/
```

## 23. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否先完成前置调研 | 是，见 `docs/10D-context-management-research.md` |
| 是否直接实现代码 | 否，本文只定义计划 |
| 是否引入 `agent_session_state` 必选表 | 否 |
| 是否保持短期记忆请求级恢复、run 内内存维护 | 是 |
| 是否把压缩摘要当作长期记忆 | 否 |
| 是否要求真实模型 API 才能测试 | 否 |
| 是否定义 token 预算和触发条件 | 是 |
| 是否把工具结果治理独立出来 | 是 |
| 是否明确 Headroom 只做内容级压缩 | 是 |
| 是否采用单个工具输出 1000 tokens 压缩阈值 | 是 |
| 是否保护 tool call / tool result 配对 | 是 |
| 是否定义压缩摘要结构 | 是 |
| 是否定义持久化和恢复方式 | 是 |
| 是否定义压缩质量评测 | 是 |
| 是否避免修改 RAG / Memory / ToolGateway | 是 |

## 24. 阶段通过标准

10D 通过标准：

1. token estimator 可用并有测试。
2. context budget 可配置。
3. ContextManager / reducer 层存在，ContextAssembler 不再承担全部治理逻辑。
4. 大工具结果不会无限进入上下文。
5. 单个工具输出 `> 1000 tokens` 时，原文先持久化，压缩结果带 `raw_ref` 进入上下文。
6. 单个工具输出 `<= 1000 tokens` 时不做内容压缩。
7. 历史超过阈值时触发会话级压缩。
8. 压缩后上下文包含 summary + 最近 N 轮原文。
9. 压缩摘要写入 `HISTORY_TRIMMED` 事件。
10. 下次请求能从事件流恢复 summary + recent history。
11. tool call / tool result 配对不被破坏。
12. `CONTEXT_ASSEMBLED` trace 包含 token budget、compaction 状态和 component usage。
13. 压缩 prompt 版本化。
14. 压缩质量测试覆盖关键信息保留和敏感信息过滤。
15. 全量测试、eval runner、compileall 通过。
