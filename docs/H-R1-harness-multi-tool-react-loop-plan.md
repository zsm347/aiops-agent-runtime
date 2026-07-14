# H-R1 Harness 多工具与多轮 ReAct 循环实施计划

## 1. 阶段定位

H-R1 是一次独立的 Harness 共享能力增强，不属于 `10G.2A 长期记忆专项评测` 的实现范围，也不是 `10H 观测、LangSmith、部署配置`。

阶段目标：

```text
让一个 Agent run 能完整执行：
模型一次返回的多个工具调用
+ 工具结果后的后续模型/工具轮次
+ 明确的 run 级工具预算
+ 预算耗尽后的可解释收尾
```

该能力同时服务于：

- 长期记忆 + 实时日志/告警组合任务。
- RAG + 实时工具组合任务。
- 首次检索为空后的改写 query 重试。
- 工具参数错误返回模型后的参数修正。
- 工具失败后的替代工具或降级路径。

H-R1 不调整长期记忆 Dataset/Judge，不修改 memory prompt，不接真实 Embedding，也不实现 PostgreSQL memory store。

## 2. 立项依据

第一次长期记忆 Track B dev 运行中：

| Case | 模型行为 | 当前 Harness 行为 |
|---|---|---|
| R02 | 第一次 `searchMemory` 为空后生成第二次搜索 | 第二次调用没有执行，最终答案为空 |
| U02 | 同一响应生成告警、日志、记忆多个调用 | 只执行第一个调用 |
| U05 | 同一响应生成实时告警和记忆检索 | 只执行第一个调用，后续搜索也被截断 |
| I01 | 先 list topics，再生成 search | list 完成后 search 没有执行 |

直接根因位于 `SkeletonAgentGraph`：

```text
_tool_dispatch 只执行 response.tool_calls[0]
tool_dispatched 一旦为 true，后续 model tool_calls 直接路由到 final
```

这不是模型没有产生工具调用，也不是 Memory Eval capture 缺陷。

## 3. 当前契约与约束

H-R1 必须复用并保留：

- LangGraph `StateGraph` 主链路。
- `ModelGateway.complete(messages, tools)`。
- `ToolGateway.execute(run_context, tool_call)`。
- Pydantic 工具参数校验。
- 权限、policy、timeout、runtime retry、failure governance。
- 每个 `tool_call_id` 的 trace。
- `ToolResultReducer` 的脱敏、raw_ref 和大结果压缩。
- ConversationLockManager 的同会话串行化。
- OpenAI-compatible assistant tool_calls / tool message 配对协议。
- 现有同步 chat 和当前伪 streaming API 行为。

### 3.1 为什么继续使用自定义 LangGraph node

LangGraph 已提供 `ToolNode`，但当前项目不能直接替换：

- 本项目的所有工具必须经过自定义 `ToolGateway`。
- ToolGateway 需要 `RunContext` 才能完成 tenant 权限、run 级失败治理和 trace。
- timeout、runtime retry、Pydantic violations、raw_ref 和安全脱敏已经在本地契约中实现。
- 直接使用通用 ToolNode 会形成双层工具执行器，或者绕过现有治理。

因此 H-R1 继续使用 LangGraph `StateGraph` 的循环能力，只增强现有 `tool_dispatch` node。未来如果 LangGraph ToolNode 支持无损委托到本项目 ToolGateway，再单独评估替换。

不能为了实现循环：

- 绕过 ToolGateway 直接调用 handler。
- 只在 memory eval runner 中模拟多轮。
- 让模型生成 tenant/user/agent/run identity。
- 静默丢弃未执行的 tool call。
- 达到预算后返回空字符串。
- 用 LangGraph 默认 recursion error 代替业务可解释预算。

## 4. 设计结论

### 4.1 状态机

目标状态机：

```text
model_call
  -> 无 tool_calls：final
  -> 有 tool_calls 且预算可用：tool_dispatch_batch
       -> 为本响应中的每个 tool_call 生成一个 ToolResult
       -> model_call
  -> 有 tool_calls 但预算耗尽：tool_budget_block
       -> 为每个未执行调用生成配对的 budget error ToolResult
       -> final_model_call（tools disabled）
       -> final
```

新的 graph state 至少包含：

```text
run_context
messages
model_response
final_answer
tool_round_count
tool_call_count
seen_tool_call_ids
force_final_without_tools
tool_budget_exhausted
tool_result_reduction
```

删除当前永久性的 `tool_dispatched: bool`。是否继续 dispatch 应由当前 `model_response.tool_calls` 和显式预算决定，而不是由“本 run 以前是否调用过工具”决定。

### 4.2 多工具执行方式

模型一次返回 `N` 个工具调用时：

```text
1. 保留一个 assistant message，内含原始 N 个 tool_calls。
2. 按模型返回顺序逐个调用 ToolGateway.execute。
3. 为每个 tool_call 追加一个对应的 tool message。
4. 所有结果追加完成后，统一运行一次 ToolResultReducer。
5. 将完整 messages 送入下一次 model_call。
```

初版选择顺序执行，不并发：

- 工具可能有副作用。
- ToolGateway 有本 run failure governance 状态。
- trace sequence 和 raw_ref 需要稳定顺序。
- 部分工具的后端系统可能不适合瞬时并发。
- 当前没有 `read_only/parallel_safe` 工具能力标记。

以后只有在 ToolPolicy 明确增加 `parallel_safe=true` 后，才单独设计并行工具批次。

### 4.3 调用配对

每个通过协议校验的模型 tool call 必须得到以下二者之一：

```text
真实 ToolGateway execution result
或
明确的 ToolGateway blocked/budget result
```

不能只把 model tool call 放入 assistant message，却不追加对应 tool message。否则 OpenAI-compatible 下一轮消息不合法，TraceJudge 也会报告 `tool result missing`。

## 5. Run 级预算

### 5.1 配置

新增独立配置：

```text
agent_max_tool_rounds = 4
agent_max_tool_calls_per_run = 8
```

定义：

- `tool_round`：一次模型响应中包含一个或多个 tool calls，并完成其全部 execution/blocked result。
- `tool_call_count`：本 run 中模型请求的 tool call 总数，包括因 run budget 被阻止的调用。

默认值依据：

- 当前组合排障通常需要 memory、alerts、logs、docs 中的 2-4 个来源。
- 允许一次参数修正或空结果 query 改写后，常见调用数约 4-6。
- 4 轮、8 次调用能够覆盖当前 R02/U02/U05/I01，同时限制无效循环。
- 配置可调，但不能由模型传入。

该预算与 ToolGateway 内部预算不同：

| 预算 | 所属层 | 含义 |
|---|---|---|
| runtime retry budget | ToolGateway | 同一次工具执行内部的网络/超时自动重试 |
| model repair/failure governance | ToolGateway | 同一 run 内模型修参数或同失败模式治理 |
| tool rounds/calls | Harness Graph | 整个 Agent run 的工具循环总上限 |

### 5.2 批次超过剩余 call 预算

若当前模型一次返回的调用数超过剩余预算：

1. 整个批次都不执行，避免一组有关联或有副作用的调用只完成前半部分。
2. 对本批次每个调用生成 `agent_tool_budget_exhausted` blocked result。
3. 所有调用仍有一一配对的 tool message。
4. 本轮结束后进入禁用工具的 final model call。

不能直接丢弃超额调用，不能只执行一个任意前缀，也不能把模型给出的 tool_call_id 改成新 ID。以后只有在 ToolPolicy 明确表达事务组或 `parallel_safe/read_only` 语义后，才评估部分执行。

### 5.3 达到 round 预算

若模型在最后一个允许轮次后仍返回 tool calls：

- 不执行新的真实工具。
- 对本响应全部 tool calls 生成 budget-blocked result。
- 记录 run 级预算耗尽事件。
- 进入一次 `tools=[]` 的 final model call。

`tool_round_count` 在模型返回一批协议合法的 tool calls 时递增；无论该批次被真实执行还是因 run budget 被整体阻止，都计为一次请求轮，避免预算路径出现 off-by-one 或无限重试。

## 6. 预算错误结果

### 6.1 归属

Graph 负责判断 run budget，ToolGateway 负责生成标准 ToolErrorResult 和 `TOOL_CALL_BLOCKED` trace。

建议给 ToolGateway 增加窄接口：

```text
block_tool_call(
  run_context,
  tool_call,
  reason,
  message,
  allowed_next_actions,
)
```

该接口不执行 handler，但复用 ToolErrorResult、脱敏和 trace 规范。

### 6.2 结果结构

预算耗尽结果使用现有 `TOOL_BLOCKED` 类型，避免仅为一个 reason 扩大错误枚举：

```json
{
  "success": false,
  "error_type": "TOOL_BLOCKED",
  "reason": "agent_tool_budget_exhausted",
  "message": "本次 Agent run 的工具调用预算已耗尽，该调用未执行。",
  "retryable_by_runtime": false,
  "retryable_by_model": false,
  "allowed_next_actions": [
    "answer_with_available_evidence",
    "state_unverified_gaps",
    "ask_user_to_start_a_narrower_follow_up"
  ],
  "disallowed_next_actions": [
    "do_not_claim_the_tool_was_executed",
    "do_not_invent_missing_evidence"
  ]
}
```

### 6.3 Trace

新增 run 级事件：

```text
AGENT_TOOL_BUDGET_EXHAUSTED
```

payload：

```text
toolRoundCount
toolCallCount
maxToolRounds
maxToolCallsPerRun
blockedToolCallCount
status=exhausted
```

每个被阻止调用仍记录现有：

```text
TOOL_CALL_BLOCKED
reason=agent_tool_budget_exhausted
tool_call_id=<model supplied id>
```

## 7. 最终收尾

### 7.1 禁用工具的最后一次模型调用

预算耗尽后，模型仍需要看到每个 blocked ToolResult，再做一次收尾回答：

```text
ModelGateway.complete(messages, tools=[])
```

该调用要求模型：

- 只使用已经获得的证据。
- 说明哪些工具没有执行。
- 标记未确认信息。
- 不再请求工具。

### 7.2 模型仍返回 tool_calls

即使 `tools=[]`，provider 或模型仍可能返回 tool calls。此时：

- 不再进入 graph 循环。
- 记录 provider/model protocol anomaly。
- 如果 response.content 非空，使用其内容并追加未执行说明。
- 如果 response.content 为空，返回确定性的用户友好降级回答，不能返回空字符串。

建议新增 run 级事件：

```text
AGENT_FINALIZATION_FALLBACK
```

### 7.3 普通空回答

模型没有 tool calls 且 content 为空时，也不得让 Harness 以空答案成功结束。返回统一的非编造错误说明，并记录 `AGENT_FINALIZATION_FALLBACK`。

## 8. Tool call 协议防护

每个 tool call 的 `id` 和 `name` 必须非空；一个 run 内的 `tool_call_id` 必须唯一。该检查发生在 assistant tool_calls 被追加到 messages 之前。

当前 OpenAI-compatible adapter 会把缺失/空 ID 自动改写为 `toolcall-{index}`，并会静默跳过空 name。H-R1 必须取消这两种掩盖行为：

- adapter 在 `ModelResponse` 中保留空 ID/name，使 Graph 能统一拒绝协议非法批次。
- adapter 的 raw metadata 继续记录 `missing_tool_call_id/invalid_tool_name` parse error。
- adapter 不执行工具，也不自行决定降级路径。

处理规则：

- 同一模型响应中存在空 ID、空 name 或重复 ID：整批视为 model protocol error，不执行任何调用。
- 跨轮重复 ID：本轮整批视为 model protocol error，不执行任何调用。
- 无效 assistant tool_calls 消息不追加到下一轮 messages，因此不会制造无法唯一配对的 ToolMessage。
- 记录 `AGENT_MODEL_PROTOCOL_ERROR`，向 messages 追加一条脱敏的 runtime system notice，说明本轮工具请求因协议错误未执行，随后进入一次 `tools=[]` 的 final model call。
- 不由后端静默改写模型 ID；否则 trace、tool message 和 provider消息无法对齐。

该路径不是具体工具被策略拒绝，不复用 `TOOL_CALL_BLOCKED`；使用 run 级 `AGENT_MODEL_PROTOCOL_ERROR` 更准确。

## 9. LangGraph recursion limit

显式业务预算必须先于 LangGraph 自身 recursion limit 生效。

`ainvoke` 使用派生上限：

```text
recursion_limit = 2 * agent_max_tool_rounds + 6
```

原因：一个普通工具轮包含 `model_call + tool_dispatch_batch` 两个节点，并额外预留 entry、budget block、final model 和 END 路径。

LangGraph recursion error 只能作为防御性异常，不能成为正常预算控制方式。

## 10. ToolResultReducer 兼容

H-R1 不修改内容压缩策略，但必须验证：

- 一个 assistant message 中多个 tool_calls 均能保留。
- 每个 tool message 与对应 ID 一一匹配。
- 大结果分别生成各自 `raw_ref`。
- 一次 batch 追加后只做一次 reducer pass。
- reducer 不会留下 assistant 中存在但 tool message 已被移除的悬空调用。
- 多轮压缩后下一次模型输入仍符合 OpenAI tool message 顺序。

## 11. Run 状态清理

当前至少有两类进程内 run 状态：

```text
ToolGateway._failure_records / _validation_counts
ConversationRuntime._active_history_by_run / _history_events_by_run
```

现有 Runtime 只在 `complete_run/fail_run` 的 trace 写入成功后清理。`start_run` 写 trace 失败、完成事件写入失败或流式请求被 `CancelledError` 取消时，都可能残留状态。

H-R1 增加：

```text
ToolGateway.clear_run_state(run_id)
ConversationRuntime.cleanup_run(run_id)
```

清理规则：

- Graph 在正常结束、预算结束、协议错误和异常结束的 `finally` 中调用 `ToolGateway.clear_run_state`。
- `ConversationRuntime.complete_run/fail_run` 使用 `try/finally`，即使 trace 写入失败也清理 Runtime 状态。
- `ConversationRuntime.start_run` 在初始化后续步骤失败时立即回滚本次 run 状态。
- `AgentHarnessService.chat/chat_stream` 使用最终幂等 cleanup 作为兜底。
- `chat_stream` 明确处理 `asyncio.CancelledError`：最佳努力记录取消/失败 trace，重新抛出取消，并确保 Runtime/ToolGateway 状态均清理。

清理只删除治理计数，不删除 trace。

## 12. 文件范围

### 12.1 允许修改

```text
src/superbiz_agent/config.py
src/superbiz_agent/harness/graph.py
src/superbiz_agent/harness/events.py
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/harness/service.py
src/superbiz_agent/model_gateway/openai_compatible.py
src/superbiz_agent/tools/gateway.py
tests/test_harness_react_loop.py
tests/test_skeleton.py
tests/test_context_window_compaction.py
tests/test_model_gateway_openai_compatible.py
tests/test_tool_gateway_reliability.py
docs/04-python-migration-roadmap.md
docs/10-advanced-capabilities-plan.md
```

`service.py/runtime.py` 只允许注入 H-R1 配置、增加幂等 run cleanup 和取消路径防护，不重构 API/chat 的业务契约。

### 12.2 禁止修改

```text
prompts/
src/superbiz_agent/memory/
src/superbiz_agent/rag/
src/superbiz_agent/evals/memory_cases.py
evals/datasets/long_term_memory_v1.json
alembic/
.env
```

## 13. 实施批次

### H-R1-A：Graph 状态与顺序 batch dispatch

- 删除永久 `tool_dispatched`。
- 增加 round/call/seen ID 状态。
- 一次响应执行全部 tool calls。
- 一次 batch 后统一 reducer。
- 后续 model tool round 可以继续。

专项验收：

- 单轮两个工具均执行并配对。
- 两个连续工具轮后正常 final。
- 第一个工具失败时同 batch 第二个工具仍执行。
- tool call 顺序和 trace sequence 稳定。

### H-R1-B：预算、blocked result 和 finalization

- 增加 Settings 配置与校验。
- 增加 ToolGateway budget block 接口。
- 增加预算事件。
- 增加 tools-disabled final call。
- 增加空回答 fallback。

专项验收：

- round budget 生效。
- call budget 不足以容纳整批时执行 all-or-block，不产生部分副作用。
- 所有超额调用均有 blocked ToolResult。
- final call 看得到 blocked results 且 tools 为空。
- final model 仍返回 tool calls 时不会继续循环。

### H-R1-C：Reducer、history、eval 回归

- 多 tool result 压缩和 raw_ref 测试。
- trace/history replay 配对测试。
- OpenAI adapter 不再伪造缺失 tool_call_id 或静默丢弃空 name。
- start/complete/fail trace 异常和 chat_stream cancellation 后 run 状态均清理。
- 用 scripted gateway 复现 R02/U02/U05/I01 的调用形态。
- 不运行真实模型，不修改 Memory Dataset。

## 14. 验收矩阵

### 14.1 功能

- 一个响应 0/1/N 个 tool calls 均正确。
- 至少 2 个连续工具轮可以完成。
- ToolGateway error result 能被模型看到并继续决策。
- 剩余 call 预算不足以容纳当前批次时，整批不执行且每个调用都有 blocked result。
- 最终答案不为空。

### 14.2 协议

- 每个 assistant tool_call 有且仅有一个对应 tool message。
- tool_call_id 不被后端改写。
- duplicate/blank ID 或 blank name 使整批在进入消息历史前被明确拒绝并记录协议事件。
- OpenAI-compatible message serialization 测试通过。

### 14.3 治理与安全

- 所有真实执行仍经过 ToolGateway。
- 参数校验、权限、policy、timeout、retry 不回退。
- tenant/user/agent/run 仍由后端注入。
- blocked result 和 trace 已脱敏。
- run 状态最终清理。
- 流式取消和 trace 写入异常不会遗留 Runtime/ToolGateway run 状态。

### 14.4 上下文

- 多工具大结果仍压缩。
- raw_ref 与 tool_call_id 对应。
- reducer 后无悬空调用。
- context overflow/compaction 测试不回退。

### 14.5 全量

```text
H-R1 专项 pytest
MODEL_PROVIDER=stub 全量 pytest
基础 eval runner 14/14
Ruff
compileall
```

## 15. 完成定义

H-R1 只有同时满足以下条件才可完成：

1. 同一响应中的所有 tool calls 均有 execution/blocked result。
2. 后续工具轮不会被永久布尔状态截断。
3. round/call 预算均可配置且有 trace。
4. 预算耗尽后完成 tools-disabled finalization。
5. 无空最终答案。
6. ToolGateway 既有治理能力全部保留。
7. 多结果 reducer 和 OpenAI 配对测试通过。
8. run 级 ToolGateway 状态完成清理。
9. 没有修改 memory/RAG/prompt/Dataset。
10. 专项、全量 pytest、基础 eval、Ruff、compileall 全部通过。

H-R1 完成后只说明 Harness 已具备有限多工具/多轮循环；不代表 C03/A03 prompt 行为、真实 memory retrieval 或 PostgreSQL 持久化已经完成。

## 16. 实施与主验收结果

状态：

```text
H-R1 complete
```

已完成实现：

- 删除永久 `tool_dispatched`，支持单次模型响应中的多个工具调用按顺序完整执行，并支持后续有限工具轮。
- 增加 run 级 round/call 预算、整批 all-or-block、逐调用 blocked result 和禁用工具后的最终收尾调用。
- 增加工具调用 ID/name 协议校验；空值、同轮重复 ID 和跨轮重复 ID 会在写入 assistant tool-call 消息前整批拒绝。
- OpenAI-compatible adapter 不再伪造缺失 ID，也不再静默丢弃空 name。
- 保留 ToolGateway 参数校验、权限、策略、超时、重试、失败治理、脱敏和 trace 契约。
- 多结果继续经过一次 ToolResultReducer，保持 tool message 配对、压缩和 `raw_ref`。
- ConversationRuntime、ToolGateway 和 Service 已补齐正常结束、异常、trace 写入失败及流式取消路径的幂等 run 状态清理。

主验收结果：

```text
H-R1 专项测试                         -> 57 passed
MODEL_PROVIDER=stub 全量 pytest       -> 182 passed
基础 eval runner                      -> 14/14 passed
Ruff                                  -> passed
compileall                            -> passed
Dataset / prompt 哈希                 -> unchanged
生产代码阻塞缺陷                      -> 未发现
```

本次主验收未调用真实模型，且未修改 memory、RAG、prompt、Dataset 或 `.env`。H-R1 的完成不表示长期记忆正式 baseline 已完成；后续阶段为 `M-R1`。
