# 10C ToolGateway 可靠性增强实施计划

## 1. 阶段定位

本文是第 10 步高级能力中的第三个代码子阶段：`10C ToolGateway 可靠性增强`。

本阶段目标不是替换当前 Agent Harness，也不是整体引入另一个 Agent 框架，而是在现有 Python 单 Agent 链路上，把 `ToolGateway` 从“参数校验 + 执行 + trace”增强为可解释、可控、可评测的工具治理层。

本阶段必须保持：

- LangGraph / Harness 主链路不变。
- `ModelGateway` 不重构。
- 工具注册方式不重构。
- 现有业务工具契约不破坏。
- 本地 deterministic eval 不退化。

## 2. 当前基线

当前 Python 版本已经具备：

- `ToolRegistry` 注册工具定义。
- `ToolDefinition` 包含：
  - `name`
  - `description`
  - `args_model`
  - `handler`
  - `policy`
  - `requires_context`
- `ToolPolicy` 包含：
  - `tool_name`
  - `danger_level`
  - `action`
  - `timeout_seconds`
  - `max_retries`
  - `idempotent`
- `ToolGateway.execute()` 已做：
  - 未知工具拦截。
  - policy deny 拦截。
  - Pydantic 参数校验。
  - 工具执行。
  - `TOOL_CALL_STARTED`
  - `TOOL_CALL_COMPLETED`
  - `TOOL_CALL_FAILED`
  - `TOOL_CALL_BLOCKED`
- `RolloutEventType` 已包含 10C 所需事件：
  - `TOOL_CALL_TIMEOUT`
  - `TOOL_CALL_RETRY`
  - `TOOL_CALL_LATE_RESULT_DISCARDED`

当前不足：

- `timeout_seconds` 只进入 trace，没有实际执行超时控制。
- `max_retries` 只进入 trace，没有实际工程层重试。
- 参数校验错误只返回一条简单 message，没有字段级 `violations`。
- 工具执行异常只返回 `str(exc)`，没有统一脱敏和模型可理解结构。
- 没有区分 runtime 自动重试预算和模型修正重试预算。
- 没有本 run 内同一工具重复失败治理。

## 3. 参考对象与取舍

### 3.1 参考 PydanticAI

参考点：

- 工具参数使用 Pydantic schema 校验。
- 参数校验失败后，把具体校验错误作为 retry prompt 反馈给模型。
- 工具有 retry 预算，超过预算后停止继续修正。
- 工具可以通过类似 `ModelRetry` 的语义要求模型修正输入后重试。

本项目取舍：

- 不整体引入 PydanticAI。
- 继续使用当前 LangGraph / Harness / ToolGateway。
- 只吸收其工具校验反馈和 retry budget 设计。

原因：

- PydanticAI 更像完整 Agent Harness，不只是工具参数校验库。
- 当前项目已有自己的 `ModelGateway`、`ToolGateway`、tenant context、trace、eval 和 Java 兼容契约。
- 如果整体引入 PydanticAI，容易出现双层 Agent Runtime 和职责重叠。

### 3.2 参考 LangGraph

参考点：

- 节点级 retry / timeout / error handling 适合保护图节点。

本项目取舍：

- LangGraph 节点级保护可以后续用于 `tool_dispatch_node` 外层。
- 10C 的细粒度工具治理仍放在 `ToolGateway`。

原因：

- LangGraph 看到的是 node。
- 项目真正需要的是每个具体工具自己的 timeout、幂等重试、toolCallId trace、模型可见错误结果。

### 3.3 参考 Vercel AI SDK

参考点：

- tool call validation failure 可以进入 repair / retry 流程。
- 模型可根据结构化错误修复工具调用。

本项目取舍：

- 10C 不实现完整 `repairToolCall`。
- 只实现字段级参数错误反馈和模型重试预算。

## 4. 核心设计结论

本阶段不追求穷尽所有错误类型。

更合理的设计是：

```text
原始异常 / 校验错误 / 工具策略拒绝
  -> 脱敏
  -> 粗粒度归因
  -> 生成固定 JSON schema 的 ToolErrorResult
  -> 返回给模型
  -> 模型在 allowed_next_actions 范围内决策
  -> ToolGateway 控制重试预算和本 run 内失败治理
```

也就是说：

- `ToolErrorResult` 的 JSON schema 要提前定义。
- 错误类型可以粗粒度，不要求穷尽。
- 具体 message、violations、retry budget、allowed actions 根据运行时动态生成。
- 模型可以选择下一步，但选择空间由后端约束。
- ToolGateway 保留最终控制权。

## 5. ToolErrorResult 结构设计

### 5.1 字段设计

建议把当前 `ToolErrorResult` 从：

```json
{
  "success": false,
  "error_type": "TOOL_ERROR",
  "message": "...",
  "suggestion": "..."
}
```

增强为：

```json
{
  "success": false,
  "error_type": "PARAM_VALIDATION_FAILED",
  "reason": "tool_arguments_invalid",
  "message": "queryLogs 工具参数不合法。请根据 violations 修正；如果缺少用户信息，不要猜测，先追问用户。",
  "violations": [
    {
      "field": "limit",
      "problem": "Input should be less than or equal to 100",
      "expected": "1 到 100 之间的整数",
      "received": 101,
      "fix_hint": "把 limit 调整到 100 以内。"
    }
  ],
  "retryable_by_runtime": false,
  "retryable_by_model": true,
  "allowed_next_actions": [
    "retry_same_tool_with_fixed_arguments",
    "ask_user_for_missing_required_fields"
  ],
  "disallowed_next_actions": [
    "do_not_guess_missing_required_fields",
    "do_not_call_unrelated_tools"
  ],
  "retry_budget": {
    "runtime_used": 0,
    "runtime_max": 0,
    "model_used": 1,
    "model_max": 2
  }
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `success` | 固定为 `false` |
| `error_type` | 粗粒度错误类型，主要用于 trace / eval / 统计 |
| `reason` | 给模型看的稳定原因短语 |
| `message` | 给模型看的安全错误说明 |
| `violations` | 参数校验或业务校验的字段级错误 |
| `retryable_by_runtime` | 后端是否还可以自动重试 |
| `retryable_by_model` | 模型是否可以修正参数或换条件后重试 |
| `allowed_next_actions` | 模型允许采取的下一步动作 |
| `disallowed_next_actions` | 明确禁止模型做的动作 |
| `retry_budget` | runtime 和 model 两类预算 |

### 5.2 保持兼容

为了不破坏现有测试和工具契约：

- `success`
- `error_type`
- `message`
- `suggestion`

建议暂时保留。

新增字段必须是可选字段或有默认值。

`suggestion` 可以作为 `allowed_next_actions` 的自然语言摘要，兼容旧逻辑。

## 6. 参数校验失败设计

### 6.1 触发时机

工具执行前：

```python
args = definition.args_model.model_validate(tool_call.arguments)
```

如果抛出 `ValidationError`，不进入工具 handler，不做 runtime retry。

### 6.2 生成 violations

从 `ValidationError.errors()` 生成字段级 violations。

示例：

```json
{
  "field": "region",
  "problem": "Input should be 'ap-guangzhou', 'ap-shanghai', 'ap-beijing' or 'ap-chengdu'",
  "expected": "枚举值之一：ap-guangzhou, ap-shanghai, ap-beijing, ap-chengdu",
  "received": "us-west-1",
  "fix_hint": "请从工具 schema 允许的 region 中选择；如果用户没有指定地域，使用默认地域或追问用户。"
}
```

生成依据：

- `err["loc"]` -> `field`
- `err["msg"]` -> `problem`
- `err.get("input")` -> `received`
- `args_model.model_json_schema()` -> `expected`
- 工具名 + 字段名 + 错误类型 -> `fix_hint`

### 6.3 allowed actions

参数校验失败时，建议：

```json
{
  "allowed_next_actions": [
    "retry_same_tool_with_fixed_arguments",
    "ask_user_for_missing_required_fields"
  ],
  "disallowed_next_actions": [
    "do_not_guess_missing_required_fields",
    "do_not_call_unrelated_tools"
  ]
}
```

说明：

- 如果字段是格式错、枚举错、范围错，模型可以修正后重试。
- 如果缺少必填字段，而且用户问题里没有足够信息，模型必须追问用户。
- 不允许模型为了绕过校验去调用无关工具。

### 6.4 模型重试预算

参数校验失败使用 `model_retry_budget`，建议默认：

```text
同一 run、同一 tool、同一 tool_call 修正最多 2 次
```

达到预算后：

- `retryable_by_model = false`
- `allowed_next_actions` 改为：
  - `ask_user_for_missing_required_fields`
  - `degrade_with_user_friendly_error`
- 本 run 内可标记该工具暂不可继续盲目调用。

## 7. 工具执行失败设计

### 7.1 执行流程

工具参数校验通过后：

```text
TOOL_CALL_STARTED
  -> 执行 handler with timeout
  -> 成功：TOOL_CALL_COMPLETED
  -> 失败：进入异常翻译和重试逻辑
```

### 7.2 runtime 自动重试

runtime 自动重试只适用于：

- 工具是幂等的：`policy.idempotent == true`
- 工具允许重试：`policy.max_retries > 0`
- 错误属于瞬时失败或可能因重试恢复：
  - timeout
  - 网络连接错误
  - 上游 429
  - 上游 502 / 503 / 504
  - 可识别的临时不可用

runtime 不自动重试：

- 参数校验失败。
- policy deny。
- 权限错误。
- 认证错误。
- 非幂等工具。
- 明确业务拒绝。
- 敏感信息风险。

### 7.3 backoff + jitter

runtime retry 不要立即连续打上游。

建议：

```text
第 1 次重试：等待 100ms - 300ms
第 2 次重试：等待 300ms - 700ms
后续按指数退避，但 10C 不需要支持很多次
```

`jitter` 是随机抖动，用来避免大量请求同时重试打爆上游。

### 7.4 timeout

`policy.timeout_seconds` 必须实际生效。

建议：

- async handler 用 `asyncio.wait_for`。
- sync handler 用 `asyncio.to_thread` 包装后再 `wait_for`。

限制：

- Python 线程里的同步阻塞函数即使 wait_for 超时，底层线程可能仍在跑。
- 10C 要记录 `TOOL_CALL_TIMEOUT`。
- 如果后续拿到迟到结果，不应再写入主链路，可记录 `TOOL_CALL_LATE_RESULT_DISCARDED`。
- P0 可先不实现迟到结果捕捉，只在文档中保留该事件用途。

## 8. 异常翻译器设计

新增一个轻量组件，例如：

```text
ToolErrorTranslator
```

职责：

```text
输入：
  tool_name
  exception or validation error
  tool policy
  attempt state
  run failure state

输出：
  ToolErrorResult dict
```

### 8.1 粗粒度错误类型

错误类型不追求穷尽，只做可治理大类：

| error_type | 来源示例 | 处理语义 |
|---|---|---|
| `PARAM_VALIDATION_FAILED` | Pydantic `ValidationError` | 模型修参数或追问用户 |
| `TOOL_BLOCKED` | unknown tool / policy deny | 不重试 |
| `TOOL_TIMEOUT` | `TimeoutError` / `asyncio.TimeoutError` | 幂等工具可 runtime retry；耗尽后模型缩小范围或换路径 |
| `TOOL_RETRY_EXHAUSTED` | runtime retry 用完 | 本次 run 不再自动重试 |
| `UPSTREAM_UNAVAILABLE` | 429 / 502 / 503 / 504 / connection error | 幂等工具可 runtime retry；耗尽后换路径 |
| `PERMISSION_DENIED` | 401 / 403 / `PermissionError` | 不重试，提示无权限 |
| `TOOL_ERROR` | 其他未知异常 | 脱敏后返回，模型最多谨慎换条件或换路径 |

说明：

- `UNKNOWN_ERROR` 可选；也可以继续使用现有 `TOOL_ERROR` 表达未知工具异常。
- 错误分类不是为了穷举异常，而是为了确定处理语义。

### 8.2 message 生成

`message` 应该：

- 面向模型可读。
- 描述当前工具失败事实。
- 不暴露堆栈。
- 不暴露内部路径。
- 不暴露密钥、token、手机号、身份证等敏感信息。
- 不把未验证猜测写成事实。

示例：

```json
{
  "message": "queryLogs 工具执行超时。可能是查询范围过大，或者日志系统响应较慢。"
}
```

### 8.3 allowed_next_actions 生成

后端不需要为所有异常写死复杂策略，但要给模型划定范围。

示例：

timeout：

```json
[
  "retry_same_tool_with_narrower_scope",
  "use_alternative_tool",
  "search_memory_for_historical_reference",
  "search_rag_for_runbook",
  "degrade_with_unconfirmed_items"
]
```

permission denied：

```json
[
  "degrade_with_user_friendly_error"
]
```

unknown tool error：

```json
[
  "retry_same_tool_once_with_clearer_arguments",
  "use_alternative_tool",
  "search_memory_for_historical_reference",
  "search_rag_for_runbook",
  "degrade_with_unconfirmed_items"
]
```

### 8.4 disallowed_next_actions 生成

示例：

```json
[
  "do_not_expose_raw_exception",
  "do_not_claim_root_cause_without_evidence",
  "do_not_retry_after_budget_exhausted"
]
```

## 9. 本 run 内失败治理

需要在 `ToolGateway` 或其辅助状态中维护本次 run 的工具失败计数。

建议状态粒度：

```text
run_id + tool_name
```

记录：

- validation failure count
- timeout count
- upstream failure count
- unknown failure count
- total failure count

建议默认阈值：

```text
同一 run 同一工具连续失败 >= 3：
  本 run 内停止同一失败模式的盲目重试
```

达到治理阈值后如果模型继续调用：

- 如果仍是同一类失败模式，例如同样缺字段、同样超时、同样权限错误，不执行 handler。
- 返回 `TOOL_BLOCKED` 或 `TOOL_RETRY_EXHAUSTED`。
- `message` 说明该工具在本次 run 内多次失败，已停止继续盲目调用。
- `allowed_next_actions` 引导：
  - 使用替代工具。
  - 查长期记忆。
  - 查 RAG。
  - 向用户说明证据不足。

说明：

- 这是本 run 内临时治理，不是全局熔断。
- 不能影响其他用户、其他 run、其他租户。
- 如果模型拿到了用户补充信息，或者调用参数发生了实质变化，可以允许重新尝试；这个判断可以在 10C P0 中先保守实现为“达到阈值后只返回追问或降级建议”，后续再增强为参数指纹级判断。

## 10. 降级路径设计

当工具不可用或达到预算后，模型允许的降级路径包括：

```text
1. 使用替代工具。
2. 查询长期记忆，获取历史经验或用户稳定背景。
3. 查询 RAG，获取 runbook / 文档 / 操作手册。
4. 追问用户补充必要信息。
5. 用户友好错误说明。
```

约束：

- 长期记忆只能作为历史参考，不能当作当前实时证据。
- RAG 文档只能提供排障方法和知识依据，不能替代实时日志/告警证据。
- 如果实时工具失败，最终回答必须明确“哪些证据没有拿到”。
- 不允许编造已经查询成功的日志、告警、指标。

## 11. Trace 设计

本阶段使用已有事件类型。

### 11.1 TOOL_CALL_STARTED

payload 增强：

```json
{
  "toolName": "queryLogs",
  "arguments": {},
  "timeoutSeconds": 8,
  "maxRetries": 1,
  "idempotent": true
}
```

### 11.2 TOOL_CALL_RETRY

每次 runtime 自动重试记录：

```json
{
  "toolName": "queryLogs",
  "attempt": 1,
  "maxRetries": 1,
  "errorType": "TOOL_TIMEOUT",
  "backoffMs": 200,
  "status": "retrying"
}
```

### 11.3 TOOL_CALL_TIMEOUT

最终或单次 timeout 记录：

```json
{
  "toolName": "queryLogs",
  "attempt": 1,
  "timeoutSeconds": 8,
  "status": "timeout"
}
```

### 11.4 TOOL_CALL_FAILED

最终失败记录：

```json
{
  "toolName": "queryLogs",
  "errorType": "TOOL_RETRY_EXHAUSTED",
  "result": {},
  "attempts": 2,
  "status": "error"
}
```

### 11.5 TOOL_CALL_LATE_RESULT_DISCARDED

本阶段可不强制实现。

用途：

- 同步阻塞工具被 timeout 后，底层线程迟到返回。
- 迟到结果不得进入主链路。
- 如能捕捉，记录该事件。

## 12. 需要修改的文件

建议新增：

```text
src/superbiz_agent/tools/error_translator.py
src/superbiz_agent/tools/retry.py
tests/test_tool_gateway_reliability.py
```

建议修改：

```text
src/superbiz_agent/tools/errors.py
src/superbiz_agent/tools/gateway.py
src/superbiz_agent/tools/policies.py
tests/test_business_tools.py
tests/test_skeleton.py
docs/04-python-migration-roadmap.md
```

说明：

- `policies.py` 如果当前字段足够，可以不改。
- `events.py` 已有事件类型，原则上不需要新增枚举。
- 测试只应适配增强后的错误结构，不应降低已有断言。

## 13. 测试计划

### 13.1 参数校验

覆盖：

- 缺少必填字段。
- 枚举值错误。
- 数值范围错误。
- 字符串最小长度错误。

断言：

- 不执行 handler。
- 不 runtime retry。
- 返回 `PARAM_VALIDATION_FAILED`。
- 返回 `violations`。
- `allowed_next_actions` 包含修正参数和追问用户。
- trace 记录 `TOOL_CALL_FAILED`。

### 13.2 工具 timeout

构造慢工具。

断言：

- 超过 `timeout_seconds` 后返回错误。
- 记录 `TOOL_CALL_TIMEOUT`。
- 幂等工具可按 `max_retries` 重试。
- 非幂等工具不重试。

### 13.3 幂等 retry

构造前一次失败、第二次成功的幂等工具。

断言：

- 记录 `TOOL_CALL_RETRY`。
- 最终 `TOOL_CALL_COMPLETED`。
- 返回成功结果。

### 13.4 retry 耗尽

构造一直失败的幂等工具。

断言：

- 重试次数不超过 `policy.max_retries`。
- 最终返回结构化错误。
- error_type 可为 `TOOL_RETRY_EXHAUSTED` 或原始大类 + retry exhausted reason。
- trace 有 retry 和 failed。

### 13.5 非幂等工具

构造非幂等工具失败。

断言：

- 不做 runtime retry。
- 直接返回模型可见错误。

### 13.6 本 run 内失败上限

同一 run 连续调用同一失败工具。

断言：

- 达到阈值后不再执行同一失败模式的 handler。
- 返回本 run 内停止盲目重试提示。
- 不影响另一个 run。

### 13.7 脱敏

构造异常 message 包含：

- `api_key`
- `token`
- `password`
- 内部路径

断言：

- 返回给模型的 `message` 不包含敏感原文。
- trace result 中也不包含敏感原文。

## 14. 验收命令

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

## 15. 不做清单

本阶段不做：

- 不整体引入 PydanticAI。
- 不替换 LangGraph。
- 不重写 Agent ReAct loop。
- 不实现完整 tool-call repair 框架。
- 不做全局熔断中心。
- 不做 RAG 管线改造。
- 不改长期记忆设计。
- 不做上下文压缩。
- 不接 LangSmith。
- 不做生产级鉴权。
- 不改变系统提示词主结构。
- 不强制约束模型最终自然语言答案格式。

## 16. subAgent 实施边界

如果交给 subAgent，实现边界必须明确：

允许写：

```text
src/superbiz_agent/tools/errors.py
src/superbiz_agent/tools/error_translator.py
src/superbiz_agent/tools/retry.py
src/superbiz_agent/tools/gateway.py
tests/test_tool_gateway_reliability.py
tests/test_business_tools.py
tests/test_skeleton.py
```

原则上不写：

```text
src/superbiz_agent/model_gateway/
src/superbiz_agent/memory/
src/superbiz_agent/rag/
src/superbiz_agent/harness/graph.py
src/superbiz_agent/api/
```

除非测试发现必须小幅适配。

## 17. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否直接替换当前 Harness | 否 |
| 是否整体引入 PydanticAI | 否 |
| 是否参考 PydanticAI 的字段级校验反馈 | 是 |
| 是否把所有异常完全交给模型自由处理 | 否 |
| 是否要求穷尽所有错误类型 | 否 |
| 是否固定 ToolErrorResult schema | 是 |
| 是否允许 runtime 和 model 两类重试预算 | 是 |
| 是否只对幂等工具做 runtime retry | 是 |
| 是否保留 trace / eval 可观测性 | 是 |
| 是否避免修改 RAG / Memory / ModelGateway | 是 |

## 18. 阶段通过标准

10C 通过标准：

1. `ToolErrorResult` 支持字段级 violations 和 allowed/disallowed actions。
2. 参数校验失败能让模型知道具体哪个字段错、为什么错、如何修。
3. `timeout_seconds` 实际生效。
4. 幂等工具按 `max_retries` 做 runtime retry。
5. 非幂等工具不自动重试。
6. retry / timeout / failed trace 可回放。
7. 同一 run 内重复失败不会无限循环。
8. 工具错误 message 和 trace result 做脱敏。
9. 原有成功工具调用不回退。
10. 全量测试、eval runner、compileall 通过。
