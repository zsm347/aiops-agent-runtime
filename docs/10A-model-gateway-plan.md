# 10A 真实 ModelGateway 与模型调用治理实施计划

## 1. 阶段定位

本文是第 10 步高级能力中的第一个代码子阶段：`10A 真实 ModelGateway 与模型调用治理`。

本阶段只做真实模型 provider adapter 的工程闭环，不做 streaming、不做上下文压缩、不做 ToolGateway retry、不做 RAG 真实管线。

目标是让 Python Harness 在保持当前 deterministic stub 测试稳定的前提下，具备接入阿里 Qwen / DashScope OpenAI-compatible 接口的能力。

## 2. 当前问题

当前代码中：

- `ModelGateway` 只有 `complete(messages)`。
- `StubModelGateway` 是唯一实现。
- `AgentHarnessService.build_default()` 无条件创建 `StubModelGateway()`。
- `SkeletonAgentGraph` 负责记录 `MODEL_CALL_STARTED` 和 `MODEL_CALL_COMPLETED`，但没有 token usage、timeout、retry 的细分能力。
- settings 已有基础模型配置：
  - `model_provider`
  - `model_base_url`
  - `model_api_key`
  - `model_name`
  - `model_timeout_ms`
  - `model_max_retries`

因此，本阶段要做的是：

```text
settings
  -> model gateway factory
  -> StubModelGateway or OpenAICompatibleModelGateway
  -> graph model call trace
  -> tests with fake client
```

## 3. 设计原则

- 默认 provider 仍是 `stub`，保证本地测试和 eval 不依赖真实 API key。
- 真实 provider adapter 使用成熟 SDK，不自研 HTTP client。
- 真实 API key 只从环境变量 / settings 读取，不写入代码、测试或文档示例真实值。
- 测试用 fake client / mock response，不访问外网。
- 先支持非流式 complete；streaming 留给 10B。
- 保留现有 `ModelMessage`、`ModelToolCall`、`ModelResponse` 数据结构，除非确实需要小幅扩展。
- 真实 provider 的底层错误需要归一化成可测试的模型网关错误类型。

## 4. 实施范围

### 4.1 新增文件

建议新增：

```text
src/superbiz_agent/model_gateway/errors.py
src/superbiz_agent/model_gateway/openai_compatible.py
src/superbiz_agent/model_gateway/factory.py
tests/test_model_gateway_openai_compatible.py
```

### 4.2 修改文件

建议修改：

```text
src/superbiz_agent/model_gateway/base.py
src/superbiz_agent/harness/graph.py
src/superbiz_agent/harness/service.py
src/superbiz_agent/config.py
tests/test_skeleton.py
tests/test_eval_runner.py
```

说明：

- `config.py` 只在确实缺配置时补充；当前已有配置则少改。
- `tests/test_skeleton.py` / `tests/test_eval_runner.py` 只允许为 trace payload 小幅变化更新断言，不应降低原有覆盖。

## 5. ModelGateway 错误模型

建议定义：

```text
ModelGatewayError
ModelGatewayTimeoutError
ModelGatewayAuthError
ModelGatewayRateLimitError
ModelGatewayContextOverflowError
ModelGatewayProviderError
```

用途：

- `OpenAICompatibleModelGateway` 把 SDK 异常映射为这些错误。
- `SkeletonAgentGraph` 根据错误类型记录更准确的 event：
  - timeout -> `MODEL_CALL_TIMEOUT`
  - context overflow -> `MODEL_CALL_CONTEXT_OVERFLOW`
  - retry attempt -> `MODEL_CALL_RETRY`
  - final failure -> `MODEL_CALL_FAILED`

本阶段可以先做最小映射：

- `TimeoutError` / SDK timeout -> `ModelGatewayTimeoutError`
- 401/403 -> `ModelGatewayAuthError`
- 429 -> `ModelGatewayRateLimitError`
- context length / maximum context -> `ModelGatewayContextOverflowError`
- 其他 -> `ModelGatewayProviderError`

## 6. OpenAI-compatible adapter

### 6.1 依赖

使用 `openai` Python SDK。`pyproject.toml` 当前已有：

```text
openai>=1.40.0
```

不要新增自研 HTTP 调用。

### 6.2 构造参数

建议：

```python
OpenAICompatibleModelGateway(
    model_name: str,
    base_url: str | None,
    api_key: str,
    timeout_ms: int,
    max_retries: int,
    client: Any | None = None,
)
```

`client` 用于测试注入 fake client。

### 6.3 message 转换

把内部 `ModelMessage` 转为 OpenAI-compatible chat messages：

```json
{"role": "system|user|assistant|tool", "content": "..."}
```

如果 `message.role == "tool"` 且有 `name`，保留 name 或 tool_call_id 的兼容字段需要根据 SDK 支持决定。本阶段现有 graph 中 tool result 只用于第二次模型调用，stub 已可用；真实 provider 若需要严格 `tool_call_id`，可在 10A 先记录为已知限制，10B/10C 再完善完整 tool message schema。

### 6.4 tool schema 转换

本阶段有两种可选实现：

1. 10A 暂不传 tool schema 给 provider，只验证普通 completion。
2. 在 `ModelGateway.complete()` 增加可选 `tools` 参数，将 `ToolRegistry` schema 转换为 OpenAI tools。

取舍建议：

- 为了支持真实模型主动调用工具，最终必须有 `tools` 参数。
- 但当前 `SkeletonAgentGraph` 调用 `complete(messages)`，没有传 registry。
- 因此 10A 可以先做最小兼容：
  - 保持 protocol 的 `complete(messages)` 不破坏现有代码。
  - 在 adapter 内支持可选 tools 参数，但 graph 暂不传。
  - 或新增 `complete(messages, tools=None)`，并同步更新 stub 和 graph。

推荐做法：

```python
async def complete(
    self,
    messages: list[ModelMessage],
    tools: list[ToolDefinition] | None = None,
) -> ModelResponse:
    ...
```

同时：

- `StubModelGateway.complete()` 接受可选 tools，但可以不使用。
- `SkeletonAgentGraph` 把 `ToolGateway.registry.list()` 传给 ModelGateway。
- 真实 adapter 把 Pydantic schema 转为 OpenAI `tools`。

这样 10A 完成后，真实模型具备工具调用入口。

### 6.5 response 解析

需要解析：

- final content。
- tool calls：
  - id
  - function name
  - JSON arguments
- usage：
  - prompt tokens
  - completion tokens
  - total tokens
- finish reason。

如果 arguments 不是合法 JSON：

- 不在 adapter 中瞎修。
- 返回空 dict 或抛 provider parse error二选一。
- 推荐返回空 dict 并在 `raw` 中记录 parse failure，交给 ToolGateway 参数校验返回结构化错误。

## 7. Retry / timeout 边界

本阶段只做模型调用级最小治理：

- `max_retries` 控制 retry 次数。
- retry 只针对 timeout、rate limit、5xx / transient provider error。
- auth error、参数配置错误不 retry。
- 每次 retry 记录 `MODEL_CALL_RETRY`。
- 最终 timeout 记录 `MODEL_CALL_TIMEOUT`。
- 最终失败记录 `MODEL_CALL_FAILED`。

注意：

- `SkeletonAgentGraph` 当前负责记录 model events。
- 如果 retry 在 gateway 内部发生，gateway 需要拿到 trace_store/run_context，或 graph 需要包 retry loop。
- 为了不让 ModelGateway 依赖 trace，本阶段推荐把 retry loop 放在 `SkeletonAgentGraph._model_call` 外层，或增加一个 `TracedModelGateway` wrapper。

推荐最小实现：

- `OpenAICompatibleModelGateway` 只负责一次 SDK 调用和错误归类。
- `SkeletonAgentGraph` 根据 `run_context.model_provider` 和 gateway error 做 retry 和 trace。
- 这样 trace 仍集中在 Harness 层。

如果实现复杂度过高，10A 可以先只做：

- adapter 归类错误。
- graph 记录 failed/timeout。
- retry 留给 10A.2。

但如果跳过 retry，必须在文档和验收中明确。

## 8. Service provider factory

新增 factory：

```text
build_model_gateway(settings) -> ModelGateway
```

规则：

- `model_provider == "stub"` -> `StubModelGateway`
- `model_provider in ("openai-compatible", "qwen-openai-compatible", "dashscope-openai-compatible")`
  -> `OpenAICompatibleModelGateway`
- 真实 provider 缺少 `model_api_key` 时抛配置错误，但不能影响默认 stub。

`AgentHarnessService.build_default()` 使用 factory，不再直接写死 `StubModelGateway()`。

## 9. 测试计划

新增测试：

1. `test_model_gateway_factory_defaults_to_stub`
   - 默认 settings 返回 StubModelGateway。

2. `test_openai_compatible_gateway_parses_content`
   - fake client 返回普通 answer。
   - 断言 `ModelResponse.content`。

3. `test_openai_compatible_gateway_parses_tool_call`
   - fake client 返回 tool call。
   - 断言 tool name 和 arguments。

4. `test_openai_compatible_gateway_captures_usage`
   - fake response 带 usage。
   - 断言 `response.raw["usage"]` 或扩展字段。

5. `test_model_gateway_missing_api_key_fails_for_real_provider`
   - provider 是真实 adapter 且无 key，应该快速失败。

6. `test_graph_records_model_timeout_or_failed`
   - fake gateway 抛 timeout/provider error。
   - 断言 trace event。

原有测试必须继续通过：

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
PYTHONPATH=src python3 -m compileall -q src tests
```

## 10. 不做项

本阶段明确不做：

- 不做 `/api/chat_stream`。
- 不做 streaming model interface。
- 不做上下文压缩。
- 不做真实工具 timeout/retry。
- 不做 RAG 真实管线。
- 不接 LangSmith。
- 不改长期记忆逻辑。
- 不让测试依赖真实 API key 或外网。

## 11. subAgent 任务边界

如果交给 subAgent，任务应限定为：

```text
只实现 docs/10A-model-gateway-plan.md 定义的真实 ModelGateway 与模型调用治理。
不得实现 streaming、上下文压缩、ToolGateway retry、RAG 真实管线、LangSmith、长期记忆改造。
默认 provider 必须仍是 stub，现有 pytest/eval 不得退化。
真实 provider 测试必须使用 fake client，不访问外网，不需要真实 API key。
修改完成后运行 python3 -m pytest、eval runner、compileall；ruff 如未安装说明即可。
```

## 12. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否保持默认 stub | 是 |
| 是否避免真实 API key 成为测试依赖 | 是 |
| 是否优先使用 OpenAI SDK | 是 |
| 是否避免自研 HTTP client | 是 |
| 是否未提前做 streaming | 是 |
| 是否未提前做上下文压缩 | 是 |
| 是否未提前做 ToolGateway retry | 是 |
| 是否保留 LangGraph / Harness trace 责任 | 是 |
| 是否有明确测试计划 | 是 |

