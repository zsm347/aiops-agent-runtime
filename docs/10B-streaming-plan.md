# 10B Streaming / chat_stream 实施计划

## 1. 阶段定位

本文是第 10 步高级能力中的第二个代码子阶段：`10B Streaming / chat_stream`。

本阶段目标是在不破坏现有 `/api/chat` 非流式链路的前提下，新增 Java 兼容的 SSE 流式接口：

```text
POST /api/chat_stream
```

本阶段只做流式响应工程闭环，不做 ToolGateway retry、不做上下文压缩、不做 RAG 真实管线、不做 LangSmith、不做复杂多工具流式编排。

## 2. 当前基线

已完成：

- `/api/chat` 非流式接口。
- `SseMessage` schema。
- `AgentHarnessService.chat()`。
- `SkeletonAgentGraph.run()` 返回最终答案。
- `ModelGateway.complete(messages, tools=None)`。
- `StubModelGateway` deterministic 非流式。
- `OpenAICompatibleModelGateway` 非流式 complete。
- rollout trace 完整记录 run/model/tool/final events。

当前没有：

- `/api/chat_stream` route。
- `AgentHarnessService.chat_stream()`。
- `SkeletonAgentGraph.run_stream()`。
- `ModelGateway.stream()`。
- SSE 编码工具。
- 流式测试。

## 3. 依据

依据：

- `docs/02-python-migration-spec.md`：
  - `POST /api/chat_stream`
  - SSE event name 为 `message`
  - payload 类型：`content | retrying | final | error | done`
- `docs/03-python-architecture-design.md`：
  - FastAPI 负责 API/SSE。
  - ModelGateway adapter 负责非流式/流式统一接口。
  - Harness 保留 run/context/tool/trace。
- `docs/10-advanced-capabilities-plan.md`：
  - 10B 是 streaming 子阶段。
  - 10B 需要保持 `/api/chat` 不回退。

## 4. 推荐设计

### 4.1 流式事件结构

内部定义 `AgentStreamEvent` 或复用 `SseMessage` 语义：

```json
{
  "type": "content | retrying | final | error | done",
  "data": {}
}
```

SSE 输出格式：

```text
event: message
data: {"type":"content","data":"..."}

event: message
data: {"type":"final","data":{"data":"完整答案"}}

event: message
data: {"type":"done","data":null}
```

### 4.2 最小流式策略

本阶段采用两层策略：

1. 无工具调用的普通回答：
   - 如果 ModelGateway 支持 `stream()`，使用真实模型 token/chunk 流式输出。
   - StubModelGateway 提供 deterministic chunk 流。

2. 涉及工具调用的回答：
   - 本阶段不做工具调用阶段逐 token 输出。
   - 仍复用现有 Graph：模型选择工具 -> ToolGateway 执行 -> 模型生成最终答案。
   - 工具链路完成后，把最终答案按 chunk 输出为 `content` 事件。

这样做的原因：

- 保留当前可靠 trace 和 ToolGateway 治理。
- 避免 10B 同时重写 ReAct loop。
- 为后续 10C/10D 留出更细的模型/工具事件治理空间。

### 4.3 ModelGateway stream interface

建议扩展 `ModelGateway`：

```python
async def stream(
    self,
    messages: list[ModelMessage],
    tools: list[ToolDefinition] | None = None,
) -> AsyncIterator[ModelStreamChunk]:
    ...
```

新增 `ModelStreamChunk`：

```python
@dataclass(frozen=True)
class ModelStreamChunk:
    content_delta: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)
```

说明：

- 10B 可以只使用 `content_delta`。
- 如果 provider 返回 streaming tool calls，本阶段可以先聚合或忽略为后续工作；不要在 10B 实现复杂 tool-call delta 合并。
- `StubModelGateway.stream()` 可通过 `complete()` 得到最终内容后按固定大小切 chunk。

### 4.4 Graph streaming

新增 `SkeletonAgentGraph.run_stream()`：

```text
输入 run_context + assembled messages
输出 async iterator of AgentStreamEvent
```

推荐实现：

- 如果第一轮模型流式直接产出 content，无工具调用：
  - 记录 `MODEL_CALL_STARTED`
  - 持续 yield `content`
  - 记录 `MODEL_CALL_COMPLETED`
  - yield `final`
- 如果模型需要工具调用：
  - 可以先调用现有 `run()` 得到最终答案。
  - 再把最终答案分 chunk yield。
  - trace 使用现有 run() 产生的 model/tool events。

为了避免本阶段判断 streaming tool call delta，最小可接受实现是：

- `run_stream()` 内部调用现有 `run()` 得到最终答案。
- 然后把最终答案按 chunk 输出。

但如果这样做，则要在文档中明确：

- 这是 API 层流式输出，不是 provider token 级 streaming。
- 10B.2 再增强为真正 provider stream。

本项目更推荐折中实现：

- `StubModelGateway.stream()` 可用。
- `OpenAICompatibleModelGateway.stream()` 可以用 fake client 测普通 content chunk。
- 但 `SkeletonAgentGraph.run_stream()` 如遇工具调用仍可 fallback 到 `run()`。

10B 的强制交付边界：

- 必须实现 `/api/chat_stream`，并输出合法 SSE。
- 必须保证 trace 与非流式主链路一致。
- 必须保证工具场景可用，即使工具场景是先完成现有 `run()` 再把最终答案切 chunk 输出。
- 可以实现 provider stream interface，但不要求在 10B 内完成复杂 streaming tool-call delta 合并。
- 如果 provider stream 接入复杂度影响主线，以 API 层流式输出作为 10B.1 验收目标，provider token 级 streaming 作为 10B.2 增强。

### 4.5 Service streaming

新增：

```python
AgentHarnessService.chat_stream(context, question) -> AsyncIterator[SseMessage]
```

职责：

- 空问题直接 yield：
  - `error`
  - `done`
- 正常问题：
  - acquire conversation lock。
  - start run。
  - append user message。
  - assemble context。
  - 调用 graph streaming。
  - 聚合完整 answer。
  - append assistant message。
  - complete run。
  - yield final。
  - yield done。
- 发生异常：
  - fail run。
  - yield error。
  - yield done。

注意：

- 非流式 `chat()` 不应改为依赖 streaming。
- streaming 和非流式可以共享内部 helper，但不能牺牲现有稳定性。

### 4.6 API route

新增：

```text
POST /api/chat_stream
```

返回 `StreamingResponse`：

```python
StreamingResponse(generator, media_type="text/event-stream")
```

每条 SSE：

```text
event: message
data: <json>

```

要求：

- JSON 使用 `ensure_ascii=False`。
- `done` 必须总是最后输出。
- 失败也要输出 `done`。

## 5. Trace 设计

本阶段必须保留现有 trace：

- `RUN_STARTED`
- `USER_MESSAGE_APPENDED`
- `CONTEXT_ASSEMBLED`
- `MEMORY_INJECTED`
- `MODEL_CALL_STARTED`
- `MODEL_CALL_COMPLETED`
- `MODEL_CALL_FAILED`
- `TOOL_CALL_*`
- `ASSISTANT_MESSAGE_APPENDED`
- `RUN_COMPLETED`
- `RUN_FAILED`

可选新增 payload 字段：

- `streaming: true`
- `streamChunkCount`
- `answerLength`

本阶段不强制新增独立 `STREAM_*` 事件，避免扩大事件枚举；如果后续需要前端/观测分析，再单独立项。

## 6. 测试计划

新增测试建议：

1. `test_chat_stream_empty_question_returns_error_and_done`
   - POST `/api/chat_stream` 空问题。
   - 断言 SSE 包含 `error`、`done`。
   - 不产生 run events。

2. `test_chat_stream_plain_chat_emits_content_final_done`
   - 输入 `hello`。
   - 断言 SSE 顺序包含：
     - `content`
     - `final`
     - `done`
   - final data 是完整答案。

3. `test_chat_stream_tool_question_preserves_trace`
   - 输入 `现在几点？`。
   - 断言最终答案包含 `Asia/Shanghai`。
   - trace 包含 tool call events。

4. `test_chat_stream_failure_emits_error_done`
   - 使用 fake service / fake graph 抛异常。
   - 断言 `error`、`done`，并有 `RUN_FAILED`。

5. `test_stub_model_gateway_stream_chunks`
   - 直接测试 stub stream。

6. `test_openai_compatible_gateway_stream_parses_content_chunks`
   - 使用 fake client，不访问外网。
   - 只测普通 content chunk。

原有测试必须继续通过：

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
PYTHONPATH=src python3 -m compileall -q src tests
```

## 7. 不做项

本阶段明确不做：

- 不做复杂 streaming tool-call delta 合并。
- 不做多工具并发流式。
- 不做前端页面。
- 不做 ToolGateway timeout/retry。
- 不做模型 retry/backoff 完整实现。
- 不做上下文压缩。
- 不做 RAG 真实管线。
- 不做 LangSmith。
- 不做生产认证。

## 8. subAgent 任务边界

如果交给 subAgent，任务应限定为：

```text
只实现 docs/10B-streaming-plan.md 定义的 Streaming / chat_stream。
不得实现 ToolGateway retry、上下文压缩、RAG 真实管线、LangSmith、生产认证、长期记忆改造。
必须保持 /api/chat 非流式接口和现有 eval 不回退。
真实 provider streaming 测试必须使用 fake client，不访问外网。
完成后运行 python3 -m pytest、eval runner、compileall；ruff 如未安装说明即可。
```

## 9. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否新增 Java 兼容 `/api/chat_stream` | 是 |
| 是否保持 `/api/chat` 不回退 | 是 |
| 是否明确 SSE event name 和 payload | 是 |
| 是否默认不依赖真实 provider | 是 |
| 是否避免复杂 tool-call delta 合并 | 是 |
| 是否未提前做 ToolGateway retry | 是 |
| 是否未提前做上下文压缩 | 是 |
| 是否有明确测试计划 | 是 |
