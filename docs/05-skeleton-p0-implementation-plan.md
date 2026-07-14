# Skeleton P0 实施计划

## 1. 阶段定位

本文是 Python 迁移路线第 5 步 `Skeleton P0` 的实施计划。

本阶段目标不是完整迁移 Java Harness，也不是接真实模型，而是把当前 Python 骨架从简单 echo stub 升级为一个可测试的最小 Agent Harness 闭环：

```text
chat -> context -> model stub -> tool call -> answer -> trace
```

本阶段完成后，系统应该能证明：

- API 能进入 Harness，而不是直接返回 `[stub] question`。
- 每次请求能生成 `run_id`。
- 能按固定顺序组装最小模型上下文。
- 能通过 `StubModelGateway` 产生确定性工具调用。
- 工具调用必须经过 `ToolGateway`。
- 能调用一个内置工具 `getCurrentDateTime`。
- 能记录内存 trace events。
- 最终答案来自模型 stub 对工具结果的二次处理。

## 2. 依据

本计划只依据以下文档和当前代码，不自由扩展范围：

- `docs/02-python-migration-spec.md`
- `docs/03-python-architecture-design.md`
- `docs/04-python-migration-roadmap.md`
- 当前 Python 骨架：
  - `src/superbiz_agent/api/`
  - `src/superbiz_agent/harness/`
  - `src/superbiz_agent/model_gateway/`
  - `src/superbiz_agent/tools/`
  - `src/superbiz_agent/prompts/`
  - `tests/test_skeleton.py`

阶段边界以 `docs/04-python-migration-roadmap.md` 为准。

## 3. 本阶段做什么

### 3.1 API 行为升级

当前 `/api/chat` 调用 `AgentHarnessService.placeholder()`，返回：

```text
[stub] {question}
```

Skeleton P0 要改成：

```text
/api/chat
  -> resolve_request_context
  -> AgentHarnessService.chat
  -> ConversationRuntime.start_run
  -> ContextAssembler.assemble
  -> SkeletonAgentGraph / LangGraph adapter
  -> StubModelGateway
  -> ToolGateway
  -> getCurrentDateTime
  -> StubModelGateway final answer
  -> ConversationRuntime.complete_run
```

服务构造要求：

- 不在每个 request handler 内部临时 new 一个完整 service。
- `create_app()` 或 route dependency 应创建一个可复用的 `AgentHarnessService`，并放在 `app.state` 或通过 FastAPI dependency 注入。
- 测试可以拿到同一个 service 或同一个 `InMemoryRolloutEventStore` 来检查 trace。
- 不使用不可控的模块级全局可变单例承载租户数据。

### 3.2 最小上下文组装

新增最小 `ContextAssembler`，按以下顺序组装 `ModelMessage`：

```text
system prompt
active history
current user message
```

说明：

- `system prompt` 从 `PromptRegistry` 加载，默认版本来自 settings：`ops-agent-system-v2`。
- Skeleton P0 暂不注入 `core_memory`。
- Skeleton P0 暂不注入 `memory_index`。
- Skeleton P0 暂不做 token 精确计数。
- `CONTEXT_ASSEMBLED` 事件至少记录：
  - `promptVersion`
  - `messageCount`
  - `estimatedChars`
  - `hasCoreMemory=false`
  - `hasMemoryIndex=false`

### 3.3 最小 Agent 编排

本阶段应尊重架构 ADR 中的“LangGraph 优先”原则。

推荐实现一个很薄的 `SkeletonAgentGraph`：

```text
model_call node
  -> 如果模型返回 tool_call，进入 tool_dispatch node
  -> 如果模型返回 final answer，进入 finalization

tool_dispatch node
  -> 调用 ToolGateway
  -> 把 tool result 追加给下一次 model_call

finalization
  -> 返回最终 answer
```

实现要求：

- 可以使用 LangGraph 的最小 `StateGraph`。
- 只使用 LangGraph 做最小 graph stepping，不启用 checkpoint、persistence、streaming、memory 等高级能力。
- 不实现通用 ReAct 框架。
- 不实现复杂循环控制。
- 本阶段最多允许一次工具调用，再进行一次 final model call。
- 如果 LangGraph 接入出现版本 API 问题，subAgent 必须停止并报告，不允许私自改成完整自研 ReAct loop。
- 如果本地环境缺少 LangGraph，但 `pyproject.toml` 已声明依赖，应先按项目依赖安装或报告环境问题，不应因此把长期架构改成自研 loop。

### 3.4 StubModelGateway

扩展 `ModelGateway` 数据模型，使它能表达工具调用：

```text
ModelMessage(role, content, name?)
ModelToolCall(id, name, arguments)
ModelResponse(content, tool_calls, raw)
```

`StubModelGateway` 行为必须确定性：

- 如果最新 user message 包含时间/日期类意图，例如：
  - `时间`
  - `日期`
  - `几点`
  - `time`
  - `date`
- 且当前 messages 中还没有 `getCurrentDateTime` 的 tool result，则返回一个 tool call：

```json
{
  "name": "getCurrentDateTime",
  "arguments": {
    "timezone": "Asia/Shanghai"
  }
}
```

- 如果 messages 中已经存在 `getCurrentDateTime` 的 tool result，则生成最终自然语言答案。
- 非时间类问题可以返回一个确定性 stub answer，但也必须经过 `MODEL_CALL_STARTED` 和 `MODEL_CALL_COMPLETED` trace。

注意：

- `run_id`、`tenant_id`、`user_id`、`agent_id` 不能作为模型可见工具参数。
- StubModelGateway 不调用真实 Qwen。
- StubModelGateway 不接 OpenAI-compatible SDK。

### 3.5 最小 ToolRegistry

新增 `ToolRegistry`，注册一个工具：

```text
getCurrentDateTime
```

工具定义包含：

- name
- description
- Pydantic args model
- handler
- ToolPolicy

建议 args model：

```text
timezone: str = "Asia/Shanghai"
```

工具输出使用 JSON dict：

```json
{
  "success": true,
  "data": {
    "timezone": "Asia/Shanghai",
    "isoTime": "...",
    "epochMillis": 0
  }
}
```

### 3.6 最小 ToolGateway

新增 `ToolGateway`，只实现 Skeleton P0 所需能力：

- 根据 tool name 从 `ToolRegistry` 找工具。
- unknown tool 返回结构化错误。
- 用 Pydantic 校验 arguments。
- 执行前记录 `TOOL_CALL_STARTED`。
- 执行成功记录 `TOOL_CALL_COMPLETED`。
- 参数错误记录 `TOOL_CALL_FAILED`。
- policy deny 记录 `TOOL_CALL_BLOCKED`。
- timeout/retry 只保留接口和 policy 字段，不做完整实现。

工具错误结构对齐迁移规格：

```json
{
  "success": false,
  "error_type": "PARAM_VALIDATION_FAILED | TOOL_BLOCKED | TOOL_ERROR",
  "message": "...",
  "suggestion": "..."
}
```

本阶段不做：

- 自动重试。
- backoff/jitter。
- circuit breaker。
- 复杂异常分类。
- executor pool。

### 3.7 最小 ConversationRuntime

新增 `ConversationRuntime`，只做本阶段最小职责：

- 创建 `RunContext`。
- 生成 `run_id`。
- 记录 `RUN_STARTED`。
- 记录 `USER_MESSAGE_APPENDED`。
- 调用 `ContextAssembler`。
- 维护本次 run 内 active history。
- 记录 `ASSISTANT_MESSAGE_APPENDED`。
- 记录 `RUN_COMPLETED`。
- 异常时记录 `RUN_FAILED`。

本阶段 active history 只在当前 run 内存中存在。

不做：

- 从 PostgreSQL 恢复历史。
- 事件 replay。
- 会话级分布式锁。
- 跨请求短期记忆。
- 历史压缩。

### 3.8 InMemoryRolloutEventStore

新增内存 trace store，用于测试和本地闭环：

```text
append(event)
list_by_run(run_id)
list_by_session(context)
clear_session(context)
```

事件模型需要补齐迁移规格里的最小字段：

- `sequence`
- `occurred_at`

要求：

- `sequence` 在内存 store 中递增。
- event 必须包含 `tenant_id/user_id/agent_id/session_id/run_id`。
- API response 不直接返回 trace。
- 单元测试可以直接通过 service/store 检查 trace。

## 4. 本阶段不做什么

Skeleton P0 明确不做：

- 不接真实 Qwen。
- 不接 OpenAI-compatible SDK。
- 不接 LiteLLM。
- 不接 DashScope SDK。
- 不接 PostgreSQL event store。
- 不做事件 replay。
- 不做 `/api/chat_stream`。
- 不做上下文压缩。
- 不做真实 RAG。
- 不接 LlamaIndex/Milvus。
- 不做长期记忆。
- 不做 Core Memory / Archival Memory。
- 不做 LangSmith。
- 不做完整 eval runner。
- 不做生产级多租户鉴权。

## 5. 文件级实施范围

### 5.1 可能新增文件

```text
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/harness/context_assembler.py
src/superbiz_agent/harness/graph.py
src/superbiz_agent/harness/trace_store.py
src/superbiz_agent/model_gateway/stub.py
src/superbiz_agent/tools/registry.py
src/superbiz_agent/tools/gateway.py
src/superbiz_agent/tools/builtin/__init__.py
src/superbiz_agent/tools/builtin/datetime_tool.py
```

### 5.2 需要修改文件

```text
src/superbiz_agent/api/routes_chat.py
src/superbiz_agent/api/app.py
src/superbiz_agent/harness/service.py
src/superbiz_agent/harness/context.py
src/superbiz_agent/harness/events.py
src/superbiz_agent/model_gateway/base.py
src/superbiz_agent/tools/errors.py
tests/test_skeleton.py
```

### 5.3 不应修改文件

```text
prompts/ops-agent-system-v1.md
prompts/ops-agent-system-v2.md
docs/01-java-capability-inventory.md
docs/02-python-migration-spec.md
docs/03-python-architecture-design.md
docs/04-python-migration-roadmap.md
```

除非发现文档和代码存在明确冲突，否则本阶段不修改前置设计文档。

## 6. 目标运行流程

时间类问题示例：

```text
用户：现在几点？
```

目标内部流程：

```text
1. API 解析 AgentRequestContext
2. AgentHarnessService 创建 run_id
3. runtime 记录 RUN_STARTED
4. runtime 记录 USER_MESSAGE_APPENDED
5. ContextAssembler 组装 system prompt + current user message
6. runtime 记录 CONTEXT_ASSEMBLED
7. graph 调用 StubModelGateway
8. runtime 记录 MODEL_CALL_STARTED
9. StubModelGateway 返回 getCurrentDateTime tool_call
10. runtime 记录 MODEL_CALL_COMPLETED
11. ToolGateway 校验参数并调用 getCurrentDateTime
12. runtime 记录 TOOL_CALL_STARTED / TOOL_CALL_COMPLETED
13. graph 把 tool result 交回 StubModelGateway
14. runtime 记录第二次 MODEL_CALL_STARTED / MODEL_CALL_COMPLETED
15. StubModelGateway 返回最终答案
16. runtime 记录 ASSISTANT_MESSAGE_APPENDED
17. runtime 记录 RUN_COMPLETED
18. API 返回 Java 兼容响应
```

目标响应结构仍然是：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "success": true,
    "answer": "...",
    "errorMessage": null
  }
}
```

## 7. Trace 验收要求

时间类问题至少产生以下事件：

```text
RUN_STARTED
USER_MESSAGE_APPENDED
CONTEXT_ASSEMBLED
MODEL_CALL_STARTED
MODEL_CALL_COMPLETED
TOOL_CALL_STARTED
TOOL_CALL_COMPLETED
MODEL_CALL_STARTED
MODEL_CALL_COMPLETED
ASSISTANT_MESSAGE_APPENDED
RUN_COMPLETED
```

每条事件至少满足：

- `tenant_id` 不为空。
- `user_id` 不为空。
- `agent_id` 不为空。
- `session_id` 不为空。
- `run_id` 不为空，除非事件类型未来明确允许为空。
- `event_id` 不为空。
- `sequence` 单调递增。
- `occurred_at` 不为空。
- `payload` 是 dict。

## 8. 测试计划

### 8.1 API 测试

更新或新增测试：

- `/api/chat` 时间类问题返回 `success=true`。
- answer 中包含时间工具结果的关键信息，例如 `Asia/Shanghai` 或 `当前时间`。
- 空问题仍返回 `success=false`，错误信息为 `问题内容不能为空`。
- request headers 中的 tenant/user/agent 能进入 trace。
- `create_app()` 使用同一个可检查的 HarnessService / trace store，避免 API 测试无法验收 trace。

### 8.2 ContextAssembler 测试

新增测试：

- prompt version 能加载。
- messages 顺序是 system -> user。
- `CONTEXT_ASSEMBLED` payload 包含 prompt version 和估算大小。

### 8.3 StubModelGateway 测试

新增测试：

- 时间类问题第一次 model call 返回 `getCurrentDateTime` tool call。
- 已存在 tool result 后返回 final answer。
- 非时间类问题不调用工具，也能返回 deterministic answer。

### 8.4 ToolGateway 测试

新增测试：

- `getCurrentDateTime` 正常执行。
- unknown tool 返回 `TOOL_BLOCKED` 或等价结构化错误。
- 参数校验失败返回 `PARAM_VALIDATION_FAILED`。
- 成功工具调用产生 `TOOL_CALL_STARTED` 和 `TOOL_CALL_COMPLETED`。

### 8.5 HarnessService 测试

新增测试：

- `AgentHarnessService.chat` 返回非空 `run_id`。
- trace 中存在 run/model/tool/assistant events。
- `clear(context)` 能清理内存 session trace。

## 9. 验收命令

subAgent 实施完成后必须先运行：

```bash
python3 -m pytest
```

如果本地支持 ruff，也建议运行：

```bash
python3 -m ruff check src tests
```

但 Skeleton P0 的硬性验收以 pytest 通过为准。

## 10. subAgent 实施任务单

交给 subAgent 的任务应限制为：

```text
只实现 docs/05-skeleton-p0-implementation-plan.md 定义的 Skeleton P0。
不得实现真实模型、PostgreSQL、RAG、长期记忆、streaming、eval runner。
实现完成后必须运行 python3 -m pytest。
提交结果时说明：
1. 修改了哪些文件。
2. 最小闭环怎么跑通。
3. trace 中有哪些事件。
4. 测试命令和结果。
5. 是否有偏离计划的地方。
```

如果实现过程中发现计划和当前代码冲突，subAgent 必须停止并报告冲突，不得扩大范围自行重设计。

## 11. 自审清单

本计划进入实现前需要检查：

| 检查项 | 结论 |
|---|---|
| 是否符合第 5 步 Skeleton P0 | 是 |
| 是否提前做 Migration P0 | 否 |
| 是否提前接 PostgreSQL event store | 否 |
| 是否提前做真实模型 provider | 否 |
| 是否提前做 RAG / LlamaIndex / Milvus | 否 |
| 是否提前做长期记忆 | 否 |
| 是否保留 prompt version | 是 |
| 是否保留 run_id 和 trace 语义 | 是 |
| 是否让工具调用经过 ToolGateway | 是 |
| 是否使用 Pydantic 做工具参数校验 | 是 |
| 是否遵守框架优先原则 | 基本遵守，要求用薄 LangGraph skeleton graph |
| 是否避免完整自研 ReAct 框架 | 是 |
| 测试是否覆盖最小闭环 | 是 |

## 12. 阶段完成标准

Skeleton P0 可以验收通过的条件：

1. `/api/chat` 不再只是 echo stub。
2. 时间类问题会触发 `getCurrentDateTime` 工具调用。
3. 工具调用经过 `ToolGateway`。
4. 最终 answer 来自工具结果后的 final model response。
5. 每次 chat 有 `run_id`。
6. 内存 trace 中能看到 run/model/tool events。
7. API 响应结构保持 Java 兼容。
8. `python3 -m pytest` 全部通过。
9. 没有实现本阶段明确排除的能力。
