# Python 迁移规格

## 1. 目的

本文定义把当前 Java 单 Agent Harness 重建为 Python 版本时必须遵守的迁移契约。

Python 版本追求的是“行为等价”，不是逐行翻译 Java 代码，也不是重新设计一个新的多 Agent 系统。

Java 参考实现：

```text
/Users/zsm/Documents/SuperBizAgent-release-2026-01-02
```

Python 目标工作区：

```text
/Users/zsm/Documents/SuperBizAgent-python
```

依据：

- `docs/01-java-capability-inventory.md`
- Java 源码：`src/main/java`
- Java 配置：`src/main/resources/application.yml`
- Java 提示词：`src/main/resources/prompts`
- Java 数据库迁移：`src/main/resources/db/migration`

## 2. 迁移目标

构建一个 Python 单 Agent Harness，保留 Java 版本的工程职责：

- 主聊天接口和会话接口 API 兼容。
- 后端拥有 request context，包含 tenant/user/agent/session/run 身份。
- 基于事件流的 conversation trace。
- 复用 LangGraph 承担 ReAct/graph 编排，项目保留薄 Harness wrapper。
- 复用 OpenAI-compatible SDK、LiteLLM 或 DashScope SDK 做底层模型访问，项目保留 ModelGateway adapter。
- 复用 Pydantic/PydanticAI 风格 schema 能力做工具参数校验，项目保留 ToolGateway policy/trace adapter。
- 上下文组装包含 system prompt、core memory、memory index、active history。
- 基于 PostgreSQL 恢复短期会话历史。
- RAG 编排优先使用 LlamaIndex + Milvus，项目保留 tenant filter、trace、citation、tool output adapter。
- 保留 core memory + archival memory 的长期记忆设计。
- 保留 prompt versioning。
- 评测优先复用 LangSmith 等成熟体系，同时保留本地 deterministic trace eval gate。

## 2.1 框架优先原则

Python 迁移的默认原则是：成熟框架优先，项目只自研薄适配层和必须自控的工程约束。

优先复用：

- Agent 编排：LangGraph。
- RAG 编排：LlamaIndex。
- 模型访问：OpenAI-compatible SDK、LiteLLM、DashScope SDK。
- 参数/schema 校验：Pydantic，后续可参考 PydanticAI。
- 评测和观测：本地 deterministic eval + 后续 LangSmith。

项目必须自己保留的部分：

- `tenant_id/user_id/agent_id/session_id/run_id` 的后端上下文控制。
- `agent_rollout_event` 事件流和 Java 兼容 trace schema。
- 工具执行 policy、tenant guard、错误归一化、审计事件。
- context assembly 顺序和 memory injection 契约。
- RAG/Memory 工具输出 JSON 结构、citation metadata 和 eval 兼容。

也就是说，不从零自研通用 Agent/RAG/Model/Eval 框架；只做项目适配层。

## 3. 不做什么

第一版 Python 不应悄悄扩大为：

- 多 Agent 编排。
- UI 重构。
- 完整生产级 SSO/JWT 认证。
- 接入所有真实企业日志/指标系统。
- 只展示框架能力、隐藏 Harness 行为的 demo。
- 用框架默认 memory/agent 抽象替代本项目语义。
- 从零自研通用 RAG 框架、通用 ReAct 框架、通用模型网关或通用评测平台。

## 4. 阶段范围

### 4.1 Skeleton P0

创建 Python 仓库结构、依赖声明、配置模型、prompt registry、包边界和可导入应用骨架。

Skeleton P0 不需要真实模型调用。

### 4.2 Migration P0

Migration P0 是第一个有实际迁移意义的后端版本，必须包含：

- FastAPI API 层，覆盖 Java 主接口。
- 基于 trusted headers 的 request context resolver。
- PostgreSQL `agent_rollout_event` 事件存储。
- run 内 active history 状态和事件 replay。
- context assembler。
- model gateway adapter，可以先用 stub；真实 provider adapter 放到 P1。
- tool gateway policy/trace adapter，参数校验使用 Pydantic。
- eval 和基础 chat 需要的核心工具。
- P0 可以保留 `queryInternalDocs` 工具契约或 mock fixture，但不实现完整 LlamaIndex/Milvus RAG。
- 版本化 prompt 加载。
- 可运行 deterministic fixture case 并收集 trace 的本地 eval gate。

### 4.3 P1

P1 增加：

- Qwen/DashScope-compatible 真实模型 provider。
- P1 默认使用 OpenAI-compatible SDK 接入 Qwen；只有出现多 provider、fallback、budget 或统一日志诉求时，再引入 LiteLLM。
- 流式响应路径。
- 上下文压缩。
- 完整工具 timeout/retry policy。
- Core Memory 和 Archival Memory 持久化。
- 长期记忆检索。

### 4.4 P2

P2 增加：

- RAG ingestion 和 retrieval。
- LlamaIndex RAG 编排。
- Milvus 向量检索。
- Milvus 原生 BM25/hybrid search，并带 tenant filter。
- rerank。
- citation metadata。
- RAG eval suite。

### 4.5 P3

P3 增加：

- 更强认证集成。
- 可选 PostgreSQL RLS。
- 如有需要，引入分布式后台 worker。
- 经确认后再做 Recall Memory / 历史对话召回。
- 更完整的可观测性 dashboard。

## 5. API 兼容契约

### 5.1 `POST /api/chat`

请求体：

```json
{
  "Id": "session-id",
  "Question": "user question"
}
```

兼容别名：

- `Id`, `id`, `ID`
- `Question`, `question`, `QUESTION`

请求头：

```text
X-Tenant-Id
X-User-Id
X-Agent-Id
X-Request-Id
```

成功响应：

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

如果 Harness 失败，但 HTTP handler 本身正常，也保留 Java 风格响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "success": false,
    "answer": null,
    "errorMessage": "..."
  }
}
```

### 5.2 `POST /api/chat_stream`

使用 SSE，event name 为 `message`。

每条 SSE payload 是 JSON：

```json
{
  "type": "content | retrying | final | error | done",
  "data": {}
}
```

payload 结构：

- `content`：`data` 是字符串 chunk。
- `retrying`：`data` 是 `{ "attempt": 1, "reset": true, "reason": "..." }`。
- `final`：`data` 是 `{ "data": "full answer" }`。
- `error`：`data` 是错误字符串。
- `done`：`data` 是 `null`。

### 5.3 `POST /api/chat/clear`

请求体：

```json
{
  "Id": "session-id"
}
```

成功响应：

```json
{
  "code": 200,
  "message": "success",
  "data": "会话历史已清空"
}
```

清理行为：

- 清理当前 tenant/user/agent/session 的内存运行时状态。
- 清理同一 tenant/user/agent/session 的持久化 rollout events。
- 如果后续实现 sidecar store，也清理 sidecar artifacts。

### 5.4 `GET /api/chat/session/{sessionId}`

响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "sessionId": "session-id",
    "messagePairCount": 0,
    "createTime": 0
  }
}
```

## 6. Request Context 契约

Python 必须实现 `AgentRequestContext`，字段包括：

- `tenant_id`
- `user_id`
- `agent_id`
- `session_id`
- `request_id`
- `roles`
- `permissions`

默认值：

- `tenant_id`: `default-tenant`
- `user_id`: `default-user`
- `agent_id`: API resolver 边界默认 `ops-agent`
- `request_id`: 缺失时生成 UUID
- `roles`: 空集合/列表
- `permissions`: 空集合/列表

重要规则：

- `tenant_id`、`user_id`、`agent_id`、`request_id`、`run_id` 都是后端运行时元数据。
- 模型不能通过工具参数传入或覆盖这些字段。
- P0 使用 trusted headers 只适合作为开发边界；生产认证放在后续阶段。

## 7. 运行时和事件契约

Python 必须保留 event-sourced trace 语义。

事件记录字段：

- `tenant_id`
- `user_id`
- `agent_id`
- `event_id`
- `event_type`
- `session_id`
- `run_id`
- `message_id`
- `tool_call_id`
- `sequence`
- `occurred_at`
- `payload`

必须支持的事件类型：

```text
THREAD_CREATED
THREAD_RECOVERED
RUN_STARTED
USER_MESSAGE_APPENDED
CONTEXT_ASSEMBLED
TOKEN_USAGE
MODEL_CALL_STARTED
MODEL_CALL_COMPLETED
MODEL_CALL_FAILED
MODEL_CALL_TIMEOUT
MODEL_CALL_RETRY
MODEL_CALL_CONTEXT_OVERFLOW
TOOL_CALL_STARTED
TOOL_CALL_COMPLETED
TOOL_CALL_FAILED
TOOL_CALL_BLOCKED
TOOL_CALL_TIMEOUT
TOOL_CALL_RETRY
TOOL_CALL_LATE_RESULT_DISCARDED
ASSISTANT_MESSAGE_APPENDED
HISTORY_TRIMMED
MEMORY_WRITTEN
MEMORY_DUPLICATE_SKIPPED
MEMORY_WRITE_FAILED
MEMORY_WRITE_REJECTED
MEMORY_SEARCHED
MEMORY_INJECTED
CORE_MEMORY_UPDATED
CORE_MEMORY_UPDATE_REJECTED
ARCHIVAL_MEMORY_WRITTEN
ARCHIVAL_MEMORY_DUPLICATE_SKIPPED
ARCHIVAL_MEMORY_WRITE_FAILED
RUN_COMPLETED
RUN_FAILED
```

运行时规则：

- 每个用户请求生成一个 `run_id`。
- active history 只在当前 run 内保存在内存。
- run 开始时从 PostgreSQL events 恢复历史。
- 按 `tenant_id + user_id + agent_id + session_id` 串行化会话变更。
- 执行过程中持续持久化事件，失败 run 也必须可诊断。

## 8. 上下文组装契约

Python `ContextAssembler` 必须按以下顺序组装模型输入：

```text
stable system prompt
<core_memory>
<memory_metadata>
active history / recent messages / compaction summary
current user request
```

规则：

- 不把历史会话拼进静态 prompt 文件。
- prompt 文本必须从版本化文件加载。
- Core Memory 和 Memory Metadata 是运行时上下文，不是写死在 system prompt 里的文本。
- 兼容说明：10G.1 后模型可见标签使用 `<memory_metadata>`；内部字段和 trace 为兼容既有 eval，可暂时保留 `memory_index_xml`、`hasMemoryIndex`、`memoryIndexTopicCount` 等命名，并新增 `hasMemoryMetadata`。
- context assembly 必须记录 `CONTEXT_ASSEMBLED` 事件，至少包含 prompt version、history item count、估算大小。
- Skeleton P0 后再引入 tokenizer 精确计数；初期可保留 char/token-ratio 风格估算。

## 9. Prompt 契约

prompt 文件位置：

```text
prompts/
  ops-agent-system-v1.md
  ops-agent-system-v2.md
```

当前默认：

```text
ops-agent-system-v2
```

trace 和 eval report 必须记录运行时实际使用的 prompt version。

## 10. 模型网关契约

即使 LangGraph、OpenAI-compatible SDK、LiteLLM 或 DashScope SDK 负责底层模型调用，Python 版本也必须保留项目自己的 ModelGateway adapter。

ModelGateway adapter 职责：

- 统一不同 provider 的非流式/流式调用接口。
- 调用 SDK 或框架提供的底层能力，不重复实现 HTTP client。
- timeout、retry/backoff、retry-after 语义归一化。
- context-window overflow 分类。
- 如果还没开始工具调用，可选强制压缩后重试。
- provider 返回 usage 时提取 token usage。
- 记录 `MODEL_CALL_*` 和 `TOKEN_USAGE` 事件。

模型 provider 必须可替换。初始 provider 可以是 Qwen/DashScope 的 OpenAI-compatible API adapter，也可以是 LiteLLM adapter 或 DashScope SDK adapter。

## 11. 工具网关契约

Python 必须实现项目级 `ToolGateway` adapter，但通用 schema/validation 能力优先复用 Pydantic/PydanticAI 风格设计。

ToolGateway adapter 职责：

- 工具执行前用 Pydantic 做参数校验。
- 工具 allow/deny policy。
- 每个工具独立 timeout。
- 每个工具独立 retry count。
- 只有工具配置为幂等、且错误可重试时才 retry。
- 返回结构化、模型可见的错误。
- 记录工具开始、完成、失败、blocked、timeout、retry 事件。
- 注入隐藏运行时上下文。

说明：

- 不从零自研通用 function calling 框架。
- LangGraph/LangChain/PydanticAI 能提供工具 schema、调用和部分错误处理能力时应优先复用。
- 项目 adapter 只负责本项目必须自控的 policy、tenant guard、trace、错误格式和 Java 兼容契约。

标准工具错误结构：

```json
{
  "success": false,
  "error_type": "PARAM_VALIDATION_FAILED | TOOL_TIMEOUT | TOOL_ERROR | TOOL_BLOCKED",
  "message": "...",
  "suggestion": "optional model-facing next action"
}
```

兼容说明：

- Java 当前参数校验错误返回中文自然文本，并带 metadata。
- Python 应统一为 JSON，同时保留清楚的模型修正提示。

## 12. 必需工具契约

### 12.0 默认工具策略

Python tool gateway 默认值应对齐 Java P0：

```text
default_action: allow
default_danger_level: medium
default_timeout_seconds: 10
default_max_retries: 0
executor_pool_size: 8
queue_capacity: 32
unknown_tool_policy: deny
```

初期保留的 Java 单工具默认值：

| Tool | Timeout | Retries | Idempotent |
|---|---:|---:|---|
| `getCurrentDateTime` | 1s | 0 | true |
| `getAvailableLogTopics` | 2s | 0 | true |
| `queryPrometheusAlerts` | 5s | 1 | true |
| `queryInternalDocs` | 8s | 1 | true |
| `queryLogs` | 8s | 1 | true |
| `listMemoryTopics` | 2s | 0 | true |
| `searchMemory` | 4s | 0 | true |
| `updateCoreMemory` | 6s | 0 | false |
| `saveArchivalMemory` | 6s | 0 | false |
| `writeMemory` | 6s | 0 | false |

### 12.1 `queryInternalDocs`

用途：

- 检索内部文档和知识库。
- 用于内部流程、最佳实践、操作步骤、排障文档、接口/流程规则。
- 不用于当前日志、指标、告警或长期记忆。

输入：

```json
{
  "query": "specific search query"
}
```

输出：

```json
{
  "status": "ok",
  "count": 3,
  "chunks": [
    {
      "id": "...",
      "ref": 1,
      "source": "...",
      "content": "...",
      "score": 0.0
    }
  ]
}
```

无结果：

```json
{
  "status": "no_results",
  "message": "No relevant documents found in the knowledge base."
}
```

### 12.2 `listMemoryTopics`

输入：

```json
{
  "type": "experience | knowledge | all"
}
```

输出：

```json
{
  "success": true,
  "count": 0,
  "note": "Topics are only a routing aid; call searchMemory before using memory as evidence.",
  "topics": [
    {
      "type": "experience",
      "topic": "orders/timeout",
      "count": 2
    }
  ]
}
```

### 12.3 `searchMemory`

输入：

```json
{
  "query": "specific natural language query",
  "type": "experience | knowledge | all",
  "scopeService": "order-service",
  "scopeEnv": "production",
  "tags": "timeout,order"
}
```

输出：

```json
{
  "success": true,
  "count": 0,
  "note": "Historical memory is not current evidence; verify with realtime tools when diagnosing live issues.",
  "memories": [
    {
      "id": "...",
      "type": "experience",
      "topic": "order-service/payment-timeout",
      "similarity": 0.72,
      "confidenceLabel": "high",
      "content": "..."
    }
  ]
}
```

默认检索目标：

- 过滤后 top 3。
- 实现后应支持 similarity threshold 和 confidence label。

### 12.4 `updateCoreMemory`

输入：

```json
{
  "blockKey": "user_rules | user_ops_profile | service_notes",
  "newContent": "full updated block content",
  "changeReason": "brief reason"
}
```

成功输出：

```json
{
  "success": true,
  "status": "updated",
  "blockKey": "user_rules",
  "version": 2,
  "message": "Core memory block updated."
}
```

拒绝输出：

```json
{
  "success": false,
  "status": "rejected",
  "error_type": "POLICY_ERROR",
  "message": "...",
  "suggestion": "..."
}
```

重要规则：

- `newContent` 是更新后的完整 block 内容，不是 diff patch。

### 12.5 `saveArchivalMemory`

输入：

```json
{
  "topic": "order-service/oom-timeout",
  "content": "concise durable memory",
  "evidenceSummary": "what verified this memory",
  "scopeService": "order-service",
  "scopeEnv": "production",
  "tags": "timeout,oom,order"
}
```

输出：

```json
{
  "success": true,
  "status": "written | duplicate_skipped | error",
  "duplicate": false,
  "memoryId": "...",
  "message": "...",
  "note": "Long-term memory is historical reference, not current evidence."
}
```

### 12.6 已弃用 `writeMemory`

迁移期间仅保留向后兼容。

优先使用：

- `updateCoreMemory`
- `saveArchivalMemory`

## 13. RAG 迁移契约

Java 当前行为：

- Milvus dense vector retrieval。
- 应用侧 Lucene BM25。
- RRF fusion。
- DashScope rerank，默认 `qwen3-rerank`。
- 默认 rerank topK 为 3。
- 返回 chunks 包含 source metadata 和 ref id。

Python 目标：

- P2 默认使用 LlamaIndex 作为 RAG 编排层，Milvus 作为向量/混合检索存储。
- LlamaIndex 负责 document loading、chunk/index、retriever/query engine、metadata filter、rerank 接入等通用 RAG pipeline。
- Milvus 负责 dense vector、BM25 sparse vector、hybrid search 和 scalar filtering。
- 项目只保留 `queryInternalDocs` 工具 adapter、tenant filter、trace、citation metadata 和 tool output JSON 契约。
- 使用 `tenant_id`、`knowledge_base_id`、`document_id`、`source`、title/path 等 scalar metadata 字段做过滤和引用。
- 检索时必须强制 tenant isolation。
- rerank 是候选召回后的独立阶段。
- 工具输出保留 chunk id、ref、source、content、score。

## 14. 长期记忆契约

Python 必须保留当前两层长期记忆设计。

### 14.1 Core Memory

Core Memory：

- 始终可见的运行时上下文。
- 存储在 PostgreSQL。
- request/run 开始时加载。
- 注入 `<core_memory>`。
- 通过 `updateCoreMemory` 更新。

默认 block：

- `user_rules`
- `user_ops_profile`
- `service_notes`

### 14.2 Archival Memory

Archival Memory：

- 可检索的长期记忆。
- 存储在 `long_term_memory`。
- 按 tenant/user/agent 隔离，并支持 service/environment/tags 过滤。
- 通过 `saveArchivalMemory` 写入。
- 通过 `searchMemory` 检索。
- 保留 exact-hash 去重和 embedding 近邻去重。

### 14.3 Rollout Extraction

当前 Java rollout extraction 不是完整后台 LLM 抽取流程，只是轻量/rule-driven baseline。

除非明确提升优先级，否则 Python 迁移中应把 LLM 后台抽取标记为后续工作。

## 15. 数据模型兼容

Python 使用 Alembic 做迁移，并保持与 Java migration 的表语义兼容。

核心表：

- `agent_rollout_event`
- `memory_rollout_session_state`，如果保留 rollout scanning
- `long_term_memory`
- `agent_core_memory_block`

### 15.1 `agent_rollout_event`

字段：

```text
sequence BIGSERIAL / monotonic sequence
tenant_id
event_id
event_type
session_id
run_id
message_id
tool_call_id
user_id
agent_id
occurred_at
payload JSON/JSONB
```

访问模式：

- 按 `tenant_id + user_id + agent_id + session_id` 读取事件，并按 sequence 排序。
- 按 `tenant_id + run_id` 查询事件。
- 只清理当前 tenant/user/agent/session。

### 15.2 `long_term_memory`

字段：

```text
id
tenant_id
type
topic
content
source
session_id
agent_id
user_id
embedding
embedding_model
embedding_dimension
embedding_metric
embedding_version
created_at
updated_at
usage_count
last_used_at
status
archived_at
archive_reason
tags
scope_service
scope_env
content_hash
```

兼容说明：

- Java 仍保留 legacy `type` 约束：`rule`、`experience`、`knowledge`。
- 即使后续弱化强分类，Python 也应优先保持数据库兼容。

### 15.3 `agent_core_memory_block`

字段：

```text
id
tenant_id
user_id
agent_id
block_key
description
content
max_tokens
version
read_only
source
content_hash
created_at
updated_at
status
```

唯一键：

```text
tenant_id + user_id + agent_id + block_key
```

### 15.4 `memory_rollout_session_state`

Migration P0 不强制需要这张表，除非实现 rollout scanning。

如果保留，需要保留字段：

```text
tenant_id
session_id
user_id
agent_id
active_run_id
active_run_started_at
last_event_sequence
last_event_at
last_content_event_sequence
last_content_event_at
content_event_count
last_rollout_content_event_sequence
last_rollout_at
updated_at
```

必需租户字段：

- `tenant_id`
- `user_id`
- `agent_id`
- 适用时包含 `session_id`

Migration P0 不引入 `agent_session_state` 作为必选表。

## 16. 配置契约

Python 必须通过环境变量和 typed settings 暴露等价配置。

重要配置组：

- server
- database
- model gateway
- prompt version
- context/window/compaction
- tool gateway
- RAG
- memory
- eval
- observability

密钥不能硬编码在 Python 配置文件中。Java `application.yml` 中有 API-key 示例，Python 必须改用环境变量。

## 17. 评测契约

Python eval 必须保留 trace-based gate，同时优先复用成熟评测体系。

默认策略：

- 本地 deterministic eval 作为迁移兼容门禁。
- 后续接 LangSmith 管理 dataset、experiment、trace 和部分 evaluator。
- 不把评测完全绑定到外部 SaaS；本地 runner 必须能独立跑通。

Dataset case 字段应支持：

- user input
- initial context/session state
- available tools
- mock tool returns
- expected behavior
- forbidden behavior
- scoring rules

每次 eval run 必须记录：

- run id
- session id
- prompt version
- model/provider version
- tool schema version
- model input/output summary
- tool calls
- tool arguments
- tool results
- rollout events
- final answer
- token usage
- latency
- errors

Evaluator 类型：

- deterministic rule evaluator
- reference/expectation evaluator
- 可选 LLM-as-judge evaluator

Migration P0 只要求 deterministic fixture-based eval。

## 18. 验收标准

阶段验收：

- Skeleton P0：package 可导入，config 可加载，prompt registry 能找到 prompt 文件，测试能运行。
- Migration P0：`/api/chat` 可使用 stub 或已配置模型 provider；事件可持久化；工具调用经过 gateway；deterministic eval 可产出报告。
- P1：真实模型和 streaming path 带 trace 跑通。
- P2：RAG 检索按租户隔离，并返回 citation metadata。

行为验收：

- 模型永远不能提供 tenant/run/tool execution metadata。
- 工具参数校验错误对模型可见且可修正。
- 工具运行时错误可分类并记录 trace。
- 会话恢复只读取当前 tenant/user/agent/session 的数据。
- prompt version 和 tool schema version 在 trace/eval 中可见。

## 19. 待决策项

对应阶段实现前需要决定：

- 使用 LangGraph prebuilt ReAct agent，还是自定义 graph loop。
- DashScope 走 OpenAI-compatible API，还是 DashScope SDK。
- Python P1 的长期记忆向量检索用 PostgreSQL pgvector，还是单独向量库。
- Milvus 原生 BM25 和 metadata filtering 的 RAG schema。
- 是否为压缩或记忆抽取引入分布式 worker。
