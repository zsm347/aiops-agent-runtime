# Java Agent Harness 能力盘点

## 1. 目的

本文用于盘点当前 Java 单 Agent Harness 已经具备的能力，作为 Python 版本迁移的参考基线。

这份文档的目标不是重新设计系统，而是回答两个问题：

- Java 版本现在到底做了什么。
- Python 版本至少要迁移哪些行为，才能算是同一个 Agent Harness 的 Python 实现。

范围：

- 只关注当前单 Agent Harness 链路。
- 多 Agent Python 重构文档不在本次盘点范围内。
- Java 代码库是当前参考实现和行为基线。
- 已实现能力和未来设想必须分开描述。

Java 参考项目：

```text
/Users/zsm/Documents/SuperBizAgent-release-2026-01-02
```

## 2. 当前整体运行链路

当前请求大致链路：

```text
HTTP/API 层
  -> AgentHarnessService.chat / stream
  -> ConversationRuntime.startRun
  -> ModelGateway / StreamingModelGateway
  -> CompactionCoordinator.compactBeforeModelCall
  -> ContextAssembler.assemble
       稳定 system prompt
       core memory
       memory index
       active history
  -> Spring AI Alibaba ReactAgent
  -> TraceToolInterceptor
  -> DefaultToolGateway
  -> 业务工具 / memory 工具 / RAG 工具
  -> ConversationRuntime 记录事件
  -> 最终回答 / 流式 chunk
```

关键入口：

| 模块 | Java 类 |
|---|---|
| Harness API 服务 | `org.example.agent.harness.service.AgentHarnessService` |
| ReAct 执行适配 | `DefaultAgentChatExecutor` |
| 运行时状态和事件 | `ConversationRuntime` |
| 上下文组装 | `ContextAssembler` |
| 模型网关 | `ModelGateway`, `StreamingModelGateway` |
| 工具网关 | `DefaultToolGateway`, `TraceToolInterceptor` |
| 长期记忆工具 | `LongTermMemoryTools` |
| RAG 工具 | `InternalDocsTools` |

对 Python 迁移的含义：

- Python 版本不能只是一个简单模型 wrapper。
- 需要保留显式 Harness 层，包住 LangGraph 或其他 ReAct runtime。
- `tenantId`、`userId`、`agentId`、`sessionId`、`runId`、`toolCallId` 等运行时元数据必须由后端上下文控制，不能让模型通过工具参数传入。

## 3. API 契约盘点

`ChatController` 当前实现的主要接口：

| 接口 | 方法 | 作用 | 备注 |
|---|---|---|---|
| `/api/chat` | `POST` | 非流式 Agent Harness 对话 | 从 header 解析租户上下文，返回完整答案 |
| `/api/chat_stream` | `POST` | SSE 流式 Agent Harness 对话 | 通过 `SseEmitter` 输出 chunk |
| `/api/chat/clear` | `POST` | 清空一个会话 | 清空运行时状态、事件流和 sidecar 文件 |
| `/api/chat/session/{sessionId}` | `GET` | 查看会话线程视图 | 用于调试和会话查看 |
| `/api/ai_ops` | `POST` | 较早的 AI Ops 流式链路 | 不是本次单 Agent 迁移主目标 |
| `/api/upload` | `POST` | 上传文档 | 属于 RAG ingestion |

请求上下文 header：

```text
X-Tenant-Id
X-User-Id
X-Agent-Id
X-Request-Id
```

聊天请求体包含会话 ID 和用户问题，具体 DTO 在迁移规格文档中定义。

Python 迁移优先级：

- 优先保留 `/api/chat`、`/api/chat_stream`、`/api/chat/clear`、会话查看接口。
- `/api/ai_ops` 暂时视为 legacy，除非后续明确纳入。
- SSE 事件格式要保持兼容，或者提供清楚的兼容适配。

## 4. 已实现能力图

### 4.1 Agent 编排

已实现：

- `AgentHarnessService` 创建 run，生成 `runId`，设置 MDC 字段，调用模型网关，并完成或失败 run。
- 底层 ReAct 循环由 Spring AI Alibaba `ReactAgent` 提供。
- 项目工程化 Harness 层包住 ReAct 循环，负责上下文组装、模型网关适配、工具治理适配、压缩、trace、memory context 和 token usage 捕获。
- 同步对话和流式对话共享大体运行结构。

Python 迁移目标：

- 可以用 `LangGraph` 提供基础 ReAct 循环，但外层必须保留项目 Harness adapter。
- 保留后端拥有的 `RunContext` / `ToolContext`。
- 同步和流式链路保持行为一致。

待定：

- ReAct 循环使用 LangGraph prebuilt agent，还是自定义 graph node。

### 4.2 运行时状态、短期记忆和事件流

已实现：

- `ConversationRuntime` 在运行期间维护每个会话的 `ThreadState`。
- 通过 `ConversationKey` 按 tenant/user/agent/session 隔离状态。
- 通过 `RolloutEventStore` 记录事件。
- 支持 JSONL 和 PostgreSQL 两种事件存储路径。
- 重启后第一次访问时，可以从持久化事件恢复状态。
- 按会话加锁，避免同一会话并发 run 互相覆盖。
- active history 包含用户消息、助手消息、工具调用、工具结果和压缩提示。

重要事件类型包括：

- run 生命周期事件
- 模型调用开始、完成、失败、超时、重试、上下文溢出事件
- 工具调用开始、完成、失败、超时、重试事件
- token usage 事件
- 上下文压缩事件
- 记忆检索、写入、core memory 事件

Python 迁移目标：

- 实现等价的 `ConversationRuntime`。
- PostgreSQL 事件流作为主存储。
- 每次 run 内在内存维护 active history。
- 保留会话锁、事件 replay、eval trace 导出。

### 4.3 上下文管理

已实现：

- `ContextAssembler` 组装：
  - `SystemPromptProvider` 提供的稳定 system prompt
  - `<core_memory>`
  - `<memory_index>`
  - 裁剪后的 active history
- `ActiveHistoryPolicy` 估算 token 并裁剪历史。
- `CompactionCoordinator` 在模型调用前和工具结果后触发摘要压缩。
- 如果模型网关发现 context overflow，并且还没有开始工具调用，可以强制压缩后重试。
- 短期记忆不是常驻全局内存；它从事件流恢复，并在本次执行过程中维护在内存里。

Python 迁移目标：

- 实现同样插入顺序的 `ContextAssembler`：
  - system prompt
  - core memory
  - memory index
  - active history
- 后续可以引入 tokenizer 精确计数，但第一阶段先保留当前阈值语义。
- 保留压缩 trace 和紧急 fallback 语义。

### 4.4 系统提示词

已实现：

```text
src/main/resources/prompts/
  ops-agent-system-v1.md
  ops-agent-system-v2.md
```

- `ops-agent-system-v2.md` 是当前主提示词。
- 内容包含证据优先级、工具使用规则、长期记忆规则、失败处理和回答规则。
- 当前 v2 还明确了：
  - `updateCoreMemory.newContent` 是完整 block 重写，不是 patch。
  - Core Memory 超预算时，应压缩/重写、转存 archival，或者不写。

Python 迁移目标：

- 复制提示词到 `prompts/` 并保留版本文件。
- trace 中记录实际使用的 prompt version。
- 不把 prompt 文本硬编码在 Python 类里。

### 4.5 模型网关

已实现：

- `ModelGateway` 记录模型调用开始、完成、失败、超时、重试事件。
- 通过 `ModelCallExceptionClassifier` 分类模型失败。
- 支持 retry/backoff。
- 上下文窗口溢出时，可以先强制压缩再重试。
- 配置为工具调用已开始后不再重试。
- `StreamingModelGateway` 处理流式路径。
- `TokenUsageExtractor` 和 `TokenUsageModelInterceptor` 在可用时捕获 token usage。

Python 迁移目标：

- 实现 provider-neutral 的 `ModelGateway`：
  - retry/backoff
  - provider 返回时支持 retry-after
  - context overflow 处理
  - timeout 分类
  - token usage trace
  - 统一同步和流式行为

初始模型提供方：

- 优先考虑 Alibaba/Qwen OpenAI-compatible API。
- 也可以在 embedding/rerank/tokenizer 等场景使用 DashScope SDK。

### 4.6 工具治理

已实现：

- `DefaultToolGateway` 是统一工具执行入口。
- 按工具解析策略：
  - allow/deny
  - timeout
  - retry count
  - idempotency
  - danger level
- 工具在有界线程池中执行。
- 支持：
  - policy blocked
  - timeout
  - 幂等工具的可重试 IO 异常
  - 参数校验失败
  - 通用执行异常
- 不把原始异常直接泄露给模型，而是返回模型可见的工具错误结果。
- `TraceToolInterceptor` 记录 trace，并阻止同一工具连续失败导致的无效循环。

Python 迁移目标：

- 用 Pydantic 做工具参数校验。
- 工具错误统一为结构化 JSON：

```json
{
  "success": false,
  "error_type": "PARAM_VALIDATION_FAILED | TOOL_TIMEOUT | TOOL_ERROR | TOOL_BLOCKED",
  "message": "...",
  "suggestion": "optional model-facing next action"
}
```

- 保留隐藏运行时元数据注入：模型不能提供 `tenantId`、`runId`、`toolCallId`。

### 4.7 业务工具

已实现工具：

| 工具类 | 能力 |
|---|---|
| `DateTimeTools` | 查询当前日期/时间 |
| `QueryMetricsTools` | 查询当前 Prometheus 告警 |
| `QueryLogsTools` | 查询日志主题和日志内容 |
| `InternalDocsTools` | 内部文档/RAG 检索 |
| `LongTermMemoryTools` | 记忆 topic、检索、写入、core/archive memory |

Python 迁移目标：

- 用 typed Python function/class + Pydantic schema 重建工具。
- 工具描述先贴近 Java 行为。
- 后续再把 mock 服务替换为真实集成。

### 4.8 RAG

已实现：

- RAG 入口工具是 `InternalDocsTools.queryInternalDocs`。
- Java 服务包括：
  - `RagService`
  - `HybridSearchService`
  - `VectorSearchService`
  - `Bm25SearchService`
  - `VectorEmbeddingService`
  - `VectorIndexService`
  - `DocumentChunkService`
- 当前 Java 仍包含应用侧 Lucene BM25。
- 后续设计建议改为 Milvus 原生 BM25，方便使用 Milvus metadata 做多租户过滤。
- `HybridSearchService` 已包含 DashScope rerank 路径，配置项为 `rag.rerank.model`，当前是 `qwen3-rerank`。
- RAG 结果应携带 source/chunk 元数据，方便最终回答使用 `[1]`、`[2]` 形式引用。

Python 迁移目标：

- 明确 Python P0 是严格复刻当前 Java，还是直接采用目标态 Milvus 原生 hybrid。
- 最小目标：
  - 文档 chunk 和 metadata
  - 向量检索
  - source/citation metadata 返回给模型
  - tenant 和 knowledge base 过滤

建议：

- 如果不是要求逐行等价，Python RAG 可以直接按 Milvus 原生 BM25/hybrid 目标态设计。

### 4.9 长期记忆

已实现基础：

- `long_term_memory` 使用 PostgreSQL + pgvector。
- embedding 模型配置为 Alibaba `text-embedding-v4`，维度 1024，cosine。
- 多租户迁移已增加 `tenant_id`。
- P0 已增加 `agent_core_memory_block` 表。
- `long_term_memory` 已增加 archival metadata：
  - `tags`
  - `scope_service`
  - `scope_env`
  - `content_hash`

已实现服务：

- `CoreMemoryService`
- `CoreMemoryRepository`
- `MemoryWritePolicy`
- `LongTermMemoryWriteService`
- `ArchivalMemoryWriteRepository`
- `MemorySearchService`
- `MemoryIndexService`
- `MemoryLifecycleJob`
- `MemoryRolloutService`

已实现工具：

- `updateCoreMemory`
- `saveArchivalMemory`
- `searchMemory`
- `listMemoryTopics`
- deprecated `writeMemory`

P0 语义：

- Core Memory 每次都会注入上下文。
- Core Memory block key：
  - `user_rules`
  - `user_ops_profile`
  - `service_notes`
- Core Memory 更新是完整 block 重写。
- Archival Memory 按需检索。
- Archival search 支持：
  - service scope
  - environment scope
  - tags
  - minimum similarity
- Archival retrieval 默认小 topK。
- 写入策略会拒绝密钥、原始日志、非法 block、超预算内容和低质量写入。

非 P0 / 未来能力：

- 真正后台 LLM 记忆抽取。当前 `MemoryRolloutService` 和 `PromptMemoryExtractionClient` 只是 rollout scanning / rule-driven extraction baseline，不是完整 LLM extraction。
- `memory_extraction_job`
- `memory_candidate`
- Recall Memory / 历史对话召回
- LLM 冲突分类器
- 完整用户记忆管理 UI/API

Python 迁移目标：

- 先迁移 P0 行为，再加 P1/P2 增强。
- 除非有强理由，不改变当前存储模型。
- 保留 memory trace events 和 eval 预期。

### 4.10 多租户上下文

已实现：

- `AgentRequestContext` 携带 `tenantId`、`userId`、`agentId`、`sessionId`、`requestId`、roles、permissions。
- `AgentRequestContextResolver` 解析请求上下文。
- 当前 P0 resolver 读取可信 HTTP header：
  - `X-Tenant-Id`
  - `X-User-Id`
  - `X-Agent-Id`
  - `X-Request-Id`
- header 缺失时默认 `default-tenant`、`default-user`、`ops-agent`。
- 当前 resolver 明确不从 request body 读取 tenant identity。
- 这还不是生产级认证；生产态应从 JWT、SSO、网关认证或其他可信认证上下文解析 tenant/user/roles。
- 工具执行和 memory 工具从后端运行时上下文读取身份。
- 数据库迁移已给长期记忆和事件流增加 `tenant_id`。

Python 迁移目标：

- 用 request middleware/dependency 创建后端拥有的 `TenantContext`。
- repository 层强制 DB filter 和 guard。
- 不把 tenant/user/agent 暴露为模型可传的工具参数。

### 4.11 评测 Harness

已实现：

- Agent eval 类在 `org.example.agent.harness.eval.agent`。
- 数据集在：

```text
src/test/resources/agent-eval/suites/
  capability.json
  prompt-regression.json
  safety.json
  smoke.json
```

Eval case 结构包括：

- id/name/category/tags
- request
- setup
- expectations
- judge config

Trace 记录：

- run status
- events
- tool calls
- model calls
- token usage
- memory events
- context events

Judge 包括：

- rule judge
- evidence rule judge
- trace judge
- parameter rule evaluator
- gate policy
- baseline comparator
- metric aggregator

Python 迁移目标：

- 复用或翻译现有 JSON/YAML 数据集。
- 早期就实现 eval runner。
- 用 eval suite 作为 Java/Python 兼容门禁。
- 后续可集成 LangSmith/OpenEvals/Ragas/DeepEval，但不能丢掉当前 trace-based deterministic judges。

## 5. 数据存储盘点

重要迁移：

```text
V20260609_01__long_term_memory_phase4.sql
V20260628_01__tenant_isolation_p0.sql
V20260704_01__core_memory_p0.sql
```

重要表：

| 表 | 作用 |
|---|---|
| `long_term_memory` | archival memory，带 pgvector embedding |
| `agent_core_memory_block` | 常驻上下文的 core memory block |
| `agent_rollout_event` | 持久事件流 / replay 来源 |
| `memory_rollout_session_state` | rollout extraction 水位 |

Python 迁移目标：

- 使用 SQLAlchemy + Alembic。
- 要么复用相同表名，要么设计清楚兼容映射。
- 如果 Java 和 Python 过渡期共用数据库，表兼容必须明确。

## 6. 配置盘点

重要配置来源：

```text
src/main/resources/application.yml
```

需要保留或明确修改的关键默认值：

| 配置项 | 当前 Java 值 / 行为 | 迁移说明 |
|---|---|---|
| server port | `9900` | Python 可换端口，但 API path 尽量兼容 |
| PostgreSQL | `POSTGRES_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`，Flyway | Python 用 SQLAlchemy + Alembic，并明确迁移所有权 |
| 主模型 | Harness 和 RAG 中使用 Qwen 相关模型 | Python provider 必须可配置 |
| 模型超时 | DashScope chat timeout `180000 ms` | 保留为模型网关默认超时 |
| context window | `model-context-window-tokens=32000` | 当前工程默认，不等于模型真实最大值 |
| auto compaction | `model-auto-compact-token-limit=28800` | 除非规格修改，否则保留阈值语义 |
| token estimate | `token-byte-ratio=2.5` | 后续可换 tokenizer，先保持行为清楚 |
| summary max tokens | `compact-summary-max-tokens=300` | 压缩摘要输出预算 |
| tool output limit | `tool-output-char-limit=8000` | 工具结果需要截断/摘要 |
| model gateway retry | `max-retries=1`, backoff `1000..10000 ms` | 先保留 retry policy |
| context overflow | `force-compact-on-context-overflow=true` | 重要 Harness 行为 |
| event store | `agent.harness.event-store.type=postgres` | Migration P0 使用 PostgreSQL event store |
| tool gateway pool | pool `8`, queue `32` | Python 可用 async/semaphore，但必须限制并发 |
| unknown tool policy | `deny` | 重要安全默认值 |
| memory embedding | `text-embedding-v4`, dimension `1024`, cosine | 保留 memory 兼容 |
| memory duplicate threshold | `0.92` | 保留 archival dedupe 行为 |
| memory search | topK `3`, candidate topK `10`, max return tokens `800` | 保留检索契约 |
| memory rollout | enabled, idle threshold `30 min`, rule-driven baseline | 不纳入 Migration P0，除非明确要求 |
| lifecycle | 90 天未使用自动归档 | 后台能力，后续处理 |
| RAG search | semantic topK `10`, BM25 topK `10`, RRF topK `15`, rerank topK `3` | 保留或明确替换为 Milvus native |
| rerank model | `qwen3-rerank` | RAG parity 重要配置 |

安全说明：

- 当前 Java `application.yml` 中有疑似 API key 示例。
- Python 版本必须从一开始使用环境变量或 secret manager，不能硬编码密钥。

## 7. 建议 Python 模块映射

```text
src/superbiz_agent/
  api/
  harness/
  model_gateway/
  tools/
  rag/
  memory/
  evals/
  persistence/
  prompts/
```

详细包结构见 `03-python-architecture-design.md`。

## 8. 迁移优先级建议

这里有两个 P0，不能混淆。

### 8.1 Skeleton P0

Skeleton P0 是技术骨架验证，可以小于 Java Harness。

包含：

1. FastAPI chat endpoint。
2. 后端拥有的 request context。
3. in-memory conversation runtime 和基础 trace。
4. system prompt + active history 的 context assembler。
5. 一个 model gateway。
6. 带 Pydantic 校验和错误处理的 tool gateway。
7. DateTime + mock logs/metrics/docs tools。
8. trace artifact 输出。
9. 小型 eval runner。

### 8.2 Migration P0

Migration P0 是能合理称为 Java 单 Agent Harness Python 迁移版的最小版本。

必须包含：

1. `/api/chat` 兼容。
2. 后端拥有 tenant/user/agent/session/run context。
3. PostgreSQL `agent_rollout_event` 作为主 trace source。
4. 从事件流 replay/recovery。
5. context assembler 包含 stable system prompt、core memory、memory index、active history。
6. model gateway 包含 retry/backoff、失败分类、context-overflow compaction hook、token usage trace。
7. tool gateway 包含 policy、timeout、Pydantic validation、幂等瞬时失败重试、结构化错误。
8. 主工具：时间、告警、日志、内部文档、memory topics/search/update/save。
9. prompt versioning。
10. smoke/capability eval runner。

Migration P0 可以使用 mock logs/metrics/RAG 数据，但工具契约和 trace 语义必须与 Java 行为一致。

P1：

1. `/api/chat_stream` SSE 兼容。
2. 完整上下文压缩和 forced-compaction recovery。
3. RAG retrieval/ingestion parity。
4. 更完整 eval judges 和 baseline 对比。

P2：

1. trusted headers 之外的多租户加固。
2. Milvus-native hybrid RAG。
3. memory lifecycle 和后台任务。
4. 生产观测能力。

## 9. 关键风险

- 如果 Python 迁移没有保留 trace 语义，评测和排障能力会变弱。
- 使用 LangGraph 不代表可以删除 Harness 层的上下文、模型、工具、记忆治理。
- 一次迁移太多内容，会分不清失败来自模型、prompt、工具、RAG、memory 还是框架。
- Python 版本应先通过兼容契约和 eval gate，再引入更多框架便利能力。

## 10. 下一份文档

下一份文档是：

```text
02-python-migration-spec.md
```

它定义：

- Python P0 精确范围
- 不做什么
- API 契约
- 数据模型兼容
- 工具 schema
- trace event schema
- eval 验收标准
