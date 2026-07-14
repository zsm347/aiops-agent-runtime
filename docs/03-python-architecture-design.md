# Python 架构与技术栈设计

## 1. 范围

本文选择 Python 版本的架构和核心技术栈，用于重建当前 Java 单 Agent Harness。

依据：

- Java 能力盘点：`docs/01-java-capability-inventory.md`
- 迁移规格：`docs/02-python-migration-spec.md`
- 相关框架官方文档

核心目标不是“尽量多用框架”，也不是“凡事自研”。默认原则是成熟框架优先，项目只保留薄适配层和必须自控的工程约束。

具体含义：

- Agent 编排优先使用 LangGraph。
- RAG 编排优先使用 LlamaIndex。
- 模型访问优先使用 OpenAI-compatible SDK、LiteLLM 或 DashScope SDK。
- schema/参数校验优先使用 Pydantic，后续可参考 PydanticAI。
- 评测优先保留本地 deterministic eval gate，并可接入 LangSmith。
- 项目自研只覆盖 tenant context、trace event、tool policy、context assembly、memory/RAG 输出契约等项目特有部分。

## 2. 框架事实依据

FastAPI 适合作为 API 层，因为它基于 Python type hints，支持 OpenAPI，适合构建异步 typed API。官方文档：<https://fastapi.tiangolo.com/>

LangGraph 适合作为 Agent 编排候选，因为它面向 agent orchestration，支持 durable execution、streaming、human-in-the-loop、persistence 等能力。官方文档也说明 LangGraph 负责 orchestration，LangSmith 负责 tracing/evaluation/prompts。官方文档：<https://docs.langchain.com/oss/python/langgraph/overview>

LlamaIndex 适合作为 RAG 编排候选，因为它面向 context-augmented LLM applications，覆盖数据接入、索引、retriever、query engine、agent/workflow、evaluation/observability 等能力，并且有 Milvus、hybrid search、rerank、metadata filter 等集成路径。官方文档：<https://developers.llamaindex.ai/python/framework/>

Pydantic 适合作为 schema/validation 层，因为它可以定义 typed model，并生成 JSON Schema/OpenAPI 兼容 schema。官方文档：<https://pydantic.dev/docs/validation/dev/concepts/json_schema/>

SQLAlchemy 支持 asyncio 数据库访问，Alembic 是配套迁移工具。官方文档：<https://docs.sqlalchemy.org/en/latest/orm/extensions/asyncio.html>，<https://alembic.sqlalchemy.org/>

Milvus 支持 dense/sparse 多向量 hybrid retrieval、BM25 function 和 scalar filtering。官方文档：<https://milvus.io/docs/multi-vector-search.md>，<https://milvus.io/docs/filtered-search.md>

LangSmith 的评测流程与 dataset -> evaluator -> experiment -> analysis 很接近，也支持 code rule、LLM-as-judge、human review、pairwise comparison 等评估方式。官方文档：<https://docs.langchain.com/langsmith/evaluation>

阿里云 Model Studio 支持 Qwen 的 OpenAI-compatible 接口，DashScope Python SDK 覆盖 generation、embedding、rerank 等能力。官方文档：<https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope>，<https://github.com/dashscope/dashscope-sdk-python>

LiteLLM 可作为多模型 provider adapter 候选，用于统一不同模型供应商的调用、fallback、logging、budget 等能力。是否引入取决于第一阶段 provider 数量和复杂度。官方文档：<https://docs.litellm.ai/>

## 3. 推荐技术栈

| 层级 | 选择 | 原因 |
|---|---|---|
| API | FastAPI + Uvicorn | 适合 typed request/response DTO、SSE、OpenAPI 和异步接口 |
| Schema | Pydantic v2 | 用于工具参数、API DTO、配置、JSON Schema 和校验错误 |
| Settings | pydantic-settings | 类型化环境变量配置，避免硬编码密钥 |
| Agent runtime | LangGraph + 项目 Harness adapter | LangGraph 负责图编排；adapter 负责 run/context/tool/trace 兼容 |
| Model SDK | P1 默认 OpenAI-compatible SDK；后续按需 LiteLLM / DashScope SDK | 优先复用成熟 provider SDK；LiteLLM 用于多 provider/fallback/budget 场景 |
| Database | PostgreSQL | 保留 Java event store 和 memory persistence 模型 |
| ORM/query | SQLAlchemy 2 async + asyncpg | 成熟的异步 PostgreSQL 访问方案 |
| Migration | Alembic | SQLAlchemy 标准迁移工具 |
| 长期记忆向量检索 | P1 先用 PostgreSQL pgvector | 记忆和租户 metadata 放在同一存储里，贴近 Java 表语义 |
| RAG 编排 | LlamaIndex | 优先复用成熟 RAG pipeline，不自研通用 Retriever |
| RAG 向量库 | Milvus | 保留 Java RAG 方向，并支持原生 BM25/hybrid search |
| Eval | 本地 deterministic eval gate + 后续 LangSmith | 本地门禁保证迁移可复现，LangSmith 管理实验和 trace |
| Observability | structlog/logging + OpenTelemetry hooks + 后续 Prometheus metrics | 先简单可用，保留扩展点 |
| Tests | pytest + pytest-asyncio + httpx | Python 异步 API 测试标准组合 |
| Packaging | pyproject.toml + src layout | 包边界清晰，支持 editable install |

## 4. 架构边界

Python 系统组织为：

```text
HTTP/API
  -> AgentHarnessService
  -> ConversationRuntime
  -> ContextAssembler
  -> ModelGateway adapter
  -> LangGraph ReAct loop 或 custom graph
  -> ToolGateway policy/trace adapter
  -> Domain tools / RAG tools / memory tools
  -> LlamaIndex RAG pipeline
  -> RolloutEventStore
  -> Eval trace collector
```

关键决策：

- LangGraph 负责 graph stepping。
- LlamaIndex 负责通用 RAG pipeline。
- OpenAI-compatible SDK、LiteLLM 或 DashScope SDK 负责底层模型访问。
- 项目 Harness adapter 负责 run identity、tenant context、prompt assembly、tool governance、event trace、memory injection、eval artifacts。

这与 Java 模式一致：框架提供基础能力，项目 Harness 负责生产级工程控制和兼容契约。

## 5. 包结构

推荐包结构：

```text
src/superbiz_agent/
  api/
    app.py
    routes_chat.py
    schemas.py
    sse.py
  harness/
    service.py
    runtime.py
    context.py
    events.py
    locks.py
  model_gateway/
    base.py
    openai_compatible.py
    dashscope_native.py
    errors.py
  tools/
    gateway.py
    registry.py
    policies.py
    errors.py
    builtin/
  memory/
    core.py
    archival.py
    index.py
    search.py
    schemas.py
  rag/
    ingestion.py
    retrieval.py
    milvus_store.py
    rerank.py
    citations.py
  persistence/
    database.py
    models.py
    repositories/
  prompts/
    registry.py
  evals/
    cases.py
    runner.py
    judges.py
    traces.py
  observability/
    logging.py
    metrics.py
  config.py
  main.py
```

## 6. 核心运行时设计

### 6.1 Request Context

`AgentRequestContext` 在 API 边界从 trusted headers 解析：

- `X-Tenant-Id`
- `X-User-Id`
- `X-Agent-Id`
- `X-Request-Id`

请求体只提供 session id 和用户问题。

运行时内部元数据：

- `run_id`
- `tool_call_id`
- 内部 trace ids

这些字段必须通过后端上下文对象传递，不能暴露为模型可传的工具参数。

### 6.2 Run Context

每个请求创建一个 `RunContext`：

```text
request_context
run_id
prompt_version
tool_schema_version
model_provider
started_at
tool_call_started flag
```

`RunContext` 在 Harness service 中显式传递，并通过 `ToolExecutionContext` 注入工具执行。

不要依赖全局可变状态实现租户隔离。

### 6.3 Conversation Runtime

职责：

- start run。
- 从 `agent_rollout_event` 恢复历史会话状态。
- 本次 run 内维护 active history。
- 追加 user/assistant/tool events。
- 持久化 trace events。
- 按 conversation key 串行化变更。
- complete/fail run。

实现说明：

- 初期可以使用进程内 `asyncio.Lock`，key 为 `tenant_id:user_id:agent_id:session_id`。
- 多实例部署前不需要分布式锁。

## 7. LangGraph 使用方式

推荐初始路径：

- 优先评估 LangGraph prebuilt ReAct agent 能否满足需求。
- 如果 prebuilt agent 难以插入项目 trace/tool policy/context assembly，再构建小型 custom graph。
- 节点：
  - model call node
  - tool dispatch node
  - finalization node
- 边：
  - model 请求工具 -> tool dispatch
  - tool result -> model call
  - final answer -> finalization

无论用 prebuilt agent 还是 custom graph，都必须满足：

- 需要 Java 兼容的 trace events。
- 工具调用必须经过项目 ToolGateway policy/trace adapter。
- 需要在框架默认 memory 之外做 context assembly。
- 需要 model gateway retry/compaction 语义。

结论：

- 不从零自研 ReAct 框架。
- LangGraph 是默认 Agent runtime。
- 项目只保留 LangGraph adapter，用于接入 trace、tenant context、context assembly 和 tool policy。

## 8. 模型访问与 ModelGateway Adapter 设计

推荐 provider 策略：

### P0

- 实现轻量 `ModelGateway` adapter interface。
- 提供 `StubModelGateway` 做 deterministic tests。
- 提供兼容的 message 和 tool-call 数据模型。

### P1

- 增加 `OpenAICompatibleQwenGateway`，使用阿里 Model Studio OpenAI-compatible endpoint。
- P1 默认先用 OpenAI-compatible SDK。
- 如果后续需要多 provider、fallback、budget、统一日志，再接入 LiteLLM。
- 使用环境变量：
  - `MODEL_BASE_URL`
  - `MODEL_API_KEY`
  - `MODEL_NAME`

### P1/P2 原生 DashScope

DashScope SDK 用在它明显更合适的地方：

- embeddings
- rerank
- tokenizer
- provider-specific features

项目不重复实现通用 HTTP/model client。底层模型访问由 SDK 或 LiteLLM 完成。

项目 `ModelGateway` adapter 负责：

- retry/backoff
- timeout
- context overflow classification
- streaming aggregation
- token usage extraction
- trace events

## 9. 工具治理设计

工具注册应包含：

- Pydantic args model
- Python callable
- tool description
- policy metadata

概念示例：

```python
ToolDefinition(
    name="searchMemory",
    args_model=SearchMemoryArgs,
    handler=search_memory,
    description="...",
    policy=ToolPolicy(timeout_seconds=4, max_retries=0, idempotent=True),
)
```

执行路径：

```text
model tool call
  -> ToolGateway
  -> validate args with Pydantic
  -> resolve tool policy
  -> inject ToolExecutionContext
  -> execute with timeout
  -> retry if idempotent and retryable
  -> normalize result/error
  -> record rollout event
  -> return JSON tool result to model
```

错误类型：

- `PARAM_VALIDATION_FAILED`
- `TOOL_BLOCKED`
- `TOOL_TIMEOUT`
- `TOOL_RETRY_EXHAUSTED`
- `UPSTREAM_UNAVAILABLE`
- `TOOL_ERROR`

模型可见错误必须包含简洁 `message`，可选包含 `suggestion`。

实现原则：

- schema、tool definition、基础 validation 优先使用 Pydantic/LangGraph/LangChain/PydanticAI 能力。
- 项目不自研通用 function calling 框架。
- 项目 ToolGateway adapter 只负责本项目必需的 policy、tenant guard、trace event、错误归一化和 Java 兼容输出。

后续如果 PydanticAI 的 retry/retry prompt 机制更适合工具参数修复，可以引入其设计或局部使用。

## 10. 上下文和记忆设计

### 10.1 Context Assembler

模型上下文组装顺序：

```text
system prompt
core memory
memory index
active history
current user message
```

Prompt registry：

- 基于文件。
- 版本化。
- 不在 service class 中硬编码 prompt 内容。

### 10.2 短期记忆

短期记忆指当前会话在本次 run 的上下文：

- run 开始时从 PostgreSQL event stream 加载。
- ReAct 循环期间保存在内存。
- run 过程中持久化新事件。
- run 结束后释放内存状态。

不强制需要 `agent_session_state` 表。

### 10.3 Core Memory

Core Memory：

- PostgreSQL 表：`agent_core_memory_block`
- run 开始时加载
- 注入上下文
- 通过 `updateCoreMemory` 更新
- 完整 block 重写语义

### 10.4 Archival Memory

Archival Memory：

- PostgreSQL 表：`long_term_memory`
- P1 通过 pgvector 做向量字段
- exact hash dedupe
- embedding 近邻去重
- 默认 top 3 检索
- 必须带 tenant/user/agent filters

### 10.5 RAG

RAG 和 Memory 分离：

- RAG 用于内部文档、流程规则、排障手册、接口说明、runbook。
- Memory 用于用户偏好、稳定服务背景、历史经验。

P2 默认使用 LlamaIndex + Milvus：

- LlamaIndex 负责 document loading、chunk/index、retriever/query engine、metadata filter 和 rerank 接入。
- Milvus 负责 dense/sparse vector、BM25、hybrid search 和 scalar filtering。
- 项目负责 `queryInternalDocs` 工具 adapter、tenant filter 强制注入、trace、citation metadata 和 tool output JSON。
- shared collection
- scalar fields：`tenant_id`、`knowledge_base_id`、`document_id`、source/title metadata
- 原生 BM25 sparse vector field
- dense vector field
- hybrid search
- search 前 scalar filter
- rerank topK 3
- chunks 返回 citation metadata

## 11. 持久化设计

使用 SQLAlchemy model 和 Alembic migration。

初始 migration 创建或镜像：

- `agent_rollout_event`
- `long_term_memory`
- `agent_core_memory_block`
- 可选 `memory_rollout_session_state`

Repository 层：

```text
RolloutEventRepository
CoreMemoryRepository
ArchivalMemoryRepository
MemorySearchRepository
```

API handler 不应直接写 raw SQL。

## 12. 评测设计

先做本地 deterministic eval gate，因为迁移验收依赖 Java 兼容 trace；后续接入 LangSmith 管理 dataset、experiment、trace 和部分 evaluator。

核心对象：

- `EvalCase`
- `EvalSuite`
- `EvalRun`
- `TraceArtifact`
- `RuleJudge`
- `EvidenceJudge`
- optional `LLMJudge`

P0 evaluator 范围：

- 必须调用的工具是否调用。
- 禁止调用的工具是否未调用。
- 必填参数是否存在。
- 最终答案是否包含预期关键事实。
- trace 是否有 run/model/tool events。
- 是否明显编造工具里没有的数据。

LangSmith 后续可以接入，因为它的官方 workflow 与 dataset/evaluator/experiment/analysis 对齐。但本地 runner 必须保留，避免评测完全依赖外部 SaaS。

## 13. 安全和租户隔离

P0 最小规则：

- 在 API 边界解析 tenant/user/agent。
- 每个 repository query 必须包含 tenant filters。
- 每个读取持久化数据的工具都从 `ToolExecutionContext` 获取 tenant context。
- 模型不能传 tenant id 或 run id。
- 不能硬编码 API key。
- trace artifacts 需要脱敏。

后续：

- JWT/SSO。
- PostgreSQL RLS。
- per-tenant resource binding。
- tenant-scoped Milvus knowledge bases 和外部日志/指标凭证。

## 14. 选型矩阵

| 领域 | 选择 | 替代方案 | 决策理由 |
|---|---|---|---|
| API | FastAPI | Flask, Django | 更适合 async typed API 和 OpenAPI |
| Agent runtime | LangGraph + Harness adapter | 完全手写 loop | LangGraph 负责编排，adapter 保留项目语义 |
| Validation | Pydantic | dataclasses/manual validation | 直接支持 schema、validation、API/tool DTO |
| DB | SQLAlchemy async | raw asyncpg | 提供 models/repositories，raw asyncpg 只用于热点路径 |
| Migrations | Alembic | 纯自定义 SQL | 与 SQLAlchemy 配套，也能承载手写 SQL |
| Chat provider | P1 OpenAI-compatible SDK，后续 LiteLLM / DashScope SDK | 自研 HTTP client | 优先复用 provider SDK；LiteLLM 用于多 provider/fallback/budget |
| Eval | local deterministic gate + LangSmith later | 只用本地或只用 SaaS | 本地保留 Java trace 兼容，LangSmith 管理实验更方便 |
| Memory vector | pgvector first | Milvus memory collection | memory 和 tenant metadata 放在同一存储 |
| RAG orchestration | LlamaIndex | 自研 Retriever | LlamaIndex 已有成熟 RAG pipeline 和 Milvus 集成 |
| RAG vector | Milvus | pgvector | Milvus 更适合大文档库 hybrid search |

## 15. 实施顺序

1. 项目骨架。
2. typed config 和 prompt registry。
3. API DTO 兼容。
4. request context resolver。
5. rollout event models 和 repositories。
6. runtime start/complete/fail path。
7. context assembler。
8. stub model gateway adapter。
9. LangGraph adapter。
10. tool registry 和 ToolGateway policy/trace adapter。
11. deterministic eval runner。
12. 真实 Qwen model adapter。
13. Streaming。
14. Compaction。
15. Core/Archival Memory。
16. 长期记忆检索。
17. LlamaIndex + Milvus RAG adapter。
18. 完整 RAG eval。

## 16. 风险

| 风险 | 缓解方式 |
|---|---|
| LangGraph 抽象隐藏过多 trace 细节 | 使用 custom graph nodes，并包住 model/tool calls |
| 框架 memory 与项目 memory 设计冲突 | 不把默认框架 memory 当事实源 |
| LlamaIndex 默认检索流程漏掉 tenant filter | 项目 RAG adapter 强制注入 tenant_id/knowledge_base_id filter |
| async 复杂度泄漏到业务代码 | repository/service 保持 async，隔离阻塞 SDK 调用 |
| 工具错误格式偏离 Java | 定义 Pydantic error schema 和测试 |
| RAG 多租户隔离不完整 | 每个 Milvus query 必须包含 scalar tenant filter |
| Eval 依赖外部 SaaS | local eval runner 作为 canonical |

## 17. 架构验收标准

此架构可接受的条件：

- 能实现 `02-python-migration-spec.md` 的全部契约。
- 保留 Harness 拥有的 context、tool、memory、trace 语义。
- 优先复用 LangGraph、LlamaIndex、Pydantic、OpenAI-compatible SDK/LiteLLM/DashScope、LangSmith 等成熟能力。
- 不把 provider-specific 逻辑硬编码进业务模块。
- 不依赖真实模型也能跑 deterministic tests。
- 后续可以添加真实 Qwen、Milvus、rerank、LangSmith，而不改变核心契约。
