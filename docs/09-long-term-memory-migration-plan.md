# 长期记忆迁移实施计划

## 1. 阶段定位

本文是 Python 迁移路线第 9 步 `长期记忆迁移` 的实施计划。

本阶段目标是迁移 Java 单 Agent Harness 中长期记忆的核心闭环：

```text
request/run context
  -> load core memory
  -> build memory index
  -> context assembler 注入 <core_memory> / <memory_index>
  -> model tool call
  -> ToolGateway 参数校验和 trace
  -> memory service policy / write / search
  -> memory trace events
  -> deterministic eval gate
```

本阶段不是要一次性做成完整生产级记忆系统，而是先把 Java 版本已经稳定下来的核心语义迁移到 Python：

- Core Memory 始终可见。
- Archival Memory 可写入、可检索。
- memory index 告诉模型外部记忆概况和使用规则。
- 记忆写入必须经过确定性 guardrail。
- 记忆工具必须通过 ToolGateway，而不是绕过工具治理。
- 记忆相关行为必须能被 trace 和 eval 验收。

10G.1 后续修订说明：

- 第 9 步最初按 Java 兼容语义设计了模型可见 `<memory_index>`。
- 10G.1 已将模型可见内容升级为 `<memory_metadata>`，用于更清楚表达“外部长期记忆概况 / 路由元信息”，不注入 Archival Memory 正文。
- 为兼容既有 eval 和 trace，代码内部可以继续保留 `MemoryIndexService`、`memory_index_xml`、`hasMemoryIndex`、`memoryIndexTopicCount` 等命名，并新增 `hasMemoryMetadata`。
- 因此本文中历史出现的 `<memory_index>` 可理解为第 9 步基线；当前模型可见标签以 10G.1 的 `<memory_metadata>` 为准。

## 2. 依据

本计划依据：

- `docs/01-java-capability-inventory.md`
  - 长期记忆已实现能力盘点。
  - Core Memory / Archival Memory / MemoryIndex / Policy / Tool contracts。
- `docs/02-python-migration-spec.md`
  - `listMemoryTopics`
  - `searchMemory`
  - `updateCoreMemory`
  - `saveArchivalMemory`
  - `long_term_memory`
  - `agent_core_memory_block`
- `docs/03-python-architecture-design.md`
  - memory 包结构。
  - PostgreSQL + pgvector first 的目标方向。
  - 项目只做 memory adapter / policy / trace，不替换成框架默认 memory。
- `docs/04-python-migration-roadmap.md`
  - 第 9 步长期记忆迁移范围。
- Java 参考：
  - `CoreMemoryService`
  - `MemoryIndexService`
  - `MemoryWritePolicy`
  - `LongTermMemoryWriteService`
  - `MemorySearchService`
  - `LongTermMemoryTools`
  - `V20260609_01__long_term_memory_phase4.sql`
  - `V20260704_01__core_memory_p0.sql`
- 当前 Python：
  - `ContextAssembler`
  - `ConversationRuntime`
  - `ToolGateway`
  - `StubModelGateway`
  - `RolloutEventStore`
  - deterministic eval runner。

## 3. 本阶段做什么

### 3.1 迁移能力列表

本阶段要实现：

```text
Core Memory
Archival Memory
Memory Index
Memory Write Policy
Memory Retrieval
Memory Trace Events
Memory Tool Contracts
```

本阶段新增并注册工具：

```text
listMemoryTopics
searchMemory
updateCoreMemory
saveArchivalMemory
```

继续保留已有工具：

```text
getCurrentDateTime
getAvailableLogTopics
queryLogs
queryPrometheusAlerts
queryInternalDocs
```

本阶段不默认注册 deprecated `writeMemory`。如果后续需要兼容老 prompt 或老 eval，再作为单独兼容任务补。

### 3.2 Core Memory

Core Memory 是始终注入模型上下文的长期上下文。

默认 block：

```text
user_rules
user_ops_profile
service_notes
```

block 配置对齐 Java：

| blockKey | maxTokens | 用途 |
|---|---:|---|
| `user_rules` | 300 | 跨会话必须遵守的用户规则，例如回答格式、长期约束 |
| `user_ops_profile` | 500 | 用户稳定运维画像，例如负责服务、常用环境、偏好排查顺序 |
| `service_notes` | 600 | 高频稳定服务背景，例如服务依赖、架构事实、环境说明 |

运行时行为：

- request/run 开始组装上下文时加载当前 `tenant_id + user_id + agent_id` 的 active core blocks。
- 缺失 block 自动初始化为空 block。
- 注入 `<core_memory>`，位置在 system prompt 之后、memory index 之前。
- 空 block 也渲染 description 和 metadata，让模型知道可写 block。
- `updateCoreMemory` 使用完整 block 重写语义，不支持 patch/diff。

### 3.3 Archival Memory

Archival Memory 是上下文外的可检索长期记忆。

用途：

- 已验证历史排障经验。
- 稳定服务背景。
- 可复用知识。
- 用户偏好里不适合长期常驻上下文的细节。

存储字段要兼容 Java 语义：

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

本阶段写入 `saveArchivalMemory` 默认保存为 `type=experience`，与 Java P0 archival write path 一致。

### 3.4 Memory Index

Memory Index 是注入模型上下文的轻量索引，不直接注入完整 archival 内容。

10G.1 后模型可见标签改为 `<memory_metadata>`，语义从“索引”收敛为“外部记忆概况和检索路由元信息”。内部类名可暂时保留 `MemoryIndexService`。

渲染格式参考 Java：

```text
<memory_index>
archival memory total: 2
rule memories:
top topics:
experience:
- order-service/payment-timeout (1)
knowledge:
available tags: timeout, order, production
usage rules:
- call listMemoryTopics if the current task may need memories outside these topics.
- call searchMemory to retrieve specific memories before using them as evidence.
- memory is historical reference, not current evidence.
</memory_index>
```

说明：

- `rule memories` 可保留空区块，兼容 Java 格式。
- topic 用于提示模型可能存在外部记忆。
- 模型不能只凭 memory index 回答具体事实；必须调用 `searchMemory` 取回具体记忆内容。
- 当前 10G.1 实现中，模型不能只凭 memory metadata 回答具体事实；必须调用 `searchMemory` 取回具体 archival memory content。

### 3.5 Memory Write Policy

写入策略必须是后端确定性 guardrail，不只靠 prompt。

Core Memory 更新校验：

- 必须有运行时上下文。
- `blockKey` 必须属于白名单。
- `newContent` 不得超过 block token 预算。
- 拒绝 read-only block。
- 拒绝疑似密钥、token、密码、private key。
- 拒绝原始日志、堆栈、告警/指标 dump、大段工具输出。

Archival Memory 写入校验：

- 必须有运行时上下文。
- `topic` 非空，长度不超过 120 字符。
- `content` 非空，长度不超过 4000 字符。
- `evidenceSummary` 可选，但如果存在也要做敏感信息扫描。
- 拒绝疑似密钥、token、密码、private key。
- 拒绝原始日志、堆栈、告警/指标 dump、大段工具输出。

Policy 只做确定性规则，不让模型参与是否允许写入的安全判定。

### 3.6 去重和检索

本阶段保留两个去重层次：

1. exact hash 去重：
   - 对归一化 content 计算 SHA-256。
   - 同一 `tenant_id + user_id + agent_id + content_hash` 已存在 active memory，则返回 `duplicate_skipped`。

2. 近邻去重：
   - 本阶段先实现 deterministic local embedding，用于测试和本地可复现。
   - 目标是验证近邻去重和搜索流程，不代表最终线上 embedding 质量。
   - 相似度超过 `0.92` 时返回 `duplicate_skipped`。

长期记忆检索：

- 默认 topK = 3。
- 默认 minSimilarity = 0.5。
- `0.5 <= similarity < 0.7` 标为 `medium`。
- `similarity >= 0.7` 标为 `high`。
- 支持 `scopeService`、`scopeEnv`、`tags` 过滤。
- 只返回 active memory。
- 每次返回的 memory best-effort 增加 `usage_count` 并更新 `last_used_at`。

### 3.7 存储后端

本阶段采用“双层实现”：

1. 默认 `memory` backend：
   - 使用进程内 repository。
   - 用于本地测试、eval、无数据库运行。
   - 必须严格按 `tenant_id + user_id + agent_id` 隔离。

2. PostgreSQL 兼容迁移：
   - 增加 Alembic migration。
   - 增加 SQLAlchemy models。
   - 表结构与 Java migration 语义兼容。

新增配置建议：

```text
memory_enabled = true
memory_store_backend = "memory"
memory_search_top_k = 3
memory_search_min_similarity = 0.5
memory_duplicate_similarity = 0.92
```

说明：

- 当前本地 Python 环境不稳定地缺少部分数据库依赖，所以默认运行不能依赖真实 PostgreSQL。
- PostgreSQL repository 可以后续单独补全；本阶段至少要固定数据模型和迁移文件，避免设计漂移。
- 如果本阶段实现 PostgreSQL repository，也必须是可选路径，不能影响默认测试。
- 如果 SQLAlchemy 模型需要表达 pgvector，但当前项目没有引入 `pgvector` Python 依赖，可以定义很薄的 SQLAlchemy `UserDefinedType`，只用于把列类型编译成 `VECTOR(1024)`；不要为了这个阶段引入新的大型依赖。
- in-memory backend 不应做成不可控的全局静态变量，应由 `AgentHarnessService` 构建并注入 memory services，保证测试和 eval 可控。

### 3.8 ToolGateway 运行时上下文

memory 工具不能让模型传 `tenant_id/user_id/agent_id/session_id/run_id`。

因此需要让 ToolGateway 支持把 `RunContext` 注入工具 handler。

推荐实现：

- `ToolDefinition` 增加 `requires_context: bool = False`。
- 普通工具保持原有 `handler(args)`。
- memory 工具使用 `handler(args, run_context)`。
- ToolGateway 根据 `requires_context` 决定调用方式。

这样可以保证：

- 模型只传业务参数。
- memory 工具从后端运行时上下文拿租户、用户、agent、session、run。
- trace 和多租户隔离不会被模型污染。

注意：

- 不要把 `tenantId`、`userId`、`agentId`、`sessionId`、`runId` 放进 memory tool 的 Pydantic args model。
- 即使模型传了同名额外字段，也应因为 `extra="forbid"` 被参数校验拒绝。

### 3.9 ContextAssembler 注入顺序

当前组装顺序：

```text
system prompt
active history
current user message
```

本阶段调整为：

```text
system prompt
core memory
memory index
active history
current user message
```

实现要求：

- `ContextAssembler` 接收可选 memory context provider。
- `ConversationRuntime.assemble_context` 把 `RunContext.request_context` 传入 assembler。
- `AssembledContext.trace_payload` 正确设置：
  - `hasCoreMemory`
  - `hasMemoryIndex`
  - 可选 `coreMemoryBlockCount`
  - 可选 `memoryIndexTopicCount`
- 组装后如果注入了 memory，记录 `MEMORY_INJECTED` 事件。

### 3.10 Memory Trace Events

本阶段至少记录：

| 行为 | 事件 |
|---|---|
| Core Memory 注入上下文 | `MEMORY_INJECTED` |
| Core Memory 更新成功 | `CORE_MEMORY_UPDATED` |
| Core Memory 更新拒绝 | `CORE_MEMORY_UPDATE_REJECTED` |
| Archival Memory 写入成功 | `ARCHIVAL_MEMORY_WRITTEN` |
| Archival Memory hash/near duplicate | `ARCHIVAL_MEMORY_DUPLICATE_SKIPPED` |
| Archival Memory 写入失败 | `ARCHIVAL_MEMORY_WRITE_FAILED` |
| Memory search | `MEMORY_SEARCHED` |
| Policy 拒绝 archival 写入 | `MEMORY_WRITE_REJECTED` |

ToolGateway 仍会记录：

```text
TOOL_CALL_STARTED
TOOL_CALL_COMPLETED
TOOL_CALL_FAILED
TOOL_CALL_BLOCKED
```

memory 事件是业务语义 trace；tool call 事件是工具执行 trace，两者都要保留。

## 4. 工具契约

### 4.1 `listMemoryTopics`

输入：

```json
{
  "type": "experience | knowledge | all"
}
```

默认：

```json
{
  "type": "all"
}
```

输出：

```json
{
  "success": true,
  "count": 1,
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

Policy：

| Tool | Danger | Timeout | Retries | Idempotent |
|---|---|---:|---:|---|
| `listMemoryTopics` | low | 2s | 0 | true |

### 4.2 `searchMemory`

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
  "count": 1,
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

Policy：

| Tool | Danger | Timeout | Retries | Idempotent |
|---|---|---:|---:|---|
| `searchMemory` | low | 4s | 0 | true |

### 4.3 `updateCoreMemory`

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

Policy：

| Tool | Danger | Timeout | Retries | Idempotent |
|---|---|---:|---:|---|
| `updateCoreMemory` | medium | 6s | 0 | false |

### 4.4 `saveArchivalMemory`

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

Policy：

| Tool | Danger | Timeout | Retries | Idempotent |
|---|---|---:|---:|---|
| `saveArchivalMemory` | medium | 6s | 0 | false |

## 5. StubModelGateway 更新

本阶段只做 deterministic 规则，方便 eval 验证工具链，不做真实模型智能。

新增规则：

- 用户说“记住”“以后都”“以后回答按...”这类稳定规则 -> `updateCoreMemory`
  - 默认写 `user_rules`。
  - `newContent` 是完整 block 内容。
- 用户说“根因确认”“保存经验”“写入长期记忆”“这次经验是...” -> `saveArchivalMemory`
  - topic 从服务名/问题类型粗略推断。
  - content 使用用户原句的简洁稳定表达。
- 用户说“参考历史经验”“以前有没有类似”“查一下长期记忆” -> `searchMemory`
  - query 使用用户原句。
  - 默认 type=`all`。
- 用户说“有哪些记忆主题”“长期记忆 topic” -> `listMemoryTopics`
  - 默认 type=`all`。

限制：

- 仍然最多一次工具调用 + final answer。
- 不实现多工具链式“先 list 再 search”。
- 不实现后台 LLM extraction。
- 不实现模型主动合并复杂 core block 的能力；这里只是 deterministic stub 行为。

## 6. 本阶段不做什么

明确不做：

- 不做 Recall Memory / 历史对话召回。
- 不做完整后台 LLM memory extraction。
- 不做 `memory_extraction_job`。
- 不做 `memory_candidate`。
- 不做 memory UI。
- 不做完整生命周期遗忘/归档 job。
- 不接真实 embedding provider。
- 不要求真实 pgvector 检索质量。
- 不做多轮 memory planning。
- 不实现 deprecated `writeMemory`，除非后续明确需要兼容。
- 不修改 Java 项目。
- 不改 RAG/Milvus 业务实现。

## 7. 文件级实施范围

### 7.1 可能新增文件

```text
alembic/versions/20260705_02_create_long_term_memory.py
src/superbiz_agent/memory/schemas.py
src/superbiz_agent/memory/embedding.py
src/superbiz_agent/memory/policy.py
src/superbiz_agent/memory/store.py
src/superbiz_agent/memory/core.py
src/superbiz_agent/memory/archival.py
src/superbiz_agent/memory/search.py
src/superbiz_agent/memory/index.py
src/superbiz_agent/memory/tools.py
tests/test_long_term_memory.py
```

### 7.2 需要修改文件

```text
src/superbiz_agent/config.py
src/superbiz_agent/persistence/models.py
src/superbiz_agent/tools/registry.py
src/superbiz_agent/tools/gateway.py
src/superbiz_agent/tools/builtin/__init__.py
src/superbiz_agent/harness/context_assembler.py
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/harness/service.py
src/superbiz_agent/model_gateway/stub.py
src/superbiz_agent/evals/cases.py
tests/test_eval_runner.py
tests/test_skeleton.py
```

### 7.3 不应修改文件

```text
src/superbiz_agent/rag/
docs/01-java-capability-inventory.md
docs/02-python-migration-spec.md
docs/03-python-architecture-design.md
docs/04-python-migration-roadmap.md
docs/05-skeleton-p0-implementation-plan.md
docs/06-basic-eval-runner-plan.md
docs/07-migration-p0-implementation-plan.md
docs/08-business-tools-migration-plan.md
```

除非发现明确冲突，不修改前置设计文档。

## 8. 实施批次

### 批次 A：memory model、policy、repository

目标：

- 定义 Core Memory / Archival Memory 数据模型。
- 实现 deterministic embedding。
- 实现 in-memory repository。
- 实现 MemoryWritePolicy。
- 增加 SQLAlchemy models 和 Alembic migration。

验收：

- Core block 缺失时自动初始化。
- 超预算 core 更新被拒绝。
- 敏感信息和原始日志写入被拒绝。
- archival exact hash 去重生效。
- tenant/user/agent 隔离生效。

### 批次 B：memory services 和 context injection

目标：

- 实现 `CoreMemoryService`。
- 实现 `ArchivalMemoryService`。
- 实现 `MemorySearchService`。
- 实现 `MemoryIndexService`。
- `ContextAssembler` 注入 `<core_memory>` 和外部记忆元信息。第 9 步基线是 `<memory_index>`；10G.1 后模型可见标签为 `<memory_metadata>`。
- `ConversationRuntime` 记录 `MEMORY_INJECTED`。

验收：

- 首次请求上下文包含 3 个 core blocks。
- 更新 core memory 后，下一轮上下文能看到新内容。
- memory index 能显示 topic、tag 和 usage rules。

### 批次 C：memory tools 和 ToolGateway context injection

目标：

- `ToolDefinition` 支持 `requires_context`。
- ToolGateway 将 `RunContext` 注入 memory handler。
- 注册 4 个 memory tools。
- 工具返回结构对齐迁移规格。
- 工具内部记录 memory 业务事件。

验收：

- 模型不能传 tenant/user/agent/run 字段。
- memory tool 通过后端 context 访问当前租户数据。
- 4 个工具的 policy 与 Java 配置一致。

### 批次 D：StubModelGateway 和 eval

目标：

- 增加 deterministic memory tool selection。
- 增加 tool result -> final answer 格式化。
- 增加 memory eval cases。

建议 eval case：

1. `core_memory_update_required`
   - 输入：`请记住，以后排障回答按现象、证据、判断、建议、未确认项组织。`
   - 必须调用 `updateCoreMemory`
   - 必须产生 `CORE_MEMORY_UPDATED`

2. `core_memory_injected_on_next_turn`
   - prime：先更新 core memory。
   - 当前输入：`hello`
   - 必须产生 `MEMORY_INJECTED`
   - context history 或 trace payload 能证明 `hasCoreMemory=true`

3. `archival_memory_save_required`
   - 输入：`这次 order-service 5xx 根因确认是 payment-service 连接池耗尽，请保存为长期经验。`
   - 必须调用 `saveArchivalMemory`
   - 必须产生 `ARCHIVAL_MEMORY_WRITTEN`

4. `memory_search_required`
   - prime：先保存 archival memory。
   - 当前输入：`参考历史经验，order-service 5xx 以前有没有类似原因？`
   - 必须调用 `searchMemory`
   - answer 包含 `payment-service`

5. `memory_topics_required`
   - prime：先保存 archival memory。
   - 当前输入：`有哪些长期记忆主题？`
   - 必须调用 `listMemoryTopics`
   - answer 包含 `order-service`

6. `memory_policy_rejects_secret`
   - 输入：`请记住 api_key=abc123secret456`
   - 必须调用 memory write tool。
   - tool result 为 rejected。
   - 必须产生 reject event。

Eval runner 需要支持：

- 保留现有 `initial_context.prime_user_input`。
- 增加可选 `initial_context.prime_user_inputs` 列表，用于一个 case 前置多轮准备。
- 生成 artifact 时仍收集同一 session 的完整 trace；judge 可以基于 required tool/event 判断阶段行为。
- 如需判断 context 注入状态，可在 `TraceArtifact` 中增加 `context_flags` 或 `max_context_core_memory_block_count` 这类 deterministic 字段。

## 9. 测试计划

新增/更新测试：

- Core Memory：
  - 默认加载 3 个 block。
  - build context XML 转义正常。
  - update 成功后 version +1。
  - invalid blockKey 拒绝。
  - 超预算拒绝。
  - secret/raw log 拒绝。

- Archival Memory：
  - save 成功。
  - exact hash duplicate skipped。
  - near duplicate skipped。
  - scope/tags 持久化。
  - tenant/user/agent 隔离。

- Search：
  - topK=3。
  - minSimilarity=0.5。
  - confidenceLabel medium/high。
  - scopeService/scopeEnv/tags filter。
  - returned memory usage_count 增加。

- Memory Index：
  - archival total 正确。
  - top topics 正确。
  - tags 正确。
  - usage rules 存在。

- Tools：
  - 4 个 memory tools 注册。
  - 4 个 memory tools policy 正确。
  - 参数校验错误结构化。
  - ToolGateway context 注入生效。

- Context：
  - `<core_memory>` 在 system prompt 后。
  - `<memory_index>` 在 core memory 后。
  - trace `hasCoreMemory=true`、`hasMemoryIndex=true`。
  - 产生 `MEMORY_INJECTED`。

- Eval：
  - 新增 memory case 全部通过。
  - 原有 8 个 case 继续通过。

## 10. 验收命令

subAgent 实施完成后必须运行：

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
```

如果本地安装了 ruff，也运行：

```bash
python3 -m ruff check src tests
```

如果 ruff 未安装，说明即可，不作为失败。

可选：

```bash
PYTHONPATH=src python3 -m compileall -q src tests
```

## 11. subAgent 实施任务单

交给 subAgent 的任务应限制为：

```text
只实现 docs/09-long-term-memory-migration-plan.md 定义的长期记忆迁移。
不得实现 Recall Memory、后台 LLM extraction、memory_extraction_job、memory_candidate、memory UI、生命周期遗忘 job、真实 embedding provider、真实 pgvector 检索质量、真实模型或 streaming。
不得修改 Java 项目。
不得修改 src/superbiz_agent/rag/。
实现完成后必须运行 python3 -m pytest 和 eval runner。
提交结果时说明：
1. 修改/新增了哪些文件。
2. Core Memory、Archival Memory、Memory Index、Policy 怎么实现。
3. 4 个 memory tools 的 schema、policy、输出结构。
4. ContextAssembler 如何注入 memory。
5. 新增了哪些 trace events 和 eval cases。
6. 测试命令和结果。
7. 是否有偏离计划或遗留问题。
```

如果实现过程中发现计划和当前代码冲突，subAgent 必须停止并报告冲突，不得扩大范围自行重设计。

## 12. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否符合第 9 步长期记忆迁移 | 是 |
| 是否提前做 Recall Memory | 否 |
| 是否提前做后台 LLM extraction | 否 |
| 是否提前做 memory_extraction_job / memory_candidate | 否 |
| 是否绕过 ToolGateway | 否 |
| 是否让模型传 tenant/user/agent/run | 否 |
| 是否保留 Core/Archival 双层结构 | 是 |
| 是否保留 write policy | 是 |
| 是否保留 memory trace events | 是 |
| 是否默认测试不依赖真实 DB / embedding provider | 是 |
| 是否增加 eval 覆盖 | 是 |

## 13. 阶段完成标准

长期记忆迁移可以验收通过的条件：

1. `updateCoreMemory`、`saveArchivalMemory`、`searchMemory`、`listMemoryTopics` 已注册到 ToolRegistry。
2. 四个 memory tools 都有 Pydantic 参数 schema。
3. 四个 memory tools policy 与迁移规格一致。
4. ToolGateway 支持后端 `RunContext` 注入，模型不能传运行时身份字段。
5. Core Memory 默认 3 个 block，并能注入 `<core_memory>`。
6. Memory Index / Memory Metadata 能注入模型上下文。第 9 步基线是 `<memory_index>`；10G.1 后模型可见标签为 `<memory_metadata>`。
7. Archival Memory 支持写入、hash 去重、近邻去重、scope/tags 过滤、top3 检索、confidence label。
8. Memory Write Policy 能拒绝 secrets、原始日志、大段工具输出和超预算 core 更新。
9. memory trace events 能被 eval artifact 观察到。
10. 新增 memory eval cases 全部通过，原有 case 不回退。
11. `python3 -m pytest` 全部通过。
12. 未接入本阶段明确排除的真实外部系统或后续能力。
