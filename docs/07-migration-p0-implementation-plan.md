# Migration P0 实施计划

## 1. 阶段定位

本文是 Python 迁移路线第 7 步 `Migration P0` 的实施计划。

Migration P0 的目标是把当前 Skeleton P0 升级为“有实际迁移意义”的最小 Python 后端：

```text
FastAPI chat
  -> trusted-header request context
  -> conversation lock
  -> PostgreSQL-compatible rollout event store
  -> session replay / active history recovery
  -> context assembler
  -> stub model gateway
  -> tool gateway
  -> deterministic eval gate
```

本阶段不是生产完整版本。它要补齐 Java 单 Agent Harness 的关键工程骨架：事件流事实源、会话恢复、租户过滤、trace 兼容和 eval 可回归。

## 2. 依据

本计划依据：

- `docs/02-python-migration-spec.md`
  - Migration P0 范围。
  - Request Context 契约。
  - 运行时和事件契约。
  - `agent_rollout_event` 表契约。
  - eval 契约。
- `docs/03-python-architecture-design.md`
  - Run Context。
  - Conversation Runtime。
  - Repository 层。
  - LangGraph + Harness adapter 边界。
- `docs/04-python-migration-roadmap.md`
  - 第 7 步 Migration P0 范围。
- Java 参考：
  - `agent_rollout_event` 表迁移。
  - `ConversationRuntime.recover/applyEvent`。
  - `RolloutEventStore`。
- 当前 Python：
  - Skeleton P0。
  - 基础 Eval Runner。

## 3. 本阶段做什么

### 3.1 Rollout Event Store 抽象

当前 `InMemoryRolloutEventStore` 是 Skeleton P0 专用实现。Migration P0 要抽象出统一事件存储接口：

```text
RolloutEventStore
  append_event(...)
  append(...)
  list_by_session(context)
  list_by_run(tenant_id, run_id)
  clear_session(context)
```

实现：

- `InMemoryRolloutEventStore` 保留，用于单元测试和本地无数据库运行。
- 新增 `PostgresRolloutEventStore`，使用 SQLAlchemy async 访问 PostgreSQL。

要求：

- Runtime 和 ToolGateway 依赖抽象接口，不依赖具体 in-memory 类。
- 事件字段与 `RolloutEvent` 模型一致。
- 所有 session 查询必须带 `tenant_id + user_id + agent_id + session_id`。
- run 查询必须带 `tenant_id + run_id`，`tenant_id` 不能可选。

### 3.2 PostgreSQL Schema / Alembic 迁移

新增 Alembic 迁移，创建 Python 版本需要的 `agent_rollout_event` 表。

字段：

```text
sequence BIGSERIAL PRIMARY KEY
tenant_id VARCHAR NOT NULL
event_id VARCHAR NOT NULL UNIQUE
event_type VARCHAR NOT NULL
session_id VARCHAR NOT NULL
run_id VARCHAR
message_id VARCHAR
tool_call_id VARCHAR
user_id VARCHAR NOT NULL
agent_id VARCHAR NOT NULL
occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
payload JSONB NOT NULL DEFAULT '{}'::jsonb
```

索引：

```text
idx_rollout_tenant_session_sequence
  (tenant_id, user_id, agent_id, session_id, sequence)

idx_rollout_tenant_run
  (tenant_id, run_id)

idx_rollout_event_type
  (event_type)
```

说明：

- 本阶段只迁移 `agent_rollout_event`。
- 不创建 `long_term_memory`。
- 不创建 `agent_core_memory_block`。
- 不创建 `agent_session_state`。
- 不启用 PostgreSQL RLS。

### 3.3 SQLAlchemy 模型与 Repository

新增：

```text
src/superbiz_agent/persistence/models.py
src/superbiz_agent/persistence/repositories/rollout_events.py
```

Repository 职责：

- insert event。
- read session events ordered by `sequence asc`。
- read run events ordered by `sequence asc`。
- clear current tenant/user/agent/session。

实现要求：

- API handler 不写 SQL。
- Runtime 不写 SQL。
- Repository 必须强制 tenant/user/agent/session 过滤。
- JSON payload 用 dict 读写。
- DB 访问错误应转成项目内异常，例如 `RolloutEventStoreError`。

### 3.4 Service 构造与配置

新增配置：

```text
rollout_store_backend = "memory" | "postgres"
```

默认：

```text
memory
```

原因：

- 本地测试不要求必须启动 PostgreSQL。
- Migration P0 需要具备 PostgreSQL 实现，但默认开发路径可继续用 memory。

当配置为 `postgres`：

```text
create_engine(settings.database_url)
create_sessionmaker(engine)
PostgresRolloutEventStore(sessionmaker)
```

当配置为 `memory`：

```text
InMemoryRolloutEventStore()
```

### 3.5 Conversation Runtime Replay

Migration P0 要让 run 开始时从 event store 恢复当前会话的 active history。

新增最小 replay 逻辑：

```text
start_run(context, question)
  -> list_by_session(context)
  -> apply previous events only
  -> recovered active_history
  -> append RUN_STARTED
  -> append THREAD_RECOVERED when previous events exist
```

本阶段需要恢复的历史类型：

- `USER_MESSAGE_APPENDED`
- `ASSISTANT_MESSAGE_APPENDED`
- `TOOL_CALL_STARTED`
- `TOOL_CALL_COMPLETED`

推荐恢复为 `ModelMessage`：

```text
USER_MESSAGE_APPENDED -> ModelMessage(role="user", content=payload.content)
ASSISTANT_MESSAGE_APPENDED -> ModelMessage(role="assistant", content=payload.content)
TOOL_CALL_COMPLETED -> ModelMessage(role="tool", name=payload.toolName, content=json(result))
```

注意：

- 当前 Skeleton P0 的 `USER_MESSAGE_APPENDED` 只记录 `messageLength`，Migration P0 必须改为记录 `content`。
- 当前 Skeleton P0 的 `ASSISTANT_MESSAGE_APPENDED` 只记录 `messageLength`，Migration P0 必须改为记录 `content`。
- `RUN_STARTED` payload 应包含 `question`、`promptVersion`、`toolSchemaVersion`、`modelProvider`、`status`。
- `RUN_COMPLETED` payload 应包含 `answer`、`status`。
- 当前 run 的 `USER_MESSAGE_APPENDED` 不能被重复放入 recovered history；ContextAssembler 仍通过 `current_user_message` 追加当前用户消息。
- `THREAD_RECOVERED` 是恢复审计事件，不应作为模型上下文历史。

### 3.6 Run 内 Active History

运行时规则：

- 每个请求开始时读取持久化 session events，恢复历史。
- 本次 run 内在内存维护 active history，避免每次 model call 都查数据库。
- 本次 run 结束后移除 run 内 active history。
- 下一次请求再从 event store 恢复。

这与当前短期记忆设计一致：短期记忆不是常驻进程内存事实源，事实源是 event store。

### 3.7 Conversation Lock

新增进程内会话锁：

```text
key = tenant_id:user_id:agent_id:session_id
```

规则：

- 同一 conversation key 的 run 串行执行。
- 不同 conversation key 可并发。
- 本阶段只用进程内 `asyncio.Lock`。
- 不做分布式锁。

### 3.8 ToolGateway Trace 兼容

当前 ToolGateway 已记录工具事件，但 Migration P0 需要确保 payload 更利于 replay/eval：

`TOOL_CALL_STARTED` payload 至少包含：

```text
toolName
arguments
timeoutSeconds
maxRetries
```

`TOOL_CALL_COMPLETED` payload 至少包含：

```text
toolName
result
status="success"
```

`TOOL_CALL_FAILED/BLOCKED` payload 至少包含：

```text
toolName
errorType / reason
result
status="error"
```

### 3.9 ContextAssembler

Migration P0 的 ContextAssembler 仍不注入长期记忆，但要贴近正式顺序：

```text
system prompt
active history / recent messages
current user request
```

`CONTEXT_ASSEMBLED` payload 至少包含：

```text
promptVersion
historyItemCount
messageCount
estimatedChars
hasCoreMemory=false
hasMemoryIndex=false
```

### 3.10 API 兼容补齐

检查并补齐主 API：

- `POST /api/chat`
- `POST /api/chat/clear`
- `GET /api/chat/session/{sessionId}`

本阶段不做：

- `POST /api/chat_stream`

`GET /api/chat/session/{sessionId}` 返回：

```json
{
  "sessionId": "session-id",
  "messagePairCount": 0,
  "createTime": 0
}
```

`messagePairCount` 可以基于 recovered user/assistant turn 粗略计算。

### 3.11 Eval Runner 回归

更新基础 eval，使其继续通过，并增加 Migration P0 能力检查：

- trace artifact 能看到 prompt version。
- trace artifact 能看到 tool schema version。
- session replay case：
  - 第一次请求写入 event store。
  - 第二次请求恢复上一轮 user/assistant 历史。
  - `CONTEXT_ASSEMBLED.historyItemCount > 0`。

如果默认 backend 是 memory，replay case 可用同一个 service 复用同一个 memory store 验证。

PostgreSQL repository 的测试可以采用：

- repository 单元测试 mock/fake async session。
- SQLAlchemy statement 编译测试。
- 如果环境存在 PostgreSQL，再加可选 integration test。

不要让普通 `python3 -m pytest` 依赖本机 PostgreSQL。

## 4. 本阶段不做什么

Migration P0 明确不做：

- 不接真实 Qwen。
- 不接 OpenAI-compatible SDK。
- 不接 LiteLLM。
- 不接 DashScope SDK。
- 不做 streaming。
- 不做上下文压缩。
- 不做真实 RAG。
- 不接 LlamaIndex/Milvus。
- 不做长期记忆。
- 不做 Core Memory / Archival Memory。
- 不做 Recall Memory。
- 不做 LangSmith。
- 不做 LLM-as-judge。
- 不做生产级 JWT/SSO。
- 不启用 PostgreSQL RLS。
- 不做分布式锁。
- 不做后台任务。

## 5. 文件级实施范围

### 5.1 可能新增文件

```text
alembic.ini
alembic/env.py
alembic/versions/20260705_01_create_agent_rollout_event.py
src/superbiz_agent/harness/history.py
src/superbiz_agent/harness/locks.py
src/superbiz_agent/harness/stores.py
src/superbiz_agent/persistence/models.py
src/superbiz_agent/persistence/repositories/__init__.py
src/superbiz_agent/persistence/repositories/rollout_events.py
tests/test_migration_p0_runtime.py
tests/test_rollout_event_repository.py
```

### 5.2 需要修改文件

```text
src/superbiz_agent/config.py
src/superbiz_agent/api/app.py
src/superbiz_agent/api/routes_chat.py
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/harness/service.py
src/superbiz_agent/harness/context_assembler.py
src/superbiz_agent/harness/trace_store.py
src/superbiz_agent/tools/gateway.py
src/superbiz_agent/evals/traces.py
tests/test_skeleton.py
tests/test_eval_runner.py
```

### 5.3 不应修改文件

```text
prompts/
src/superbiz_agent/rag/
src/superbiz_agent/memory/
docs/01-java-capability-inventory.md
docs/02-python-migration-spec.md
docs/03-python-architecture-design.md
docs/04-python-migration-roadmap.md
```

除非发现明确冲突，本阶段不修改前置设计文档。

## 6. 实施批次

### 批次 A：事件存储抽象和 replay 数据结构

目标：

- 引入 `RolloutEventStore` protocol。
- 让 `InMemoryRolloutEventStore` 实现该 protocol。
- 新增 replay helper，从 events 还原 active history。
- 不改 API 行为。

验收：

- 现有 19 个测试继续通过。
- 新增 replay helper 单测。

### 批次 B：Runtime 持久化 payload 与 session recovery

目标：

- `start_run` 先读取 session events 并恢复历史。
- replay 只读取当前 run 之前已经存在的 events，不能把当前 run 刚写入的 user message 重复拼进上下文。
- 写入 content payload。
- 记录 `THREAD_RECOVERED`。
- 引入 conversation lock。

验收：

- 同一 service 下连续两次 chat，第二次 context 有历史。
- clear 后历史清空。

### 批次 C：PostgreSQL repository 和 Alembic schema

目标：

- 建表迁移。
- SQLAlchemy model。
- `PostgresRolloutEventStore`。
- repository 强制 tenant filters。

验收：

- 不依赖真实 PostgreSQL 的单测通过。
- 如环境有 PostgreSQL，可手动运行 integration test，但不作为默认 pytest 必需条件。

### 批次 D：API 和 eval 回归

目标：

- 补 `GET /api/chat/session/{sessionId}`。
- 更新 eval trace artifact 和 suite。
- 增加 replay capability case。

验收：

- `python3 -m pytest` 全部通过。
- `PYTHONPATH=src python3 -m superbiz_agent.evals.runner` 通过。

## 7. 测试计划

新增/更新测试：

- `RolloutEventStore` memory implementation：
  - append/list_by_session/list_by_run/clear。
  - tenant/user/agent/session 隔离。
- replay helper：
  - user/assistant/tool result 事件能恢复为 `ModelMessage`。
  - 不应用 `RUN_FAILED` 等治理事件作为模型历史。
- runtime：
  - 首次请求无 recovered history。
  - 二次请求可恢复上轮历史。
  - `THREAD_RECOVERED` 只在有历史时记录。
  - content payload 被写入。
- conversation lock：
  - 同一 conversation key 复用同一 lock。
  - 不同 key 不互相阻塞。
- repository：
  - 生成的 SQL / 参数包含 tenant filters。
  - clear 只按 tenant/user/agent/session 删除。
- API：
  - `/api/chat` 仍兼容。
  - `/api/chat/clear` 清理 event store。
  - `/api/chat/session/{sessionId}` 返回 Java 兼容结构。
- eval：
  - `skeleton_p0_smoke` 继续通过。
  - 新增 replay case 通过。

## 8. 验收命令

subAgent 实施完成后必须运行：

```bash
python3 -m pytest
```

如果本地安装了 ruff，也运行：

```bash
python3 -m ruff check src tests
```

如果 ruff 未安装，说明即可，不作为失败。

可选手动命令：

```bash
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
```

## 9. subAgent 实施任务单

交给 subAgent 的任务应限制为：

```text
只实现 docs/07-migration-p0-implementation-plan.md 定义的 Migration P0。
优先分批实现：A -> B -> C -> D。
不得接真实模型、streaming、RAG、长期记忆、LangSmith、LLM-as-judge、生产认证、RLS、分布式锁。
默认 pytest 不得依赖本机 PostgreSQL。
实现完成后必须运行 python3 -m pytest。
提交结果时说明：
1. 修改/新增了哪些文件。
2. event store 抽象如何设计。
3. session replay 如何恢复 active history。
4. PostgreSQL schema/repository 做到什么程度。
5. API/eval 有哪些新增验收。
6. 测试命令和结果。
7. 是否有偏离计划的地方。
```

如果实现过程中发现计划和当前代码冲突，subAgent 必须停止并报告冲突，不得扩大范围自行重设计。

## 10. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否符合第 7 步 Migration P0 | 是 |
| 是否提前做 P1 真实模型 | 否 |
| 是否提前做 streaming | 否 |
| 是否提前做 RAG / LlamaIndex / Milvus | 否 |
| 是否提前做长期记忆 | 否 |
| 是否依赖 PostgreSQL 才能跑默认测试 | 否 |
| 是否保留 memory store 用于测试 | 是 |
| 是否实现 PostgreSQL-compatible rollout event store | 是，计划要求 |
| 是否实现 session replay/recovery | 是，计划要求 |
| 是否保持 tenant/user/agent/session 隔离 | 是，计划要求 |
| 是否保持 eval gate 可跑 | 是 |

## 11. 阶段完成标准

Migration P0 可以验收通过的条件：

1. 存在 `RolloutEventStore` 抽象。
2. Memory store 和 PostgreSQL store 都实现该抽象。
3. `agent_rollout_event` schema 与迁移规格一致。
4. Runtime run 开始时能按 tenant/user/agent/session 恢复历史。
5. Runtime 写入的 user/assistant/tool payload 可用于 replay。
6. clear 只清理当前 conversation。
7. API chat/clear/session 基本兼容。
8. deterministic eval runner 继续通过，并覆盖 replay case。
9. `python3 -m pytest` 全部通过。
10. 未实现本阶段明确排除的能力。
