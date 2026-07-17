# M-P2 PostgreSQL 长期记忆持久化实施计划

## 1. 文档状态

```text
阶段：M-P2
状态：complete / independent acceptance passed
目标：把 Core Memory 与 Archival Memory 的唯一事实源从进程内存迁移到 PostgreSQL
前置：M-R1 complete、M-P0 complete
后续：M-P1 真实 Embedding 与生产检索、M-P3 语义去重、10G.2B 生命周期治理
```

本计划冻结 M-P2 的实现和验收边界。当前实现已通过真实 PostgreSQL gate 与技术负责人
独立验收，M-P2 状态为 complete；这不表示长期记忆整体完成。

## 2. 设计依据与优先级

设计依据按以下优先级解释：

1. `docs/M-P0-archival-dedup-contract.md`：当前 Archival exact dedupe 的最高优先级契约。
2. `docs/02-python-migration-spec.md`：Core/Archival 使用 PostgreSQL、tenant/user/agent 隔离的迁移目标。
3. 当前 Python 实现与测试：工具返回、trace、Core block 顺序、hash 和 metadata merge 的事实行为。
4. `docs/09-long-term-memory-migration-plan.md`：只保留未被 M-P0 覆盖的部分。
5. Java PostgreSQL repository：只作为字段语义和兼容性参考，不照搬其弱于 M-P0 的去重与并发实现。

以下旧设计已经失效，M-P2 不得恢复：

- 只按 `tenant + user + agent + content_hash` 去重。
- 使用 local-deterministic embedding 或 `0.92` 阈值跳过写入。
- 把同进程锁当作跨 worker 原子保证。
- 配置为 `postgres` 时静默退回内存。

## 3. 阶段目标

M-P2 必须实现：

1. `memory_store_backend=memory|postgres` 的严格后端选择。
2. PostgreSQL 持久化 Core Memory block。
3. PostgreSQL 持久化 Archival Memory 内容及 metadata。
4. M-P0 exact key 的数据库级跨连接唯一性与原子 tag merge。
5. Core block 的数据库乐观并发控制，避免已知版本被静默覆盖。
6. 进程重启、两个独立 runtime 和两个独立数据库连接之间的恢复与一致性。
7. 所有 Memory 生产 I/O 使用异步接口，不在 FastAPI event loop 中执行同步数据库请求。
8. PostgreSQL runtime 的 engine 生命周期、关闭和 fail-closed 行为。
9. memory backend、迁移和真实 PostgreSQL 的专项验收门禁。

## 4. 明确不做

M-P2 不实现：

- 真实 Embedding provider。
- PostgreSQL pgvector 相似度查询、IVFFlat/HNSW 或生产检索排名。
- 语义去重、same/related/distinct/conflict Judge。
- Archival update/delete/archive 工具。
- 后台 extraction、consolidation、遗忘、TTL 或衰减。
- Recall Memory 或历史对话语义召回。
- Tool schema、生产 prompt、Memory Dataset 或 Judge 修改。
- 正式 Track B `48 x 3`、Holdout 或真实模型调用。
- PostgreSQL RLS、跨 Memory 与 trace 的 outbox/unit-of-work。
- RAG、上下文压缩、真实 streaming 或其他高级能力修复。

## 5. 当前事实与设计阻塞

### 5.1 同步 Memory API 与异步 PostgreSQL 不兼容

当前 Core、Archival、Search、Index、ContextProvider 和 `InMemoryMemoryStore` 都是同步接口；项目数据库基础设施使用 `AsyncEngine`、`AsyncSession` 和 `async_sessionmaker`。

M-P2 不允许：

- 在 async 请求线程中调用同步 PostgreSQL driver。
- 使用 `run_until_complete` 嵌套 event loop。
- 用 `asyncio.to_thread` 长期包装整套数据库访问。
- 为了少改调用方而维护一套 sync PostgreSQL repository。

### 5.2 现有数据库索引不能保证 M-P0 exact dedupe

M-P0 exact key 为：

```text
tenant_id
+ user_id
+ agent_id
+ type
+ scope_service
+ scope_env
+ canonical_content_hash
```

仅 `status=active` 的记录参与唯一性；`topic/tags/session/source` 不参与。

现有数据库只有普通 identity/hash 索引，无法防止两个进程同时插入同一 exact key。

### 5.3 64 维临时 embedding 与 `VECTOR(1024)` 不兼容

当前运行时使用 64 维 `local-deterministic` embedding；数据库列固定为 `VECTOR(1024)`。M-P2 在 M-P1 之前实施时，不能把 64 维向量填充、截断或伪装成 1024 维写入。

### 5.4 Core version 目前没有端到端并发保护

Core block 虽有 `version`，但当前更新没有 expected version 条件。不同 session 可以同时基于旧 block 生成完整 replacement，最后提交者可能覆盖先提交者。

### 5.5 PostgreSQL Memory 写入与 rollout trace 不是同一事务

Memory 写入先完成，trace 后追加。M-P2 不在本阶段增加 outbox，因此必须明确失败语义，不能宣称两者原子一致。

## 6. 总体架构

```text
Memory tools / ContextProvider / eval snapshot
                    |
            async Memory services
                    |
            MemoryRepository Port
              /             \
 InMemoryMemoryRepository   PostgresMemoryRepository
        |                           |
 existing sync M-P0 store    AsyncSession / PostgreSQL
```

设计原则：

- Service 只依赖 async Port，不依赖 SQLAlchemy model。
- 现有同步 `InMemoryMemoryStore` 保留为 M-P0 低层实现和确定性测试对象。
- 增加薄的 async in-memory adapter；不得重写 M-P0 的 canonical、exact key 和 tag merge 算法。
- PostgreSQL adapter 独立实现数据库事务，但必须通过同一组 adapter contract tests。
- fixture/admin seeding 与生产 repository 接口分离；正式 PostgreSQL runtime 不接受 `eval_fixture` 等测试专用 source。

## 7. Async Memory Repository 契约

新增一个最小的 async Port，至少覆盖：

```python
class MemoryRepository(Protocol):
    async def ensure_default_core_blocks(...): ...
    async def list_active_core_blocks(...): ...
    async def list_core_blocks_for_scope(...): ...
    async def cas_replace_core_content(...): ...

    async def write_archival_exact(...): ...
    async def list_active_memories(...): ...
    async def list_memories_for_scope(...): ...
    async def mark_returned(...): ...
```

topic/tag/scope metadata 可以由 repository 查询或由 service 从 scoped active records 聚合，但两个 adapter 必须保持同一排序和过滤语义。

所有 repository 查询必须显式包含：

```text
tenant_id + user_id + agent_id
```

即使已经有 memory `id`，也不能只按 `id` 更新、计数或读取。

### 7.1 异步传播边界

具体签名传播冻结为：`ContextManager.prepare` 与其 `_build_memory_context` 改为 async，
`ConversationRuntime._prepare_context_if_enabled` 必须 await；`ContextAssembler.assemble` 的
non-prepared fallback 也改为 async 并 await provider，而 `assemble_prepared` 保持纯同步。
两个 `MemoryContextProvider` Protocol 定义统一接收 `RunContext`，不能继续只传三个 identity
字符串而丢失 run_id。

以下方法迁移为 async：

- Core：load、update、build context。
- Archival：save。
- Search：search、list topics、usage update。
- Memory Index/Metadata：build、count。
- Memory ContextProvider：build context。
- Memory eval Snapshot/Fixture 调用链中依赖生产 Service/Repository 的部分。
- ContextManager 的 Memory 预取路径。
- ContextAssembler 的非 prepared fallback 路径。
- MemoryToolHandlers 对 Core/Archival/Search/Topics 的调用。
- MemoryEvalRunner 对 fixture、snapshot 和 service 生命周期的调用。

纯函数继续同步：

- canonicalization/hash。
- policy 规则。
- XML/metadata render。
- cosine similarity。
- tool schema 生成。

### 7.2 Runtime、inspection 与 fixture 边界

`MemoryRuntimeComponents` 不再把 `InMemoryMemoryStore` 当作所有 backend 的公共事实接口，
而是显式暴露 async `repository`、只读 `inspection_repository` 和 run snapshot registry。
测试专用 `fixture_admin` 只允许在 `memory` backend 构造；`postgres` backend 请求 fixture
写入时必须在执行 SQL 前拒绝。生产代码不得通过可选的 `.store` 绕过 repository 事务。

`MemoryFixtureSeeder`、`MemorySnapshotProvider` 和 `MemoryEvalRunner` 随调用链 async 化。
Runner 必须在 `finally` 中 `await service.aclose()`。本阶段的 Memory Track A/Track B 仍只使用
`memory` backend；真实 PostgreSQL 验收使用独立的 repository/runtime integration tests，
不得把 `eval_fixture` source 放入生产表，也不得因此修改 Dataset 或 Judge。

### 7.3 Run 级 Core version snapshot

新增进程内、backend-only 的 `CoreVersionSnapshotRegistry`，由一个 Memory runtime 独占：

```text
run_id -> tenant/user/agent + {block_key: version}
```

- `MemoryContextProvider.build_context` 改为接收 `RunContext`，Core blocks 只读取一次；同一批
  blocks 用于 XML render、block count 和 version capture，不能为了计数再次读取数据库。
- snapshot 只能在整个 Memory context 成功组装后 `capture_once`；同一 run 再次 capture 不得
  用数据库新版本静默替换模型实际看到的版本，并且 identity 不一致时 fail closed。
- Core CAS 成功或确认目标内容已经存在后，只推进对应 block 的 snapshot version。
- `ConversationRuntime.cleanup_run` 必须释放 snapshot；complete、failed、cancelled、异常和
  显式重复 cleanup 都要覆盖，释放操作必须幂等。
- snapshot 不持久化、不放入 tool args、不允许模型提供或修改。Core XML 中既有 version
  metadata 保持不变，但 CAS 使用 registry 中的可信版本。

## 8. Backend 选择与生命周期

### 8.1 配置

`memory_store_backend` 只允许：

```text
memory
postgres
```

未知值在 Settings 构造或 Harness 启动时直接失败。

```text
memory_enabled=false
```

时不构造 Memory runtime；否则：

- `memory`：构造原有 store + async adapter。
- `postgres`：构造独立 AsyncEngine、sessionmaker 和 PostgreSQL repository。

PostgreSQL 初始化、查询或写入失败时不得回退到 memory backend，也不得返回空记忆伪装成功。

### 8.2 Runtime 生命周期

`MemoryRuntimeComponents` 增加并发安全、可重复调用的 `aclose()`：

- memory backend：no-op。
- postgres backend：dispose 自己拥有的 engine。
- waiter 取消不能中断共享关闭任务。
- 第一次关闭部分失败时，后续调用可以继续清理。

`AgentHarnessService.aclose()` 必须关闭 Memory runtime 和 RAG runtime，并汇总资源标签；不得因一个资源关闭失败而跳过另一个资源。

Runtime 不自动执行 Alembic migration。部署或测试必须在启动 Harness 前显式执行 `alembic upgrade head`；schema 缺失或版本不兼容时 fail closed。

### 8.3 PostgreSQL readiness/capability probe

不能只靠部署说明假设 migration 已执行。PostgreSQL Memory runtime 增加共享、可取消等待且
只执行一次的 async readiness task；FastAPI lifespan 通过 `AgentHarnessService.astart()` 主动
执行，`chat/chat_stream` 也在直接使用 service 时防御性确保 ready。

probe 通过 `pg_catalog`/`information_schema` 验证 M-P2 命名的 active exact unique index、
关键 hash/scope/tags/Core integrity constraints、Core unique constraint 和 vector 列维度仍与
冻结契约一致。不得只判断表和列存在，因为旧 revision 已经包含这些列，却没有 M-P2 的
跨进程唯一性。未来 Alembic head 可以继续前进，但这些 capability 不得消失或漂移。

probe 失败映射为安全的 `memory_store_contract_error`，不得开始 context 读取或 Memory 写入，
不得回退内存。真实 PostgreSQL 测试必须证明停留在 M-P2 前一个 revision 时 runtime 拒绝
ready；应用 migration 后同一门禁通过。probe 和关闭共用与 RAG runtime 一致的共享 task、
waiter cancellation 和 retry-safe cleanup 规则。

## 9. M-P2 在 M-P1 之前的 Embedding 过渡契约

这是用户明确选择 M-P2 先于 M-P1 后的阶段性兼容方案。

### 9.1 写入

当 embedding provider 为 `local-deterministic`（当前默认 dimension 为 64）：

- PostgreSQL `embedding` 写 `NULL`。
- 仍持久化实际 `embedding_model`、实际正整数 `embedding_dimension`、metric 和 version。
- 禁止 padding 到 1024。
- 禁止修改现有 `VECTOR(1024)` 列为 64 维。
- 禁止增加临时 JSONB embedding 列。

### 9.2 读取与搜索

M-P2 的 PostgreSQL repository 读取内容/metadata；SearchService 在内存中：

1. 对查询生成与当前 deterministic adapter 同维度的 embedding。
2. 对缺失或与当前 deterministic adapter 不兼容的 memory embedding，根据正文临时重算。
3. 保持现有 cosine、topK、threshold、filter 和排序语义。
4. 不把临时重算结果回写 PostgreSQL。

因此该过渡契约不把配置硬编码为 64，但任何 local-deterministic 向量都必须写为 SQL
`NULL`；不得因为把 dimension 改为其他值就尝试写入 `VECTOR(1024)`。repository 还必须
拒绝维度 metadata 非正数、provider/model/version 缺失或非有限向量等畸形领域输入。

M-P2 的 scalar read path 不依赖 SQLAlchemy/asyncpg 解码 `VECTOR(1024)`：查询使用明确的 scalar column projection，不把 vector 列作为本阶段领域对象的事实来源；映射后的临时 `embedding=[]`，随后由 SearchService 重算。已有 1024 维 vector 原值不得被 M-P2 的普通读取、duplicate merge 或 Core 操作覆盖。

这只保证 M-P2 行为兼容和重启恢复，不构成生产检索基线。报告必须继续标记：

```text
production retrieval ranking = not_evaluated
```

M-P1 后再负责真实 1024 维 embedding、backfill、pgvector 查询和排名校准。

## 10. Core Memory PostgreSQL 契约

### 10.1 默认 block 原子初始化

第一次加载 identity scope 时：

```text
INSERT default blocks ON CONFLICT
  (tenant_id, user_id, agent_id, block_key) DO NOTHING
-> SELECT active default blocks
-> 按 user_rules / user_ops_profile / service_notes 顺序返回
```

整个 insert、active select、archived/缺失检查必须位于同一事务；发现任一 default key 已
archived、未知或最终仍缺失时回滚本次插入，不能留下另外两个半初始化 row。并发 runtime
初始化后，每个 key 只能有一行。

已有 row 的 description、max_tokens、source、created_at 不得在普通读取时隐式改写。

如果 default key 已存在但为 `archived`，M-P2 不擅自复活或覆盖；返回类型化 contract error。Core archive/reactivate 语义留给生命周期阶段。

### 10.2 不变行为

- 相同 normalized Core hash 返回 `unchanged`。
- `unchanged` 不改变 version 和 `updated_at`。
- 成功更新 version 恰好 `+1`。
- read-only block 不更新。
- Core hash 继续使用当前 `content_hash()` 规则，不改用 Archival canonical hash：`None` 或
  `content.strip()` 为空时 hash 为 `""`；否则对 `content.strip()` 的 UTF-8 计算 SHA-256。

### 10.3 CAS

数据库更新必须包含：

```text
tenant_id
+ user_id
+ agent_id
+ block_key
+ status=active
+ read_only=false
+ version=expected_version
```

并执行：

```text
SET content=?, content_hash=?, version=version+1, updated_at=CURRENT_TIMESTAMP
```

为避免让模型生成 `expectedVersion`，M-P2 不修改 tool schema。ContextProvider 按 7.3 的
registry 保存模型实际看到的 block version；CoreService 更新时只从可信 snapshot 读取
expected version。生产 tool 路径缺少 snapshot 时返回 `update_context_missing` 并 fail closed，
禁止“执行时读取最新版本后覆盖”的弱 fallback。测试初始化和 admin fixture 必须走独立
fixture/admin port，不得借生产更新路径绕过 snapshot。

CAS 失败后：

- 如果 read_only/status 改变，返回对应拒绝。
- 否则如果数据库当前 hash 已等于目标 hash，返回 `unchanged`，并把本 run 对应 snapshot
  推进到数据库当前 version。
- 其他情况返回 `update_conflict`，不得自动用新版本重试完整 replacement。
- 面向模型仍使用既有 `rejected` 信封，不新增模型可控 identity/version 参数。

同一 run 成功更新后，snapshot 更新为新 version，支持该 run 后续合法的第二次更新。
并发把 block 切换为 read-only 即使没有增加 version，也必须因 SQL predicate 拒绝；验收
必须覆盖该竞争条件。

## 11. Archival exact dedupe PostgreSQL 契约

### 11.1 数据库唯一性

新增 active partial unique expression index：

```text
tenant_id
+ user_id
+ agent_id
+ type
+ COALESCE(scope_service, '')
+ COALESCE(scope_env, '')
+ content_hash
WHERE status = 'active'
```

冻结对象名称，供 migration、SQLAlchemy model、DML inference 和 readiness 共用：

```text
uq_long_term_memory_active_exact                 # partial unique expression index
chk_long_term_memory_active_content_hash
chk_long_term_memory_scope_service_nonblank
chk_long_term_memory_scope_env_nonblank
chk_long_term_memory_tags_array
chk_core_memory_block_key
chk_core_memory_version_positive
chk_core_memory_max_tokens_positive
chk_core_memory_content_hash_format
```

既有 `uq_core_memory_block` 保持不变。实现不得让 migration、model 和 readiness 各自复制
不同名称或不同 predicate。

repository 和 migration 使用与现有 `_blank_to_none` 相同的 Python `strip()` 规范 scope；
数据库同时增加 `scope IS NULL OR btrim(scope) <> ''` 的最低 check，禁止常见纯空白字符串，
确保 `NULL` 与 `''` 不会产生歧义。repository 读取到任何 Python `strip()` 后为空的非 NULL
scope 时仍须 fail closed，不能把 DB check 当成完整 Unicode 空白验证。

active Archival 的 `content_hash` 必须是 64 位小写 SHA-256。数据库只能校验格式；repository 必须根据正文重新计算 canonical hash，不能信任 caller 传入值。

### 11.2 原子 insert-or-merge

单个事务内：

```text
规范 scope/tags
-> 重新计算 canonical hash
-> INSERT ... ON CONFLICT (<完整 expression key>) WHERE status='active'
   DO NOTHING RETURNING row
-> 插入成功：written
-> 未插入：按完整 exact key SELECT ... FOR UPDATE
-> 稳定顺序合并 tags
-> 仅 tags 变化时更新 tags/updated_at
-> duplicate_skipped
-> COMMIT 后返回
```

conflict target 只能由 repository 使用固定字段、固定 `COALESCE(scope, '')` expression 和
固定 active predicate 构造，不能接受模型/调用方提供的 column、filter 或 raw expression。
SQLAlchemy PostgreSQL dialect 的实际编译 SQL 和真实 PostgreSQL inference 必须先通过专项
门禁；若当前依赖版本不能可靠命中该 partial expression unique index，实施必须停止并回到
设计审核，不能退化为“先查后插”或进程锁。

显式 exact target 的目的，是让 primary key 或其他 unique constraint 冲突抛出
`IntegrityError` 并映射为 `memory_exact_conflict_unresolved`。不得使用不带 target 的裸
`ON CONFLICT DO NOTHING`：当 exact row 与 candidate ID 冲突同时存在时，只查询 exact key
无法证明究竟命中了哪个约束，可能把主键冲突误报成 duplicate。

写事务固定使用 PostgreSQL `READ COMMITTED`。同 exact key 的 concurrent insert 由 unique
index 串行收敛；随后 `SELECT FOR UPDATE` 串行执行 stable tag merge。不得依赖更高隔离级别
下未经设计的 serialization 行为。

### 11.3 必须保留的 M-P0 语义

- topic、tags、session、source 不参与 exact key。
- 保留旧 id/topic/content/session/source/embedding/created_at。
- tags 为旧顺序优先、追加 incoming 新 tag、区分大小写。
- 没有新 tag 时 `metadataMerged=false` 且不改 `updated_at`。
- archived 不阻止同 exact key 新 active 写入。
- 不同 identity/type/scope 必须分别写入。
- canonical 不同即使 deterministic similarity=1.0 也必须分别写入。

## 12. 查询、Metadata 与 usage 契约

### 12.1 查询

只读取 `status=active`，并保持：

- type filter。
- scopeService/scopeEnv 精确、区分大小写。
- tags any-match、区分大小写。
- topK/minSimilarity。
- similarity desc，再以 id 做确定性 tie-break。

### 12.2 Memory Metadata

必须保持：

- archival total。
- topic count。
- available tags。
- available service/env scopes。
- 现有 token budget 和 fallback。
- 模型可见名称为 Memory Metadata，内部兼容字段名可以继续保留。

为避免把即将持久化的数据直接破坏 XML 边界，M-P2 至少补齐：

- tags/scopeService/scopeEnv 的敏感信息检查。
- topic/tag/scope 输出 XML escaping。

更完整的 prompt-injection 策略、字符 allowlist 和 provenance 不在本阶段扩展。

### 12.3 usage

返回检索结果后，使用 scoped SQL 原子执行：

```text
usage_count = usage_count + 1
last_used_at = CURRENT_TIMESTAMP
updated_at = CURRENT_TIMESTAMP
```

usage 更新保持 best-effort：失败不能抹掉已经成功取得的检索结果，但必须产生安全日志或诊断信号。不得跨 identity 更新同 id 记录。

## 13. Migration 设计

新增 Alembic revision，`down_revision` 指向当前 head；不得修改已经发布的 `20260705_02` 或 RAG migration。

upgrade 顺序：

1. 把空白 `scope_service/scope_env` 规范为 `NULL`。
2. 使用与 M-P0 相同、在 revision 内冻结的 Python canonicalizer 分批回填 active memory
   `content_hash`；不得一次把无界正文表全部载入进程。
3. 使用当前 Core `content_hash()` 规则回填 `agent_core_memory_block.content_hash`：空白内容
   为 `""`，非空内容为 stripped content 的 SHA-256。
4. 拒绝 canonical 后为空的 active Archival，不能替它生成有效 hash。
5. 检查 tags 必须为 JSON array 且每项为字符串；发现旧脏数据时 fail closed。
6. 检测完整 exact key 的 active 重复组。
7. 如果发现重复，迁移 fail closed 并报告有限数量的记录 ID；M-P2 不猜测合并 topic/source/session，也不静默删除数据。
8. 增加命名的 integrity checks：active Archival hash 为 64 位小写 hex；非 NULL scope 经
   PostgreSQL `btrim` 后非空；tags 在数据库层至少为 JSON array；Core block_key 只允许三个
   defaults、version >= 1、max_tokens > 0、content_hash 非 NULL 且为 `""` 或 64 位小写 hex。
   PostgreSQL 普通 CHECK 不得虚构“遍历 JSONB 每个元素”的能力；每项为字符串由 migration
   preflight 和 repository 读写校验共同保证，除非实际验证过的 immutable DB expression
   可以无辅助函数实现。不得为此临时创建未经审核的数据库函数。
9. 创建 active partial unique expression index。
10. 更新 SQLAlchemy model 与 migration 约束保持一致。

revision 内 canonicalizer 不 import 当前应用 helper，避免未来 helper 修改导致旧 migration 漂移；专项测试必须证明 revision 算法与 M-P0 当前算法一致。

preflight 或 migration 异常只能报告有限数量的 row ID、总计数和安全错误类型，不得输出
正文、tags、DSN、凭证或原始 driver error。模型和应用 runtime 不负责自动修复脏数据。

downgrade：

- 删除本 revision 新增的 index/check。
- 不尝试恢复旧 content hash 或空白 scope；数据规范化不可逆，必须在文档中明确。

## 14. 错误与事务语义

新增安全的类型化 persistence errors，至少区分：

```text
memory_store_unavailable
memory_store_contract_error
memory_store_isolation_error
core_update_conflict
core_block_inactive
update_context_missing
memory_exact_conflict_unresolved
```

新增 `memory/errors.py`，复用现有 ToolGateway 的异常属性协议，不扩展 tool schema：unavailable
错误只暴露固定安全消息和 503/retryable 属性；contract/isolation/conflict 错误声明
`retryable=false`、`retryable_by_model=false` 和受控 next actions。异常构造器不接收并拼接
原始 SQL/driver 文本。Core 的 conflict/inactive/missing-snapshot 继续返回既有 `rejected`
工具信封；repository 连接/契约异常由 ToolGateway 转成现有结构化 ToolErrorResult。

SQL、DSN、凭证、原始 driver error 不进入模型工具结果。

Memory transaction commit 成功后才允许返回 `written/updated`。commit 失败不得返回成功。

Memory 与 rollout trace 在 M-P2 仍为两个事务：

- Memory 成功、trace 失败时，Memory 不回滚。
- Archival 重试依靠 exact dedupe 收敛为 duplicate。
- Core 相同内容重试收敛为 unchanged。
- 本阶段报告必须披露该限制；outbox 留到观测/可靠性阶段。

## 15. 文件边界

预计允许修改或新增：

```text
src/superbiz_agent/config.py
src/superbiz_agent/api/app.py
src/superbiz_agent/memory/ports.py                       # new
src/superbiz_agent/memory/errors.py                      # new
src/superbiz_agent/memory/run_snapshots.py               # new
src/superbiz_agent/memory/adapters/__init__.py           # new
src/superbiz_agent/memory/adapters/in_memory.py           # new
src/superbiz_agent/memory/store.py
src/superbiz_agent/memory/core.py
src/superbiz_agent/memory/archival.py
src/superbiz_agent/memory/search.py
src/superbiz_agent/memory/index.py
src/superbiz_agent/memory/runtime.py
src/superbiz_agent/memory/tools.py
src/superbiz_agent/memory/policy.py
src/superbiz_agent/memory/schemas.py
src/superbiz_agent/harness/context_assembler.py
src/superbiz_agent/harness/context_manager.py
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/harness/service.py
src/superbiz_agent/persistence/models.py
src/superbiz_agent/persistence/repositories/memory.py       # new
src/superbiz_agent/persistence/repositories/__init__.py
alembic/versions/<new_m-p2_revision>.py                    # new
src/superbiz_agent/evals/memory_fixtures.py
src/superbiz_agent/evals/memory_snapshots.py
src/superbiz_agent/evals/memory_runner.py
tests/test_long_term_memory.py
tests/test_memory_persistence.py                            # new
tests/test_memory_eval_*.py                                # 仅 async 传播所需
.env.example
README.md                                                  # 仅 backend/migration 运行说明
docs/04-python-migration-roadmap.md                        # 验收后更新状态
```

实际实施可以缩小文件集合，但不得扩展到 prompt、Dataset、Judge、RAG 或模型行为。

## 16. 实施批次与内部门禁

以下是同一个大批次内的实施顺序，不要求每个小项与用户反复交接：

### D1：Port 与 async 传播

- 定义 async repository port 和结果 DTO。
- 增加 in-memory adapter。
- Core/Archival/Search/Index/ContextProvider async 化。
- 实现 run-scoped Core version registry、strict missing-snapshot rejection 和全退出路径 cleanup。
- Eval fixture/snapshot/runner async 化，并保持只允许 memory backend。
- 保持 M-P0 低层 sync store contract tests。
- 跑 memory unit、eval unit 和 stub 回归。

### D2：Schema、migration 与 PostgreSQL repository

- 增加 migration/model constraints。
- 实现 Core initialize/CAS。
- 实现 Archival insert-or-merge。
- 实现 scoped read、metadata、usage。
- 实现命名 schema capability probe；旧 revision 必须 fail closed。
- 通过 SQL 编译、fake-session transaction 和 migration tests。

### D3：Runtime 与生命周期

- 严格 backend 选择。
- 构造/注入 PostgreSQL repository。
- engine close、Harness close、no-fallback。
- FastAPI lifespan `astart()` 与 direct service 防御性 readiness。
- 更新 `.env.example` 和运行说明；补充 Memory 配置，并把已经过期的示例 `ops-tools-v2` 更正为当前可启动的 `ops-tools-v3`，但不修改实际 tool schema。

### D4：真实 PostgreSQL 验收

- migration upgrade。
- 重启恢复。
- 双 runtime/双连接并发。
- 两个 OS subprocess 的写入、退出、恢复与 concurrent exact write。
- exact dedupe/tag merge/CAS/tenant isolation。
- migration downgrade 仅在隔离数据库验证。

任一内部门禁失败，必须先修复并重跑；不能带着失败继续到下一子批次。

## 17. 测试矩阵

### 17.1 Adapter contract

同一测试语义必须覆盖 in-memory adapter 和 PostgreSQL adapter：

- 三个 Core defaults 的顺序和幂等初始化。
- Core updated/unchanged/read-only/conflict。
- Core snapshot capture-once、成功推进、identity mismatch 和 complete/fail/cancel cleanup。
- 缺少 Core snapshot 的生产 tool update fail closed。
- Archival exact key 全字段隔离。
- NULL scopes 和空白 scope 规范化。
- stable tag union、metadata preservation、updated_at。
- archived 不阻止 active。
- type/scope/identity 分离。
- active filter、tags any-match。
- usage 原子增加。
- inspection API 不初始化数据。

### 17.2 Repository unit

- SQL statement 必须带 tenant/user/agent。
- Core UPDATE 必须带 expected version、active status 和 `read_only=false`。
- insert 必须使用固定 partial expression conflict target；裸 `ON CONFLICT DO NOTHING` 禁止。
- commit 一次后才返回成功。
- commit/execute 失败映射为安全类型化错误。
- primary-key 冲突不能误报 exact duplicate。
- malformed JSONB/vector/metadata fail closed。
- 一个 Core default archived、另外两个缺失时整次初始化回滚，不留下半初始化 row。

### 17.3 Migration

- upgrade/downgrade schema 操作对称。
- canonical backfill 与 M-P0 snapshot 一致。
- duplicate preflight 会阻止 unique index 创建。
- canonical-empty、非数组/非字符串 tags、active duplicates、非法 Core version/max_tokens/hash
  的 preflight 会 fail closed，且错误不输出内容。
- hash/scope/tags/version/block-key constraints 与 model 和 readiness probe 一致。
- Alembic head 单一。

### 17.4 真实 PostgreSQL，禁止 SQLite 替代

使用显式 `M_P2_TEST_DATABASE_URL`，不读取或输出 `.env`/凭证；测试数据库或 schema 必须与开发/生产数据隔离。必须验证：

1. 全量 migration 可应用。
2. runtime A 写入 Core 和 Archival，close 后 runtime B 可恢复。
3. M-P2 前一 revision 上 readiness 拒绝；upgrade 后 readiness 通过。
4. 两个独立 engine/sessionmaker 并发初始化只产生三个 Core row；archived default 的失败
   不留下其他半初始化 row。
5. 24 个并发 exact write 最终只有一个 active，所有 tag 均被合并。
6. 两个相同 expected version、不同目标内容的 Core CAS 必须恰好一个 `updated`、一个
   `conflict`；相同目标允许一个 `updated`、一个 `unchanged`，禁止两者都失败冒充通过。
7. 管理员并发切换 `read_only=true` 且不增加 version 时旧 CAS 仍被拒绝。
8. 至少两个使用 `spawn` 的 OS subprocess 各自创建 engine：A 写入并退出后 B 能恢复；两进程
   concurrent exact write 最终一条 active 且 tags 全合并。DSN 只传递不输出。
9. 不同 tenant/user/agent/type/scope 不互相去重或读取。
10. archived exact row 不阻止新 active。
11. usage_count 在并发返回后无丢增量。
12. local-deterministic 64 维及另一个非 1024 测试维度写入后 `embedding IS NULL`，实际
    model/dimension/version metadata 正确；已有 1024 向量在 scalar read、exact duplicate
    tag merge 和 Core 操作前后的 `embedding::text` 完全不变。
13. 数据库不可用时不回退内存。
14. engine close 后连接资源释放，重复关闭安全。

canonical-empty、畸形 tags、active exact duplicates 和非法 Core version/max_tokens/hash 的
migration negative cases 必须在真实 PostgreSQL 隔离 schema 上执行，不能只验证生成的 SQL
字符串或用 SQLite 替代。每个失败 case 都要证明 revision 未被错误标记为已应用，且日志和
异常不包含脏正文或 DSN。

真实数据库测试必须用 `Settings(_env_file=None, ...)` 或等价显式构造，防止 BaseSettings
隐式读取项目 `.env`。测试仅从 `M_P2_TEST_DATABASE_URL` 获取隔离 DSN；变量缺失时可以 skip，
但报告必须把 PostgreSQL gate 标记 pending，不能把 skip 算通过。

### 17.5 回归

- 长期记忆专项。
- Memory eval unit/snapshot/runner unit。
- 全量 `MODEL_PROVIDER=stub`。
- 基础 eval 14/14。
- Ruff。
- `compileall src tests`。
- `pip check`。
- 实施开始前和结束后比较禁止范围：prompt 文件、Memory Dataset、Judge 源码不得变化；
  canonical OpenAI tool schema JSON hash 必须一致。哈希由实施 Agent 在修改前现场记录，
  不能用历史 artifact fingerprint 代替当前工作区事实。

不得运行：

- Track B 正式 baseline。
- Holdout。
- 真实对话模型。
- 真实 Embedding。

## 18. 完成定义

只有同时满足以下条件，M-P2 才能标记 complete：

1. `memory` backend 回归全部通过。
2. `postgres` backend 真实数据库测试全部通过。
3. readiness 在旧 revision fail closed、在 M-P2 schema 通过。
4. 重启恢复、两个独立连接和两个 OS subprocess 并发通过。
5. M-P0 exact dedupe 在数据库约束和事务层均成立，主键冲突不误报 duplicate。
6. Core CAS 不发生静默覆盖，缺少 run snapshot 时严格拒绝。
7. local-deterministic embedding 未写入 `VECTOR(1024)`。
8. PostgreSQL 失败不会静默回退内存。
9. Tool schema、prompt、Dataset、Judge 哈希保持不变。
10. 无 `.env`、凭证或内部 SQL 错误泄露。
11. 独立验收明确通过。

如果缺少真实 PostgreSQL 环境，只能标记：

```text
implementation complete / PostgreSQL acceptance pending
```

当前分支在 2026-07-16 使用 PostgreSQL 16.14 与 pgvector 0.8.5 完成权威入口验收：

```text
python scripts/run_memory_postgres_acceptance.py
planned=39 executed=39 skipped=0 passed=39 failed=0 status=passed
```

同轮回归为 M-P2 persistence/runtime/snapshot `137 passed`、长期记忆 `24 passed`、Memory
eval `86 passed`、RAG B/C/D `198 passed`、全量 stub `519 passed, 39 skipped`、基础 eval
`14/14`。

2026-07-17，技术负责人使用全新一次性 PostgreSQL 16.14 + pgvector 0.8.5 隔离集群
独立复跑权威入口，结果同为 `39 passed / 0 skipped`；并复跑 persistence/runtime/snapshot
`137 passed`、长期记忆 `24 passed`、Memory eval `86 passed`、全量 stub
`519 passed / 39 skipped`、基础 eval `14/14`。代码审查未发现 M-P2 阻断项，冻结资产无越界
变更。因此当前状态为：

```text
complete / independent acceptance passed
```

## 19. 已知残余风险

M-P2 完成后仍存在：

- local-deterministic 检索不代表生产语义质量。
- M-P2 的 scoped candidate scan 和 Python cosine 不代表生产规模能力。
- PostgreSQL vector 列尚未由 M-P1 正式接管。
- Memory 与 trace 不是同一事务。
- PostgreSQL rollout event engine 的统一所有权和关闭仍是独立已知问题。
- 没有 Archival 生命周期和冲突治理。
- 没有 PostgreSQL RLS。
- run snapshot 是进程内并发控制辅助状态；进程崩溃会丢失，但 run 同时终止，不影响
  PostgreSQL 记忆事实。正常退出路径必须清理，防止常驻进程泄漏。
- 更完整的持久化 prompt-injection、provenance 和 metadata allowlist 仍待安全阶段。

这些限制必须写入验收报告，不能被测试通过掩盖。

## 20. 实施停止规则

出现以下任一情况，实施 Agent 必须停止并报告，不得自由改设计：

- SQLAlchemy PostgreSQL dialect 无法编译并由真实 PostgreSQL 正确 inference 冻结的 partial
  expression conflict target。
- 实际 PostgreSQL 版本或 pgvector extension 与 migration 不兼容。
- 现有数据库发现 active exact duplicates 或 canonical-empty 数据。
- async 传播需要修改 prompt/tool schema/Dataset 才能继续。
- 无法为 ContextProvider 建立并在所有退出路径清理严格的 run-level Core snapshot。
- 真实 PostgreSQL 无法获得，导致关键门禁不能执行。
- 需要写入任意 local-deterministic 向量到 `VECTOR(1024)` 才能让测试通过。
- 需要恢复语义去重或修改 M-P0 exact key。

## 21. 设计自审清单

- [x] 没有越过 M-P2 提前实现 M-P1/M-P3/生命周期。
- [x] 没有重新发明数据库框架，复用 SQLAlchemy async/Alembic。
- [x] M-P0 exact key、canonical 和 metadata merge 全部保留。
- [x] 没有让模型生成 tenant/user/agent 或 expected version。
- [x] 没有把 64 维向量伪装成 1024 维。
- [x] run snapshot 的 capture、推进、identity 绑定和全退出路径 cleanup 已冻结。
- [x] 缺失 snapshot、旧 schema、数据库故障均 fail closed。
- [x] Core CAS 同时约束 active、read_only 和 expected version。
- [x] exact insert 使用显式 partial expression conflict target，主键冲突不会误报 duplicate。
- [x] Core blank hash 与现有 `content_hash()` 行为一致。
- [x] migration 没有虚构 PostgreSQL JSONB CHECK 能力。
- [x] 真实 PostgreSQL 门禁包含旧 revision、脏数据和两个 OS subprocess。
- [x] PostgreSQL 失败语义为 fail closed。
- [x] 真实 PostgreSQL 是完成门禁，不用 SQLite 冒充。
- [x] 技术负责人已在全新隔离 PostgreSQL 集群独立复跑并明确验收通过。
- [x] prompt、tool schema、Dataset、Judge 不在修改范围。
- [x] 已披露 Memory/trace 非原子限制。
- [x] 已区分 implementation complete 与 M-P2 complete。
