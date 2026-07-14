# 10F-C Milvus Store 与同步 Ingestion 实施计划

## 1. 阶段定位

本阶段对应 `10F RAG 真实管线` 的 Batch C：

```text
Milvus Store + Embedding Adapter + PostgreSQL RAG Metadata + 同步 Ingestion
```

Batch B 已完成文档加载与 chunking。本阶段把 `RagPreparedChunk` 变成带 dense embedding、BM25 sparse 字段和可过滤租户元数据的 Milvus 记录，并用 PostgreSQL 保存知识库与文档状态。

本阶段不实现模型工具检索、queryInternalDocs 真实切换、rerank、citation 输出、RAG trace 或专项 eval；这些分别属于 Batch D-F。

## 2. 实施依据

### 2.1 已验证依赖版本

当前环境：

```text
llama-index-core                    0.14.23
llama-index-vector-stores-milvus    1.1.0
pymilvus                            2.6.16
jieba                               0.42.1
openai                              2.45.0
```

本机 Milvus Lite 已验证以下组合可创建 collection 并 upsert：

```text
MilvusVectorStore
BM25BuiltInFunction
dense FLOAT_VECTOR
sparse SPARSE_FLOAT_VECTOR
显式 scalar fields
upsert_mode=True
Jieba search-mode analyzer + lowercase/removepunct
```

`jieba` 必须加入项目 `rag` optional dependency。缺失时初始化明确失败，不能静默退回默认 tokenizer。Milvus Lite 支持该 analyzer 的建表和写入，但不实现 `run_analyzer` RPC，因此 Lite 验收以真实 collection schema、中文写入和查询为准，不把 `run_analyzer` 作为门禁。

### 2.2 LlamaIndex 物理字段约束

`MilvusVectorStore.add()` 从 `TextNode.dict()[text_key]` 读取正文。当前 `TextNode` 的实际字段是 `text`，不是 `content`。

因此冻结两层契约：

```text
项目领域字段：RagPreparedChunk.content
Milvus 物理字段：text
```

`MilvusHybridChunkStore` 负责映射。Batch D 取回记录后仍归一化为项目既有 `content` 工具字段。不能为追求物理列名一致而绕过 LlamaIndex 或依赖私有 API。

## 3. Batch C 完成边界

完成后应具备：

```text
受信任后端输入
  -> 校验 tenant / knowledge base
  -> PostgreSQL claim document
  -> Batch B loader/chunker
  -> embedding batch
  -> LlamaIndex TextNode
  -> Milvus dense + BM25 upsert
  -> PostgreSQL document active
```

不具备：

```text
用户上传 API
后台 job / worker
模型主动入库
真实 queryInternalDocs 检索
hybrid query / tenant retrieval filter
rerank
RAG answer eval
旧文档物理删除或原地替换
```

## 4. Embedding 契约

### 4.1 项目接口

新增 `RagEmbeddingService` Protocol：

```text
async embed_documents(texts: Sequence[str]) -> RagEmbeddingBatch
```

`RagEmbeddingBatch` 至少包含：

```text
vectors
provider
model
version
dimension
```

硬校验：

- 输入不得为空或含空白文本。
- 输出数量必须与输入数量相同。
- provider 返回的 index 必须完整、唯一并按输入顺序重组。
- 分批调用时，每批独立校验并按批内 index 重排，再按 batch 顺序合并；provider index 不视为跨批次全局 index。
- 每个向量维度必须等于配置维度。
- 向量值必须是有限数，拒绝 NaN/Infinity。
- 凭证、完整文档和向量正文不写日志/trace。

### 4.2 实现

生产 adapter：

```text
OpenAICompatibleRagEmbeddingService
AsyncOpenAI.embeddings.create
model=text-embedding-v4
dimensions=1024
```

配置解析：

- API key 优先 `rag_embedding_api_key`，未配置时可回退同一受信任运行环境中的 `model_api_key`。
- Base URL 优先 `rag_embedding_base_url`，未配置时可回退 `model_base_url`。
- key/base URL 不能来自工具参数、prompt 或文档 metadata。
- 不在测试中调用真实 embedding API。

测试 adapter：

```text
DeterministicRagEmbeddingService
```

只用于 unit/Milvus Lite 测试，必须稳定、可指定 dimension，不能被生产 factory 在 `rag_enabled=true` 时静默选中。

### 4.3 配置补充

新增：

```text
rag_embedding_version: optional，未配置时解析为 rag_embedding_model
rag_embedding_batch_size: default 32, >=1
rag_chunk_size_tokens: default 1024, >=1
rag_chunk_overlap_tokens: default 100, >=0 且 < chunk_size
rag_ingestion_stale_after_seconds: default 900, >=1
```

现有 embedding model/dimension 配置继续使用。

## 5. PostgreSQL 数据模型

### 5.1 rag_knowledge_base

字段：

```text
id                  backend-generated UUID string, global primary key
tenant_id           required
name                required
description         nullable
status              active | archived
is_default          boolean
created_by          required trusted backend identity
created_at
updated_at
```

约束：

- 每个 tenant 最多一个 `status=active AND is_default=true` 的知识库。
- 使用 PostgreSQL partial unique index；不只靠应用层先查后写。
- 增加 `UNIQUE(tenant_id, id)`，供文档表建立数据库级租户归属复合外键。
- `status` 使用 CheckConstraint 限定为 `active | archived`。
- repository 所有查询和更新都必须携带 tenant_id。
- 模型和普通 RAG 检索工具不能创建或选择知识库。

### 5.2 rag_document

字段：

```text
id                  backend-generated UUID string, global primary key
tenant_id
knowledge_base_id
source_uri
document_name
content_type
parser
parser_version
embedding_provider
embedding_model
embedding_version
embedding_dimension
status              indexing | active | failed | archived
content_hash
chunk_count
claim_token          nullable UUID string，仅 indexing 状态持有
claimed_at           nullable timestamptz，仅 indexing 状态持有
failure_code         nullable
failure_message      nullable, sanitized and bounded
created_by
created_at
updated_at
```

约束：

- 使用复合外键 `(tenant_id, knowledge_base_id) -> rag_knowledge_base(tenant_id, id)`，数据库层拒绝跨 tenant 归属；应用层 tenant filter 仍必须保留。
- `(tenant_id, knowledge_base_id, content_hash)` 全状态唯一。
- 同一知识库、相同规范化输入永远复用同一 document row 和全局 document_id。
- document_id 不能由租户或模型自定义；由后端 UUID 生成，保证共享 Milvus collection 的 chunk primary key 不跨租户碰撞。
- `chunk_count >= 0`。
- status 和 embedding_dimension 有 CheckConstraint。

### 5.3 文档 content_hash

文档级 hash 与 chunk `content_hash` 不同。

计算：

```text
canonical_content_type
+ normalized full content（CRLF/CR -> LF，首尾 trim）
-> deterministic JSON
-> SHA-256
```

content type 必须进入 hash，因为相同文本按 Markdown 与 parsed text 解析可能产生不同 chunk。

P0 冻结同一 Milvus collection 的 chunking/embedding 配置。变更 parser、chunking 或 embedding 模型需要新 collection/reindex 方案，不能在本阶段把同一 content_hash 偷偷混入两套向量配置。

## 6. Repository 契约与状态机

### 6.1 Knowledge base repository

至少提供：

```text
create_default(...)
get_active_for_tenant(tenant_id, knowledge_base_id)
get_default_for_tenant(tenant_id)
```

所有 statement 可独立编译检查 tenant filter。

### 6.2 Document claim

`claim_for_ingestion` 在一个独立短事务中按 tenant + knowledge base + content_hash claim。首次创建使用 PostgreSQL `INSERT ... ON CONFLICT DO NOTHING RETURNING`，避免唯一约束错误污染当前事务；未返回行时再读取竞争者已创建的记录。已有行的状态转换使用 `SELECT ... FOR UPDATE`，不使用 `SKIP LOCKED`：

同一短事务内必须先按 `tenant_id + knowledge_base_id + status=active` 重新读取并锁定知识库，再执行 document claim；不能只依赖 ingestion service 事务外的预检查，避免知识库在检查与 claim 之间被归档。

```text
不存在：生成全局 document UUID 和 claim_token，插入 indexing，claimed_at 使用数据库 CURRENT_TIMESTAMP
active：返回 duplicate_skipped，不调用 chunk/embed/Milvus
indexing 且未过 stale threshold：返回 in_progress，不重复执行
failed：复用同一 row/document_id，生成新 claim_token，切回 indexing
stale indexing：基于数据库时钟和 claimed_at 判断，生成新 claim_token 后接管
archived：P0 不自动复活，明确拒绝并要求后续生命周期策略
```

`claim_for_ingestion` 必须完成写入和 commit 后才能返回；commit 失败不得返回 claim。chunk、embedding、Milvus 等外部操作期间不得持有数据库事务或行锁。

failed/stale 重试必须更新 `claim_token/claimed_at/updated_at`，清空 failure 字段并把 chunk_count 重置为 0；不得改变 document_id/content_hash/created_at。archived 不更新任何字段。

`create_default` 的并发冲突同样不能泄漏原始 `IntegrityError`：使用 savepoint 或 rollback 后的新事务读取既有 default KB，并返回明确结果。

### 6.3 终态更新

```text
mark_active(tenant_id, document_id, claim_token, chunk_count)
mark_failed(tenant_id, document_id, claim_token, failure_code, sanitized_message)
```

两个终态更新各自使用独立短事务并 commit，必须限定 `tenant_id + document_id + status=indexing + claim_token`。更新 rowcount 必须为 1；否则抛出 `claim_lost`，避免 stale worker 覆盖新 worker。`mark_active` 因 claim 丢失失败时不得再执行 `mark_failed`；补偿错误不能覆盖原始异常。

`mark_active/mark_failed` 必须在同一原子更新中清空 `claim_token/claimed_at`。数据库增加状态一致性 CheckConstraint：`status=indexing` 当且仅当 claim_token 和 claimed_at 均非空。

新增 `refresh_claim(tenant_id, document_id, claim_token)`：按相同 fencing 条件用数据库时间刷新 claimed_at，rowcount 必须为 1。ingestion 至少在 embedding 前、Milvus upsert 前续租；续租失败立即停止后续副作用。默认 900 秒是 lease timeout，不是单次 ingestion 的硬超时。

## 7. MilvusHybridChunkStore

### 7.1 项目接口

定义窄 Protocol：

```text
async ensure_ready() -> None
async upsert(chunks, embedding_batch) -> tuple[str, ...]
```

Batch C 不暴露 query 接口；Batch D 在同一 adapter 上增加受 tenant filter 约束的检索。

### 7.2 Collection 初始化

固定：

```text
overwrite=False
upsert_mode=True
doc_id_field=document_id
text_key=text
embedding_field=dense_vector
sparse_embedding_field=sparse_vector
enable_dense=True
enable_sparse=True
sparse_embedding_function=BM25BuiltInFunction(text -> sparse_vector)
similarity_metric=COSINE
dense index=FLAT + COSINE
sparse index=SPARSE_INVERTED_INDEX + BM25
```

BM25 function 使用固定名称和中文 analyzer：

```python
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

BM25BuiltInFunction(
    function_name="rag_text_bm25",
    input_field_names="text",
    output_field_names="sparse_vector",
    analyzer_params={
        "tokenizer": {
            "type": "jieba",
            "dict": ["_default_"],
            "mode": "search",
            "hmm": True,
        },
        "filter": ["lowercase", "removepunct"],
    },
)
```

显式 scalar fields：

```text
tenant_id                VARCHAR
knowledge_base_id        VARCHAR
document_name            VARCHAR
source_uri               VARCHAR
heading_path_json        VARCHAR
section_title            VARCHAR，空字符串表示无标题
page_start               INT64，0 表示未知
page_end                 INT64，0 表示未知
page_metadata_json       VARCHAR
chunk_index              INT64
content_hash             VARCHAR
embedding_provider       VARCHAR
embedding_model          VARCHAR
embedding_version        VARCHAR
embedding_dimension      INT64
created_at               VARCHAR ISO-8601 UTC
```

说明：当前 LlamaIndex scalar-field API 不暴露逐字段 nullable 配置，因此可空 page/title 使用受控 sentinel；Batch D 输出时必须归一化回 `None`。

初始化顺序必须是：先用独立 `MilvusClient` preflight；collection 不存在才由 LlamaIndex 创建，已存在则必须通过全部兼容性校验后才构造 `MilvusVectorStore`。构造完成后无条件执行一次相同的 postflight 全量校验，关闭“preflight 后被其他实例创建不兼容 collection”的 TOCTOU 窗口。不能只在构造后检查，因为构造过程可能创建索引或改变既有 collection。

若 collection 已存在，preflight 必须校验：

- dense dimension。
- dense/sparse 字段和类型。
- 固定名称 `rag_text_bm25` 的 BM25 function 类型、input/output。
- `text` 启用了 analyzer，且 analyzer 为冻结的 Jieba search-mode 配置。
- 所有安全过滤 scalar fields。
- 主键、document_id 和 text 字段。
- dense index 固定 `FLAT + COSINE`，sparse index 固定 `SPARSE_INVERTED_INDEX + BM25`；按字段定位并要求每个向量字段只有一个预期索引。

不兼容时 fail closed；不得 overwrite、自动降维或创建无 tenant scalar 的降级 collection。

### 7.3 TextNode 映射

每个 chunk 映射成 LlamaIndex `TextNode`：

```text
id_ = chunk_id
text = chunk.content
embedding = 对应 dense vector
SOURCE relationship node_id = document_id
metadata = 所有显式 scalar fields
```

必须设置 SOURCE relationship，否则 LlamaIndex 会把自定义 `document_id` 覆盖成字符串 `None`。

adapter 返回 ID 必须与输入 chunk_id 完全一致；但 LlamaIndex 的返回 ID 来自输入节点，不能单独证明服务端写入成功。Milvus Lite 集成测试必须在 upsert 后按主键读取实际记录并验证。

生产写入优先使用框架原生 `await store.async_add(nodes)`；preflight/初始化中的同步 client 调用通过 `asyncio.to_thread` 隔离。不得阻塞 async ingestion event loop。

## 8. 同步 Ingestion Service

### 8.1 输入

新增不含 document_id 的受信任请求：

```text
RagIngestionRequest
tenant_id
knowledge_base_id
document_name
source_uri
content
content_type
created_by
parser / parser_version
page metadata（可选）
```

`tenant_id` 和 `knowledge_base_id` 由可信管理入口传入，不来自模型。repository 再验证知识库确属该 tenant 且 active。

### 8.2 执行顺序

`ingest_sync` 表示调用方等待整个工作流完成，不表示使用同步阻塞 I/O；实现为 async method：

```text
1. 规范化输入并计算 document content_hash
2. 验证 active knowledge base
3. claim document
4. duplicate/in_progress 直接返回，不调用后续组件
5. 用 claim 得到的全局 document_id 构造 RagSourceDocument
6. Batch B chunker
7. 空 chunk -> mark_failed + raise
8. 分批 embedding，合并并校验结果
9. Milvus upsert
10. PostgreSQL mark_active
11. 返回 indexed result
```

### 8.3 失败与补偿

PostgreSQL 与 Milvus 无法组成一个原子事务。P0 使用可重试状态机：

- chunk/embed/Milvus 任一步失败，当前 worker 持有 claim 时最佳努力 `mark_failed`，然后抛出 `RagIngestionError`。
- `mark_active` 返回 claim_lost 时不得再 mark_failed，避免旧 worker 覆盖新 claim。
- 错误只保存 stage/code 和经过统一清洗、长度限制的 message，不保存原始文档、向量或凭证。
- Milvus 可能已经部分 upsert；重试 failed/stale row 时复用同一 document_id，因此产生相同 chunk_id，`upsert_mode=True` 覆盖同一记录，不制造新 orphan ID。
- 如果 PostgreSQL 本身不可用导致 `mark_failed` 也失败，row 可能保持 indexing；超过 stale threshold 后允许复用相同 row 重试。
- 本阶段不物理删除未知范围的 Milvus 记录，避免补偿删除误伤已成功重试数据。

## 9. 返回契约

```text
indexed:
  document_id, content_hash, chunk_count

duplicate_skipped:
  existing document_id, content_hash, chunk_count

in_progress:
  existing document_id, content_hash
```

失败抛出带 `document_id`（若已 claim）、stage 和安全 message 的领域异常；不得返回看似成功的空结果。

## 10. 测试策略

### 10.1 Embedding

- OpenAI-compatible request model/input/dimensions 正确。
- provider 乱序响应按 index 恢复。
- 缺失/重复 index、数量错、维度错、NaN/Infinity 拒绝。
- fake embedding 稳定且不被生产 factory 误用。

### 10.2 PostgreSQL contract

- migration upgrade/downgrade 对称。
- partial default-KB unique index 存在。
- document scope/hash unique constraint 存在。
- status/chunk_count/dimension check 存在。
- repository select/update statement 均包含 tenant_id。
- fake session/repository 覆盖 claim 状态机；普通单测不要求常驻 PostgreSQL。
- Batch C 标记 complete 前必须执行一次真实 PostgreSQL 验收，覆盖两个独立 session 并发首次 claim、stale claim fencing、partial default index、复合 tenant FK 和 migration upgrade/downgrade。若当前环境没有可用 PostgreSQL，则 Batch C 只能记录为“代码完成、真实 PostgreSQL 门禁待验”，不能标记 complete。

### 10.3 Milvus adapter

- mock 构造参数准确且 `overwrite=False/upsert_mode=True/BM25/Jieba`。
- TextNode ID、text、SOURCE relationship、scalar metadata 正确。
- page/title sentinel 与 JSON metadata 正确。
- ID 数量/集合不匹配时失败。
- Milvus Lite integration：创建实际 dense+sparse collection、中文 upsert、按主键读取、按 tenant scalar query、重复 ID 覆盖；没有 rag extra/jieba 时显式 skip，不回退 fake collection。
- existing incompatible schema fail closed。

### 10.4 Ingestion

- indexed 正常顺序：claim -> chunk -> embed -> upsert -> active。
- active duplicate 不调用 chunk/embed/store。
- non-stale indexing 不重复执行。
- failed/stale 重试复用相同 document_id 和稳定 chunk_id。
- 空 chunk、embedding、Milvus、mark_active 失败均进入失败路径。
- partial Milvus 后重试不生成新 IDs。
- tenant 与 knowledge base 不匹配在任何 embedding/upsert 前拒绝。
- 不记录正文、向量和凭证到 failure metadata。

### 10.5 回归

- Batch B loader/chunking 全部通过。
- queryInternalDocs fixture/disabled 行为不变。
- memory、Harness、全量测试不回退。

## 11. 文件边界

允许新增/修改：

```text
src/superbiz_agent/config.py
.env.example
pyproject.toml
src/superbiz_agent/rag/models.py
src/superbiz_agent/rag/embedding.py
src/superbiz_agent/rag/milvus_store.py
src/superbiz_agent/rag/ingestion.py
src/superbiz_agent/rag/__init__.py
src/superbiz_agent/persistence/models.py
src/superbiz_agent/persistence/repositories/rag.py
src/superbiz_agent/persistence/repositories/__init__.py
alembic/versions/20260712_01_create_rag_metadata.py
tests/test_rag_embedding.py
tests/test_rag_milvus_store.py
tests/test_rag_ingestion.py
tests/test_rag_persistence.py
docs/10F-rag-real-pipeline-plan.md
docs/04-python-migration-roadmap.md
docs/10-advanced-capabilities-plan.md
docs/current-work-handoff.md
```

禁止修改：

```text
.env
prompts/
src/superbiz_agent/model_gateway/
src/superbiz_agent/harness/graph.py
src/superbiz_agent/tools/
src/superbiz_agent/memory/
src/superbiz_agent/evals/memory_*
evals/datasets/long_term_memory_v1.json
src/superbiz_agent/rag/retrieval.py
tests/test_rag_tool.py
```

Batch C 不接 Agent 工具，不修改 queryInternalDocs。

## 12. 实施拆分

### C1：Persistence

独立写集：

```text
persistence/models.py
persistence/repositories/rag.py
Alembic migration
tests/test_rag_persistence.py
```

### C2：Embedding + Milvus Store

独立写集：

```text
src/superbiz_agent/config.py
.env.example
pyproject.toml
rag/embedding.py
rag/milvus_store.py
tests/test_rag_embedding.py
tests/test_rag_milvus_store.py
```

C1/C2 可以并行。

### C3：Ingestion integration

在 C1/C2 合同稳定并主审后再实现：

```text
rag/models.py
rag/ingestion.py
rag/__init__.py
tests/test_rag_ingestion.py
```

不得让 C3 worker 自行改写 C1/C2 的基础合同；发现冲突先返回设计层。

## 13. 验收命令

```bash
MODEL_PROVIDER=stub PROMPT_VERSION=ops-agent-system-v3 TOOL_SCHEMA_VERSION=ops-tools-v2 python -m pytest tests/test_rag_chunking.py tests/test_rag_embedding.py tests/test_rag_milvus_store.py tests/test_rag_ingestion.py tests/test_rag_persistence.py -q
MODEL_PROVIDER=stub PROMPT_VERSION=ops-agent-system-v3 TOOL_SCHEMA_VERSION=ops-tools-v2 python -m pytest -q
MODEL_PROVIDER=stub PROMPT_VERSION=ops-agent-system-v3 TOOL_SCHEMA_VERSION=ops-tools-v2 PYTHONPATH=src python -m superbiz_agent.evals.runner
python -m ruff check <Batch C changed files>
python -m compileall -q src tests
```

普通验收不得调用真实 embedding API、远程 Milvus 或本机 `.env` 凭证。

## 14. 完成定义

Batch C 只有同时满足以下条件才可标记 complete：

1. PostgreSQL metadata migration、model、repository 合同完整。
2. document_id 全局生成，content hash 幂等和并发 claim 有数据库约束。
3. embedding adapter/fake adapter 通过完整 shape/安全校验。
4. Milvus dense + BM25 schema/upsert 通过 mock 与 Milvus Lite contract。
5. 所有安全过滤字段为显式 scalar，existing incompatible schema fail closed。
6. ingestion 状态机可从 failed/stale/partial upsert 使用相同 IDs 重试。
7. duplicate/in-progress 不产生 embedding 或 Milvus 副作用。
8. 没有实现或修改 Batch D-F 能力。
9. 专项、全量、基础 eval、Ruff、compileall 全部通过。
10. 真实 PostgreSQL 并发、claim fencing、partial index 和复合 tenant FK 门禁通过；环境不可用时不得把 Batch C 标记 complete。

Batch C complete 不代表真实检索、tenant query filter、rerank、citation 或 RAG eval 已完成。

## 15. 实施与验收状态（2026-07-12）

当前状态：

```text
implementation complete / Milvus gate passed / real PostgreSQL gate pending
```

实施期间经过三轮设计复审、C1/C2 独立代码审查、两轮返工和 C3 独立安全复审。已修复的关键问题包括：

- claim 使用 fencing token、数据库时钟 lease 和 heartbeat，避免 stale worker 覆盖新 worker。
- Milvus 使用固定 Jieba BM25、preflight/postflight 和冻结索引配置，避免 schema 漂移与 TOCTOU。
- RAG 凭证不进入 Settings 默认 repr；embedding provider 异常响应统一受控失败。
- content hash 在 repository 与 PostgreSQL constraint 两层限定为小写 SHA-256。
- ingestion 安全错误不保留敏感异常链；取消会最佳努力释放 claim 后重新抛出。
- C3 再校验每个向量的 dimension、数值类型和 finite，替代 adapter 不能绕过门禁。

最终主验收：

```text
Batch B/C RAG pytest: 89 passed
full stub pytest: 259 passed, 1 existing warning
base eval: 14/14 passed
Milvus Lite integration: passed
Ruff: passed
compileall: passed
C3 final independent review: PASS
```

当前机器没有可用 PostgreSQL、Docker 或 Podman，完成定义第 10 项尚未执行。因此不得把 Batch C 标记为完整 `complete`。下一实施阶段可以推进 Batch D，但发布前必须补齐真实 PostgreSQL 门禁。
