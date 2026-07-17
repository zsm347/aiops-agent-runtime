# M-P1 真实 Embedding、pgvector 检索与生产检索基线

## 1. 文档状态

```text
阶段：M-P1
状态：implementation in progress
前置：M-P0 exact dedupe complete、M-P2 PostgreSQL persistence complete
目标：用真实 1024 维 embedding 和 PostgreSQL pgvector 建立可复现的生产检索基线
后续：M-P3 semantic dedupe、10G.2B 生命周期治理
```

本阶段不修改 M-P0 canonical exact key，不实现 semantic dedupe、rerank、后台遗忘、
48 x 3、Holdout 或聊天模型驱动的检索评测。

## 2. 现场审计

实施前事实：

- `long_term_memory.embedding` 已冻结为 `VECTOR(1024)`，允许 `NULL`。
- M-P2 对 local-deterministic 64 维向量只写 `NULL`。
- `MemorySearchService` 同步绑定 `DeterministicEmbeddingService`，读取全部候选后在 Python
  计算 cosine。
- 当前 `memory_search_min_similarity=0.5` 未经过真实检索标定。
- production retrieval ranking 为 `not_evaluated`。
- exact dedupe 已由数据库 partial unique index 和 M-P0 canonical SHA-256 冻结。

## 3. 依据

采用以下一手来源：

1. pgvector-python v0.5.0：
   <https://github.com/pgvector/pgvector-python/tree/v0.5.0>
   - `pgvector.sqlalchemy.VECTOR`
   - `VECTOR.cosine_distance()`
   - SQLAlchemy asyncpg 类型处理
2. OpenAI Embeddings API：
   <https://platform.openai.com/docs/api-reference/embeddings/create>
   - `input`、`model`、`dimensions`、`encoding_format`
   - response `data[index].embedding` 与 response `model`
3. DashScope text-embedding-v4：
   <https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api>
   - OpenAI-compatible API 支持 `model/input/dimensions/encoding_format`
   - 原生 API 的 `parameters.text_type` 区分 `query` 与 `document`
   - 1024 是 text-embedding-v4 的受支持维度

本仓库固定并验证 `pgvector==0.5.0`。OpenAI SDK 沿用项目依赖，验收记录实际版本。

## 4. 边界

### 4.1 必须实现

- 异步 `MemoryEmbeddingService` Protocol。
- 分离的 `embed_query` / `embed_documents` 领域语义。
- OpenAI-compatible adapter、严格请求/响应校验和安全错误。
- PostgreSQL 1024 维向量写入与 DB-side cosine nearest-neighbor。
- active Archival 的 dry-run / batch / resumable / CAS backfill。
- 独立 retrieval Dataset、manifest、runner、dev threshold/topK scan。
- 真实 API + 真实 PostgreSQL 门禁及逐项指标。

### 4.2 禁止实现

- embedding 相似度参与写入去重。
- M-P3 semantic dedupe 或近似重复治理。
- rerank、HNSW/IVFFlat 参数调优、生命周期 job。
- 修改 prompt、Memory Dataset v1、Judge、Graph、tool schema 或 RAG。
- 使用 deterministic/stub/skip 声称生产 ranking evaluated。

## 5. Embedding 契约

### 5.1 类型

`MemoryEmbeddingService` 暴露：

```text
embed_documents(texts) -> MemoryEmbeddingBatch
embed_query(text)      -> MemoryQueryEmbedding
aclose()               -> None
```

每个结果绑定：

```text
provider / model / version / dimension / ordered vectors
```

生产 dimension 必须精确为 1024。禁止 padding、截断、类型强转掩盖、版本伪造。

### 5.2 provider 请求

OpenAI-compatible 请求固定发送：

```text
input / model / dimensions=1024 / encoding_format=float
```

领域接口始终区分 query 与 document。DashScope 官方 OpenAI-compatible endpoint 当前未文档化
`text_type`，因此 adapter 不发送未受支持的私有字段。需要原生 `parameters.text_type` 的部署必须
新增独立 DashScope-native adapter 和门禁，不能在 compatible 请求中伪造支持。

### 5.3 校验

adapter 必须验证：

- 输入是非空字符串或非空字符串序列。
- batch 大小合法。
- response model 与配置 model 一致。
- response 数量与请求数量一致。
- index 唯一、完整、范围合法，恢复原请求顺序。
- 每个 vector 恰为 1024 维，且全部是有限实数、非 bool。
- provider 异常转换为稳定安全错误，不携带 request、URL、key 或正文。

local deterministic adapter 只允许 unit、fixture、Track A 和 local/test memory backend。
production PostgreSQL 配置不得构建 deterministic adapter。

## 6. 配置

Memory embedding 使用独立设置，不回退 model/RAG key：

```text
MEMORY_EMBEDDING_PROVIDER
MEMORY_EMBEDDING_MODEL
MEMORY_EMBEDDING_VERSION
MEMORY_EMBEDDING_DIMENSION
MEMORY_EMBEDDING_BATCH_SIZE
MEMORY_EMBEDDING_BASE_URL
MEMORY_EMBEDDING_API_KEY
MEMORY_EMBEDDING_TIMEOUT_MS
```

`MEMORY_STORE_BACKEND=postgres` 且处于非 local/test 环境时，local-deterministic 配置 fail
closed。真实 provider 缺 key、model/version/base 配置畸形或 dimension 非 1024 时 fail closed。

## 7. 写入与 exact dedupe

`ArchivalMemoryService` 在 policy 通过后调用 `embed_documents([content])`，把 provider 返回的
identity 和 1024 维 vector 写入候选。repository 继续只用 M-P0 canonical hash、identity、type、
scope 作为 exact conflict target。

embedding 不进入 dedupe key，不用于“相似即跳过”，不改变 duplicate tag merge。

## 8. PostgreSQL 检索

`MemorySearchService`：

1. 调用 `embed_query`。
2. 把 query vector、provider identity、filters、threshold、limit 交给 repository。
3. repository 在 SQL 中同时强制：
   - `tenant_id + user_id + agent_id`
   - `status='active'`
   - searchable type
   - optional service/env/tag filters
   - embedding 非 NULL
   - model/version/dimension/metric 精确匹配
4. 用 `embedding <=> query_vector` 排序，以 `1 - cosine_distance` 作为 similarity。
5. SQL 内应用 threshold 和 limit；禁止先取全量再 Python 排序。
6. 返回前重新验证 identity、active 状态和 embedding metadata。

usage 标记仍是 best-effort，但失败日志只能使用稳定事件名，不包含正文或身份。

in-memory repository 可为 unit/fixture 执行 deterministic Python scan；PostgreSQL repository
不得调用该实现或退回 scalar candidate scan。

## 9. Backfill

backfill 只选择 active 且 embedding missing/stale 的 Archival：

```text
embedding IS NULL
OR model/version/dimension/metric 与目标 identity 不一致
```

每批按稳定 ID 排序。流程：

1. 读取有限批候选及内部 CAS token（id、content hash、updated_at）。
2. dry-run 只统计，不调用 provider、不写数据库。
3. 对正文批量调用 `embed_documents`。
4. 每行 `UPDATE ... WHERE` 同时检查 active、原 content hash、原 updated_at、仍为 missing/stale。
5. CAS miss 计为 `concurrent_skipped`，绝不覆盖并发修改的正文或 metadata。
6. provider/DB 失败返回稳定错误码；重复运行只处理剩余 missing/stale，因此可恢复、幂等。

报告只包含 scanned/eligible/embedded/updated/concurrent_skipped/failed/batches 和稳定错误码，
不输出正文、identity、DSN、host、key。

## 10. Retrieval Dataset

新增独立 dev-only Dataset 和 manifest，不修改 `long_term_memory_v1.json` 或 Holdout。每个 fixture
使用稳定 evidence ID，case 覆盖：

- exact positive
- paraphrase
- hard negative
- no-match
- type/service/env/tag filter
- tenant/user/agent isolation

gold 只能引用 manifest 中的 fixture/evidence ID。Dataset 变更必须更新自身 SHA，不得为提高指标
修改 gold。

## 11. Runner 与指标

runner 直接构造生产 `MemorySearchService + PostgresMemoryRepository`，不调用聊天模型或 LLM
Judge。它在专用 PostgreSQL 数据库写入 fixture，执行 dev scan 并输出：

```text
HitRate@3
Recall@3
MRR
no-match false-positive rate
identity isolation violations
p50 / p95 latency
infrastructure failures
```

扫描多个 topK/min_similarity 点，输出信息保留、召回和误召回曲线。推荐阈值只能来自 dev；
不得使用 Holdout，不预设 0.5 正确，也不虚构质量目标。

只有同时满足以下条件，report 才可标记
`production_retrieval_ranking=evaluated`：

- provider 非 deterministic/stub。
- provider response identity 与 1024 维契约全部通过。
- 使用真实 PostgreSQL pgvector 写入和查询。
- executed case 完整、infrastructure failures 为 0。

质量低不是基础设施失败；必须如实提交 metrics 和 error analysis。

## 12. 验收

- fake client adapter 合同与反测试。
- 专用 PostgreSQL 的真实 vector write/backfill/query/restart/isolation。
- M-P2 39-case PostgreSQL gate 39/39、0 skip。
- 长期记忆、Memory eval、全量 stub、基础 eval。
- Ruff、compileall、pip check、diff check。
- prompt、Memory Dataset v1、Judge、Graph、OpenAI tool schema before/after SHA 相同。

## 13. 提交与状态

建议分阶段提交：

1. design + dependency + embedding contracts。
2. pgvector repository + runtime + backfill。
3. retrieval Dataset/runner + real acceptance。
4. evidence/status documentation。

PR 必须保持 Draft。真实 gate 未完成时状态只能是：

```text
implementation complete / real embedding or PostgreSQL baseline pending
```

真实 gate 完成后也只能标记：

```text
real embedding and PostgreSQL baseline executed / pending independent acceptance
```

不得直接标记 M-P1 complete。
