# M-P1 真实 Embedding、pgvector 检索与生产检索基线

## 1. 文档状态

```text
阶段：M-P1
状态：implementation complete / pending independent acceptance
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

## 14. 2026-07-17 原始实施与验收证据（已由 M-P1-R1 取代）

> 本节保留原始执行历史。原 report 缺少逐 case observation，identity canary 被计入 ranking
> denominator，且 `0.4` 选点未通过 hard-negative quality gate，因此不得再作为可复核的
> production candidate 证据。当前结论以第 15 节为准。

当前分支已完成实现和执行门禁，状态为：

```text
historical baseline executed / superseded by M-P1-R1
```

这不是 M-P1 complete；该历史结果已被 M-P1-R1 撤回为非 candidate 证据。

### 14.1 实现结果

- 新增异步 `MemoryEmbeddingService`，生产服务不再绑定 deterministic 具体类型。
- 新增独立 Memory provider/model/version/dimension/batch/base URL/API key/timeout 配置；真实
  provider 未配置独立 key 时 fail closed，产品代码不回退 model 或 RAG credential。
- 固定 `pgvector==0.5.0`，SQLAlchemy 模型和查询使用官方 `VECTOR(1024)` 与
  `cosine_distance()`。
- PostgreSQL query 在 SQL 内强制 tenant/user/agent、active、type、scope、tag、embedding
  identity、threshold、order 与 limit；返回后由 repository 和 service 两层复核。
- local-deterministic 仅保留 unit/fixture/Track A 路径；PostgreSQL 过渡写入继续保存 `NULL`
  vector，不 padding、不截断。
- backfill 支持 dry-run、批次、ID keyset、重复执行、失败恢复与 content/hash/updated_at CAS；
  报告只含计数和稳定错误码。
- exact dedupe 仍只由 M-P0 canonical hash 和 frozen exact key 决定，embedding 不参与写入
  去重。
- 新增独立 dev Dataset/manifest 和稳定 fixture/evidence ID；未修改 Memory Dataset v1、Judge
  或 Holdout。

### 14.2 真实 PostgreSQL 门禁

环境：PostgreSQL `16.14 (Homebrew)`、server pgvector `0.8.5`、Python pgvector `0.5.0`。
所有 URL 与凭证只注入进程，未写入文档、报告或环境文件。

```text
python scripts/run_memory_retrieval_postgres_acceptance.py
planned=7 executed=7 skipped=0 passed=7 failed=0 status=passed

python scripts/run_memory_postgres_acceptance.py
planned=39 executed=39 skipped=0 passed=39 failed=0 status=passed
```

首次 M-P1 PG 全量运行在 setup 阶段失败，7 个 case 均未执行。根因是 M-P2 权威门禁的最终
downgrade case 按设计把一次性数据库留在 base，M-P1 fixture 在 Alembic upgrade 前尝试清理
不存在的表。修复为 M-P1 权威脚本在 URL、数据库名、数据库 comment 和显式确认四重保护后
先执行 `upgrade head`。最小 vector write/restart case 随后 `1 passed`，完整复跑 `7 passed`。

### 14.3 真实 embedding + PostgreSQL dev baseline

原始执行没有独立 `MEMORY_EMBEDDING_*` credential，而是在验收包装进程内映射了 Chat
compatible credential。M-P1-R1 已禁止这种做法；该运行不能充当整改后的真实 baseline。

```text
planned=12 executed=12 skipped=0 infrastructure_failures=0
production_retrieval_ranking=evaluated
historical selected topK=3 min_similarity=0.4 (withdrawn)
```

选定 dev 点的结果：

| metric | value |
|---|---:|
| HitRate@3 | 1.000 |
| Recall@3 | 0.950 |
| MRR | 1.000 |
| no-match false-positive rate | 0.000 |
| identity isolation violations | 0 |
| forbidden result violations | 1 |
| mean results/query | 1.667 |
| p50 latency | 185.93 ms |
| p95 latency | 287.42 ms |
| infrastructure failures | 0 |

`topK=3` threshold 曲线：

| min similarity | HitRate@3 | Recall@3 | MRR | no-match FPR | forbidden |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| 0.2 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| 0.4 | 1.00 | 0.95 | 1.00 | 0.00 | 1 |
| 0.5 | 0.90 | 0.90 | 0.90 | 0.00 | 1 |
| 0.6 | 0.80 | 0.80 | 0.80 | 0.00 | 0 |
| 0.7 | 0.60 | 0.60 | 0.60 | 0.00 | 0 |

该历史选择规则没有把 hard-negative forbidden hit 作为硬门禁，因此 `0.4` 结论已撤回。

质量错误分析保持原 gold 不变：

- `MPR03` hard negative 在选定点仍返回 1 个 forbidden evidence。
- `MPR05` scope filter 的两个 relevant evidence 只召回一个，因此总体 Recall@3 为 `0.95`。
- 小型 dev fixture 的延迟不是生产规模容量结论；尚未引入 ANN index、rerank 或 M-P3 semantic
  dedupe。

### 14.4 回归

```text
M-P1/M-P2 persistence/runtime/embedding/retrieval unit: 148 passed
long-term memory:                                      24 passed
Memory eval dataset/judges/runner/snapshots:           86 passed
RAG B/C/D with constraints/rag-integration.txt:       198 passed
MODEL_PROVIDER=stub full suite:                       559 passed, 46 skipped
basic eval runner:                                     14/14 passed
Ruff changed Python files:                             passed
compileall -q src tests scripts:                       passed
pip check:                                             passed
git diff --check main...HEAD:                         passed
```

全量 stub 的 46 skips 是未注入专用 URL 时明确 pending 的 39 个 M-P2 PG case 与 7 个 M-P1 PG
case；真实门禁统计以上方独立权威入口为准。

### 14.5 冻结资产 SHA-256 before/after

```text
84cef93089ae4932350842786ead4cf8c92df2964e05213aeae49db9fe568b49  prompts/ops-agent-system-v3.md
df34b4b851f89c827e2bfdf67ffcfc167a5dd3b2f349d2423b2df3926953ff0f  evals/datasets/long_term_memory_v1.json
029f18c70971146164255a93594b6d9007072a6385a36953a061b05caba302ea  src/superbiz_agent/evals/memory_judges.py
4593330fbc29c9186e6192ca1f0728dfde519c83542cd65e62a67096b0124b70  src/superbiz_agent/harness/graph.py
eadf90360e0268fe60f0b6259cbd5285a9d00ca6900e3637b198afbc1d5ca817  canonical OpenAI tool schema JSON
```

before 与 after 完全一致。新增 retrieval Dataset SHA-256 为
`ba3e3854f68cb0cc0e12ab287026c45a4f70008a6bf3540b40a5d4b3dc321a88`，并由独立 manifest
固定。

## 15. M-P1-R1 评测整改

### 15.1 当前状态

```text
implementation complete / pending independent acceptance
```

PR #3 继续保持 Draft。没有修改生产 prompt、主 Memory Dataset/Judge、Graph、RAG、tool
schema 或生产默认 threshold；没有进入 M-P3。

### 15.2 Dataset 语义修正

Dataset 从 `1.0.0` 更新到 `1.0.1`。唯一语义变更是 MPR10-MPR12：

- 查询身份仍为 primary，跨 tenant/user/agent canary 仍分别保留在 forbidden IDs。
- primary scope 下没有真正相关 gold，因此 `relevant_evidence_ids` 改为空。
- `expected_empty=true`，任何非空结果都会使 isolation case 失败。
- MPR01-MPR09、全部 fixture/query/content 和三条跨身份 forbidden canary 均未修改。

```text
before: ba3e3854f68cb0cc0e12ab287026c45a4f70008a6bf3540b40a5d4b3dc321a88
after:  007b2c17505784f53ab8937f19d243949a7899a7d6da256de62639d147ac3cbe
```

### 15.3 三条独立评测轨道

- Semantic ranking：MPR01-MPR07；只有这 7 个 case 进入 HitRate@3、Recall@3、MRR 分母。
- No-match：MPR08-MPR09；单独计算 false-positive case 数和 FPR。
- Identity isolation：MPR10-MPR12；单独计算 case pass/fail、identity violation、forbidden
  evidence violation 和 unexpected non-empty result。

每个 report observation 现在固定包含 case/category/track、query SHA-256、实际 topK/threshold、
有序 evidence ID/score/rank、query latency、relevant/forbidden IDs 与 matches、unexpected IDs、
identity violation 和 case pass/fail；不记录 query 原文、正文、DSN、host 或 credential。

### 15.4 Scan、quality gate 与 candidate

一次 superset scan 使用 `topK=max(scan topK)`、`min_similarity=-1`。它的延迟字段明确命名为
`superset_scan_p50/p95_latency_ms`，只用于描述 scan，不再冒充 candidate latency。

production candidate quality gate 必须同时满足：

1. 全部 case 执行且 infrastructure failure 为 0。
2. no-match FPR 为 0。
3. semantic hard-negative forbidden violation 为 0。
4. identity isolation violation 为 0。
5. isolation forbidden evidence violation 为 0。
6. 三个 isolation case 全部通过。

只有 gate 通过的 `topK=3` 点才参与 MRR/Recall/HitRate 排序。没有通过点时 selected config 为
`null`、candidate status 为 `no_candidate`。候选产生后，runner 会以该候选的真实 topK 和
threshold 再执行全部 query，单独记录 actual candidate p50/p95，并再次检查同一 quality gate。
report 同时记录 Pareto frontier 和质量残差。

这只是 12-case dev pilot，不是 Holdout、规模压测、ANN 结论或正式 production calibration。
即使产生 dev pilot candidate，也不修改生产默认 `memory_search_min_similarity=0.5`。

### 15.5 可复核 artifact

report 使用稳定、排序、紧凑 JSON 序列化。每个成功真实 baseline 同时生成
`<report>.manifest.json`，固定 report filename、report SHA-256 和 Dataset SHA-256。对应脱敏
report/manifest 必须纳入 PR 证据。测试覆盖 report schema、逐 case observation、稳定序列化、
Dataset drift 和 report manifest hash。

本轮运行时 Settings 未配置独立 `MEMORY_EMBEDDING_*` provider/dimension/key/base URL。权威
runner 在任何 API 调用前返回：

```text
status=pending reason=MEMORY_EMBEDDING_configuration_missing exit_code=3
```

因此本轮没有真实 embedding API 调用，没有生成整改后的 report/manifest，report SHA 为
`not_generated (pending)`，production candidate 结论为 `pending`。禁止使用 stub 或从
Chat/RAG credential 映射来填补该证据。

### 15.6 验收结果

新建空 PostgreSQL 16.14 + pgvector 0.8.5 一次性数据库，执行后已删除数据库和角色并停止
服务：

```text
M-P1 retrieval unit/embedding/backfill: 42 passed
M-P1 PostgreSQL gate:                  7 passed, 0 skipped
M-P2 PostgreSQL regression:           39 passed, 0 skipped
MODEL_PROVIDER=stub full suite:        564 passed, 46 skipped
```

全量 46 skips 是未注入专用 URL 时的 39 个 M-P2 和 7 个 M-P1 PostgreSQL cases；两个真实 PG
门禁已如上独立完整执行。Ruff、compileall、pip check、`git diff --check` 与冻结资产 hash 在
最终提交前复核。
