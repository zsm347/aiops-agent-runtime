# 当前工作交接快照

更新时间：2026-07-17（Asia/Shanghai）

本文只记录当前可验证状态，供会话切换后继续推进；不替代各阶段正式计划和验收文档。

## 1. 已完成阶段

### H-R1 Harness 多工具与多轮 ReAct

状态：`complete`

- 支持单响应多工具顺序执行和有限多轮工具循环。
- 增加 run 级 round/call 预算、all-or-block、tools-disabled finalization、协议防护和 run 状态清理。
- 主验收：专项 57 passed、全量 stub 182 passed、基础 eval 14/14、Ruff/compileall 通过。
- 正式记录：`docs/H-R1-harness-multi-tool-react-loop-plan.md`。

## 2. 长期记忆当前状态

### M-P2 PostgreSQL 持久化

状态：`complete / independent acceptance passed`

- PR：`https://github.com/zsm347/aiops-agent-runtime/pull/1`。
- PostgreSQL 16.14、pgvector 0.8.5 一次性隔离数据库验收：39 planned、39 executed、
  0 skipped、39 passed、0 failed，权威脚本输出 `status=passed`。
- 首次真实运行发现 Alembic validator 隐式事务未提交；修复 Alembic migration engine 的显式
  事务所有权后，补充了 asyncpg catalog `"char"` 类型兼容和连接错误安全归一化。
- 回归：M-P2 persistence/runtime/snapshot 137、长期记忆 24、Memory eval 86、RAG B/C/D
  198、全量 stub 519 passed / 39 skipped、基础 eval 14/14；Ruff、compileall、pip check、
  diff check 通过。
- prompt、Memory Dataset、Judge、Graph 与 OpenAI tool schema 冻结哈希未变化。
- 2026-07-17 技术负责人使用全新一次性 PostgreSQL 16.14 + pgvector 0.8.5
  隔离集群独立复跑：真实 PostgreSQL `39 passed / 0 skipped`、全量 stub
  `519 passed / 39 skipped`、基础 eval `14/14`，M-P2 独立验收通过。

### M-P1 真实 Embedding 与 pgvector 检索

状态：`implementation complete / pending independent acceptance`

- 分支：`feat/m-p1-real-embedding-retrieval`。
- PostgreSQL `16.14`、server pgvector `0.8.5`、Python pgvector `0.5.0`。
- M-P1 真实 PostgreSQL gate：`7/7 passed, 0 skipped`；M-P2 回归 gate：`39/39 passed,
  0 skipped`。
- M-P1-R1 已撤回原 `topK=3/min_similarity=0.4` candidate：原 report 不含逐 case evidence，
  isolation canary 被计入 ranking denominator，hard-negative forbidden hit 未作为硬门禁，且
  latency 是 superset scan 口径。
- Dataset `1.0.1` 将 MPR10-MPR12 修正为空结果 isolation contract，保留 tenant/user/agent
  forbidden canary；SHA 从 `ba3e...21a88` 更新为 `007b...c3cbe`。
- 新 evaluator 分离 semantic MPR01-07、no-match MPR08-09、isolation MPR10-12；report 新增
  逐 case 脱敏 observations、scan config、硬 quality gate、Pareto frontier、候选实配复跑延迟
  和 report SHA manifest。
- production candidate 必须满足 no-match FPR=0、hard-negative forbidden=0、identity 与
  isolation forbidden=0、三个 isolation case 全过；否则 `no_candidate`。生产默认 threshold
  `0.5` 未修改。
- 若 scan 完整但 candidate 实配复跑发生 database/Embedding/network error，report 必须为
  `infrastructure_failed`、candidate 为 `candidate_not_evaluated`、ranking 为
  `not_evaluated`、CLI 非零；不得伪装成质量 `no_candidate`。该状态机单测已覆盖。
- 当前 Settings 没有独立 `MEMORY_EMBEDDING_*` 配置，真实 baseline 返回 pending/exit 3；没有
  从 Chat/RAG credential 回退，没有 API 调用，没有生成新 report，report SHA 为
  `not_generated (pending)`，candidate 结论为 `pending`。
- 本轮回归：retrieval/embedding/backfill `43 passed`、M-P1 PG `7/7`、M-P2 PG `39/39`、
  全量 stub `564 passed / 46 skipped`；最终静态与冻结 hash 门禁见 M-P1 plan 第 15 节。
- 冻结 prompt、Memory Dataset v1、Judge、Graph、OpenAI tool schema SHA-256 均未变化。
- 下一步是注入独立 Memory embedding 配置后只运行一次轻量真实 baseline，并提交脱敏
  report/manifest，再由技术负责人独立验收；不要转 Ready、不要合并、不要进入 M-P3。

详见 `docs/M-P1-real-embedding-retrieval-plan.md`。

### M-R1 生产 prompt、tool 语义与 Core no-op

实现状态：`implementation submitted / deterministic acceptance passed / real guardrails pending`

已实现：

- 新增 `ops-agent-system-v3`，保留 v2；默认 tool schema 升级为 `ops-tools-v2`。
- 明确长期记忆写入双门槛、Core/Archival 路由、写后声明和检索过滤语义。
- Core 规范化 hash 相同内容返回 `unchanged`，记录 `CORE_MEMORY_UNCHANGED`，不再伪报 updated。
- evaluator version 升级为 `1.2.0`，artifact capture 支持 unchanged 事件。
- 新增 `scripts/run_memory_eval_mr1.py` targeted runner。

确定性主验收：

```text
M-R1 专项：55 passed
全量 stub：188 passed
基础 eval：14/14 passed
Ruff：passed
compileall：passed
```

禁改基线在 M-R1 实施后保持：

```text
Dataset SHA-256:
9251dae6db2a525b7c0c02ad7ef7f16a7cc08da60db9824cdae1d1749c53de87

prompt v2 SHA-256:
90b10a434a147745396f81d16bc6b78c6cabacf247f113c4c82d32a0f3928ad9

第一次 Track B dev 原始报告 SHA-256:
4e210463d323a6b910fe746bd4701fc1a944444db332acb3ac67db1d8ad690c9
```

### M-R1 真实模型 positive

原始报告：

```text
artifacts/evals/memory/mr1_positive/20260711T144827Z-1.0.0-track_b.json
SHA-256:
6c8bfb8d50b19bb4e8117510f57d42f2f9cfa1de65237818a0a31ddabec1c073
```

运行事实：

```text
planned/executed: 6/6
infrastructure failures: 0
C03: 3/3 正确调用 updateCoreMemory 并成功写入 Core
A03: 3/3 正确调用 saveArchivalMemory 并成功写入 Archival
```

报告显示 5/6，唯一失败是 A03 repetition 2：模型写入“单个分区热点”，Dataset 只接受连续子串“单分区热点”。这是已确认的 evaluator false negative，不是模型行为失败。

### 10G.2A-R1B

状态：`plan drafted / review interrupted by external 502 / not implemented`

计划：`docs/10G2A-R1B-a03-equivalent-fact-remediation-plan.md`

计划内容：

- A03 保留 `Kafka / lag / 10 万 / rebalance` 必须事实。
- 将“单分区热点/单个分区热点”改为 `required_fact_any_of`。
- 不改 Judge，不重跑 positive 模型。
- 对上述原始报告离线重判，预期 6/6、模型调用 0、源报告字节不变。
- targeted runner 增加 `--positive-report` 后只继续 guardrails 9 次。

下一步：

1. 完成 R1-B 计划审核。
2. 由 subAgent 实施极窄 Dataset/runner/test 修正。
3. 主 Agent 离线重判 positive 报告。
4. 运行 M-R1 guardrails 9 次并审核 preferred behavior、安全与隔离门禁。
5. 通过后才可标记 M-R1 complete。

## 3. RAG 当前切换点

### 10F Batch B

状态：`complete`

- Markdown/txt/parsed text loader、外部解析器 Protocol、`MarkdownNodeParser + SentenceSplitter` 已完成。
- heading-only 过滤、多级 `heading_path`、标题 token 预算和稳定 chunk ID/hash/metadata 已通过主验收。

### 10F Batch C

状态：

```text
implementation complete
Milvus gate passed
real PostgreSQL gate pending
```

正式计划：

```text
docs/10F-C-milvus-ingestion-plan.md
```

已实现：

```text
PostgreSQL:
- rag_knowledge_base / rag_document model 与 Alembic migration
- tenant composite FK、partial default index、SHA-256 constraint
- ON CONFLICT claim、claim_token/claimed_at fencing、heartbeat、短事务终态更新

Embedding:
- OpenAI-compatible text-embedding-v4 adapter
- batch index 重排、数量/维度/数值/finite 校验
- deterministic test adapter

Milvus:
- LlamaIndex MilvusVectorStore
- dense FLAT/COSINE + native BM25 sparse
- Jieba search analyzer
- 显式 tenant / knowledge base scalar metadata
- preflight + postflight fail-closed
- stable TextNode SOURCE/document_id 与 upsert

Ingestion:
- trusted RagIngestionRequest，无外部 document_id
- canonical content type + normalized content SHA-256
- claim -> chunk -> refresh -> embed -> refresh -> upsert -> active
- duplicate/in-progress 零后续副作用
- failed/stale/partial upsert 稳定 ID 恢复
- claim_lost fencing、取消补偿和安全错误链
```

主验收：

```text
RAG Batch B/C 联合 pytest: 89 passed
全量 stub pytest: 259 passed, 1 existing Starlette/httpx warning
基础 eval: 14/14 passed
Milvus Lite dense+sparse/Jieba/upsert: passed
Ruff: passed
compileall: passed
C3 独立最终复审: PASS
```

真实 PostgreSQL 门禁未执行：本机没有 PostgreSQL、Docker 或 Podman。恢复环境后必须验证：

```text
- 两个独立 session 并发首次 claim
- stale worker fencing
- partial default KB unique index
- composite tenant FK
- migration upgrade/downgrade
```

因此 Batch C 不能标记完整 `complete`，10F 更没有 complete。

下一 RAG 实施阶段：

```text
10F Batch D Retrieval 与 queryInternalDocs 改造
```

Batch D 边界：实现 tenant-filtered dense + BM25 hybrid retrieval 并接入既有 `queryInternalDocs`，允许输出后续引用所需的 citation-ready document/heading/page metadata；不得提前实现 Batch E rerank、最终答案引用编排/完整性治理或专用 trace，也不得实现 Batch F RAG eval。

### 10F Batch D 设计状态

状态：`design approved / implementation not started`

详细计划：

```text
docs/10F-D-hybrid-retrieval-plan.md
```

设计审核事实：

- 核对 LlamaIndex/Milvus 实际源码与 API，不依赖 README 猜测。
- Milvus Lite 实测 tenant+KB hybrid filter、RRF score、output_fields 和 cold reopen/load。
- 第一轮双路复审发现 citation/no-default/tool version 等 3 个阻塞项。
- 第二轮发现 document status、共享初始化和重试 action 等阻塞项。
- 第三轮双路复审最终均为 `PASS`。

冻结实施顺序：

```text
D1 Query Embedding + Milvus Hybrid Contract
D2 Default KB Resolver + Active-document Visibility + Retrieval Service
D3 Real Runtime + queryInternalDocs + Tool Error/Lifecycle Wiring
```

关键设计：top10 dense+BM25+RRF，PostgreSQL active 状态过滤后 final top3；无 default KB 是不可重试配置错误；工具契约升级为 `ops-tools-v3`；Batch C PostgreSQL 和远程 Milvus 生产门禁继续单独保留。
