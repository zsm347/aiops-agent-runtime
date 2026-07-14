# 10F RAG 真实管线与 RAG 专项评测实施计划

## 1. 阶段定位

本文是 Python 迁移路线第 10 步高级能力中的子阶段 `10F RAG 真实管线与 RAG 专项评测` 实施计划。

本阶段目标是把当前 fixture 版 `queryInternalDocs` 升级为真实 RAG 子系统：

```text
文档入库
  -> markdown / txt / 已解析文本
  -> LlamaIndex chunk / Node 管理
  -> embedding
  -> Milvus dense + BM25 hybrid index

用户检索
  -> queryInternalDocs tool call
  -> ToolGateway 注入 tenant context
  -> RAGRetrievalService
  -> Milvus tenant filter + hybrid search
  -> RerankAdapter，可配置降级
  -> citation-friendly chunks
  -> 返回模型
```

本阶段仍然遵守当前 Python Harness 的工程边界：

- Agent 外层编排仍使用当前 `AgentHarnessService -> ConversationRuntime -> ContextAssembler -> LangGraph -> ToolGateway -> Trace` 链路。
- LlamaIndex 只负责 RAG 内部数据和检索管线，不替代 Agent Runtime。
- Milvus 负责 dense vector 与原生 BM25 sparse/hybrid search，不回到应用侧 Lucene。
- `tenant_id`、`knowledge_base_id`、`run_id`、`tool_call_id` 等运行时字段由后端上下文注入，不允许模型伪造。
- `queryInternalDocs` 工具输出保持现有 JSON 契约兼容。
- fixture 只服务于 local/test 的确定性测试；真实 RAG 运行时失败不能静默返回 fixture 内容。

## 2. 当前基线

当前 Python 项目已经完成：

- `queryInternalDocs` 工具契约和 fixture handler。
- ToolGateway 参数校验、timeout、retry、错误结构、trace。
- 生产级认证上下文和权限校验 P0。
- ContextManager / ToolResultReducer / 大工具结果压缩。
- 本地 deterministic eval runner。

当前 RAG 相关基线：

```text
src/superbiz_agent/tools/builtin/internal_docs_tool.py
src/superbiz_agent/tools/fixtures.py
src/superbiz_agent/rag/__init__.py
src/superbiz_agent/rag/document_loader.py
src/superbiz_agent/rag/chunking.py
src/superbiz_agent/rag/models.py
```

现状：

- `queryInternalDocs` 只接 fixture。
- 10F Batch B 已完成 loader、外部解析器 Protocol 和 chunking 实现并通过主验收。
- 10F Batch C 已完成 Embedding adapter、Milvus Store、PostgreSQL metadata/repository 和同步 ingestion 的代码实现与主验收；Milvus Lite 真实门禁通过，真实 PostgreSQL 门禁因本机无服务/容器运行时待验。
- Batch D-F 尚未实现：`queryInternalDocs` 仍未接真实 hybrid retrieval，也没有 rerank、citation/trace 和 RAG 专项 eval。

## 3. 依据

本计划依据：

- `docs/02-python-migration-spec.md`
- `docs/03-python-architecture-design.md`
- `docs/04-python-migration-roadmap.md`
- `docs/08-business-tools-migration-plan.md`
- `docs/10-advanced-capabilities-plan.md`
- `docs/10D-context-window-compaction-plan.md`
- `docs/10E-auth-tenant-security-plan.md`
- 当前 Python 实现：
  - `src/superbiz_agent/tools/`
  - `src/superbiz_agent/harness/`
  - `src/superbiz_agent/security/`
  - `src/superbiz_agent/evals/`
  - `src/superbiz_agent/persistence/`

外部技术依据：

- MinerU：可选外部文档解析服务，后续用于复杂 PDF / Office 文档解析。
- LlamaIndex：RAG-first 的 Document / Node / VectorStoreIndex / Retriever / NodePostprocessor 抽象。
- Milvus：dense vector、BM25 sparse、hybrid search、metadata scalar filter。
- MTEB / BEIR：embedding 和 retrieval 模型评测指标参考。

参考资料：

- MinerU：https://github.com/opendatalab/MinerU
- MinerU quick usage：https://opendatalab.github.io/MinerU/usage/quick_usage/
- LlamaIndex Documents / Nodes：https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/
- LlamaIndex Node Parsers：https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/
- LlamaIndex Milvus hybrid search：https://developers.llamaindex.ai/python/framework/integrations/vector_stores/milvushybridindexdemo/
- Milvus + LlamaIndex full text search：https://milvus.io/docs/llamaindex_milvus_full_text_search.md
- Alibaba Cloud text embedding：https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api
- Alibaba Cloud text rerank：https://www.alibabacloud.com/help/en/model-studio/text-rerank-api
- MTEB leaderboard：https://huggingface.co/spaces/mteb/leaderboard
- BEIR metrics：https://github.com/beir-cellar/beir/wiki/Metrics-available

## 4. 技术选型结论

### 4.1 为什么使用 LlamaIndex

本项目使用 LlamaIndex 作为 RAG 内部编排层，不使用它替代 Agent 编排。

选择理由：

- LlamaIndex 是 RAG-first 框架，核心抽象围绕文档、chunk、索引和检索设计。
- `Document / Node` 适合承载 chunk 内容和 citation metadata。
- `VectorStoreIndex / Retriever / NodePostprocessor` 能覆盖索引、召回、过滤和重排链路。
- LlamaIndex 与 Milvus 集成支持 dense + sparse/BM25 hybrid search。
- 相比把 LangChain 通用组件手工拼接成 RAG，LlamaIndex 在 RAG 数据链路上胶水代码更少。

边界：

- LangGraph / 当前 Harness 仍负责 Agent 控制流。
- ToolGateway 仍负责工具治理、权限、trace、参数校验。
- RAG 结果仍以工具 JSON 形式返回模型。

### 4.2 MinerU 的阶段定位

MinerU 不作为 10F P0 的必需依赖。

10F P0 默认支持：

```text
markdown
txt
已解析文本
```

原因：

- MinerU CLI / Python 本地解析依赖较重，模型和运行环境下载量大。
- 官方 HTTP API 有文件大小、页数、限频、隐私和稳定性问题。
- P0 的核心目标是先验证 RAG 主链路：chunk、embedding、Milvus hybrid search、tenant filter、rerank、eval。
- 已解析 Markdown/text 已经足够验证 RAG 主链路。

MinerU 后续作为可选外部解析 adapter：

```text
PDF / DOCX / PPTX
  -> ExternalDocumentParserAdapter
  -> MinerU HTTP API 或自部署 mineru-api
  -> markdown
  -> 后续同 P0 markdown 入库链路
```

10F P0 不引入：

- MinerU CLI。
- MinerU Python 库。
- MinerU 本地模型下载。
- 对 MinerU 官方 API 的强依赖。

### 4.3 为什么使用 Milvus 原生 BM25

本项目 RAG 多租户隔离要求 dense 检索和关键词检索都走同一套 tenant filter。

因此本阶段使用 Milvus 原生 sparse/BM25，与 dense vector 一起放在同一个 Milvus collection 中，通过 scalar fields 过滤：

```text
tenant_id
knowledge_base_id
document_id
```

不做应用侧 Lucene / Whoosh / 内存 BM25。

原因：

- 不需要维护两套索引。
- dense 和 BM25 使用同一批 chunk 元数据。
- tenant filter 可以同时约束 dense 与 sparse 检索。
- hybrid search fusion 交给 Milvus / LlamaIndex adapter。

实现边界：`MilvusVectorStore` 当前支持 `BM25BuiltInFunction`、`HYBRID` 查询和
`RRFRanker`。它会把同一个 Milvus filter 分别带入 dense 与 sparse 的
`AnnSearchRequest`，因此可以满足本项目的统一 tenant filter 要求。

但业务代码不得散落调用 LlamaIndex/Milvus API。项目必须以
`MilvusHybridChunkStore` 适配器封装 collection schema、filter 构造、query mode 和
返回值归一化；`RAGRetrievalService` 只能依赖项目内接口。这样即使框架 API 演进，也
不会扩散到 ToolGateway、Harness 或业务工具。

当前 adapter 的官方依赖约束为 `llama-index-core>=0.13,<0.15` 与
`pymilvus[milvus_lite]>=2.6.7,<3`，因此 `rag` optional dependency 必须采用与之兼容
的区间，不能保留原先过宽的 `llama-index>=0.12` / `pymilvus>=2.4`。

### 4.4 Embedding 初始选型

Embedding 模型负责第一阶段召回，核心看召回率。

P0 初始建议：

```text
provider: Alibaba Cloud Model Studio / DashScope
model: text-embedding-v4
dimension: 1024
```

说明：

- 项目已经使用阿里系模型，接入成本低。
- `1024` 维是效果和存储成本之间的保守起点。
- 最终维度必须通过 RAG eval 验证；如果效果不足，再评估 1536 / 2048 维或其他 embedding 模型。

候选模型：

- 阿里 `text-embedding-v4`
- Qwen3-Embedding 系列
- 本地私有化备选：BGE-M3 / Qwen3-Embedding 本地部署

Embedding 模型一旦切换，历史 Milvus 向量通常需要重建，因此本阶段必须把模型名称、维度、版本写入配置和 metadata。

### 4.5 Rerank 初始选型

Rerank 模型负责第二阶段排序，核心看 top3/top5 精度。

P0 建议：

```text
first-stage hybrid topK: 10
rerank final topK: 3
preferred reranker: qwen3-rerank
```

`qwen3-rerank` 是 DashScope 的独立 rerank API，不是 Chat Completions API，不能复用
`OpenAICompatibleModelGateway.complete()`。P0 使用项目内薄封装：

```text
DashScopeRerankPostProcessor
```

它只负责：

- 接收 query + candidate nodes。
- 调用 rerank API。
- 解析 score。
- 返回排序后的 top3。

实现建议：embedding adapter 使用 OpenAI-compatible `/embeddings` API，可复用已有
OpenAI SDK、Base URL 和认证配置；rerank adapter 使用 `httpx` 调用单独的 `/reranks`
端点并解析 `index` 与 `relevance_score`。rerank 端点需要单独配置 base URL，且在阿里
云目标地域完成模型与 Workspace 开通后才可启用。该地域/Workspace 前置条件必须在部署
检查中显式校验，不能假设当前 chat API 的地域一定可用。

备选：

- 本地 BGE reranker / FlagEmbedding reranker。
- 外部 Cohere Rerank，只有在数据合规允许时考虑。

## 5. 目标架构

### 5.1 入库链路

```text
Upload / local file path
  -> RAGIngestionService.ingest_sync
  -> Markdown / txt / parsed text loader
  -> LlamaIndex Document
  -> MarkdownNodeParser
  -> MarkdownChunkPostProcessor
  -> oversized chunk secondary splitter
  -> embedding
  -> Milvus upsert
  -> Postgres document metadata update
```

### 5.2 检索链路

```text
model tool call queryInternalDocs(query)
  -> ToolGateway
  -> ToolInvocationContext（后端创建，含 RunContext + toolCallId）
  -> PermissionChecker
  -> RAGToolAdapter
  -> RAGRetrievalService.search
  -> tenant context + knowledge base scope
  -> LlamaIndex Retriever
  -> Milvus dense + BM25 hybrid search
  -> metadata scalar filter
  -> RerankAdapter，可配置降级
  -> citation chunk normalization
  -> tool JSON result
  -> model
```

### 5.3 与 Agent 主链路的关系

RAG 不直接参与上下文组装。

模型需要内部文档时，通过 `queryInternalDocs` 工具检索。工具结果返回后，由现有 ReAct 循环继续处理。

因此：

- RAG 不绕过 ToolGateway。
- RAG 不直接读取 prompt 中的 tenant_id。
- RAG 不把所有知识库内容常驻上下文。
- RAG 结果仍受 10D ToolResultReducer 约束，大结果可被压缩并带 `raw_ref`。

## 6. 数据模型设计

### 6.1 PostgreSQL 表

P0 建议新增或预留以下表。

#### `rag_knowledge_base`

知识库元信息。

```text
id
tenant_id
name
description
status
is_default
created_by
created_at
updated_at
```

说明：

- P0 每个 tenant 只允许一个 active default knowledge base；表上通过 partial unique index
  约束 `tenant_id + is_default=true + status=active`。
- `RagKnowledgeBaseScopeResolver` 由认证 tenant 解析该 default knowledge base；解析不到
  时返回不可重试的 `knowledge_base_not_configured` 工具错误，不伪装成 no-results，也不扩大为无 filter 查询。模型不能传入或选择 `knowledge_base_id`。
- 同步入库的受信任管理入口可在该 tenant 下显式创建/初始化 default knowledge base。
- 不在 P0 实现复杂 ACL UI。
- P0 不实现多知识库 ACL；多 knowledge base 授权是后续阶段，届时 resolver 返回经后端 ACL
  校验的列表。

#### `rag_document`

文档元信息。

```text
id
tenant_id
knowledge_base_id
source_uri
document_name
content_type
parser
parser_version
embedding_model
embedding_dimension
status
content_hash
chunk_count
created_at
updated_at
```

说明：

- `tenant_id` 必填。
- `knowledge_base_id` 必填。
- `content_hash` 用于重复入库判断。
- 原文件和可选外部解析输出文件可存在本地目录或对象存储，表中只保存引用路径。

#### `rag_ingestion_job`

不属于 10F P0。

P0 的 ingestion 是由受信任的管理/命令入口显式发起的同步服务，状态写入
`rag_document`。真正的上传 API、队列、后台 worker 和 `rag_ingestion_job` 表属于后续
后台任务阶段，不能因为“以后可能异步化”而在本阶段虚设无法消费的 job。

#### `rag_retrieval_trace`

RAG 检索专项 trace，可先不建独立表，P0 可写入 rollout event。

推荐记录字段：

```text
run_id
tool_call_id
tenant_id
knowledge_base_id
query_hash
hybrid_top_k
rerank_top_k
retrieved_chunk_ids
final_chunk_ids
latency_ms
created_at
```

### 6.2 Milvus Collection Schema

推荐共享 collection：

```text
collection: superbiz_rag_chunks
```

字段：

```text
chunk_id                 primary key
tenant_id                scalar string, filterable
knowledge_base_id        scalar string, filterable
document_id              scalar string, filterable
document_name            scalar string
source_uri               scalar string
heading_path_json        scalar VARCHAR，JSON 编码的标题路径数组
section_title            scalar string
page_start               scalar int
page_end                 scalar int
chunk_index              scalar int
content_hash             scalar string
embedding_model          scalar string
embedding_dimension      scalar int
content                  text
dense_vector             float vector
sparse_vector            sparse/BM25 vector, managed by Milvus integration
created_at               scalar timestamp/string
```

要求：

- `tenant_id`、`knowledge_base_id`、`document_id` 必须是可过滤 scalar 字段。
- 不能只把这些字段塞进不可过滤 JSON metadata。
- 查询必须带 `tenant_id` filter。
- 如果查询指定知识库，必须同时带 `knowledge_base_id` filter。
- `page_start`、`page_end` 对 Markdown/TXT 允许为空；只有已解析输入提供页码时才写入。
- `chunk_id` 必须由 `document_id + chunk_index + normalized_content` 稳定派生，使重试、
  upsert 和 eval 的 gold chunk 标识可复现。

## 7. 文档解析与 Chunk 策略

### 7.1 P0 输入格式

P0 支持以下输入：

```text
markdown
txt
已解析纯文本
```

Markdown 是推荐输入格式，因为它保留标题层级，适合 `MarkdownNodeParser`。

txt / 已解析纯文本没有标题层级时，可以使用普通 text splitter，并将文件名、source_uri、document_name 作为 metadata。

复杂 PDF / Office 文档解析不作为 P0 必选项。后续如果接入 MinerU，可把 MinerU 输出的 Markdown 作为同一条入库链路的输入。

MinerU 可选输出可用于补充：

- 页码。
- 图片/表格引用。
- block 类型。
- 后续审计和引用溯源。

### 7.2 MarkdownNodeParser

P0 使用 LlamaIndex `MarkdownNodeParser` 做初步切分。

原因：

- Markdown 输入与该 parser 契合。
- 能保留标题结构。
- 能把文档拆成 Node，并携带 metadata。

但不能裸用。

### 7.3 MarkdownChunkPostProcessor

本项目新增轻量后处理器：

```text
MarkdownChunkPostProcessor
```

职责：

- 删除正文为空的标题-only chunk。
- 将父级标题整理成 `heading_path` metadata。
- 将标题路径简短拼入 chunk 内容，增强 embedding 语义。
- 对超长 chunk 做二次切分。
- 对空内容、纯符号、极低信息密度内容做最小过滤。
- 计算 `content_hash`。

标题处理规则：

```text
如果上级标题只是目录容器，没有正文：
  不单独入库，只进入 heading_path。

如果某一级标题下面有正文：
  该 section 可以形成 chunk。

如果正文过长：
  用 SentenceSplitter / TokenTextSplitter 二次切分，
  每个子 chunk 继承同一 heading_path。
```

不在 P0 做复杂业务级文档清洗。

原因：

- P0 默认输入是 Markdown / txt / 已解析文本，先不处理复杂版面噪声。
- 自研复杂清洗成本高，容易误删有效内容。
- P0 先保证真实可用链路和可评测闭环。

## 8. 检索与融合策略

### 8.1 Query 输入

P0 保持 `queryInternalDocs` 工具输入兼容：

```json
{
  "query": "Pod 重启排查流程"
}
```

不让模型传：

```text
tenant_id
user_id
run_id
tool_call_id
```

这些字段全部由后端上下文注入。

`knowledge_base_id` P0 不作为必需工具参数。后端的 `RagKnowledgeBaseScopeResolver` 根据
认证 tenant 解析唯一 default knowledge base，得到检索范围。

后续如果需要多知识库选择，可以增加可选参数，但必须由后端校验 ACL。

### 8.2 Metadata Filter

每次检索必须构造 filter：

```text
tenant_id == current_tenant_id
AND knowledge_base_id IN allowed_knowledge_bases
```

P0 只检索 resolver 注入的 default knowledge base。后续多知识库阶段，如果受信任的后端
已经完成 ACL 校验并指定具体知识库，才允许构造：

```text
tenant_id == current_tenant_id
AND knowledge_base_id == backend_authorized_knowledge_base_id
```

任何无 tenant filter 的 Milvus 查询都视为安全缺陷。

工具必须声明 `requires_context=True`。P0 在 ToolGateway 内部创建不可变
`ToolInvocationContext(run_context, tool_call_id, tool_name)`，作为 context-aware handler 的
第二个参数。`queryInternalDocs` 从中获取认证后的 tenant/user/agent、runId 和 toolCallId，
并把它们交给 RAG service；模型可见的 Pydantic 参数仍只有 `query`。这不是新的常驻上下文
或模型参数，只是一次工具执行的后端传参对象。现有 memory context tools 必须迁移到同一
类型，不能保留两种隐式 handler 协议。

### 8.3 Hybrid Search

P0 使用：

```text
dense vector + Milvus native BM25 sparse
fusion: RRFRanker
hybrid_top_k: 10
```

说明：

- dense 负责语义召回。
- BM25 负责关键词、错误码、服务名、配置项召回。
- RRF 对两路召回结果做 rank-level 融合，P0 比 WeightedRanker 更稳。

后续可以通过 eval 比较：

```text
BM25 only
dense only
dense + BM25 + RRF
dense + BM25 + WeightedRanker
```

### 8.4 Rerank

P0 必须实现 rerank adapter 和 rerank 管线位置。

```text
hybrid search top10
  -> RerankAdapter
  -> final top3
```

为了避免本地测试依赖外部 API，P0 允许提供两种实现：

```text
real reranker: qwen3-rerank / DashScope API
fake reranker: deterministic rerank，用于单元测试和本地 eval
```

如果生产配置关闭 rerank，系统可以降级为 hybrid search topK 直接截断到 top3，但这只是运行时降级能力，不代表 10F 可以不实现 rerank adapter。

推荐默认：

```text
hybrid_top_k = 10
rerank_top_k = 3
```

rerank score 只用于同一次 rerank 的排序，不应跨模型或跨请求直接比较。

置信度阈值必须基于本项目 RAG eval 校准，不默认把 `0.5` 当成通用阈值。P0 在尚未完成
真实检索 baseline 前不输出 `high` / `medium` 这类看似确定的标签；若输出 `confidence`，
值必须是 `unavailable`，直到完成模型与数据集绑定的校准。

## 9. 工具输出契约

`queryInternalDocs` 成功输出必须保持现有字段：

```json
{
  "status": "ok",
  "count": 3,
  "chunks": [
    {
      "id": "chunk-id",
      "ref": 1,
      "source": "runbooks/pod-restart.md",
      "content": "...",
      "score": 0.86
    }
  ]
}
```

`score` 只表示本次响应的排序分数：启用 rerank 时是 rerank score，未启用时是 hybrid
ranker score。它不是跨请求可比较的通用相似度，也不用于硬过滤。需要保留调试信息时，
可新增 `retrievalScore`、`rerankScore`、`rerankUsed`，但模型不应根据这些数字自行臆造
置信度。

允许新增 citation-friendly 字段：

```json
{
  "documentId": "doc-id",
  "documentName": "pod-restart.md",
  "knowledgeBaseId": "kb-id",
  "headingPath": ["Pod 排障", "容器启动失败", "CrashLoopBackOff"],
  "sectionTitle": "CrashLoopBackOff",
  "pageStart": 3,
  "pageEnd": 4,
  "retrievalSource": "hybrid",
  "confidence": "unavailable"
}
```

无结果继续保持：

```json
{
  "status": "no_results",
  "message": "No relevant documents found in the knowledge base."
}
```

错误交给 ToolGateway 标准 `ToolErrorResult`，不要在 RAG handler 里随意发明错误结构。

## 10. Trace 设计

检索属于 agent run，P0 至少记录到 rollout event：

```text
RAG_RETRIEVAL_STARTED
RAG_RETRIEVAL_COMPLETED
RAG_RETRIEVAL_FAILED
RAG_RERANK_STARTED
RAG_RERANK_COMPLETED
RAG_RERANK_FAILED
```

同步 ingestion 没有 agent `runId`，因此不写入 `agent_rollout_event`；P0 通过
`rag_document.status`、结构化应用日志和管理入口返回值记录 ingestion 结果。后续异步
入库阶段再设计独立 job/audit event。

检索 trace payload 至少包含：

```text
tenant_id
tool_call_id
knowledge_base_ids
query_hash
hybrid_top_k
rerank_top_k
retrieved_count
final_count
chunk_ids
latency_ms
used_rerank
```

不记录：

- API key。
- 原始大段文档全文。
- 未脱敏敏感内容。
- 跨租户知识库 ID。

## 11. RAG 专项评测

### 11.1 为什么要做专项评测

普通 Agent eval 只能判断工具是否调用和最终回答是否包含关键词。

RAG 需要额外评估：

- 检索层是否召回正确 chunk。
- 正确 chunk 排名是否靠前。
- 最终返回给模型的 evidence 是否足够。
- 引用溯源是否正确。
- 模型回答是否忠实于 evidence。

### 11.2 Dataset Schema

新增 RAG eval case：

```json
{
  "id": "rag_pod_restart_001",
  "query": "Pod 一直重启应该怎么排查？",
  "tenantId": "tenant-a",
  "knowledgeBaseId": "default",
  "goldDocumentIds": ["doc-pod-restart"],
  "goldChunkIds": ["chunk-pod-crashloop-001"],
  "goldHeadingPaths": [["Pod 排障", "容器启动失败", "CrashLoopBackOff"]],
  "expectedAnswerKeywords": ["describe pod", "容器日志", "启动命令"],
  "forbiddenDocumentIds": ["doc-tenant-b-secret"],
  "allowNoAnswer": false
}
```

样本类型必须覆盖：

- 中文自然语言问题。
- 中英文混合术语。
- 错误码 / 服务名 / 配置项关键词问题。
- 语义改写问题。
- 无答案问题。
- 多租户隔离负样本。

### 11.3 检索层指标

RAG retrieval eval 至少输出：

```text
Hit@3
Hit@10
Recall@10
MRR@10
nDCG@10
Precision@3
tenant_leak_count
no_answer_false_positive_count
```

解释：

- `Hit@K`：topK 是否命中任一正确 chunk。
- `Recall@K`：应召回的相关 chunk 被召回多少。
- `MRR@K`：第一个正确结果排名是否靠前。
- `nDCG@K`：相关性等级排序是否合理。
- `Precision@3`：最终给模型的 top3 有多少是真的相关。

### 11.4 生成层指标

如果做 end-to-end RAG answer eval，则继续评估：

```text
answer_correctness
faithfulness
relevance
citation_correctness
unsupported_claim_count
```

P0 先做 deterministic retrieval / citation eval：

- answer 是否包含 gold keywords。
- answer 是否引用返回 chunk 的 `ref/source/headingPath`。
- answer 是否使用 forbidden source。

生成答案的 correctness / faithfulness 需要真实模型与固定证据集，作为 retrieval baseline
通过后的独立 live eval，不把它伪装成 P0 的本地通过条件。LLM-as-judge 同样留作后续增强，
不作为 P0 本地测试必需依赖。

## 12. 文件级实施范围

### 12.1 可能新增文件

```text
src/superbiz_agent/rag/config.py
src/superbiz_agent/rag/models.py
src/superbiz_agent/rag/document_loader.py
src/superbiz_agent/rag/external_parser.py
src/superbiz_agent/rag/chunking.py
src/superbiz_agent/rag/embedding.py
src/superbiz_agent/rag/milvus_store.py
src/superbiz_agent/rag/ingestion.py
src/superbiz_agent/rag/retrieval.py
src/superbiz_agent/rag/rerank.py
src/superbiz_agent/rag/schemas.py
src/superbiz_agent/evals/rag_cases.py
src/superbiz_agent/evals/rag_runner.py
tests/test_rag_chunking.py
tests/test_rag_retrieval.py
tests/test_rag_tool.py
tests/test_rag_eval.py
```

### 12.2 可能修改文件

```text
src/superbiz_agent/config.py
src/superbiz_agent/tools/builtin/internal_docs_tool.py
src/superbiz_agent/tools/builtin/__init__.py
src/superbiz_agent/tools/registry.py
src/superbiz_agent/tools/gateway.py
src/superbiz_agent/memory/tools.py
src/superbiz_agent/harness/events.py
src/superbiz_agent/evals/runner.py
src/superbiz_agent/persistence/models.py
src/superbiz_agent/persistence/repositories/
tests/
pyproject.toml
.env.example
docs/04-python-migration-roadmap.md
docs/10-advanced-capabilities-plan.md
```

### 12.3 不应修改文件

除非发现明确冲突，10F 不应修改：

```text
src/superbiz_agent/model_gateway/
src/superbiz_agent/api/routes_chat.py
prompts/
docs/01-java-capability-inventory.md
docs/02-python-migration-spec.md
docs/03-python-architecture-design.md
```

## 13. 实施批次

### 批次 A：依赖、配置和接口壳

目标：

- 增加 RAG settings。
- 定义 RAG service interface。
- 将 context-aware ToolDefinition 的后端第二参数统一为 `ToolInvocationContext`，并迁移
  现有 memory tools，确保 RAG nested trace 有准确 parent `tool_call_id`。
- fixture 仅用于 local/test 的显式 fixture mode。
- 不直接要求本地必须有 Milvus / 外部文档解析器 / API key。

配置项：

```text
rag_enabled
rag_fixture_mode
rag_milvus_uri
rag_milvus_token
rag_milvus_collection
rag_embedding_provider
rag_embedding_model
rag_embedding_dimension
rag_embedding_base_url
rag_embedding_api_key
rag_hybrid_top_k
rag_final_top_k
rag_rerank_enabled
rag_rerank_provider
rag_rerank_model
rag_rerank_base_url
rag_rerank_api_key
rag_external_parser_enabled
rag_external_parser_provider
rag_external_parser_api_base_url
```

其中 `rag_fixture_mode` 与 `rag_enabled` 互斥，且只能在 `local/test` 为 true。为保持现有
确定性 smoke / business-tool 测试，P0 的 local/test 默认可启用 fixture mode；生产配置
必须显式关闭它，启动校验应拒绝任何 production fixture mode。
`rag_embedding_api_key` 可以显式配置；仅在未配置时回退到同一受信任后端配置中的
`model_api_key`，不从工具参数或 prompt 读取。`rag_rerank_base_url` / API key 不能复用为
chat endpoint 的路径假设，必须按实际 Workspace/地域单独配置。

验收：

- 默认配置下测试不依赖真实 Milvus / 外部文档解析器。
- fixture mode 仅允许 `local` / `test` 环境显式开启；默认单元测试可使用它。
- 生产环境 `rag_enabled=false` 或真实 RAG 初始化失败时，工具返回标准可见错误，不得回落
  fixture。

### 批次 B：Document Loader 与 Chunking

目标：

- 实现 Markdown / txt / parsed text loader。
- 预留 `ExternalDocumentParserAdapter` 接口，但不强制接 MinerU。
- 实现 `MarkdownChunkPostProcessor`。
- 用小型 Markdown fixture 验证标题-only、多级标题、超长 chunk。

验收：

- 标题-only chunk 不单独入库。
- `heading_path` metadata 正确。
- 超长 chunk 可二次切分。
- chunk 保留 document/source/page metadata。

#### 批次 B 实施与主验收结果

状态：

```text
Batch B complete
```

已完成：

- 实现 Markdown、txt、parsed text loader，并构造带可信后端 metadata 的 LlamaIndex Document。
- 增加 `ExternalDocumentParserAdapter` Protocol，作为后续可信外部解析器边界；本批次未接 MinerU 或其他真实外部解析服务。
- Markdown 使用 `MarkdownNodeParser + SentenceSplitter`，txt/parsed text 使用 SentenceSplitter。
- 过滤正文为空的 heading-only section；父级标题进入 `heading_path`，不单独形成 chunk。
- 每个 Markdown 子 chunk 保留完整 `heading_path` 文本和 metadata，标题中的 `/` 不会被拆成多个标题。
- 标题上下文作为 embedding content 计入最终 token 预算；最终 chunk token 数不超过 `chunk_size`，标题上下文本身耗尽预算时明确失败。
- `chunk_id`、`content_hash`、chunk 顺序以及 document/source/page metadata 保持稳定、可复现。

主验收：

```text
RAG 专项 pytest                         -> 20 passed
MODEL_PROVIDER=stub 全量 pytest         -> 190 passed
基础 eval runner                        -> 14/14 passed
Ruff                                    -> passed
compileall                              -> passed
```

Batch B 完成不代表 10F complete。Batch C 已进入“代码与 Milvus 验收完成、真实 PostgreSQL 门禁待验”状态；Batch D Retrieval、Batch E Rerank/Citation/Trace、Batch F RAG Eval 均未实现。

### 批次 C：Milvus Store 与 Ingestion

目标：

- 实现 Milvus collection schema 初始化。
- 实现 document -> chunks -> embedding -> Milvus upsert。
- 实现 PostgreSQL knowledge base / document 元信息；不创建未被消费的 job 表。

验收：

- 同一 tenant + knowledge base 的相同 content hash 必须幂等跳过。
- P0 不实现 source 更新后的原地替换/物理删除；该能力需要单独定义 Milvus 与 PostgreSQL
  的补偿语义，留给后续 ingestion job 阶段。
- chunk 带 `tenant_id`、`knowledge_base_id`、`document_id`。
- Milvus schema 包含可过滤 scalar fields。
- 共享 collection 的 `chunk_id` 不包含 `tenant_id`，因此 Batch C 的 `rag_document.id` / `document_id` 必须由后端生成并保证全表全局唯一；不能接受仅在 tenant 内唯一的自定义 ID，否则同一主键可能跨 tenant 覆盖。现有 `chunk_id = document_id + chunk_index + normalized_content` 稳定派生算法保持不变。

#### 批次 C 实施与主验收结果

状态：

```text
implementation complete / Milvus gate passed / real PostgreSQL gate pending
```

已完成：

- PostgreSQL `rag_knowledge_base`、`rag_document` model、Alembic migration 和 tenant-scoped repository。
- `ON CONFLICT` 并发 claim、`claim_token + claimed_at` fencing、heartbeat、短事务终态更新和失败恢复契约。
- OpenAI-compatible embedding adapter、逐批 index/shape/finite 校验和 deterministic 测试 adapter。
- LlamaIndex `MilvusVectorStore` dense + BM25 schema，固定 Jieba search analyzer、显式 tenant scalar、preflight/postflight fail-closed。
- 同步 ingestion：规范化/hash、claim、chunk、refresh、embedding、refresh、Milvus upsert、mark active，以及安全失败补偿和取消处理。
- duplicate/in-progress 零后续副作用；failed/stale/partial upsert 重试复用稳定 document/chunk ID。

主验收：

```text
RAG Batch B/C 联合 pytest              -> 89 passed
MODEL_PROVIDER=stub 全量 pytest         -> 259 passed, 1 existing warning
基础 eval runner                        -> 14/14 passed
Milvus Lite dense+sparse/Jieba/upsert   -> passed
Ruff                                    -> passed
compileall                              -> passed
独立 C3 最终复审                       -> PASS
```

未完成门禁：

- 当前机器没有 PostgreSQL、Docker 或 Podman，尚未真实执行两个独立 session 的并发首次 claim、stale fencing、partial default index、复合 tenant FK 和 migration upgrade/downgrade。
- 因此不能把 Batch C 标记为完整 `complete`；恢复 PostgreSQL 环境后必须补跑该门禁。
- Batch C 没有实现 hybrid query、tenant retrieval filter、queryInternalDocs 接线、rerank、citation、RAG trace 或 RAG eval。

### 批次 D：Retrieval 与 queryInternalDocs 改造

详细实施计划：`docs/10F-D-hybrid-retrieval-plan.md`

设计状态：`analysis complete / three-round dual review PASS / implementation not started`

目标：

- 实现 `RAGRetrievalService`。
- 使用 LlamaIndex + Milvus 做 hybrid search。
- 强制 tenant filter。
- 改造 `queryInternalDocs`：通过 `ToolInvocationContext` 注入检索 scope；仅 local/test 显式 fixture
  mode 使用 fixture，真实 RAG 故障交给 ToolGateway 返回标准工具错误。

验收：

- tenant A 查不到 tenant B 文档。
- 返回结构兼容现有 `status/count/chunks`。
- 新增 citation-ready document/heading/page metadata；最终答案引用编排与完整性治理留给 Batch E。
- no results 行为兼容。

### 批次 E：Rerank 接入

目标：

- 实现 rerank adapter。
- 支持真实 rerank provider 和 fake/deterministic rerank。
- 支持运行时关闭 rerank 并降级到 hybrid search 顺序。
- rerank 只处理 first-stage topK。

验收：

- rerank enabled 时 top3 顺序来自 reranker。
- 本地测试不依赖真实 rerank API。
- rerank disabled 时使用 hybrid search 顺序。
- rerank 失败时可降级到 hybrid 结果，并记录 trace；Milvus/embedding 故障不能降级到
  fixture。

### 批次 F：RAG 专项 Eval

目标：

- 新增 RAG eval dataset。
- 新增 RAG eval runner 或扩展当前 eval runner。
- 输出 retrieval metrics 和隔离检查。

验收：

- RAG eval 能在 fixture / fake retriever 模式下稳定运行。
- 有 Milvus 测试环境时可运行真实 RAG eval。
- 报告包含 Hit@K、MRR、nDCG、Precision@3、tenant leak count。

## 14. 本阶段不做什么

10F 明确不做：

- 不做知识库管理 UI。
- 不做复杂文档权限 UI。
- 不做跨租户共享知识库。
- 不做应用侧 Lucene BM25。
- 不绕过 Milvus metadata filter。
- 不把 tenant_id / user_id / run_id 暴露为模型工具参数。
- 不把 MinerU 放进 chat 实时请求路径。
- 不把 MinerU CLI / Python 库作为 P0 必需依赖。
- 不要求本地下载 MinerU 模型。
- 不实现完整业务级文档清洗系统。
- 不做复杂 table-aware retrieval 作为 P0 必选项。
- 不把 LlamaIndex QueryEngine 的生成答案直接返回用户；最终回答仍由当前 Agent 模型生成。
- 不让真实 Milvus / 外部文档解析器 / 外部 API key 成为本地单元测试必需条件。

## 15. 测试计划

### 15.1 单元测试

新增：

- `MarkdownChunkPostProcessor`：
  - 标题-only 删除。
  - 多级标题路径。
  - 超长 chunk 二次切分。
  - content hash。

- `RAGRetrievalService`：
  - tenant filter 必填。
  - no results。
  - citation metadata normalization。
  - rerank disabled / enabled。
  - default knowledge base 仅由 tenant scope resolver 注入。

- `queryInternalDocs`：
  - local/test fixture mode 与生产拒绝 fixture。
  - real RAG handler mock。
  - tool output 兼容字段。
  - ToolGateway 参数校验不回退。
  - ToolInvocationContext 的 tenant / run / toolCallId 注入。

### 15.2 集成测试

P0 可使用 fake Milvus / fake retriever。

真实 Milvus 集成测试可标记为 optional：

```text
@pytest.mark.integration
```

避免普通本地测试强依赖 Docker / Milvus。

### 15.3 Eval 测试

新增 RAG eval smoke：

- 有正确 chunk 时命中。
- 无答案时不返回乱引用。
- tenant B 文档不会出现在 tenant A 结果中。
- top3 包含正确 source / heading_path。

## 16. 验收命令

实现完成后必须运行：

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
PYTHONPATH=src python3 -m compileall -q src tests
```

如果安装了 ruff：

```bash
python3 -m ruff check src tests
```

RAG 专项 eval：

```bash
PYTHONPATH=src python3 -m superbiz_agent.evals.rag_runner
```

如果没有真实 Milvus / 外部文档解析器 / API key，RAG runner 必须能以 fixture/fake 模式运行。

## 17. subAgent 实施任务单

10F 可以拆给一个或多个 subAgent。

建议拆分：

### subAgent A：RAG 核心管线

写入范围：

```text
src/superbiz_agent/rag/
src/superbiz_agent/tools/builtin/internal_docs_tool.py
src/superbiz_agent/tools/registry.py
src/superbiz_agent/tools/gateway.py
src/superbiz_agent/memory/tools.py
src/superbiz_agent/config.py
tests/test_rag_chunking.py
tests/test_rag_tool.py
tests/test_long_term_memory.py
```

任务：

- 实现配置。
- 实现 Markdown / txt loader，并预留 ExternalDocumentParserAdapter 接口。
- 实现 chunk postprocessor。
- 实现 retrieval service interface。
- 改造 `queryInternalDocs`。

### subAgent B：Milvus / Ingestion

写入范围：

```text
src/superbiz_agent/rag/milvus_store.py
src/superbiz_agent/rag/ingestion.py
src/superbiz_agent/persistence/
tests/test_rag_retrieval.py
tests/test_rag_ingestion.py
```

任务：

- 定义 Milvus schema。
- 实现 knowledge base / document metadata；不实现 ingestion job。
- 实现 upsert 和 tenant filter search。

### subAgent C：RAG Eval

写入范围：

```text
src/superbiz_agent/evals/rag_cases.py
src/superbiz_agent/evals/rag_runner.py
tests/test_rag_eval.py
```

任务：

- 定义 RAG eval case schema。
- 实现 retrieval metrics。
- 实现 tenant leak 检查。
- 输出 report。

如果只开一个 subAgent，也必须按 A -> B -> C 的顺序提交，不能混乱推进。

subAgent 必须在提交结果中说明：

1. 修改/新增了哪些文件。
2. 是否严格使用了 tenant filter。
3. 是否保持 `queryInternalDocs` 输出兼容。
4. 是否引入真实外部依赖作为测试必需条件。
5. 测试命令和结果。
6. 是否偏离本文档。

## 18. 风险与取舍

| 风险 | 影响 | 处理 |
|---|---|---|
| MinerU 依赖重 | 本地开发复杂 | P0 不依赖 MinerU；只预留外部 parser adapter |
| Milvus 本地环境重 | 单测不稳定 | 单测 fake store，真实 Milvus 测试标记 integration |
| Embedding 模型更换 | 需要重建索引 | 配置和 metadata 固定 model/dimension/version |
| Rerank 分数不可直接阈值化 | 误过滤正确结果 | 阈值基于 eval 校准，P0 不硬编码 0.5 |
| 文档清洗过度 | 误删有效信息 | P0 只做轻量过滤和标题处理 |
| 多租户过滤遗漏 | 严重安全问题 | Milvus 查询必须集中通过 RAGRetrievalService，单测覆盖无 filter 拒绝 |
| LlamaIndex API 变化 | 集成不稳定 | 通过项目内 adapter 包一层，业务代码不直接散落调用 LlamaIndex |
| 外部 rerank API 不可用 | 本地测试和检索降级 | P0 必须有 fake reranker；生产失败时降级到 hybrid 顺序并记录 trace |
| fixture 被误用于生产 | 返回虚假知识或掩盖系统故障 | fixture 默认仅为 local/test 提供；启动校验拒绝生产启用，真实检索故障返回 ToolErrorResult |
| rerank 地域/Workspace 未开通 | 启动后才发现 rerank 不可用 | 独立配置并在部署检查中验证；运行时降级到 hybrid |

## 19. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否先设计再实现 | 是 |
| 是否使用成熟框架而非从零自研 RAG | 是，P0 使用 LlamaIndex + Milvus，MinerU 仅作为后续可选外部解析器 |
| 是否保留当前 Agent Harness 主链路 | 是 |
| 是否保持 `queryInternalDocs` 输出兼容 | 是 |
| 是否明确 tenant filter 不能绕过 | 是 |
| 是否避免模型传入 tenant/user/run | 是 |
| 是否使用 Milvus 原生 BM25 而非应用侧 Lucene | 是 |
| 是否区分 RAG 编排和 Agent 编排 | 是 |
| 是否明确 embedding/rerank 只是初始候选，需 eval 校准 | 是 |
| 是否要求实现 rerank adapter 且不让本地测试依赖外部 API | 是 |
| 是否避免 MinerU 成为 P0 必需依赖 | 是 |
| 是否避免真实外部依赖成为单元测试必需条件 | 是 |
| 是否有 RAG 专项 eval | 是 |
| 是否适合交给 subAgent 实施 | 是 |

## 20. 阶段通过标准

10F 完成后必须满足：

1. `queryInternalDocs` 可在真实 RAG 模式下检索 Milvus。
2. fixture mode 仅可用于 local/test 的本地测试。
3. markdown / txt / 已解析文本可入库并切成带 metadata 的 chunks。
4. chunks 入库时包含 `tenant_id`、`knowledge_base_id`、`document_id`。
5. 检索时强制 tenant filter。
6. dense + Milvus native BM25 hybrid search 可用。
7. rerank adapter 可用，并支持 fake/deterministic 测试模式。
8. citation metadata 进入工具返回。
9. RAG 专项 eval 可运行并输出检索指标。
10. 全量测试和现有 eval 不回退。
11. roadmap 更新 10F 状态。
