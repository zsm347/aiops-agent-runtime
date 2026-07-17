# 高级能力阶段实施计划

## 1. 阶段定位

本文是 Python 迁移路线第 10 步 `高级能力` 的阶段分析与实施计划。

第 10 步不是一个单一功能，而是一组生产化增强能力。它的核心目标是：在第 1-9 步已经完成单 Agent Harness 主链路、业务工具、评测门禁和长期记忆闭环之后，逐步补齐真实模型、流式响应、上下文压缩、工具可靠性、安全隔离、后台任务、专项评测、观测和部署能力。

本阶段必须继续遵守前面已经固定的原则：

- 成熟框架优先，项目只做薄适配层和必须自控的工程约束。
- 不直接推翻当前 Harness，而是在现有 `AgentHarnessService -> ConversationRuntime -> ContextAssembler -> LangGraph -> ToolGateway -> Trace` 链路上增强。
- 每个子阶段必须先明确边界、验收标准和测试，再实现。
- 代码实现可以交给 subAgent，但主 agent 必须做最终技术验收。

## 2. 当前基线

第 1-9 步已经完成：

- FastAPI API 骨架。
- Java 兼容 `/api/chat`、`/api/chat/clear`、`/api/chat/session/{session_id}`。
- `AgentRequestContext` / `RunContext`。
- 基于事件流的 rollout trace。
- LangGraph custom graph 最小 ReAct 闭环。
- StubModelGateway。
- ToolRegistry / ToolGateway / Pydantic 参数校验。
- 业务工具 fixture：
  - `getCurrentDateTime`
  - `getAvailableLogTopics`
  - `queryLogs`
  - `queryPrometheusAlerts`
  - `queryInternalDocs`
- Core Memory / Archival Memory / Memory Index / Memory Policy / Memory Tools。
- 本地 deterministic eval runner。

当前验收基线：

```text
python3 -m pytest                                      -> 41 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 当前环境未安装 ruff
```

## 3. 依据

本阶段依据：

- `docs/02-python-migration-spec.md`
  - `POST /api/chat_stream`
  - ModelGateway 契约
  - ToolGateway 契约
  - eval 契约
  - observability / security / tenant isolation 方向
- `docs/03-python-architecture-design.md`
  - FastAPI
  - LangGraph
  - OpenAI-compatible SDK / LiteLLM / DashScope SDK
  - LangSmith
  - OpenTelemetry / Prometheus
  - LlamaIndex / Milvus 后续路径
- `docs/04-python-migration-roadmap.md`
  - 第 10 步高级能力范围。
- `docs/10D-context-management-research.md`
  - 10D 上下文管理前置调研。
- `docs/10D-context-window-compaction-plan.md`
  - 调研后修订的 10D 正式实施计划。
- `docs/10E-auth-tenant-security-plan.md`
  - 10E 生产级多租户鉴权与安全策略实施计划。
- 当前 Python 实现：
  - `src/superbiz_agent/api/`
  - `src/superbiz_agent/harness/`
  - `src/superbiz_agent/model_gateway/`
  - `src/superbiz_agent/tools/`
  - `src/superbiz_agent/memory/`
  - `src/superbiz_agent/evals/`

## 4. 阶段拆分

第 10 步拆成 8 个子阶段，不建议混在一个大 PR / 一次 subAgent 任务中实现。

推荐顺序：

```text
10A 真实 ModelGateway 与模型调用治理
10B Streaming / chat_stream
10C ToolGateway 可靠性增强
10D 上下文窗口与压缩
10E 安全与生产级多租户鉴权
10F RAG 真实管线与 RAG 专项评测
10G 长期记忆专项增强
10H 观测、LangSmith、部署配置
```

说明：

- `10A` 和 `10B` 是最优先，因为没有真实模型和 streaming，后续很多生产化能力无法验证。
- `10C` 与真实模型可并行规划，但最好在 `10A` 后实现，因为真实 provider 会暴露更多错误类型。
- `10D` 依赖真实模型或至少依赖可模拟的 context overflow。
- `10E` 是生产安全前提，但不能在认证规则未定义时先硬写复杂 SSO。
- `10F` 和 `10G` 是专项能力增强，应在基础 Harness 稳定后做。
- `10G` 需要拆分推进：先做 `10G.1 长期记忆提示与元数据优化`，再分别做 `10G.2A 长期记忆专项评测` 和 `10G.2B 长期记忆后台任务与生命周期治理`。
- `10H` 应贯穿各阶段逐步补，但独立验收一次。

## 5. 子阶段 10A：真实 ModelGateway 与模型调用治理

### 5.1 目标

实现真实模型调用 adapter，优先支持阿里 Qwen / DashScope 的 OpenAI-compatible 接口。

### 5.2 做什么

- 新增 `OpenAICompatibleModelGateway`。
- 保留 `StubModelGateway` 作为测试和 eval 默认 provider。
- Settings 增加或完善：
  - `model_provider`
  - `model_base_url`
  - `model_api_key`
  - `model_name`
  - `model_timeout_ms`
  - `model_max_retries`
- 支持 tool calling 所需的 message / tool schema 转换。
- 解析 provider 返回的：
  - final content
  - tool calls
  - token usage
  - finish reason
  - raw response 摘要
- 记录：
  - `MODEL_CALL_STARTED`
  - `MODEL_CALL_COMPLETED`
  - `MODEL_CALL_FAILED`
  - `MODEL_CALL_TIMEOUT`
  - `MODEL_CALL_RETRY`
  - `TOKEN_USAGE`
- 实现 retry/backoff 的最小工程语义：
  - 网络瞬时错误可重试。
  - 超时可重试。
  - 参数错误、认证错误不重试。
  - 最大重试次数由 settings 控制。

### 5.3 不做什么

- 不自研 HTTP client。
- 不把 API key 写入代码或文档示例真实值。
- 不强制接 LiteLLM；只有多 provider / fallback / budget 需求明确后再引入。
- 不把真实模型调用作为本地测试必需条件。

### 5.4 验收

- 无 API key 时，默认仍走 stub，现有测试和 eval 不退化。
- 有 fake client / mock response 时，能验证：
  - 普通 final answer。
  - tool call 解析。
  - token usage 记录。
  - timeout / retry / failed event。
- `python3 -m pytest` 通过。
- eval runner 继续通过。

## 6. 子阶段 10B：Streaming / chat_stream

### 6.1 目标

实现 Java 兼容的 SSE 流式接口：

```text
POST /api/chat_stream
```

SSE event name 为 `message`，payload 类型：

```text
content | retrying | final | error | done
```

### 6.2 做什么

- 新增 API route：`/api/chat_stream`。
- ModelGateway 增加 stream interface。
- StubModelGateway 提供 deterministic streaming。
- 如果真实 provider 支持 stream，则 adapter 转换 provider chunks。
- 流式过程中仍要记录完整 trace：
  - run started
  - user appended
  - context assembled
  - model call started/completed/failed
  - assistant appended
  - run completed/failed
- 如果中途工具调用发生，P0 可先采用“工具调用阶段不逐 token 输出，工具完成后继续流式 final answer”的策略。

### 6.3 不做什么

- 不实现复杂前端。
- 不做多工具并发流式。
- 不在没有真实 provider 支持时强行模拟所有 provider 行为。

### 6.4 验收

- `/api/chat_stream` 返回合法 SSE。
- 空问题仍返回 Java 兼容错误事件。
- 正常问题能输出 `content`、`final`、`done`。
- 失败时输出 `error`、`done`，并记录 `RUN_FAILED`。
- 非流式 `/api/chat` 不回退。

## 7. 子阶段 10C：ToolGateway 可靠性增强

### 7.1 目标

把当前 ToolGateway 从“参数校验 + 执行 + trace”增强为可解释、可测试的工具治理层。

### 7.2 做什么

- 工具执行 timeout。
- 幂等工具的工程层 retry。
- retry backoff + jitter。
- 错误分类：
  - `PARAM_VALIDATION_FAILED`
  - `TOOL_TIMEOUT`
  - `TOOL_RETRY_EXHAUSTED`
  - `UPSTREAM_UNAVAILABLE`
  - `TOOL_ERROR`
  - `TOOL_BLOCKED`
- 对模型返回结构化错误：

```json
{
  "success": false,
  "error_type": "TOOL_TIMEOUT",
  "message": "...",
  "suggestion": "..."
}
```

- 记录：
  - `TOOL_CALL_TIMEOUT`
  - `TOOL_CALL_RETRY`
  - `TOOL_CALL_FAILED`
  - `TOOL_CALL_LATE_RESULT_DISCARDED`
- 同一 run 内连续失败治理：
  - 同一工具连续失败达到阈值时，给模型返回停止建议。
  - 不让模型无限循环调用同一个失败工具。

### 7.3 不做什么

- 不实现复杂熔断中心。
- 不把所有异常都交给模型自行判断。
- 不对非幂等工具自动重试。

### 7.4 验收

- 参数错误不重试，直接返回结构化错误。
- 幂等工具 timeout / 网络错误按策略重试。
- 非幂等工具不自动重试。
- trace 中能看到 retry / timeout / failed。
- 现有工具成功路径不回退。

## 8. 子阶段 10D：上下文窗口与压缩

### 8.1 目标

实现模型调用前的上下文窗口治理，避免真实模型调用时因为上下文过长失败，同时保留排障链路中的关键证据、用户约束、当前任务状态和工具调用上下文。

10D 已完成前置调研，结论是本阶段不只是“摘要压缩”，而是要引入项目内轻量 `ContextManager / Reducer` 层。`ContextAssembler` 只负责组装已治理好的上下文，不再承担全部预算、裁剪和压缩判断。

### 8.2 做什么

- 引入近似 `TokenEstimator`，并保留 provider tokenizer / token counting API 扩展口。
- 新增 `ContextBudget` / `ContextComponentUsage` / `PreparedContext`。
- 新增 `ContextManager`，在模型调用前统一治理上下文。
- 设置模型上下文预算和组件优先级：
  - system prompt
  - core memory
  - memory index
  - active history
  - current user input
  - tool results
- 新增 `ToolResultReducer`，对工具结果做脱敏和持久化；单个工具输出超过 `1000 tokens` 时，通过 `ContentCompressionBackend` 调用 Headroom 做内容级压缩，压缩结果带 `raw_ref` 进入上下文。
- 当 active history 超预算时触发历史压缩。
- 压缩策略 P0 可保留最近 N 轮，设计上保留 sliding-window 扩展点。
- 压缩摘要持久化为 rollout event：`HISTORY_TRIMMED`。
- 压缩摘要 + 最近几轮对话继续组装上下文。
- 压缩 prompt 必须版本化。
- 恢复逻辑根据最新有效 `HISTORY_TRIMMED` 恢复 summary + recent history，避免重复恢复旧事件。
- 压缩结果需要 eval 验证关键信息不丢。

### 8.3 不做什么

- 不引入 `agent_session_state` 作为必选表。
- 不做完整 Recall Memory。
- 不把压缩摘要当作长期记忆。
- 不在本阶段做后台 LLM memory extraction，除非进入 10G。
- 不接入 Letta / LlamaIndex Memory / OpenAI Responses compaction session 替代本项目 Harness。
- 不把 LLMLingua 作为主 history compactor。
- 不把 Headroom 作为会话级 ContextManager。
- 不依赖 Headroom CCR 本地存储作为原文事实源。

### 8.4 验收

- 长会话触发压缩。
- 压缩后上下文仍包含：
  - 已确认事实。
  - 用户明确约束。
  - 未解决问题。
  - 关键工具证据摘要。
- 最近几轮对话保留。
- 单个工具输出 `> 1000 tokens` 时会先持久化原文，再压缩并带 `raw_ref` 进入上下文；`<= 1000 tokens` 不做内容压缩。
- 大工具结果不会无限进入上下文。
- tool call / tool result 配对不被破坏。
- `CONTEXT_ASSEMBLED` trace 包含 token budget、compaction 状态和 component usage。
- eval 能覆盖“压缩前后仍能回答关键问题”。

## 9. 子阶段 10E：安全与生产级多租户鉴权

### 9.1 目标

从开发态 trusted headers 过渡到生产可解释的认证上下文和租户隔离策略。

### 9.2 做什么

- 定义认证入口：
  - JWT / SSO / API Gateway trusted identity 三选一或组合。
- `tenant_id/user_id/roles/permissions` 从认证上下文解析。
- 开发态 headers 只能在 local/dev 使用。
- API 层权限校验：
  - chat 权限。
  - session clear 权限。
  - memory write 权限。
  - 高风险工具权限。
- Secret 管理：
  - API key 只来自环境变量或 secret manager。
  - 不进入 prompt。
  - 不进入工具返回。
  - trace artifacts 脱敏。
- 数据访问层 tenant filter 审核。
- PostgreSQL RLS 作为后续增强，不在没有完整认证链路前强制启用。

### 9.3 不做什么

- 不自研完整 IAM 系统。
- 不在模型工具参数里暴露 tenant/user/run。
- 不让前端随意传 tenant_id 后直接信任。

### 9.4 验收

- 本地 dev 可以继续使用 headers。
- production env 下缺少可信认证信息时拒绝请求。
- 模型无法通过工具参数伪造 tenant/user/run。
- trace 不泄露 secret。
- repository 查询仍有 tenant/user/agent/session 过滤。

### 9.5 当前实现状态

```text
已完成 P0。
```

当前说明：

```text
已新增 AuthenticatedPrincipal、AuthContextResolver、PermissionChecker 和公共 redactor。
P0 支持 dev_headers 与 trusted_gateway；jwt 只保留配置枚举和 claims contract，配置为 jwt 时返回明确未实现认证错误。
API 层已对 chat、chat_stream、clear session、session read 做权限校验。
ToolGateway 已在 handler 执行前集中检查工具权限，memory 工具使用 memory:read / memory:write，普通工具使用 tool:execute 或 tool:execute:{tool_name}。
生产环境禁止 dev_headers，禁止关闭 auth_permission_enforcement；trusted_gateway 使用 X-Auth-Gateway-Secret 常量时间比较，普通 X-Tenant-Id 不能覆盖可信身份。
```

当前验收：

```text
python3 -m pytest                                      -> 87 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

## 10. 子阶段 10F：RAG 真实管线与 RAG 专项评测

### 10.1 目标

把当前 `queryInternalDocs` fixture 逐步升级为 LlamaIndex + Milvus 的真实 RAG adapter，并建立专项评测。

### 10.2 做什么

- LlamaIndex ingestion / retrieval adapter。
- Milvus collection schema：
  - dense vector
  - sparse/BM25 vector
  - scalar fields：`tenant_id`、`knowledge_base_id`、`document_id`、source/title/path。
- Milvus 原生 BM25 hybrid search。
- 强制 tenant filter / knowledge_base filter。
- rerank adapter。
- citation metadata。
- `queryInternalDocs` 输出保持现有 JSON 契约。
- RAG eval：
  - 检索层 hit rate / MRR / recall@k。
  - 生成层 faithfulness / relevance / citation correctness。

### 10.3 不做什么

- 不回到应用侧 Lucene BM25。
- 不绕过 LlamaIndex 自研完整 RAG 框架。
- 不允许无 tenant filter 的 Milvus 查询。

### 10.4 验收

- 同一 tenant 只能检索自己的知识库。
- queryInternalDocs 返回 chunk、ref、source、score。
- citation metadata 可用于最终答案溯源。
- RAG 专项 eval 有独立 report。

### 10.5 当前实施状态

`10F Batch B Document Loader 与 Chunking` 已完成。`10F Batch C` 已实现 Embedding adapter、PostgreSQL RAG metadata/repository、LlamaIndex Milvus dense + Jieba BM25 Store 和同步 ingestion；Milvus Lite 与离线主验收通过，真实 PostgreSQL 门禁待验。

```text
RAG Batch B/C 联合 pytest               -> 89 passed
MODEL_PROVIDER=stub 全量 pytest         -> 259 passed, 1 existing warning
基础 eval runner                        -> 14/14 passed
Milvus Lite integration                 -> passed
Ruff                                    -> passed
compileall                              -> passed
```

当前不是 10F complete。Batch C 状态为 `implementation complete / Milvus gate passed / real PostgreSQL gate pending`；下一 RAG 实施阶段为 `Batch D Retrieval 与 queryInternalDocs 改造`，Batch D-F 尚未实现。

Batch D 已完成专项分析与设计审核，详细计划为 `docs/10F-D-hybrid-retrieval-plan.md`。该计划经过三轮双路独立复审并最终双 PASS；当前状态是 `design approved / implementation not started`。

## 11. 子阶段 10G：长期记忆专项增强

### 11.1 目标

在第 9 步长期记忆核心闭环基础上，先增强模型可见的记忆提示、元数据和工具描述，再单独增加更强的评测和可选后台任务。

### 11.2 做什么

10G 拆成三个子阶段：

```text
10G.1 长期记忆提示与元数据优化
10G.2A 长期记忆专项评测
10G.2B 长期记忆后台任务与生命周期治理
```

10G.1 做：

- Memory Index 改为更像 Letta 的 Memory Metadata。
- Core Memory block description 精细化。
- memory tools description / system prompt 优化。

10G.1 不做：

- 不做长期记忆专项 eval。
- 不做后台 LLM extraction。
- 不做 memory_candidate。
- 不做 Recall Memory / 历史对话召回。
- 不做生命周期遗忘 / archive job。
- 不新增 Archival Memory update/delete/archive 工具。

10G.1 阶段计划：

```text
docs/10G1-memory-prompt-metadata-plan.md
```

10G.2A 做：

- 长期记忆专项 eval：
  - 该写 core memory 时是否写。
  - 该写 archival memory 时是否写。
  - 不该写时是否拒绝。
  - 该检索 memory 时是否检索。
  - 检索结果是否被正确使用。
  - 历史经验是否被当作历史参考，而非当前证据。
- 确定性 evaluator conformance 与真实模型行为双轨评测。
- 48 个项目专用黄金样本、Memory Snapshot、Trace/State/Retrieval/Use Judges。
- Track B 固定执行 48 cases x 3 repetitions；缺少真实模型凭证时只能标记 framework complete。

10G.2A 阶段计划：

```text
docs/10G2-long-term-memory-eval-plan.md
```

10G.2B 做：

- 可选后台任务与生命周期治理：
  - 上下文压缩前记忆保存提醒。
  - memory extraction job。
  - lifecycle forgetting / archive job。
- 后台任务可以先用进程内 worker / asyncio task，生产再考虑 Celery / Dramatiq / Arq。

### 11.3 不做什么

- 10G.1 不在本阶段上线长期记忆专项 eval 或后台任务。
- 10G.2A 不实现后台任务、遗忘机制、真实 Embedding 或 Rerank。
- 10G.2B 不在没有通过长期记忆专项评测前上线大模型异步抽取。
- 不把原始日志、指标、trace dump 写入长期记忆。
- 不做 Recall Memory，除非单独立项。

### 11.4 验收

- 10G.1 验收：
  - 模型上下文中出现 `<memory_metadata>`，而不是模型可见的 `<memory_index>`。
  - `<memory_metadata>` 只暴露 archival memory 的概况、topics、tags、scopes 和使用规则，不注入 Archival Memory 正文。
  - Core Memory 三个 block 的 description 能清晰指导写入边界。
  - memory tool description 和系统提示词明确 tags/scope 是可选过滤，历史记忆不能替代实时证据。
  - 不改变 memory tools 参数 schema、权限契约和 tenant/user/run 后端注入规则。
- 10G.2A 验收：
  - memory eval suite 独立可运行。
  - 评测框架完成和真实模型 baseline 完成严格分开。
  - 只有 Dataset、Snapshot、Judges、Runner、Track A 测试完成时，才能标记 `10G.2A framework complete`。
  - 只有 Track B 的 48 x 3 真实模型实验完成并审核后，才能标记 `10G.2A complete`。
- 10G.2B 验收：
  - 后台任务如果实现，必须有 job 状态、失败重试、trace。
- 不能影响现有同步 chat 主链路。

### 11.5 当前实现状态

```text
10G.1 已完成。
10G.2A framework complete；第一次 Track B dev 36 x1 和 R1-A 评测契约修复已完成；正式 48 x3 baseline pending。
10G.2B 尚未立项。
```

当前说明：

```text
已将模型可见的长期记忆概况从 <memory_index> 升级为 <memory_metadata>。
<memory_metadata> 包含 archival_memory_total、available_topics、available_tags、available_scopes、usage_rules，不注入 Archival Memory 正文。
内部 MemoryIndexService、memory_index_xml、hasMemoryIndex、memoryIndexTopicCount 等兼容命名暂时保留，并新增 hasMemoryMetadata。
已精细化 user_rules、user_ops_profile、service_notes 三个 Core Memory block description。
已优化 listMemoryTopics、searchMemory、updateCoreMemory、saveArchivalMemory 的工具描述，并更新 ops-agent-system-v2.md 长期记忆规则。
10G.2A 已完成 Dataset、Snapshot、eval-only capture、Trace/State/Retrieval/Use Judges、Track A conformance、Runner/report/checkpoint、第一次 Track B dev 36 x1 和 R1-A。R1-A 历史离线重判后 blocking 23/35、diagnostic 0/1；safety/isolation 在当时已执行路径中均未观察到 violation，mechanical retrieval 4/12，production ranking 当时为 `not_evaluated`。此后 H-R1、M-R1 与 M-P2 已完成，M-P1 已用独立 dev Dataset 执行真实 Embedding + PostgreSQL baseline 并等待独立验收；正式 48 x3 baseline 尚未完成。
10G.2B 尚未实现后台 LLM extraction、memory_candidate、Recall Memory、遗忘/archive job，也未新增 Archival Memory update/delete/archive 工具。
```

当前验收：

```text
Python 3.11 长期记忆专项测试                           -> 57 passed
Python 3.11 全量 pytest                                -> 165 passed, 1 warning
PYTHONPATH=src python -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python -m compileall -q src tests      -> passed
Python 3.11 ruff（本阶段新增/修改文件）                -> passed
```

## 12. 子阶段 10H：观测、LangSmith、部署配置

### 12.1 目标

让系统具备工程可观测性、可部署性和可回归验证能力。

### 12.2 做什么

- structlog / logging 配置。
- request id / run id / tenant id 进入日志字段。
- OpenTelemetry hooks。
- Prometheus metrics：
  - request latency
  - model latency
  - tool latency
  - token usage
  - error count
  - eval pass rate
- LangSmith 可选接入：
  - dataset
  - experiment
  - trace
  - evaluator
- 部署配置：
  - `.env.example`
  - Dockerfile
  - docker-compose for local Postgres / Milvus
  - health check

### 12.3 不做什么

- 不让 LangSmith 成为本地 eval 的唯一依赖。
- 不把敏感内容原样上报外部 SaaS。
- 不在没有脱敏策略前记录完整 prompt/tool raw output 到外部平台。

### 12.4 验收

- 本地启动文档可用。
- health check 可用。
- metrics endpoint 可用。
- 关闭 LangSmith 时系统正常运行。
- 打开 LangSmith 时能记录实验和 trace 摘要。

## 13. subAgent 拆分建议

第 10 步适合拆给多个 subAgent，但必须按独立写入范围拆。

建议：

| 子阶段 | subAgent 类型 | 写入范围 |
|---|---|---|
| 10A | worker | `model_gateway/`、`config.py`、相关 tests |
| 10B | worker | `api/`、`model_gateway/` stream interface、相关 tests |
| 10C | worker | `tools/gateway.py`、`tools/errors.py`、相关 tests |
| 10D | worker | `harness/token_estimator.py`、`harness/context_budget.py`、`harness/context_manager.py`、`harness/tool_result_reducer.py`、`harness/compaction.py`、`harness/context_assembler.py`、`harness/runtime.py`、`harness/history.py`、`harness/events.py`、`config.py`、`prompts/`、相关 tests |
| 10E | worker | `api/`、`harness/context.py`、security module、相关 tests |
| 10F | worker | `rag/`、`tools/builtin/internal_docs_tool.py`、RAG eval |
| 10G.1 | worker | `memory/`、`prompts/`、相关 tests |
| 10G.2A | worker | `memory/` 的只读评测支撑、`evals/`、相关 tests |
| 10G.2B | worker | 后台任务与生命周期治理模块 |
| 10H | worker | `observability/`、deployment files、docs |

主 agent 负责：

- 阶段设计把关。
- subAgent 任务边界。
- 合并后验收。
- 检查是否偏离文档。
- 跑全量测试和 eval。

## 14. 推荐下一步

当前 `10G.1 长期记忆提示与元数据优化` 与 `H-R1 Harness 多工具与多轮 ReAct 循环` 已完成；`10G.2A 长期记忆专项评测` 已完成评测框架、Track A conformance、第一次 Track B dev 36 x1 和 R1-A 评测契约修复，正式 48 x3 baseline 尚未完成。

原因：

- 10A、10B.1、10C、10D、10E 已完成。
- 10D 已按前置调研和正式计划完成代码实现与主验收。
- 10E 已按计划完成生产级多租户鉴权与安全策略 P0。
- 10G.1 已在长期记忆核心闭环上完成提示与元数据优化，没有引入后台任务或专项 eval。
- 10G.2A 已完成双轨评测框架、48 个黄金样本、Dataset/Snapshot/Judges/Runner/report、Track A forced safety/isolation probes、第一次真实模型 dev 36 x1 和 R1-A 离线重判；当前仍不代表正式真实模型 baseline 已完成。
- H-R1 已完成单响应多工具顺序执行、有限多轮、run 级预算、all-or-block、工具协议防护、禁用工具收尾和 run 状态清理。主验收为专项 57 passed、全量 stub 182 passed、基础 eval 14/14，Ruff 与 compileall 通过，Dataset/prompt 哈希未变。
- 10F Batch B 已完成；Batch C 已完成代码、Milvus Lite 和离线主验收，真实 PostgreSQL 门禁待验。下一 RAG 实施阶段是 Batch D Retrieval，10F 整体仍未完成。

当前 10G.1 阶段计划：

```text
docs/10G1-memory-prompt-metadata-plan.md
```

当前 10G.2A 阶段计划：

```text
docs/10G2-long-term-memory-eval-plan.md
docs/10G2A-memory-eval-baseline-remediation-plan.md
```

R1-A、H-R1 与 M-R1 已完成。M-R1 R1-B positive 已用原报告零模型调用离线重判为 6/6；R1-C 已完成 A01/I04 精确等价候选和 positive 固定来源 SHA 门禁实现，原始 guardrails 报告经新 Dataset 零模型调用离线重判为 9/9，preferred behavior、安全和隔离门禁全部通过且无 violation。R1-C 已于 2026-07-12 通过独立验收，M-R1 状态为 `complete`。M-P2 PostgreSQL 持久化已完成独立验收；M-P1 真实 Embedding + pgvector dev baseline 已执行并等待独立验收。本轮未运行 dev 36 x3 或 holdout 12 x3，正式长期记忆 baseline 尚未完成，10G.2B 仍未立项。

## 15. 验收命令

每个子阶段完成后至少运行：

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
PYTHONPATH=src python3 -m compileall -q src tests
```

如果安装了 ruff：

```bash
python3 -m ruff check src tests
```

如果子阶段涉及 API：

```bash
python3 -m pytest tests/test_skeleton.py
```

如果子阶段涉及 eval：

```bash
python3 -m pytest tests/test_eval_runner.py
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
```

## 16. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否把第 10 步拆成可验收子阶段 | 是 |
| 是否直接自由实现高级能力 | 否 |
| 是否优先使用成熟框架 / SDK | 是 |
| 是否保留当前 Harness 主链路 | 是 |
| 是否保留 StubModelGateway 和本地 eval | 是 |
| 是否避免真实 API key 成为测试依赖 | 是 |
| 是否避免提前做 Recall Memory | 是 |
| 是否避免绕过 tenant context / ToolGateway / trace | 是 |
| 是否明确 subAgent 写入边界 | 是 |

## 17. 阶段通过标准

第 10 步整体完成标准不是“一次性全做完”，而是每个子阶段分别通过：

1. 子阶段计划明确。
2. 实现范围没有越界。
3. 新增能力有测试。
4. 原有 41 个测试不回退。
5. eval runner 不回退。
6. trace / tenant / prompt / tool schema 契约不被破坏。
7. roadmap 记录该子阶段状态。
