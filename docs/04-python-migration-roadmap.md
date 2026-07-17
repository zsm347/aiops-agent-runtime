# Python 迁移总路线

## 1. 目的

本文用于固定 SuperBizAgent Java 单 Agent Harness 迁移到 Python 版本的正式实施路线。

它解决一个问题：后续不能把 `Skeleton P0`、`基础 eval runner`、`Migration P0`、`业务工具迁移`、`长期记忆迁移` 混在一起做。

原则：

- 每个阶段只做本阶段目标。
- 每个阶段完成后先审核和验收，再进入下一阶段。
- 成熟框架优先，项目只做薄适配层和必须自控的工程约束。
- Java 项目仍是当前行为参考基线。

## 2. 当前状态

已完成：

1. Java 能力盘点。
2. Python 迁移规格文档。
3. 技术栈选型与架构 ADR。
4. Python 项目骨架。
5. Skeleton P0。
6. 基础 Eval Runner。
7. Migration P0。
8. 业务工具迁移。
9. 长期记忆迁移。

当前下一阶段：

```text
第 10 步：高级能力
```

说明：

- 第 1-9 步已完成实现和回溯审核。
- 第 9 步已按 `docs/09-long-term-memory-migration-plan.md` 完成长期记忆核心闭环，并通过主验收。
- 第 10 步高级能力已完成总计划，并完成子阶段 10A 真实 ModelGateway 与模型调用治理、10B.1 Streaming / chat_stream 最小闭环、10C ToolGateway 可靠性增强、10D 上下文窗口与压缩、10E 生产级多租户鉴权与安全策略、10G.1 长期记忆提示与元数据优化。
- `10F Batch B` 已完成；`Batch C Milvus Store & Ingestion` 已完成代码、Milvus Lite 和离线主验收，真实 PostgreSQL 并发/迁移门禁待环境恢复。`Batch D Retrieval 与 queryInternalDocs` 及其可靠性修复已完成独立验收：真实 Milvus Lite、RAG B/C/D 联合回归、全量 stub 与基础 eval 均通过。10F 尚未整体完成，下一步先进行 `10F-F0 RAG 专项评测基础与无 rerank hybrid baseline` 的设计，再决定 Batch E rerank 实施与 Batch F 正式评测。
- `H-R1 Harness 多工具与多轮 ReAct 循环` 已完成实现和主验收；阶段计划及验收记录见 `docs/H-R1-harness-multi-tool-react-loop-plan.md`。
- 10G.1 阶段计划见 `docs/10G1-memory-prompt-metadata-plan.md`。
- `10G.2A 长期记忆专项评测` 已完成 framework、Track A conformance、第一次 Track B dev 36 x1、R1-A/R1-C 评测契约修复与 M-R1 prompt/tool 语义优化。`M-P0` 确定性 exact dedupe 与进程内原子写入已完成独立验收；`M-P2` PostgreSQL 持久化实现、39-case 真实 PostgreSQL gate 与独立验收均已通过，状态为 `complete`。真实 embedding 与语义去重仍未完成；正式 48 x3 baseline 尚未完成，`10G.2B` 后台任务与生命周期治理尚未立项。

## 3. 正式实施路线

### 1. Java 能力盘点

目标：

- 搞清楚 Java 单 Agent Harness 当前真实实现了哪些能力。
- 区分已实现能力、设计能力和未来能力。

产物：

```text
docs/01-java-capability-inventory.md
```

状态：

```text
已完成
```

### 2. Python 迁移规格文档

目标：

- 定义 Python 版本要迁移什么、不迁移什么。
- 固定 API、tool schema、trace、DB、prompt、eval 等兼容契约。

产物：

```text
docs/02-python-migration-spec.md
```

状态：

```text
已完成
```

### 3. 技术栈选型与架构 ADR

目标：

- 明确 Python 版本使用哪些框架。
- 明确哪些能力交给成熟框架，哪些由项目 adapter 负责。

当前架构原则：

- API：FastAPI。
- Agent 编排：LangGraph。
- RAG 编排：LlamaIndex。
- Schema/参数校验：Pydantic，后续可参考 PydanticAI。
- 数据库：PostgreSQL + SQLAlchemy + Alembic。
- 模型访问：P1 默认 OpenAI-compatible SDK；后续按需 LiteLLM / DashScope SDK。
- Eval：本地 deterministic eval gate，后续可接 LangSmith。

产物：

```text
docs/03-python-architecture-design.md
```

状态：

```text
已完成
```

### 4. Python 项目骨架

目标：

- 创建 Python 项目基础目录和依赖。
- 建立可导入、可测试的最小工程骨架。

当前已包含：

- `pyproject.toml`
- `.env.example`
- FastAPI app skeleton
- API DTO
- request context model
- rollout event enum
- prompt registry
- stub HarnessService
- 基础测试

状态：

```text
已完成
```

验收：

```bash
python3 -m pytest
```

当前结果：

```text
基础骨架阶段已通过；当前总体验收见第 8 步状态。
```

### 5. Skeleton P0

目标：

实现最小 Agent Harness 闭环：

```text
chat -> context -> model stub -> tool call -> answer -> trace
```

本阶段要做：

- `/api/chat` 调用真实 Skeleton HarnessService，而不是简单 echo stub。
- 创建 `runId`。
- 创建最小 `ConversationRuntime`。
- 组装最小 context：
  - system prompt
  - active history
  - current user message
- 使用 `StubModelGateway`。
- 使用一个可控工具，例如 `getCurrentDateTime`。
- 建立最小 `ToolRegistry` 和 `ToolGateway`。
- 记录内存 trace events。
- 返回最终 answer。

本阶段不做：

- 不接真实 Qwen。
- 不接 PostgreSQL event store。
- 不做事件 replay。
- 不做 streaming。
- 不做 compaction。
- 不做真实 RAG。
- 不做长期记忆。
- 不接 LangSmith。

建议先写阶段计划：

```text
docs/05-skeleton-p0-implementation-plan.md
```

状态：

```text
已完成
```

### 6. 基础 Eval Runner

目标：

- 用 smoke / capability case 验证 Skeleton P0 最小闭环。
- 建立本地 deterministic eval gate。

本阶段要做：

- 定义 eval case schema。
- 支持 fixture/mock 工具输出。
- 运行 `/api/chat` 或 HarnessService。
- 收集 trace。
- 实现基础 rule judge。
- 输出 eval report。

本阶段不做：

- 不做完整 LLM-as-judge。
- 不依赖 LangSmith。
- 不做 RAG 专项评测。
- 不做长期记忆专项评测。

建议阶段计划：

```text
docs/06-basic-eval-runner-plan.md
```

状态：

```text
已完成
```

### 7. Migration P0

目标：

把 Skeleton P0 升级为能合理称为 Java 单 Agent Harness Python 迁移版的最小后端。

本阶段要做：

- tenant context。
- PostgreSQL `agent_rollout_event` event store。
- session replay/recovery。
- ModelGateway adapter 结构完善。
- ToolGateway policy/trace adapter 完善。
- prompt version 进入 trace。
- 核心 ContextAssembler。
- 保留 Java 兼容 trace schema。

本阶段可以使用：

- stub model provider。
- mock logs / metrics / RAG fixture。

本阶段不做：

- 不做完整真实 RAG。
- 不做长期记忆完整迁移。
- 不做 streaming。
- 不做 compaction 全量能力。

建议阶段计划：

```text
docs/07-migration-p0-implementation-plan.md
```

状态：

```text
已完成
```

### 8. 业务工具迁移

目标：

迁移日志、告警、RAG 等业务工具，并保持工具契约和 trace 一致。

本阶段要做：

- 日志工具契约迁移。
- 告警工具契约迁移。
- `queryInternalDocs` 工具契约迁移。
- 工具参数 schema。
- 工具输出 JSON 结构。
- 工具错误结构。
- 工具调用 trace。

RAG 说明：

- 工具契约可以先迁移。
- 完整 LlamaIndex + Milvus RAG pipeline 应按 RAG 子阶段逐步实现。

状态：

```text
已完成
```

当前验收：

```text
python3 -m pytest                         -> 34 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner -> passed=8 failed=0
```

### 9. 长期记忆迁移

目标：

迁移长期记忆核心能力。

本阶段要做：

- Core Memory。
- Archival Memory。
- memory index。
- memory write policy。
- memory retrieval。
- memory trace events。
- memory tool contracts：
  - `updateCoreMemory`
  - `saveArchivalMemory`
  - `searchMemory`
  - `listMemoryTopics`

本阶段不默认做：

- Recall Memory。
- 完整后台 LLM memory extraction。
- memory UI。

状态：

```text
已完成
```

阶段计划：

```text
docs/09-long-term-memory-migration-plan.md
```

当前说明：

```text
已实现 Core Memory、Archival Memory、Memory Index/Metadata、Memory Write Policy、Memory Retrieval、Memory Trace Events 和 4 个 memory tools。
主验收中修正了 tags 过滤语义：多标签检索按 Java 参考实现使用任一标签命中。
```

当前验收：

```text
python3 -m pytest                                      -> 41 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

### 10. 高级能力

目标：

补齐生产化和高级工程能力。

可能包含：

- streaming。
- 生产级多租户鉴权。
- 安全策略。
- 后台任务。
- 完整 eval。
- LangSmith 集成。
- RAG 专项评测。
- 长期记忆专项评测。
- OpenTelemetry / Prometheus 观测。
- 部署配置。

状态：

```text
进行中：10A 已完成，10B.1 已完成，10C 已完成，10D 已完成，10E 已完成，10F Batch B-D 已完成主验收（Batch C PostgreSQL gate pending），10G.1 已完成；下一 RAG 阶段为 10F-F0 评测基础与 baseline，随后才是 Batch E/F；10G.2A framework complete、initial Track B dev 36 x1、R1-A/R1-C、M-R1、M-P0 与 M-P2 已完成，正式 baseline incomplete
```

阶段计划：

```text
docs/10-advanced-capabilities-plan.md
docs/10A-model-gateway-plan.md
docs/10B-streaming-plan.md
docs/10C-tool-gateway-reliability-plan.md
docs/10D-context-management-research.md
docs/10D-context-window-compaction-plan.md
docs/10E-auth-tenant-security-plan.md
docs/10F-rag-real-pipeline-plan.md
docs/10F-D-hybrid-retrieval-plan.md
docs/10G1-memory-prompt-metadata-plan.md
docs/10G2-long-term-memory-eval-plan.md
docs/10G2A-memory-eval-baseline-remediation-plan.md
docs/M-P2-postgresql-memory-persistence-plan.md
```

已完成子阶段：

```text
10A 真实 ModelGateway 与模型调用治理
10B.1 Streaming / chat_stream 最小闭环
10C ToolGateway 可靠性增强
10D 上下文窗口与压缩
10E 生产级多租户鉴权与安全策略
10F Batch B Document Loader 与 Chunking
10F Batch C Milvus Store 与同步 Ingestion（代码与 Milvus 门禁完成，真实 PostgreSQL 门禁待验）
10G.1 长期记忆提示与元数据优化
```

已完成评测框架和第一次真实模型 dev 运行、正式 baseline 尚未完成：

```text
10G.2A 长期记忆专项评测
```

未立项：

```text
10G.2B 长期记忆后台任务与生命周期治理
```

10A 当前说明：

```text
已新增 OpenAI-compatible ModelGateway adapter、model gateway factory、模型网关错误分类、tools schema 传递、usage/finish_reason trace。
默认 provider 仍为 stub；真实 provider 需要通过 model_provider 和 model_api_key 显式启用。
主验收中补强了 replay 的 tool-call/history 恢复，使真实 OpenAI-compatible provider 更稳。
```

10A 当前验收：

```text
python3 -m pytest                                      -> 52 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

10B 当前说明：

```text
已新增 POST /api/chat_stream，返回 text/event-stream，SSE event name 为 message。
空问题输出 error + done，不进入 runtime。
正常和工具场景复用现有 graph.run() 得到最终答案，再切分输出 content + final + done，trace 保持与非流式主链路一致。
当前 10B.1 是 API 层 SSE 切块流式，不是 provider token 级 streaming；OpenAI-compatible provider token stream 留作后续增强。
```

10B 当前验收：

```text
python3 -m pytest                                      -> 57 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

10C 当前说明：

```text
已增强 ToolErrorResult，保留 success/error_type/message/suggestion 兼容字段，并新增 reason、violations、retryable_by_runtime、retryable_by_model、allowed_next_actions、disallowed_next_actions、retry_budget。
参数校验失败会返回字段级 violations，不执行 handler，不做 runtime retry。
ToolGateway 已支持 timeout_seconds 实际超时控制、幂等工具 runtime retry、backoff+jitter、TOOL_CALL_RETRY/TOOL_CALL_TIMEOUT/TOOL_CALL_FAILED trace、本 run 内失败模式治理和错误脱敏。
非幂等工具不会自动重试；达到失败治理阈值后只在当前 run 内停止同一失败模式的盲目重试。
```

10C 当前验收：

```text
python3 -m pytest                                      -> 66 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

10D 当前说明：

```text
已新增 ContextManager / Reducer 层、ApproxTokenEstimator、ContextBudget、ContextComponentUsage、PreparedContext、ToolResultReducer、ContentCompressionBackend 和 deterministic HistoryCompactor。
ContextAssembler 支持组装 PreparedContext，不再承担全部预算、裁剪和压缩判断。
工具结果治理分两层：单个工具输出 <=1000 tokens 时保持原工具 JSON 形状，仅做脱敏；>1000 tokens 时先保留 rollout raw_ref，再进入内容级压缩 envelope。
Headroom 只作为可选内容压缩 adapter；未安装或调用失败时降级到 deterministic backend，不作为必需依赖。
历史超预算时生成 HISTORY_TRIMMED summary，保留最近 N 轮；恢复时使用最新成功 HISTORY_TRIMMED 的 summary + toSequence 之后事件。
主验收中补强了当前 run 内工具结果治理，避免 ReAct 第二次模型调用绕过 ToolResultReducer。
```

10D 当前验收：

```text
python3 -m pytest                                      -> 74 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

10E 当前说明：

```text
已新增 AuthenticatedPrincipal、AuthContextResolver、PermissionChecker 和公共 redactor。
P0 支持 dev_headers 与 trusted_gateway；jwt 只保留配置枚举和 claims contract，配置为 jwt 时返回明确未实现认证错误。
API 层已对 chat、chat_stream、clear session、session read 做权限校验。
ToolGateway 已在 handler 执行前集中检查工具权限，memory 工具使用 memory:read / memory:write，普通工具使用 tool:execute 或 tool:execute:{tool_name}。
生产环境禁止 dev_headers，禁止关闭 auth_permission_enforcement；trusted_gateway 使用 X-Auth-Gateway-Secret 常量时间比较，普通 X-Tenant-Id 不能覆盖可信身份。
```

10E 当前验收：

```text
python3 -m pytest                                      -> 87 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

10G.1 当前说明：

```text
已将模型可见的长期记忆概况从 <memory_index> 升级为 <memory_metadata>，内容包含 archival_memory_total、available_topics、available_tags、available_scopes 和 usage_rules。
内部 MemoryIndexService、memory_index_xml、hasMemoryIndex、memoryIndexTopicCount 等兼容命名暂时保留，并新增 hasMemoryMetadata。
已精细化 Core Memory 三个 block description，优化 listMemoryTopics、searchMemory、updateCoreMemory、saveArchivalMemory 的工具描述，并更新 ops-agent-system-v2.md 长期记忆规则。
本阶段未实现长期记忆专项 eval、后台 LLM extraction、memory_candidate、Recall Memory、遗忘/archive job，也未新增 Archival Memory update/delete/archive 工具。
```

10G.1 当前验收：

```text
python3 -m pytest                                      -> 87 passed, 1 skipped
PYTHONPATH=src python3 -m superbiz_agent.evals.runner  -> passed=14 failed=0
PYTHONPATH=src python3 -m compileall -q src tests      -> passed
python3 -m ruff check src tests                        -> 未运行，当前环境未安装 ruff
```

## 4. 阶段推进规则

每个阶段遵循：

1. 先写该阶段实施计划。
2. 审核实施计划。
3. 实现代码。
4. 补测试。
5. 跑测试。
6. 审核产物。
7. 明确是否进入下一阶段。

不得跳阶段提前做后续能力。

例如：

- 在 Skeleton P0 阶段，不提前做 PostgreSQL replay。
- 在 Migration P0 阶段，不提前做完整 LlamaIndex/Milvus RAG。
- 在业务工具迁移阶段，不提前做长期记忆全量能力。
- 在长期记忆迁移阶段，不默认做 Recall Memory。

## 5. 下一步

长期记忆评测 R1-A、H-R1、M-R1 与 M-P0 已完成；R1-C 与 M-P0 已通过独立验收。长期记忆后续待办为：

```text
正式 Track B 48 x 3 baseline、M-P1 真实 Embedding 与生产检索基线、M-P3 语义去重阈值校准，以及 10G.2B 生命周期治理仍待后续推进；M-P2 PostgreSQL 持久化已完成真实数据库 gate 与独立验收
```

10G.1 阶段已完成实现和主验收，阶段计划为 `docs/10G1-memory-prompt-metadata-plan.md`。
10G.2A 阶段计划 `docs/10G2-long-term-memory-eval-plan.md` 已完成实现和主验收：48 个样本、Snapshot、Judges、Track A conformance、Runner/report/checkpoint 均已就绪。第一次 Track B dev 36 x1 和 R1-A 评测契约修复已完成；离线重判为 blocking 23/35、diagnostic 0/1，safety/isolation 在本轮观测范围内均为 0 violation，mechanical retrieval 4/12，production retrieval ranking 因 local-deterministic embedding 标记为 `not_evaluated`。H-R1 Graph 与 M-R1 prompt/tool 语义修复已经完成；后续真实 embedding、持久化和语义去重必须分别立项。

H-R1 已完成有限多工具/多轮循环、run 级预算、协议防护、禁用工具收尾和 run 状态清理，并通过专项 57、全量 stub 182、基础 eval 14/14、Ruff 与 compileall 主验收。该状态不表示正式长期记忆 baseline 已完成。

M-R1 已按 `docs/M-R1-memory-prompt-tool-semantics-plan.md` 完成 prompt v3、Core unchanged 语义、evaluator 1.2.0 和 targeted runner 实现。其历史 artifact 使用 `ops-tools-v2` 身份；Batch D 后 `ops-tools-v2` 仅可读取历史 artifact，新的 Harness run 统一使用 `ops-tools-v3`。R1-B positive 原报告已零模型调用离线重判为 6/6；R1-C 已修复 A01/I04 精确等价候选和 positive 固定来源 SHA 门禁，原始 guardrails 报告零模型调用离线重判为 9/9，preferred behavior 4/4、final-state safety 4/4、safety 4/4、isolation 1/1 均通过且无 violation。M-P0 已完成 canonical exact dedupe、metadata 合并与进程内原子写入。M-P2 已按 `docs/M-P2-postgresql-memory-persistence-plan.md` 完成实现，并以 39/39 通过真实 PostgreSQL gate；2026-07-17 独立验收复跑同样为 39/39，状态为 `complete`。正式长期记忆 Track B 48 x 3 baseline、M-P1、M-P3 及生命周期治理尚未完成。

RAG 路线中 `10F Batch C` 已完成代码实现与主验收：Milvus Lite 真实 dense+sparse/Jieba/upsert 门禁通过；真实 PostgreSQL 并发、复合 tenant FK 和 migration 门禁仍待环境恢复。`Batch D Retrieval 与 queryInternalDocs` 及其 remediation 已通过独立验收：RAG B/C/D 联合专项 198、全量 stub 374、基础 eval 14/14、真实 Milvus Lite、Batch D 范围 Ruff、compileall 与 pip check 均通过。下一阶段不是直接接 rerank，而是 `10F-F0` 先建立 RAG 专项评测与无 rerank hybrid baseline；Batch E rerank 与 Batch F 正式评测仍未完成，10F 不得标记 complete。

长期记忆评测实施顺序：

```text
批次 A-D：已完成并通过主验收
批次 E：第一次 Track B dev 36 x1 和 R1-A 已完成；正式 48 x3 baseline pending
```

完成状态必须区分：

```text
10G.2A framework complete：评测框架和 Track A 完成。
10G.2A complete：在 framework complete 基础上，Track B 48 x 3 真实模型实验也完成并审核。
```

已完成内容：

- Memory Index 到 Memory Metadata 的模型可见语义优化。
- Core Memory block description 精细化。
- memory tools description / system prompt 优化。

未做内容：

- 长期记忆真实模型 baseline（Track B 48 x 3）。
- 后台 LLM extraction。
- memory_candidate。
- Recall Memory / 历史对话召回。
- 遗忘 / archive job。
- Archival Memory update/delete/archive 工具。
