# 业务工具迁移实施计划

## 1. 阶段定位

本文是 Python 迁移路线第 8 步 `业务工具迁移` 的实施计划。

本阶段目标是迁移 Java 单 Agent Harness 中日志、告警、内部文档检索相关工具的契约，使 Python 版本具备可评测、可 trace、可治理的业务工具能力：

```text
model tool call
  -> ToolGateway
  -> Pydantic args validation
  -> business fixture/mock tool
  -> normalized JSON result
  -> TOOL_CALL_* trace
  -> eval gate
```

本阶段迁移的是工具契约和 deterministic fixture/mock 行为，不接真实企业日志系统、真实 Prometheus、真实 LlamaIndex/Milvus RAG。

## 2. 依据

本计划依据：

- `docs/02-python-migration-spec.md`
  - 必需工具契约。
  - RAG 迁移契约。
  - 工具网关契约。
- `docs/04-python-migration-roadmap.md`
  - 第 8 步业务工具迁移范围。
- Java 参考：
  - `InternalDocsTools.queryInternalDocs`
  - `QueryMetricsTools.queryPrometheusAlerts`
  - `QueryLogsTools.getAvailableLogTopics`
  - `QueryLogsTools.queryLogs`
  - `application.yml` 中工具 policy。
- 当前 Python：
  - Migration P0 ToolGateway。
  - `ToolRegistry` / `ToolDefinition`。
  - deterministic eval runner。

## 3. 本阶段做什么

### 3.1 迁移工具列表

本阶段新增并注册以下工具：

```text
getAvailableLogTopics
queryLogs
queryPrometheusAlerts
queryInternalDocs
```

继续保留：

```text
getCurrentDateTime
```

本阶段不迁移 memory 工具：

```text
listMemoryTopics
searchMemory
updateCoreMemory
saveArchivalMemory
writeMemory
```

这些属于第 9 步长期记忆迁移。

### 3.2 Tool Policy

工具 policy 对齐 Java / 迁移规格：

| Tool | Danger | Timeout | Retries | Idempotent |
|---|---|---:|---:|---|
| `getAvailableLogTopics` | low | 2s | 0 | true |
| `queryPrometheusAlerts` | medium | 5s | 1 | true |
| `queryInternalDocs` | medium | 8s | 1 | true |
| `queryLogs` | medium | 8s | 1 | true |

说明：

- 当前 ToolGateway 仍不实现完整 retry/backoff；但 policy metadata 必须准确进入 `TOOL_CALL_STARTED` trace。
- 完整 timeout/retry 执行治理属于后续增强，不在本阶段扩大。

### 3.3 `getAvailableLogTopics`

输入：

```json
{}
```

输出：

```json
{
  "success": true,
  "topics": [
    {
      "topicName": "system-metrics",
      "description": "...",
      "exampleQueries": ["cpu_usage:>80"],
      "relatedAlerts": ["HighCPUUsage"]
    }
  ],
  "availableRegions": ["ap-guangzhou", "ap-shanghai", "ap-beijing", "ap-chengdu"],
  "defaultRegion": "ap-guangzhou",
  "message": "共有 4 个可用的日志主题。建议使用默认地域 'ap-guangzhou' 或省略 region 参数"
}
```

必须包含内置主题：

- `system-metrics`
- `application-logs`
- `database-slow-query`
- `system-events`

### 3.4 `queryLogs`

输入：

```json
{
  "region": "ap-guangzhou",
  "logTopic": "application-logs",
  "query": "level:ERROR",
  "limit": 20
}
```

参数规则：

- `region` 必须是：
  - `ap-guangzhou`
  - `ap-shanghai`
  - `ap-beijing`
  - `ap-chengdu`
- `logTopic` 必填。
- `query` 可选，空字符串表示默认查询。
- `limit` 可选，范围 1-100，默认 20。

输出：

```json
{
  "success": true,
  "region": "ap-guangzhou",
  "logTopic": "application-logs",
  "query": "level:ERROR",
  "logs": [
    {
      "timestamp": "...",
      "level": "ERROR",
      "service": "order-service",
      "instance": "...",
      "message": "...",
      "fields": {}
    }
  ],
  "total": 1,
  "message": "成功查询到 1 条日志"
}
```

本阶段实现方式：

- deterministic fixture/mock。
- 不调用 CLS / ES / Loki / SLS / CloudWatch。
- 按 `logTopic + query + limit` 返回可预测结果。
- 对超出范围的参数交给 Pydantic 校验，返回 ToolGateway 结构化错误。

### 3.5 `queryPrometheusAlerts`

输入：

```json
{}
```

输出：

```json
{
  "success": true,
  "alerts": [
    {
      "alertName": "HighCPUUsage",
      "description": "...",
      "state": "firing",
      "activeAt": "...",
      "duration": "25m0s"
    }
  ],
  "message": "成功检索到 3 个活动告警"
}
```

本阶段实现方式：

- deterministic fixture/mock。
- 不请求真实 Prometheus。
- 默认返回 3 个固定告警：
  - `HighCPUUsage`
  - `HighMemoryUsage`
  - `SlowResponse`

### 3.6 `queryInternalDocs`

输入：

```json
{
  "query": "Pod 重启排查流程"
}
```

输出：

```json
{
  "status": "ok",
  "count": 3,
  "chunks": [
    {
      "id": "doc-pod-restart-1",
      "ref": 1,
      "source": "runbooks/pod-restart.md",
      "content": "...",
      "score": 0.86
    }
  ]
}
```

无结果：

```json
{
  "status": "no_results",
  "message": "No relevant documents found in the knowledge base."
}
```

本阶段实现方式：

- deterministic fixture/mock。
- 不接 LlamaIndex。
- 不接 Milvus。
- 不做 ingestion。
- 不做 hybrid search / rerank。
- 只保留 tool output JSON 契约、citation-friendly chunk 字段和 trace。

### 3.7 StubModelGateway 工具选择增强

当前 `StubModelGateway` 只会触发 `getCurrentDateTime`。

本阶段需要增加 deterministic 规则，让 eval 能覆盖新工具：

- 用户问当前活跃告警/告警状态 -> `queryPrometheusAlerts`
- 用户问有哪些日志主题/日志类型 -> `getAvailableLogTopics`
- 用户明确查日志，并提供或可推断 topic -> `queryLogs`
- 用户问内部流程/最佳实践/操作步骤/排查指南 -> `queryInternalDocs`

限制：

- 仍然最多一次工具调用 + final answer。
- 不做真实模型意图识别。
- 不实现多工具链式排障。

### 3.8 工具输出进入最终答案

`StubModelGateway` 在收到对应 tool result 后，生成 deterministic final answer：

- alert 工具：总结告警数量和主要告警名。
- logs 工具：总结日志条数、topic、关键日志 level/service。
- docs 工具：总结文档 chunk 数和 ref/source。
- log topics 工具：列出可用 topic 数量和默认 region。

目标不是生成高质量自然语言，而是让 eval 能判断：

- 工具被正确调用。
- 工具参数正确。
- 工具结果被 final answer 使用。

### 3.9 Eval Runner 更新

新增业务工具 capability cases：

1. `alerts_tool_required`
   - 输入：`现在有哪些活跃告警？`
   - 必须调用 `queryPrometheusAlerts`
   - answer 包含 `HighCPUUsage`

2. `log_topics_required`
   - 输入：`有哪些日志主题可以查？`
   - 必须调用 `getAvailableLogTopics`
   - answer 包含 `application-logs`

3. `query_logs_required`
   - 输入：`查一下 application-logs 里的 ERROR 日志`
   - 必须调用 `queryLogs`
   - 参数包含 `region`、`logTopic`、`query`
   - answer 包含 `application-logs`

4. `internal_docs_required`
   - 输入：`Pod 重启排查流程是什么？`
   - 必须调用 `queryInternalDocs`
   - 参数包含 `query`
   - answer 包含 `runbooks`

保留已有：

- datetime case
- plain chat case
- empty question case
- session replay case

### 3.10 Trace 验收

每个业务工具 case 必须产生：

```text
TOOL_CALL_STARTED
TOOL_CALL_COMPLETED
```

并且 `TOOL_CALL_STARTED.payload` 包含：

```text
toolName
arguments
timeoutSeconds
maxRetries
```

`TOOL_CALL_COMPLETED.payload` 包含：

```text
toolName
result
status="success"
```

## 4. 本阶段不做什么

业务工具迁移明确不做：

- 不接真实 Prometheus。
- 不接真实 CLS / ES / Loki / SLS / CloudWatch。
- 不接 LlamaIndex。
- 不接 Milvus。
- 不做真实 RAG ingestion。
- 不做 hybrid search / BM25 / rerank。
- 不做长期记忆工具。
- 不做 Core Memory / Archival Memory。
- 不做多工具链式 Agent 排障。
- 不做 streaming。
- 不接真实 Qwen。
- 不做 LangSmith。
- 不做生产级租户资源绑定。

## 5. 文件级实施范围

### 5.1 可能新增文件

```text
src/superbiz_agent/tools/builtin/log_topics_tool.py
src/superbiz_agent/tools/builtin/log_query_tool.py
src/superbiz_agent/tools/builtin/alerts_tool.py
src/superbiz_agent/tools/builtin/internal_docs_tool.py
src/superbiz_agent/tools/fixtures.py
tests/test_business_tools.py
```

### 5.2 需要修改文件

```text
src/superbiz_agent/tools/builtin/__init__.py
src/superbiz_agent/harness/service.py
src/superbiz_agent/model_gateway/stub.py
src/superbiz_agent/evals/cases.py
tests/test_eval_runner.py
```

### 5.3 不应修改文件

```text
prompts/
src/superbiz_agent/rag/
src/superbiz_agent/memory/
src/superbiz_agent/persistence/
alembic/
docs/01-java-capability-inventory.md
docs/02-python-migration-spec.md
docs/03-python-architecture-design.md
docs/04-python-migration-roadmap.md
docs/05-skeleton-p0-implementation-plan.md
docs/06-basic-eval-runner-plan.md
docs/07-migration-p0-implementation-plan.md
```

除非发现明确冲突，本阶段不修改前置设计文档。

## 6. 实施批次

### 批次 A：工具 schema 和 fixture handlers

目标：

- 定义 4 个工具的 Pydantic args model。
- 定义 deterministic fixture 输出。
- 注册工具 policy。

验收：

- 直接调用 ToolGateway 能执行 4 个工具。
- 参数错误返回结构化 `PARAM_VALIDATION_FAILED`。

### 批次 B：StubModelGateway 工具选择和 final answer

目标：

- 增加 deterministic tool selection。
- 增加 tool result -> final answer 格式化。

验收：

- 每类业务问题触发对应工具。
- plain chat 仍不调用工具。
- session replay 不被历史 tool result 干扰当前问题。

### 批次 C：Eval cases 和 trace 验收

目标：

- 增加 4 个业务工具 eval case。
- 验证 required tools、required arguments、answer contains、trace event。

验收：

- eval CLI 输出全部通过。
- `python3 -m pytest` 全部通过。

## 7. 测试计划

新增/更新测试：

- `getAvailableLogTopics`：
  - 返回 4 个主题。
  - policy timeout=2, retries=0。
- `queryLogs`：
  - 合法参数返回 logs。
  - invalid region 触发参数校验错误。
  - limit > 100 触发参数校验错误。
  - policy timeout=8, retries=1。
- `queryPrometheusAlerts`：
  - 返回固定告警。
  - policy timeout=5, retries=1。
- `queryInternalDocs`：
  - runbook 查询返回 chunks。
  - 无结果返回 `status=no_results`。
  - policy timeout=8, retries=1。
- StubModelGateway：
  - 业务问题触发正确 tool call。
  - 收到 tool result 后 final answer 使用结果。
- EvalRunner：
  - 新增 4 个 case 通过。
  - 总 suite 继续通过。

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
只实现 docs/08-business-tools-migration-plan.md 定义的业务工具迁移。
不得接真实 Prometheus、真实日志系统、LlamaIndex、Milvus、真实 RAG、长期记忆、真实模型、streaming、LangSmith。
不得修改 prompts、rag、memory、persistence、alembic。
实现完成后必须运行 python3 -m pytest。
提交结果时说明：
1. 修改/新增了哪些文件。
2. 4 个业务工具的 schema、policy、输出结构。
3. StubModelGateway 增加了哪些 deterministic 工具选择规则。
4. eval 新增了哪些 case。
5. 测试命令和结果。
6. 是否有偏离计划的地方。
```

如果实现过程中发现计划和当前代码冲突，subAgent 必须停止并报告冲突，不得扩大范围自行重设计。

## 10. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否符合第 8 步业务工具迁移 | 是 |
| 是否提前做完整 RAG | 否 |
| 是否提前接 LlamaIndex/Milvus | 否 |
| 是否接真实日志/告警系统 | 否 |
| 是否提前做长期记忆 | 否 |
| 是否保持 ToolGateway 治理路径 | 是 |
| 是否保留工具 policy 和 trace | 是 |
| 是否默认测试不依赖外部系统 | 是 |
| 是否增加 eval 覆盖 | 是 |

## 11. 阶段完成标准

业务工具迁移可以验收通过的条件：

1. `getAvailableLogTopics`、`queryLogs`、`queryPrometheusAlerts`、`queryInternalDocs` 已注册到 ToolRegistry。
2. 四个工具都有 Pydantic 参数 schema。
3. 四个工具 policy 与迁移规格一致。
4. 四个工具返回 JSON dict，而不是原始字符串。
5. ToolGateway trace 中记录 toolName、arguments、result、status。
6. StubModelGateway 能 deterministically 触发四个工具。
7. eval suite 覆盖四个业务工具。
8. `python3 -m pytest` 全部通过。
9. 未接入本阶段明确排除的真实外部系统或后续能力。

