# 基础 Eval Runner 实施计划

## 1. 阶段定位

本文是 Python 迁移路线第 6 步 `基础 Eval Runner` 的实施计划。

本阶段目标是在已完成的 Skeleton P0 之上，建立一个本地 deterministic eval gate，用少量 smoke / capability case 验证最小 Agent Harness 闭环是否稳定：

```text
eval case -> run HarnessService -> collect trace -> RuleJudge -> report
```

本阶段不是完整评测平台，也不是 LangSmith 接入。它只解决一个问题：后续每次改 prompt、model gateway、tool gateway、context assembler 或 runtime 时，都能用固定样本和 trace 规则快速判断最小能力有没有被破坏。

## 2. 依据

本计划依据：

- `docs/02-python-migration-spec.md` 第 17 节评测契约。
- `docs/03-python-architecture-design.md` 第 12 节评测设计。
- `docs/04-python-migration-roadmap.md` 第 6 步阶段范围。
- `docs/05-skeleton-p0-implementation-plan.md` 的 Skeleton P0 验收要求。
- 当前 Skeleton P0 实现：
  - `AgentHarnessService`
  - `InMemoryRolloutEventStore`
  - `RolloutEvent`
  - `ToolGateway`
  - `StubModelGateway`

## 3. 本阶段做什么

### 3.1 Eval Case Schema

新增 `EvalCase`，用于描述一个确定性评测样本。

最小字段：

```text
case_id
name
description
session_id
tenant_id
user_id
agent_id
user_input
expected_answer_contains
required_tools
forbidden_tools
required_events
forbidden_events
expected_success
scoring_rules
initial_context
available_tools
mock_tool_returns
required_tool_argument_keys
```

说明：

- `initial_context/session state` 本阶段先保留字段，但默认为空，不做 replay。
- `available_tools` 本阶段默认来自 `AgentHarnessService.build_default`，不额外动态裁剪。
- `mock_tool_returns` 本阶段保留字段但不启用，因为 Skeleton P0 已有 deterministic `getCurrentDateTime` 工具。
- `required_tool_argument_keys` 用于检查指定工具调用参数里必须出现哪些字段，例如 `getCurrentDateTime.timezone`。
- `scoring_rules` 是 case 内的规则开关和描述，不引入复杂 DSL。

### 3.2 Eval Suite

新增 `EvalSuite`，包含多个 `EvalCase`。

本阶段内置一个最小 suite：

```text
skeleton_p0_smoke
```

建议包含至少 3 个 case：

1. 时间类问题必须调用 `getCurrentDateTime`。
2. 普通问候不应调用任何工具。
3. 空问题应返回 Java 兼容错误，不应调用模型和工具。

空问题可以通过 API handler 测，也可以通过 runner 的 input validation path 测；如果直接跑 `HarnessService.chat`，要确保仍能得到 `success=false`。

### 3.3 Eval Runner

新增本地 runner：

```text
EvalRunner.run_suite(suite) -> EvalReport
```

执行路径：

```text
for case in suite:
  1. 创建新的 AgentHarnessService.placeholder()
  2. 构造 AgentRequestContext
  3. 调用 service.chat(context, case.user_input)
  4. 从 service.trace_store.list_by_session(context) 收集 events
  5. 生成 TraceArtifact
  6. 调用 RuleJudge
  7. 汇总 case result
```

每个 case 使用独立 service，避免内存 trace 相互污染。

### 3.4 TraceArtifact

新增 `TraceArtifact`，从 `RolloutEvent` 中提取评测所需摘要。

至少包含：

```text
run_id
session_id
prompt_version
model_provider
tool_schema_version
event_types
tool_calls
tool_arguments
tool_results
final_answer
success
error_message
latency_ms
raw_event_count
```

说明：

- `token usage` 本阶段可保留字段为 `None`，因为 StubModelGateway 没有 usage。
- `model input/output summary` 本阶段只记录模型事件数量和最终答案摘要，不保存完整 prompt。
- trace artifacts 不能泄露密钥；本阶段工具输出不包含敏感信息。

### 3.5 RuleJudge

新增确定性规则评估器 `RuleJudge`。

本阶段至少支持：

- `expected_success` 是否一致。
- final answer 是否包含 `expected_answer_contains` 中的所有片段。
- required tools 是否都调用过。
- forbidden tools 是否未调用。
- required events 是否都出现。
- forbidden events 是否未出现。
- tool arguments 是否包含 `required_tool_argument_keys` 指定的必要字段。
- 对于预期成功且产生 trace 的 case，trace 中是否有 `run_id`。
- 对于输入校验失败且不会进入 runtime 的 case，允许没有 `run_id` 和 rollout events，但必须满足 forbidden events 规则。
- trace event sequence 是否单调递增。

输出：

```text
RuleJudgeResult(
  passed: bool,
  score: float,
  failures: list[str],
  details: dict
)
```

### 3.6 Eval Report

新增 `EvalReport`，汇总 suite 结果：

```text
suite_id
case_count
passed_count
failed_count
pass_rate
case_results
started_at
finished_at
duration_ms
```

报告输出支持：

- Python 对象返回，供测试使用。
- `model_dump()` / JSON 序列化。
- 可选 CLI 打印简短 summary。

### 3.7 可选 CLI

可以提供一个轻量 CLI：

```bash
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
```

CLI 默认运行内置 `skeleton_p0_smoke` suite，并打印：

```text
suite=skeleton_p0_smoke passed=3 failed=0 pass_rate=1.00
```

CLI 是有益但不是核心；核心验收以 pytest 为准。

## 4. 本阶段不做什么

基础 Eval Runner 明确不做：

- 不接 LangSmith。
- 不做 LLM-as-judge。
- 不做复杂 EvidenceJudge。
- 不做 RAG 检索层指标，例如 hit rate、MRR、NDCG。
- 不做 RAG 生成层忠实度评测。
- 不做长期记忆专项评测。
- 不做真实模型质量评测。
- 不做线上实验管理。
- 不做 UI。
- 不做数据库持久化 eval report。
- 不读取或写入 PostgreSQL。
- 不引入外部 SaaS 依赖。

## 5. 文件级实施范围

### 5.1 可能新增文件

```text
src/superbiz_agent/evals/cases.py
src/superbiz_agent/evals/traces.py
src/superbiz_agent/evals/judges.py
src/superbiz_agent/evals/runner.py
```

### 5.2 需要修改文件

```text
src/superbiz_agent/evals/__init__.py
tests/test_eval_runner.py
```

如果为了复用 trace 字段需要小幅调整 Skeleton P0 代码，可以修改：

```text
src/superbiz_agent/harness/trace_store.py
src/superbiz_agent/harness/events.py
```

但不应改变现有 Skeleton P0 行为。

### 5.3 不应修改文件

```text
prompts/
src/superbiz_agent/rag/
src/superbiz_agent/memory/
src/superbiz_agent/persistence/
docs/01-java-capability-inventory.md
docs/02-python-migration-spec.md
docs/03-python-architecture-design.md
docs/04-python-migration-roadmap.md
docs/05-skeleton-p0-implementation-plan.md
```

## 6. 目标 Case 设计

### 6.1 datetime_tool_required

输入：

```text
现在几点？
```

期望：

- `success=true`
- answer 包含 `Asia/Shanghai`
- 必须调用 `getCurrentDateTime`
- `getCurrentDateTime` 参数必须包含 `timezone`
- 必须出现：
  - `RUN_STARTED`
  - `CONTEXT_ASSEMBLED`
  - `MODEL_CALL_STARTED`
  - `MODEL_CALL_COMPLETED`
  - `TOOL_CALL_STARTED`
  - `TOOL_CALL_COMPLETED`
  - `ASSISTANT_MESSAGE_APPENDED`
  - `RUN_COMPLETED`

### 6.2 no_tool_for_plain_chat

输入：

```text
hello
```

期望：

- `success=true`
- answer 包含 `[stub] hello`
- 禁止调用 `getCurrentDateTime`
- 禁止出现 `TOOL_CALL_STARTED`
- 必须出现 `MODEL_CALL_STARTED` 和 `MODEL_CALL_COMPLETED`

### 6.3 empty_question_error

输入：

```text
   
```

期望：

- `success=false`
- error message 包含 `问题内容不能为空`
- 禁止出现 `MODEL_CALL_STARTED`
- 禁止出现 `TOOL_CALL_STARTED`

## 7. 测试计划

新增 `tests/test_eval_runner.py`，至少覆盖：

- 内置 suite 能加载。
- EvalRunner 跑完 `skeleton_p0_smoke`。
- 3 个 case 全部通过。
- datetime case 的 trace artifact 包含 `getCurrentDateTime`。
- plain chat case 没有工具调用。
- empty question case 不触发模型调用和工具调用。
- 如果人为构造一个缺少 required event 的 artifact，RuleJudge 会失败并给出 failure reason。
- `EvalReport.pass_rate == 1.0`。

保留并继续通过已有 Skeleton P0 测试。

## 8. 验收命令

subAgent 实施完成后必须先运行：

```bash
python3 -m pytest
```

如果本地安装了 ruff，也运行：

```bash
python3 -m ruff check src tests
```

当前环境如果没有 ruff，可以说明 `No module named ruff`，不作为失败。

## 9. subAgent 实施任务单

交给 subAgent 的任务应限制为：

```text
只实现 docs/06-basic-eval-runner-plan.md 定义的基础 Eval Runner。
不得接 LangSmith、LLM-as-judge、RAG 评测、长期记忆评测、真实模型评测、PostgreSQL。
实现完成后必须运行 python3 -m pytest。
提交结果时说明：
1. 修改/新增了哪些文件。
2. 内置 suite 包含哪些 case。
3. RuleJudge 支持哪些规则。
4. EvalReport 输出什么。
5. 测试命令和结果。
6. 是否有偏离计划的地方。
```

如果实现过程中发现计划和当前代码冲突，subAgent 必须停止并报告冲突，不得扩大范围自行重设计。

## 10. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否符合第 6 步基础 Eval Runner | 是 |
| 是否提前做 Migration P0 | 否 |
| 是否接入 LangSmith | 否 |
| 是否做 LLM-as-judge | 否 |
| 是否做 RAG 专项评测 | 否 |
| 是否做长期记忆专项评测 | 否 |
| 是否依赖真实模型 | 否 |
| 是否依赖 PostgreSQL | 否 |
| 是否复用 Skeleton P0 trace | 是 |
| 是否能作为本地 deterministic gate | 是 |
| 是否有明确 case schema 和 judge | 是 |

## 11. 阶段完成标准

基础 Eval Runner 可以验收通过的条件：

1. 存在可导入的 `EvalCase`、`EvalSuite`、`EvalRunner`、`RuleJudge`、`TraceArtifact`、`EvalReport`。
2. 内置 `skeleton_p0_smoke` suite 至少包含 3 个 case。
3. EvalRunner 能运行 HarnessService 并收集 trace。
4. RuleJudge 能基于 trace 和 final answer 做确定性判断。
5. `python3 -m pytest` 全部通过。
6. 不依赖外部 SaaS、真实模型、PostgreSQL、RAG 或长期记忆。
