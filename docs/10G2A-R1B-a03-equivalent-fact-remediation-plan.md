# 10G.2A-R1B A03 同义事实误判修复计划

## 1. 阶段定位

本阶段是 M-R1 真实模型 targeted positive 运行后发现的极窄 evaluator 契约修正，不属于 M-R1 生产 prompt/tool 实现。

输入报告：

```text
artifacts/evals/memory/mr1_positive/20260711T144827Z-1.0.0-track_b.json
SHA-256: 6c8bfb8d50b19bb4e8117510f57d42f2f9cfa1de65237818a0a31ddabec1c073
```

运行事实：

```text
planned/executed: 6/6
infrastructure failures: 0
C03: 3/3 正确调用 updateCoreMemory 并写入 user_ops_profile
A03: 3/3 正确调用 saveArchivalMemory 并成功写入 Archival Memory
```

唯一失败是 A03 repetition 2 的状态 Judge：模型写入“单个分区热点”，Dataset 只接受连续子串“单分区热点”。两者表达同一技术事实，工具路由、写入状态和其他必要事实均正确，因此这是 false negative，不是模型行为失败。

## 2. 修正内容

只修改 A03 的状态期望：

```json
"required_facts": ["Kafka", "lag", "10 万", "rebalance"],
"required_fact_any_of": [["单分区热点", "单个分区热点"]]
```

继续要求同一条 Archival Memory 同时包含其余四个原子事实，不降低路由、写入触发、类型或状态要求。

复用 R1-A 已实现的 `required_fact_any_of` schema 和 Judge，不新增模糊匹配、分词、LLM judge 或关键词重写逻辑。

## 3. 不做内容

- 不修改生产 prompt v3 或 memory tool description。
- 不修改 Judge 实现。
- 不修改其他 Dataset case。
- 不接真实 Embedding，不修改检索/去重/持久化。
- 不重跑 C03/A03 真实模型。
- 不打开 Holdout，不运行正式 48 x 3 baseline。

## 4. 可复现续跑

修改 Dataset 后，对原始 positive 报告执行离线重判：

```text
offline model calls = 0
expected blocking = 6/6
source report bytes/hash unchanged
```

为避免重新消耗 6 次正向模型调用，`scripts/run_memory_eval_mr1.py` 增加窄参数：

```text
--positive-report <原始 positive report>
```

该模式：

1. 使用当前 Dataset 对指定原始报告调用现有 `rejudge_memory_eval_report`。
2. 校验 parent/source hash 不变、`offline_rejudge_model_calls=0`、case 计划正好是 C03/A03 各 3 次、6 个结果全部 completed/passed。
3. positive 离线重判通过后，只执行 guardrails 9 次。
4. guardrails 继续要求所有 case completed/passed，并额外检查 preferred behavior、安全与隔离门禁。

不传参数时，脚本仍保持原有“先真实运行 positive 6 次，再运行 guardrails 9 次”的行为。

## 5. 文件边界

允许修改：

```text
evals/datasets/long_term_memory_v1.json          # 仅 A03 一个 expectation
tests/test_memory_eval_dataset.py                # A03 schema/同义组断言
tests/test_memory_eval_runner.py                 # 离线重判与续跑入口测试
scripts/run_memory_eval_mr1.py                   # --positive-report
docs/M-R1-memory-prompt-tool-semantics-plan.md   # 记录本次验收发现
docs/04-python-migration-roadmap.md
docs/10-advanced-capabilities-plan.md
```

禁止修改：

```text
prompts/
src/superbiz_agent/evals/memory_cases.py
src/superbiz_agent/evals/memory_judges.py
src/superbiz_agent/evals/memory_metrics.py
src/superbiz_agent/evals/memory_runner.py
src/superbiz_agent/memory/
src/superbiz_agent/rag/
.env
```

## 6. 验收

1. Dataset 仍为 48 个 case、dev/holdout 仍为 36/12，只允许 A03 expectation 发生预期差异。
2. A03 `required_fact_any_of` 拒绝空组、空白候选和重复候选的既有 schema 契约不回退。
3. 原始 positive 报告 SHA-256 仍为 `6c8bfb8d50b19bb4e8117510f57d42f2f9cfa1de65237818a0a31ddabec1c073`。
4. 离线重判不调用模型，positive 结果从 5/6 修正为 6/6。
5. `--positive-report` 只有在指定报告严格属于 C03/A03 各 3 次且全部通过时才继续 guardrails。
6. Dataset 新 hash 被记录到派生报告/checkpoint，旧 checkpoint 不得跨 hash 恢复。
7. 专项、全量 stub、基础 eval、Ruff、compileall 全部通过。

R1-B 完成只修复已证实的 A03 false negative，不代表 M-R1 guardrails、正式长期记忆 baseline 或生产检索已经完成。
