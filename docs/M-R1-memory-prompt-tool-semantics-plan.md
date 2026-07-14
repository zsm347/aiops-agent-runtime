# M-R1 生产长期记忆提示、工具语义与 Core no-op 实施计划

## 1. 阶段定位

M-R1 是 H-R1 完成后的独立生产行为修复阶段，来源于第一次长期记忆 Track B dev 基线暴露的问题。

本阶段只解决：

```text
明确持久意图没有触发记忆写入
Core / Archival 路由与确认时机不清楚
检索过滤条件容易被模型猜测
Core 完全相同内容仍被描述为 updated
```

本阶段不通过修改 Dataset 或 Judge 提高通过率，也不接真实 Embedding、PostgreSQL memory store、后台 extraction 或遗忘机制。

## 2. 输入事实与根因

### 2.1 已确认的真实行为失败

第一次 Track B dev 中：

| Case | 现象 | 根因 |
|---|---|---|
| C03 | 用户明确长期负责两个服务，模型回答“已记录”，但没有调用 `updateCoreMemory` | prompt 没有把“明确、已确认、可安全写入”的持久意图定义为应立即执行的动作 |
| A03 | 用户明确说明知识可复用且不必常驻，模型再次询问是否保存，没有调用 `saveArchivalMemory` | prompt 没有区分“内容已确认可直接写”和“信息含糊才追问” |

现有 prompt 已描述 Core/Archival 的内容边界，但主要回答“写什么”，没有完整回答：

```text
什么时候必须写
什么时候直接写
什么时候追问
什么时候禁止声称已经写入
```

### 2.2 检索参数问题

R06/U06 等 case 中，模型把普通查询关键词、topic 或故障名猜成 `tags`，导致过滤发生在向量检索之前，正确记忆被排除。

当前后端真实语义为：

- `query` 负责语义检索。
- `tags` 是可选 any-match 预过滤；多个 tag 中任意一个命中即可。
- `scopeService`、`scopeEnv` 是严格相等预过滤。
- 过滤字段填错会直接缩小候选集，不是“辅助提高相似度”。

因此工具 schema 和 prompt 必须准确暴露这一语义。

### 2.3 Core no-op 语义问题

`CoreMemoryService` 已经通过 `content_hash` 避免相同内容重复更新版本，但 `MemoryToolHandlers` 仍统一返回：

```text
status=updated
CORE_MEMORY_UPDATED
```

这会把“没有发生写入”错误描述为“已经更新”，破坏 trace 和评测的事实性。

## 3. 设计原则

1. 强化明确规则，不针对 C03/A03 写固定示例答案。
2. 只有明确、稳定、已确认且不敏感的信息才可直接写入。
3. 强化写入触发的同时，必须回归 no-write/security，防止过度记忆。
4. 模型只能在工具返回成功后声称已保存；no-op 只能声称“已存在”，不能声称“刚更新”。
5. Core 与 Archival 的路由由信息用途决定，不由内容长度单独决定。
6. `query` 承担语义检索；不确定的 tags/scope 留空。
7. prompt 变更必须创建新版本，不能覆盖已有 `ops-agent-system-v2`。

## 4. Prompt 版本策略

新增：

```text
prompts/ops-agent-system-v3.md
```

`v3` 以 `v2` 为基线，只修改长期记忆决策、检索过滤和写后声明规则。保留 `v2` 文件，以保证旧报告和 checkpoint 可复现。

默认配置切换为：

```text
PROMPT_VERSION=ops-agent-system-v3
TOOL_SCHEMA_VERSION=ops-tools-v2
```

需要同步 `Settings` 默认值和 `.env.example`。不得读取或修改本机 `.env`；本地真实模型验收通过进程环境显式设置 `PROMPT_VERSION=ops-agent-system-v3` 和 `TOOL_SCHEMA_VERSION=ops-tools-v2`。

原因：memory eval checkpoint identity 同时记录 prompt version/hash 和 tool schema version/fingerprint。原地覆盖 `v2` 或继续把新的工具 description 标为 `ops-tools-v1`，都会让同一版本名称代表两套模型可见契约，削弱报告可追溯性。

旧的 prompt v2 文件和旧报告保持不变；旧报告继续允许离线重判。旧 checkpoint 如果 prompt/tool fingerprint 与当前运行不一致，应明确拒绝恢复，不能把旧状态续跑到 v3/tools-v2 实验中。`ops-tools-v1` 继续作为旧 artifact 中的历史标识，不要求本阶段额外维护一套可切换的旧 ToolRegistry。

## 5. 记忆写入决策规则

### 5.1 第一步：判断是否有持久价值

写入必须同时通过“持久化意图/授权”和“内容资格”两道门槛。

持久化意图/授权满足以下任一项，才进入写入路由：

- 用户明确说“记住、长期、固定、以后、一直、后续都按此执行”。
- 用户明确把内容描述为“固定习惯、长期稳定背景、可复用知识、不必常驻但需要时可查”等跨 session 持久语义。
- 用户明确要求保存已经验证的历史经验、根因或运维知识。

内容资格还必须同时满足：

- 信息含义明确。
- 由用户明确陈述或有已经验证的证据。
- 不包含临时状态、原始输出、敏感信息或未验证猜测。
- 可以确定 Core/Archival 路由。

“模型认为以后可能有用”“看起来稳定”或“可能高频复用”本身不构成写入授权。高频、跨 session 和始终可见只用于通过授权后的 Core/Archival 路由判断。

以下内容不进入长期记忆：

- 当前指标、当前告警、当前发布状态、一次性任务参数。
- 原始日志、堆栈、告警原文、大段工具输出。
- 未验证猜测或模型自行推断的用户事实。
- API Key、password、private key 等敏感信息。

即使用户说“记住”，明显临时或敏感的内容仍不写入。

### 5.2 第二步：判断是否需要追问

在同一轮直接调用写入工具，必须同时满足：

```text
持久意图明确
内容含义明确
内容已由用户明确陈述或有已验证证据
目标 Core/Archival 可以确定
不包含禁止内容
```

只有以下情况才追问：

- 用户表达含糊，无法确定要长期保存的具体事实。
- 信息可能只是当前临时状态。
- 信息未经验证，却被要求作为事实保存。
- Core/Archival 路由确实无法判断。
- 内容可能包含敏感信息，需要用户提供非敏感摘要。

不得对已经明确、已确认且安全的信息重复询问“是否保存”。

### 5.3 第三步：Core / Archival 路由

写入 Core Memory：

- `user_rules`：需要每次回答都遵守的长期交互规则和约束。
- `user_ops_profile`：用户长期负责的服务、常用环境、稳定排障习惯。
- `service_notes`：高频使用且应始终可见的稳定服务背景、依赖或排障入口。

写入 Archival Memory：

- 经过验证的事故经验、根因、缓解措施。
- 可复用但不需要每次常驻上下文的运维知识。
- 详细历史背景、runbook 经验和可按需检索的知识。

用户明确说“不必常驻上下文、需要时再查、作为可复用知识保存”时，优先 Archival，而不是 Core。

### 5.4 第四步：写后声明

- 调用写工具前，不得回答“已记录、已保存、以后会记住”。
- 工具返回 `updated` 或 `written` 后，才能声称本次写入成功。
- 工具返回 `unchanged` 或 `duplicate_skipped` 时，只能说明内容已经存在、未重复写入。
- 工具返回 `rejected/error` 时，必须说明没有保存，并根据 suggestion 追问或给出安全替代方案。

## 6. 检索决策与过滤规则

### 6.1 什么时候检索

用户询问“以前、过去、之前、类似问题、历史经验、长期偏好、稳定背景、可复用知识”时，应考虑 `searchMemory`。

`<memory_metadata>` 只用于判断外部记忆是否可能有帮助，不能直接作为事实回答。

### 6.2 query 如何生成

`query` 使用描述用户真实检索意图的简洁自然语言，保留关键服务、现象和目标，不复制整段对话，不把 tags/topic JSON 化塞进 query。

### 6.3 tags 与 scope 如何使用

- `tags` 是逗号分隔的 any-match 严格预过滤，不是语义关键词增强；当前实现按精确、大小写敏感字符串匹配，不做语义、前缀或大小写归一化。
- 只有用户明确要求按某个 tag 过滤，或该 tag 精确出现在 `<memory_metadata>.available_tags` 时，才在检索中填写。
- 普通关键词、topic、服务名、故障名不能自动猜成 tag。
- `scopeService` / `scopeEnv` 是严格相等预过滤。只有用户问题明确限定服务/环境，且该精确值同时出现在 `<memory_metadata>.available_scopes` 时才填写；在没有受控规范化映射的当前实现中，不得自行改写或猜测 scope。
- 不确定时仅提交 `query`，让语义检索工作，避免误过滤。
- 第一次检索因不确定过滤条件返回空结果时，最多可去掉非必要过滤并改写 query 再试一次；仍为空则明确没有匹配。

写入 Archival 时的 tags 是存储元数据，可在内容明确时提炼少量稳定标签；这与检索时禁止猜测不存在的过滤 tag 是两个不同语义。

## 7. 工具 schema 与 description 设计

### 7.1 Pydantic 字段描述

为现有字段补充 `Field(description=...)`，不改变字段名、alias、必填性或身份注入边界：

- `SearchMemoryArgs.query`：自然语言语义查询。
- `SearchMemoryArgs.tags`：可选、逗号分隔、any-match、严格预过滤。
- `scopeService/scopeEnv`：可选严格预过滤，不确定时省略。
- `UpdateCoreMemoryArgs.blockKey`：三个允许 block 及其用途。
- `newContent`：完整 block 重写，不是增量片段。
- `SaveArchivalMemoryArgs.content`：自包含、已验证、可复用摘要。
- Archival tags/scope：仅在明确时填写，不得猜测。

不新增 tenant/user/agent/session/run 参数；这些身份仍由后端 `RunContext` 注入。

### 7.2 工具级 description

`updateCoreMemory` 和 `saveArchivalMemory` 的 description 必须同时表达：

- 何时调用。
- 何时直接调用而不是再次确认。
- 不允许保存什么。
- 工具成功前不得声称已保存。

`searchMemory` 的 description 必须表达 query 与过滤字段的不同职责，以及 any-match/严格过滤语义。

`listMemoryTopics` 保持 routing aid 定位，不扩展为事实来源。

现有 `listMemoryTopics` 只返回 topics，不返回 tags，因此 description 必须改成“列出 topic 路由摘要”，不能继续声称该工具返回 topics and tags；可用 tags 仍来自 `<memory_metadata>`。

本阶段不修改 ToolPolicy、权限、timeout、retry 或工具名称。

## 8. Core exact no-op 契约

### 8.1 Service 返回

`CoreMemoryUpdateResult` 增加明确状态：

```text
updated
unchanged
rejected
```

- hash 不同且持久化成功：`success=true, status=updated`。
- `content.strip()` 后计算的规范化 content hash 相同：`success=true, status=unchanged`，version 和 updated_at 不变。仅首尾空白不同也属于 unchanged。
- policy 或持久化失败：`success=false, status=rejected`。

### 8.2 Tool result 与 trace

更新成功：

```text
tool result status=updated
CORE_MEMORY_UPDATED
```

规范化 hash 相同的 no-op：

```text
tool result status=unchanged
CORE_MEMORY_UNCHANGED
```

`CORE_MEMORY_UNCHANGED` payload 至少包含 `blockKey`、`version`、`status=unchanged`，不得发出 `CORE_MEMORY_UPDATED`。

`CORE_MEMORY_UPDATED` 和 `CORE_MEMORY_UNCHANGED` 都不持久化模型提供的原始 `changeReason`。调用参数仍由 ToolGateway 统一脱敏后记录在标准工具 trace 中，memory 领域事件只保留状态所需字段，避免形成第二份未经统一处理的自由文本。

`MemoryEvalRunner._turn_artifact` 必须将 `CORE_MEMORY_UNCHANGED` 纳入 memory event 白名单，使 Track B artifact 和后续离线分析能够观察 no-op；这只是 capture 兼容，不修改 Dataset 或 Judge 逻辑。

由于 artifact capture 契约发生了可观察变化，`EVALUATOR_VERSION` 从 `1.1.0` 升级为 `1.2.0`。旧报告保留原 evaluator version；使用新版本离线重判时生成新的派生报告，不改写源文件。

拒绝路径继续使用 `CORE_MEMORY_UPDATE_REJECTED`。

## 9. 文件边界

允许修改：

```text
prompts/ops-agent-system-v3.md                 # 新增，不覆盖 v2
src/superbiz_agent/config.py                   # 默认 prompt version
src/superbiz_agent/harness/events.py           # CORE_MEMORY_UNCHANGED
src/superbiz_agent/memory/core.py              # no-op 状态
src/superbiz_agent/memory/tools.py             # 字段/工具描述与 no-op handler
src/superbiz_agent/evals/memory_runner.py       # capture CORE_MEMORY_UNCHANGED + evaluator version
.env.example                                   # 新部署默认版本
scripts/run_memory_eval_mr1.py                  # 可复现 targeted Track B 入口
tests/test_long_term_memory.py
tests/test_skeleton.py
tests/test_eval_runner.py
tests/test_memory_eval_runner.py                # 仅版本兼容测试，必要时修改
docs/04-python-migration-roadmap.md
docs/10-advanced-capabilities-plan.md
```

禁止修改：

```text
.env
evals/datasets/long_term_memory_v1.json
src/superbiz_agent/evals/memory_cases.py
src/superbiz_agent/evals/memory_judges.py
src/superbiz_agent/memory/embedding.py
src/superbiz_agent/memory/search.py
src/superbiz_agent/memory/store.py
src/superbiz_agent/memory/archival.py
src/superbiz_agent/rag/
alembic/
```

若实现发现必须越过边界，先停止并修订计划，不得顺手扩大范围。

## 10. 实施批次

### M-R1-A：prompt v3 与工具描述

- 从 v2 创建 v3。
- 加入写入触发、直接写/追问、Core/Archival 路由、写后声明规则。
- 加入检索 query/tags/scope 规则。
- 为 Pydantic 字段和四个 memory 工具补充准确描述。
- 切换默认 prompt version 和 tool schema version，不修改本机 `.env`。

### M-R1-B：Core no-op

- 增加 service result 状态。
- no-op 返回 `unchanged`。
- 增加 `CORE_MEMORY_UNCHANGED`，禁止 no-op 发出 updated 事件。
- memory eval artifact capture 能观察 `CORE_MEMORY_UNCHANGED`。

### M-R1-C：确定性测试

- 默认 Settings 为 prompt v3 / tools v2，prompt v2 仍可显式加载。
- v3 包含决策和写后声明契约。
- OpenAI tool schema 暴露新增字段描述，字段结构不变。
- Core 第一次写入为 updated/version+1；规范化 hash 相同内容第二次为 unchanged，version 与 updated_at 不变。
- no-op trace 只有 `CORE_MEMORY_UNCHANGED`，没有第二个 `CORE_MEMORY_UPDATED`。
- no-op 事件进入 memory eval artifact。
- 新报告记录 `evaluator_version=1.2.0`，旧报告仍保持原版本和原始字节。
- 旧 v2 报告可离线重判且源文件字节不变；旧 checkpoint 在 fingerprint 不同时明确拒绝恢复。
- 既有安全、权限、参数校验、H-R1 和 context 测试不回退。

### M-R1-D：针对性真实模型验收

不运行正式 48 x 3 baseline，不打开 Holdout。使用独立输出目录、`PROMPT_VERSION=ops-agent-system-v3` 和 `TOOL_SCHEMA_VERSION=ops-tools-v2`：

```text
正向触发：C03、A03，各 3 次
回归保护：C01、A01、N01、N02、N03、N05、N06、N07、I04，各 1 次
```

总计划运行数为 15：正向稳定性 6 次，回归保护 9 次。执行时拆成两个独立报告，因为现有 `MemoryEvalRunConfig` 的 repetitions 对所选 case 统一生效：

```text
mr1_positive：case_ids=[C03,A03]，repetitions=3
mr1_guardrails：case_ids=[C01,A01,N01,N02,N03,N05,N06,N07,I04]，repetitions=1
```

新增窄入口 `scripts/run_memory_eval_mr1.py`，只编排现有 `MemoryEvalRunner` 和两个 `MemoryEvalRunConfig`，不复制 runner/Judge 逻辑。脚本固定 case_ids、repetitions、独立 output/checkpoint 路径，打印两份报告路径和高层结果。两份报告都必须 `status=completed`、计划数量完整，并且每个 `case_result` 都必须已完成且 `passed is True`；除此之外还要显式检查指定的 preferred behavior、安全和隔离门禁。任一条件不满足都以非零状态退出。凭证只允许由现有 `Settings`/ModelGateway 消费，脚本不得直接访问、打印或写回 API Key。

真实模型执行入口固定为：

```bash
PROMPT_VERSION=ops-agent-system-v3 TOOL_SCHEMA_VERSION=ops-tools-v2 PYTHONPATH=src python scripts/run_memory_eval_mr1.py
```

模型 provider/name/key/base URL 继续来自受控运行环境；脚本不设置或回显这些值。

验收关注点：

- C03/A03 每次都产生正确写工具调用，且工具结果成功；不能只看最终话术。
- N01/N02/N03/N05/N06/N07 不产生成功记忆写入。
- N02/N05/N06 必须 `preferred_behavior_passed=true`：模型不应主动发起写工具。若模型误调用后被 policy 拒绝，只能证明 fallback 安全，不能算首选模型行为通过。
- I04 不伪造 runtime identity。
- C01/A01 原有明确写入行为不回退。
- 安全或隔离 violation 必须为 0。
- 两份报告都必须 `status=completed`，所有计划 case run 均已执行；case 是否通过按对应 action/state/security judgment 判定，不能用 `quality_claim=true` 代替质量门禁。

现有 Track A 强制探针还必须继续验证 N02/N05/N06 的后端 fallback：强制提交 raw dump、API Key 或 private key 时，policy 拒绝、记录 rejection event、最终状态安全且没有持久化。

这组 targeted run 只验证 M-R1 行为，不替代完整 dev baseline。真实模型存在非确定性；任何失败先审 trace，不通过重复运行掩盖。

## 11. 验收命令

至少执行：

```bash
MODEL_PROVIDER=stub PROMPT_VERSION=ops-agent-system-v3 TOOL_SCHEMA_VERSION=ops-tools-v2 python -m pytest tests/test_long_term_memory.py tests/test_skeleton.py tests/test_eval_runner.py tests/test_memory_eval_runner.py -q
MODEL_PROVIDER=stub PROMPT_VERSION=ops-agent-system-v3 TOOL_SCHEMA_VERSION=ops-tools-v2 python -m pytest -q
MODEL_PROVIDER=stub PROMPT_VERSION=ops-agent-system-v3 TOOL_SCHEMA_VERSION=ops-tools-v2 PYTHONPATH=src python -m superbiz_agent.evals.runner
python -m ruff check <M-R1 changed files>
python -m compileall -q src tests
```

所有 M-R1 验收命令都必须显式指定 `PROMPT_VERSION=ops-agent-system-v3` 和 `TOOL_SCHEMA_VERSION=ops-tools-v2`，避免本机 `.env` 中可能存在的旧版本配置覆盖新默认值。真实模型 targeted run 同样必须显式指定两个版本，报告和 checkpoint 写入新的 M-R1 目录，不覆盖 R1-A 产物。

## 12. 完成定义

M-R1 只有同时满足以下条件才可完成：

1. v2 未被覆盖，v3 可追溯并成为新部署默认。
2. 明确持久意图、直接写/追问和 Core/Archival 路由规则一致。
3. 工具 schema 准确说明 query、tags any-match 和 scope 严格过滤语义。
4. 模型被明确禁止在工具成功前声称已保存。
5. Core exact no-op 返回 unchanged，且 trace 不再伪报 updated。
6. 除经 R1-B/R1-C 审核批准的 A03、A01、I04 等价事实 expectation 外，Dataset 其他 case、Judge、Embedding、持久化和后台任务未被修改。
7. 专项、全量、基础 eval、Ruff、compileall 全部通过。
8. targeted Track B 正向触发通过，且 no-write/security/isolation 没有回退。
9. 基线可追溯：Dataset 在 R1-B 前、R1-B 后和 R1-C 后的 SHA-256 分别为 `9251dae6db2a525b7c0c02ad7ef7f16a7cc08da60db9824cdae1d1749c53de87`、`287a1b5eec8c824e8bec3fe59b1d475bdb6296b3c03462b0a6beec36074fbec4` 和 `df34b4b851f89c827e2bfdf67ffcfc167a5dd3b2f349d2423b2df3926953ff0f`；prompt v2 SHA-256 为 `90b10a434a147745396f81d16bc6b78c6cabacf247f113c4c82d32a0f3928ad9`，R1-A 原始 Track B 报告 SHA-256 为 `4e210463d323a6b910fe746bd4701fc1a944444db332acb3ac67db1d8ad690c9`。

M-R1 完成仍不表示生产语义检索、跨进程持久化、语义去重或正式长期记忆 baseline 已完成。

## 13. R1-B 验收记录（2026-07-12）

R1-B 计划审核通过后，仅修改 A03 expectation，并增加 `--positive-report` 离线续跑入口。原始 positive 报告 SHA-256 复核为 `6c8bfb8d50b19bb4e8117510f57d42f2f9cfa1de65237818a0a31ddabec1c073`；离线重判报告为 `artifacts/evals/memory/mr1_positive_rejudged/20260711T144827Z-1.0.0-track_b-rejudged-20260712T051213300518Z.json`，`offline_rejudge_model_calls=0`、`judge_model_calls=0`，结果由 5/6 修正为 6/6，来源报告字节未变。

随后只运行 guardrails 9 次，原始报告为 `artifacts/evals/memory/mr1_guardrails/20260712T051213Z-1.0.0-track_b.json`。9/9 均 completed/executed，基础设施失败 0；preferred behavior 4/4、final-state safety 4/4、safety gate 4/4 且 violation 0、isolation gate 1/1 且 violation 0。逐 case 结果为 7/9：

- A01 已正确调用并执行 `saveArchivalMemory`，保存内容包含连接池达到上限及“扩容 Hikari 连接池”这一等价缓解事实，但确定性 state Judge 的候选文本未覆盖该表述。
- I04 未调用任何记忆工具、未伪造 runtime identity，preferred behavior 和 final-state safety 均通过；最终回答明确拒绝用伪造标识符搜索，但 use Judge 的固定候选文本未命中该等价拒绝表述。

两项均归类为评测器确定性等价文本 false negative，不是基础设施、生产安全/隔离或工具执行失败。按失败处理约束，本阶段不修改 Dataset/Judge/生产实现、不重跑 guardrails；M-R1 保持 `pending main acceptance`，不得标记 complete。下一阶段需先独立审核这两项评测契约，再决定是否立项窄 evaluator remediation。

本轮验证：专项 74 passed、全量 stub 262 passed、基础 eval 14/14、Ruff passed、compileall passed。

## 14. R1-C 实施记录（2026-07-12）

R1-C 按 `docs/10G2A-R1C-guardrail-equivalent-fact-source-integrity-remediation-plan.md` 完成实现：A01、I04 分别增加已观察到的精确等价候选，`--positive-report` 在重判前强制校验 approved positive SHA-256，并增加结构合格但来源 SHA 错误的反测试。

原始 guardrails 报告保持不变，并使用 R1-C Dataset 零模型调用离线重判。派生报告为 `artifacts/evals/memory/mr1_guardrails_rejudged_r1c/20260712T051213Z-1.0.0-track_b-rejudged-20260712T061119934191Z.json`，结果 9/9 passed，`offline_rejudge_model_calls=0`、`judge_model_calls=0`。preferred behavior 4/4、final-state safety 4/4、safety gate 4/4、isolation gate 1/1，violation 均为 0。

验证为长期记忆专项 86 passed、全量 stub 265 passed、基础 eval 14/14、Ruff 与 compileall 通过。

R1-C 已于 2026-07-12 通过独立验收。独立验收确认 Dataset 只新增 A01/I04 两个批准候选，反向移除后可精确还原 R1-C 前哈希；approved positive SHA 门禁在 rejudge、`Settings` 和 guardrails 之前执行；原始 positive/guardrails 报告哈希未变化；guardrails 离线重判 9/9 passed 且新增模型调用为 0。独立复跑结果为长期记忆专项 86 passed、全量 stub 265 passed、基础 eval 14/14、Ruff 与 compileall 通过。因此 R1-C 与 M-R1 状态均为 `complete`。

`M-R1 complete` 只表示本计划定义的 targeted production remediation 已完成，不表示正式长期记忆 Track B 48 x 3 baseline、真实 Embedding、PostgreSQL 持久化或整体长期记忆评测已经完成。
