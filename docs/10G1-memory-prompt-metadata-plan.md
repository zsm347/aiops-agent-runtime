# 10G.1 长期记忆提示与元数据优化实施计划

## 1. 阶段定位

本文是第 10 步高级能力中 `10G 长期记忆专项增强` 的拆分子阶段：

```text
10G.1 长期记忆提示与元数据优化
```

本阶段只做长期记忆的行为引导层优化：

- Memory Index 改为更像 Letta 的 Memory Metadata。
- Core Memory block description 精细化。
- memory tools description / system prompt 优化。

本阶段不做长期记忆专项 eval。专项 eval 单独拆到 `10G.2`。

## 2. 当前基线

当前长期记忆核心闭环已经完成：

- Core Memory：
  - `user_rules`
  - `user_ops_profile`
  - `service_notes`
- Archival Memory：
  - `saveArchivalMemory`
  - `searchMemory`
  - `listMemoryTopics`
- Memory Index：
  - 当前注入 `<memory_index>`。
  - 包含 archival memory total、top topics、available tags、usage rules。
- memory tools：
  - `listMemoryTopics`
  - `searchMemory`
  - `updateCoreMemory`
  - `saveArchivalMemory`

相关代码：

```text
src/superbiz_agent/memory/schemas.py
src/superbiz_agent/memory/core.py
src/superbiz_agent/memory/index.py
src/superbiz_agent/memory/tools.py
prompts/ops-agent-system-v2.md
tests/test_long_term_memory.py
```

## 3. 设计依据

本阶段依据：

- Letta 的长期记忆设计：
  - Core Memory block description 是给模型看的写入边界。
  - Archival Memory 不常驻上下文，通过工具按需检索。
  - 系统上下文里注入 memory metadata，而不是注入全部外部记忆。
  - memory metadata 主要告诉模型：历史消息数量、archival memory 数量、可用 tags 等。
- 本项目现有设计：
  - Core Memory 常驻上下文。
  - Archival Memory 通过 `searchMemory` 检索。
  - 历史记忆只能作为历史参考，不能替代当前实时证据。
  - memory 工具不能让模型传 `tenant_id/user_id/agent_id/session_id/run_id`。

## 4. 本阶段做什么

### 4.1 Memory Index 升级为 Memory Metadata

当前：

```xml
<memory_index>
archival memory total: 2
rule memories:
top topics:
experience:
- order-service/5xx (1)
available tags: order, 5xx, payment
usage rules:
- call listMemoryTopics if the current task may need memories outside these topics.
- call searchMemory to retrieve specific memories before using them as evidence.
- memory is historical reference, not current evidence.
</memory_index>
```

目标：

```xml
<memory_metadata>
- archival_memory_total: 2
- available_topics:
  experience:
  - order-service/5xx (1)
  knowledge:
  - order-service/dependencies (1)
- available_tags: order, 5xx, payment
- available_scopes:
  services: order-service, payment-service
  envs: production
- usage_rules:
  - Use this metadata only to decide whether memory lookup may help.
  - Do not answer concrete facts from metadata alone.
  - Call searchMemory before using archival memory as evidence.
  - Historical memory is reference only; verify live incidents with realtime tools.
  - Tags are optional filters. Use them only when the category is clear.
</memory_metadata>
```

说明：

- 对模型可见标签从 `<memory_index>` 改为 `<memory_metadata>`。
- 内容从“topic index”扩展为“记忆状态元信息”。
- 仍然不注入 Archival Memory 正文。
- `available_tags` 只是可选过滤提示，不代表语义检索结果。
- `available_scopes` 来自已保存记忆的 `scopeService/scopeEnv`，帮助模型判断是否带 scope 过滤。

兼容策略：

- 代码内部可以暂时保留 `MemoryIndexService` 类名和 `memory_index_xml` 字段，减少改动面。
- 对模型可见内容改为 `<memory_metadata>`。
- trace 中可新增 `hasMemoryMetadata`，但保留 `hasMemoryIndex` 兼容现有 eval。
- 后续如果要彻底重命名，可单独做小重构，不放在本阶段扩大范围。

### 4.2 Core Memory block description 精细化

当前 description 偏短：

```text
user_rules:
User rules that must be followed across sessions...

user_ops_profile:
User's stable ops background and preferences...

service_notes:
High-frequency stable service background...
```

目标是把 block description 写成更明确的写入边界。

#### `user_rules`

用途：

- 用户明确要求长期遵守的交互规则。
- 回答格式偏好。
- 长期约束。
- 禁用建议。

允许写入：

- “以后排障回答按现象、证据、判断、建议、未确认项组织。”
- “以后不要建议我重启生产服务，除非明确说明风险。”

禁止写入：

- 一次性任务状态。
- 当前告警/日志/指标。
- 未确认猜测。
- 服务运行时事实。
- 密钥或敏感信息。

更新规则：

- 保留仍然有效的旧规则。
- 合并重复规则。
- 删除过时或冲突规则。
- 内容接近上限时先压缩整理，不要无脑追加。

#### `user_ops_profile`

用途：

- 用户长期稳定的运维背景。
- 用户负责的服务。
- 常用环境。
- 常用排查习惯。
- 工具/平台偏好。

允许写入：

- “用户主要负责 order-service 和 payment-service。”
- “用户常用 production / staging 环境进行排障。”
- “用户习惯先看告警，再查日志，最后对照内部文档。”

禁止写入：

- 某次事故的临时根因。
- 当前实时日志。
- 一次性查询条件。
- 用户临时提到的服务名，除非明确是长期负责或常用背景。

#### `service_notes`

用途：

- 稳定服务背景。
- 高频跨会话复用的服务依赖。
- 稳定架构事实。
- 常用排障入口。

允许写入：

- “order-service 依赖 payment-service 完成支付确认。”
- “payment-service 常见下游依赖包括 Redis 和 MySQL。”

禁止写入：

- 未确认根因。
- 当前指标值。
- 原始 trace/log。
- 一次性变更状态。
- 详细事故教训；这类应进入 `saveArchivalMemory`。

### 4.3 Memory Tools Description 优化

优化目标：

- 让模型更清楚什么时候查。
- 让模型更清楚什么时候写 Core。
- 让模型更清楚什么时候写 Archival。
- 防止把历史记忆当作当前证据。
- 防止乱用 tags/scope 过滤导致召回变差。

#### `listMemoryTopics`

目标描述：

```text
List available long-term memory topics and tags. Use this only as a routing aid
when the visible memory metadata is insufficient. Do not answer concrete facts
from topics alone; call searchMemory before using archival memory as evidence.
```

#### `searchMemory`

目标描述：

```text
Search archival long-term memory by semantic meaning. Use this when the user asks
for historical experience, prior incidents, stable preferences, or reusable
operational knowledge. Tags, scopeService, and scopeEnv are optional filters;
use them only when the category/service/environment is clear. Returned memory is
historical reference, not current live evidence. For live diagnosis, verify with
realtime tools.
```

#### `updateCoreMemory`

目标描述：

```text
Update a core memory block that is always visible in future context. Use it only
for stable, high-value, cross-session information that should remain visible:
user rules, user ops profile, or stable service notes. newContent must be the
full updated block content, preserving useful old information and removing
outdated or conflicting information. Do not store raw logs, metrics, secrets,
unverified guesses, or one-time task state.
```

#### `saveArchivalMemory`

目标描述：

```text
Save a self-contained archival memory for durable, searchable historical
reference. Use it for verified incident lessons, confirmed root causes, reusable
runbook experience, or stable operational knowledge that does not need to stay
always visible. Do not save raw logs, stack traces, secrets, current metric
values, or unverified hypotheses. Add concise tags/scope only when they are
clear and useful for future filtering.
```

### 4.4 System Prompt 长期记忆规则优化

当前 `ops-agent-system-v2.md` 已有长期记忆规则，但可以进一步强化：

- `<memory_metadata>` 只是外部记忆概况，不是事实证据。
- 只有 `searchMemory` 返回的具体 memory content 才能作为历史记忆参考。
- tags 是过滤条件，不是语义检索本身；不确定时不要带 tags。
- `scopeService/scopeEnv` 也是过滤条件，不确定时不要乱填。
- Core Memory 和 Archival Memory 的写入边界更明确。
- 当前实时故障必须优先用实时工具验证。

## 5. 本阶段不做什么

明确不做：

- 不做长期记忆专项 eval。
- 不做后台 LLM extraction。
- 不做 `memory_extraction_job`。
- 不做 memory_candidate。
- 不做 Recall Memory / 历史对话召回。
- 不做生命周期遗忘 / archive job。
- 不新增 Archival Memory update/delete/archive 工具。
- 不设计复杂 source/provenance 字段。
- 不改变 memory tool 参数中的运行时身份注入规则。
- 不让模型传 `tenant_id/user_id/agent_id/session_id/run_id`。

## 6. 文件级实施范围

### 6.1 可能修改文件

```text
src/superbiz_agent/memory/schemas.py
src/superbiz_agent/memory/index.py
src/superbiz_agent/memory/store.py
src/superbiz_agent/memory/tools.py
src/superbiz_agent/harness/context_assembler.py
src/superbiz_agent/harness/context_manager.py
src/superbiz_agent/harness/runtime.py
src/superbiz_agent/evals/traces.py
tests/test_long_term_memory.py
tests/test_eval_runner.py
prompts/ops-agent-system-v2.md
```

### 6.2 不应修改文件

```text
src/superbiz_agent/rag/
src/superbiz_agent/model_gateway/
src/superbiz_agent/api/
src/superbiz_agent/security/
alembic/
```

除非发现明确编译冲突，本阶段不改数据库迁移。

## 7. 实施批次

### 批次 A：Memory Metadata 渲染

目标：

- 将模型可见 `<memory_index>` 升级为 `<memory_metadata>`。
- 增加 `available_scopes`。
- 保留 tags/topics 但明确其为 routing/filter aid。
- fallback 也使用 `<memory_metadata>`。

验收：

- 上下文中能看到 `<memory_metadata>`。
- Archival Memory 正文不会被直接注入 metadata。
- topics/tags/scopes 显示正确。
- 超预算 fallback 仍提醒调用 `searchMemory`。

### 批次 B：Core Block Description 精细化

目标：

- 更新 `CORE_BLOCK_SPECS` 中三个 block 的 description。
- description 明确允许写入、禁止写入、更新规则。

验收：

- `<core_memory>` 中三个 block 的 description 变清晰。
- max_tokens 不改变。
- 原有 core memory 更新策略不回退。

### 批次 C：Tool Description 与 Prompt 优化

目标：

- 更新 4 个 memory tools 的 description。
- 更新 `ops-agent-system-v2.md` 长期记忆规则。

验收：

- 工具 schema 不变。
- 工具权限和 policy 不变。
- prompt 明确 tags/scope 是可选过滤。
- prompt 明确历史记忆不是当前证据。

### 批次 D：测试更新

目标：

- 更新测试中对 `<memory_index>` 的断言。
- 保持 trace 兼容。
- 增加 metadata 内容断言。

验收：

- `python3 -m pytest` 通过。
- eval runner 不回退。

## 8. subAgent 任务单

如果进入代码实施，推荐交给一个 subAgent 完成，因为改动集中且需要保持一致。

任务边界：

```text
只实现 docs/10G1-memory-prompt-metadata-plan.md 定义的长期记忆提示与元数据优化。
不得实现长期记忆专项 eval。
不得实现后台 extraction。
不得实现遗忘/归档 job。
不得新增 Archival Memory update/delete/archive 工具。
不得改 RAG、ModelGateway、API、安全模块。
```

提交时必须说明：

1. `<memory_metadata>` 的最终格式。
2. 三个 Core block description 如何变化。
3. 4 个 memory tools description 如何变化。
4. prompt 长期记忆规则如何变化。
5. 测试命令和结果。
6. 是否偏离本文档。

## 9. 验收命令

实现完成后至少运行：

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
PYTHONPATH=src python3 -m compileall -q src tests
```

如果安装了 ruff：

```bash
python3 -m ruff check src tests
```

## 10. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否只做 10G.1 三项优化 | 是 |
| 是否没有提前做长期记忆专项 eval | 是 |
| 是否没有做后台 extraction | 是 |
| 是否没有做遗忘/归档 job | 是 |
| 是否参考 Letta 但保留本项目 Core/Archival 结构 | 是 |
| 是否避免把 Archival Memory 正文注入上下文 | 是 |
| 是否明确 tags/scope 是可选过滤而非语义检索本身 | 是 |
| 是否保留 memory tool 参数和权限契约 | 是 |
| 是否保留历史记忆不能替代实时证据的规则 | 是 |
| 是否适合交给 subAgent 实施 | 是 |

## 11. 阶段通过标准

10G.1 完成后必须满足：

1. 模型上下文中出现 `<memory_metadata>`，内容包含 archival memory total、available topics、available tags、available scopes、usage rules。
2. Core Memory 三个 block 的 description 更清楚，能指导模型写入边界。
3. memory tools description 更明确，特别是 `searchMemory`、`updateCoreMemory`、`saveArchivalMemory`。
4. `ops-agent-system-v2.md` 长期记忆规则更新。
5. 不改变 memory tools 参数 schema。
6. 不改变 tenant/user/run 后端注入规则。
7. 原有长期记忆测试和 eval runner 不回退。
