# M-P0 Archival Memory 确定性去重与原子写入契约

## 1. 阶段边界

M-P0 只实现 Archival Memory 的确定性 exact dedupe 和进程内原子写入。本阶段不使用 embedding 相似度、向量 topK 或 LLM Judge 判定重复，不接真实 embedding，也不实现 PostgreSQL 持久化。

`memory_duplicate_similarity` 仅保留为配置兼容字段，不再影响 Archival 写入。Deterministic embedding 仍可为新记忆生成搜索向量，但不能决定是否跳过写入。

## 2. Exact 等价键

两条 active Archival Memory 只有以下字段全部相等时才是 exact duplicate：

```text
tenant_id
+ user_id
+ agent_id
+ memory type
+ scope_service
+ scope_env
+ canonical_content_hash
```

比较为区分大小写的精确比较。写入服务会把空白 scope 规范成 `None`，但不会猜测或改写非空 scope。

以下字段不进入等价键：

```text
topic
tags
session_id
source
```

因此 topic、tags、session 或 source 不同但 exact key 相同的写入仍视为 duplicate；不同 tenant、user、agent、type、service 或 env 的内容绝不能互相去重。只有 `status=active` 的记录参与查重，archived 旧记录不阻止创建新的 active 记录。

## 3. Canonical Content Hash

`canonicalize_archival_content` 和 `canonical_content_hash` 定义在 `superbiz_agent.memory.dedup`。规则按以下顺序执行：

1. 把 CRLF 和 CR 统一为 LF。
2. 使用 Unicode NFC 规范化；不使用 NFKC、lower 或 casefold。
3. 把连续 Unicode 空白折叠为一个 ASCII 空格，并移除首尾空白。
4. 只移除整段内容末尾连续的 `. ! ? 。 ！ ？ …` 终止符及其后空白。
5. 对 canonical UTF-8 字节计算 SHA-256。

该规则不会删除或改写内部标点，不会改变大小写、数值、否定词、服务名、配置键、路径或正文内部的终止符。任何这些语义字段发生变化都会产生不同 canonical hash。分号不作为可忽略终止符，因为它可能属于配置或语句语义。

## 4. 原子写入语义

`InMemoryMemoryStore.write_archival_exact` 在同一个 store 锁内执行：

```text
计算/校正 canonical_content_hash
-> 查找同 exact key 的 active memory
-> 不存在时 insert
-> 存在时合并 tags 并返回 existing memory
```

同一 exact key 的并发写入具备可串行化结果：第一个获得锁的调用返回 `written`，其余调用返回 `duplicate_skipped`，最终只能存在一条 active memory。

该原子保证目前只适用于进程内 `InMemoryMemoryStore`。PostgreSQL 路径必须在 M-P2 通过数据库唯一约束或等价事务协议重新实现，不能把进程锁当作跨进程保证。

## 5. Metadata 合并

Exact duplicate 不替换已有 memory：

- 保留已有 `id`、`topic`、正文、session、source、embedding 和创建时间。
- tags 使用稳定顺序并集：先保留已有 tags 的顺序，再按 incoming 顺序追加尚未出现的非空 tag。
- tags 区分大小写，不做 casefold。
- 至少追加一个 tag 时 `metadataMerged=true`；没有变化时为 `false`。

`ARCHIVAL_MEMORY_WRITTEN` 和 `ARCHIVAL_MEMORY_DUPLICATE_SKIPPED` trace payload 记录：

```text
dedupeKind=exact
metadataMerged=true|false
```

工具返回仍保持面向模型的既有协议，不输出 canonical/content hash、tenant、user 或 agent identity，也不向模型暴露内部去重键。

## 6. 本阶段明确不做语义去重

Canonical hash 不同的两条正文必须分别写入，即使 local-deterministic embedding 的 cosine similarity 为 1.0。M-P0 不使用 `0.92` 或任何其他相似度阈值跳过写入。

后续 M-P3 才允许按独立审核后的流程实现语义判定：

```text
新记忆
-> 使用真实 embedding 检索同 identity/type/scope 的 topK
-> 用校准阈值和/或结构化 LLM Judge
-> 判定 same / related / distinct / conflict
```

M-P3 必须先具备真实 embedding 基线、误合并风险评估和明确的 metadata/conflict 处理契约，不能恢复 deterministic embedding 伪语义去重。

## 7. 验收覆盖

M-P0 测试必须覆盖：canonical 等价与保守差异、完整 exact key 隔离、archived 例外、tags 稳定合并、旧 metadata 保留、并发单 active、trace 字段和隐私边界，以及高 embedding 相似但 canonical 不同的两条正文均写入。既有写入、检索、敏感信息拒绝和租户隔离测试必须继续通过。
