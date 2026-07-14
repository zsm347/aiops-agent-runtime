# SuperBizAgent Python

这个目录用于存放现有 Java 单 Agent Harness 的 Python 版本。

Java 项目仍然是当前参考实现。这个 Python 工作区先沉淀迁移文档、架构设计和项目骨架，后续再逐步实现 Python 后端、测试、评测样本、提示词和脚本。

## 目录结构

```text
docs/      迁移盘点、迁移规格、架构设计和后续实施计划
src/       Python 应用源码
tests/     单元测试和集成测试
evals/     Agent 评测样本、runner、baseline 和报告
prompts/   版本化提示词
scripts/   本地开发和迁移脚本
```

## 已有文档

- `docs/01-java-capability-inventory.md`：Java 现有能力盘点
- `docs/02-python-migration-spec.md`：Python 迁移规格
- `docs/03-python-architecture-design.md`：Python 架构与技术栈设计

## 当前阶段

1. 盘点当前 Java Agent Harness。
2. 明确 Python 迁移范围和兼容契约。
3. 确定 Python 架构与框架选型。
4. 创建 Python 项目骨架。
5. 后续再实现最小 `chat -> context -> model -> tool -> trace` 链路。

## 骨架验证

```bash
python3 -m pytest
```

