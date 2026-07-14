# 10D 上下文管理与压缩前置调研

## 1. 调研范围

本调研服务于第 10 步高级能力中的 `10D 上下文窗口与压缩`。

目标不是泛泛讨论“摘要怎么写”，而是回答实现前必须先确认的几个问题：

- 业内成熟 Agent 框架如何管理长会话上下文。
- 现有框架是否已经提供可直接复用的 message trim、summary compaction、tool result clearing、session memory。
- 哪些能力应该直接引入，哪些只应该借鉴设计。
- 我们的 Python 单 Agent Harness 该如何在保留事件流、trace、多租户、工具证据链的前提下落地 10D。

调研对象：

- LangChain / LangGraph
- LlamaIndex
- Semantic Kernel
- AutoGen
- Letta
- OpenAI Agents SDK
- Anthropic context management
- Headroom
- LLMLingua / LLMLingua-2

不在本调研范围：

- 不设计长期记忆 Core / Archival 的新增功能。
- 不重新选择 Agent 编排框架。
- 不进入 10D 代码实现。
- 不把 prompt caching 当成上下文压缩。

## 2. 总体结论

成熟框架已经形成了一些共识：

1. 上下文管理通常发生在模型调用前，而不是等模型报错后再处理。
2. 短期历史不应该无限进入模型上下文；通常使用 trim、summary、session reducer 或 compaction。
3. 工具结果需要单独治理。尤其在 Agent 项目中，旧工具结果、搜索结果、日志、文件内容往往比自然语言对话更容易撑爆上下文。
4. 压缩应该保留最新上下文原文，把较旧历史替换为摘要，而不是把全部历史一次性压成一段。
5. 需要保护消息边界，尤其是 assistant tool call 与 tool result 不能被拆散。
6. token 估算应优先使用模型/供应商 tokenizer 或 token counting API；P0 可以保留近似估算，但要留升级口。
7. 原始历史最好继续持久化为事实源；进入模型的只是一个“上下文视图”。
8. 内容级压缩和会话级压缩要分开。Headroom 这类工具适合压缩单个大块内容，但不能替代 session summary、历史淘汰和长期记忆。

对本项目的结论：

```text
不建议直接把 LangChain/LlamaIndex/Letta/OpenAI Agents SDK 的 memory 组件整体接进来替换现有 Harness。

建议借鉴它们的策略，实现项目自己的 ContextManager / ContextReducer：
事件流仍是事实源；
run 内 active_history 仍是请求级内存对象；
模型调用前生成可控的上下文视图；
压缩摘要和工具结果清理都写 trace event；
恢复时根据 HISTORY_TRIMMED 从事件流重建 summary + recent history。
```

## 3. 候选方案对比

| 项目 | 提供能力 | 适合直接用吗 | 对 10D 的价值 |
|---|---|---|---|
| LangChain / LangGraph | short-term memory、trim messages、SummarizationMiddleware、ContextEditingMiddleware、pre_model_hook | 不整体接入 memory；可借鉴策略 | 最贴近当前 LangGraph 技术栈，适合借鉴模型调用前 reducer、中间件、工具结果清理 |
| LlamaIndex | Memory、short-term FIFO、memory blocks、token limit、flush to long-term memory | 不作为主 history store；RAG 仍可用 LlamaIndex | 借鉴 token_limit、chat_history_token_ratio、memory block priority |
| Semantic Kernel | ChatHistoryReducer、TruncationReducer、SummarizationReducer | 不直接引入 | 借鉴 reducer 抽象、target + threshold 滞回、防止频繁压缩 |
| AutoGen | ChatCompletionContext、Buffered、TokenLimited、HeadAndTail | 不直接引入 | 借鉴“上下文视图”接口和 head/tail placeholder |
| Letta | stateful agents、core memory、archival memory、message compaction | 不整体接入 | 借鉴 sliding_window compaction、self compaction、持久事实源 + 可压缩上下文 |
| OpenAI Agents SDK | Sessions、TrimmingSession、SummarizingSession、Responses compaction session | 不绑定 SDK | 借鉴 session 接口、turn-based trimming、自动/手动 compaction |
| Anthropic | server-side compaction、context editing、prompt caching、token counting | Provider 能力，不做核心依赖 | 借鉴 tool result clearing 和 provider capability 抽象 |
| Headroom | 面向 Agent 的内容级压缩，覆盖工具输出、日志、JSON、搜索结果、RAG chunk、文件和代码；支持 CCR 思路 | 不作为会话级 ContextManager；可作为 ToolResultReducer 的内容压缩 backend | 对大工具输出压缩最直接，适合补齐单个内容块过大的问题 |
| LLMLingua | prompt compression、token/片段级压缩 | 不用于主 history compaction | 可作为未来 RAG 文档/普通长文本压缩实验 |

## 4. LangChain / LangGraph

### 4.1 能力

LangGraph 官方短期记忆文档把长历史管理拆成几类：

- trim messages：在调用 LLM 前删除前 N 条或后 N 条消息。
- delete messages：从图状态中永久删除消息。
- summarize messages：把较早历史总结成摘要并替换。
- checkpoint：保存和恢复消息历史。
- custom strategies：自定义过滤或压缩策略。

LangChain 内置 middleware 中，`SummarizationMiddleware` 用于在接近 token limit 时自动总结旧历史；`ContextEditingMiddleware` 用于清理旧工具结果等上下文内容。

来源：

- LangGraph short-term memory：<https://docs.langchain.com/oss/python/langgraph/add-memory>
- LangChain built-in middleware：<https://docs.langchain.com/oss/python/langchain/middleware/built-in>
- `SummarizationMiddleware` 源码：<https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py>
- `ContextEditingMiddleware` 源码：<https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/context_editing.py>

### 4.2 实现要点

`SummarizationMiddleware` 的关键设计：

- 在模型调用前运行。
- trigger 和 keep 分离：
  - trigger 可按 tokens、messages、fraction 触发。
  - keep 可保留最近 N 条、N tokens 或模型窗口比例。
- 支持 trigger clause：同一个 clause 内是 AND，多个 clause 之间是 OR。
- 默认使用近似 token counter，也可以基于模型 profile 和 usage metadata。
- 先选择要总结的旧消息，再保留 recent messages。
- cutoff 会避免拆散 AI message 的 tool calls 与对应 ToolMessage。
- 摘要生成前还会先 trim summarization input，避免摘要模型输入本身过长。
- 最终用一条 summary message + preserved messages 替换旧消息。

`ContextEditingMiddleware` 的关键设计：

- 和 summary 分开，是另一类上下文编辑策略。
- 典型策略是 `ClearToolUsesEdit`：当上下文 token 超阈值时，清理较旧 tool result。
- 可以配置保留最近几个工具结果。
- 清理后用 placeholder 替代旧工具结果，而不是直接无痕删除。
- 可选择是否清理 tool input。

### 4.3 对本项目的借鉴

10D 应借鉴：

- 在每次模型调用前统一经过 `ContextManager`，而不是只在 `ContextAssembler` 内临时拼接。
- trigger 和 keep 分离。
- 同时支持 token trigger 和 message count trigger。
- 增加 hysteresis 或 threshold，避免每次请求都压缩。
- 压缩时必须保护 tool call / tool result 配对。
- 工具结果清理与历史摘要是两件事，不能混成一个“总结历史”动作。

不建议直接使用：

- 不直接把 LangChain `SummarizationMiddleware` 接入为状态管理核心。
- 原因是本项目有自己的 `ModelMessage`、rollout event、trace schema、tenant context、Java 兼容契约。
- 如果直接让外部 middleware 改写 LangGraph state，会弱化 `HISTORY_TRIMMED`、`MODEL_CALL_CONTEXT_OVERFLOW` 等项目级审计事件。

## 5. LlamaIndex Memory

### 5.1 能力

LlamaIndex Memory 把短期对话历史和长期 memory blocks 结合起来：

- `token_limit`：短期 + 长期内容总 token 限制。
- `chat_history_token_ratio`：短期 chat history 占总 token limit 的比例，默认 0.7。
- `token_flush_size`：当短期历史超过限制时，一次 flush 多少 token。
- memory blocks 可按 priority 注入和裁剪。
- 被 flush 的消息可以进入长期 memory block。

来源：

- LlamaIndex Memory 文档：<https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/>
- LlamaIndex Memory 示例：<https://developers.llamaindex.ai/python/examples/memory/memory/>
- LlamaIndex Memory 源码：<https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/memory/memory.py>

### 5.2 实现要点

LlamaIndex 的新 Memory 更像一个“短期 FIFO 队列 + memory blocks 瀑布流”：

```text
新消息进入短期队列
  -> 估算 token
  -> 超过 token_limit * chat_history_token_ratio
  -> 从最旧消息开始 flush
  -> flush 的消息 archive
  -> 可交给 memory blocks 处理
  -> 取回上下文时，把 active chat history 和 memory block 内容组合
```

它还引入了 block priority：

- priority 越高越优先保留。
- priority=0 可表示永不裁剪。
- 不同 memory block 可以实现自己的 truncate 方法。

### 5.3 对本项目的借鉴

可借鉴：

- 配置中明确区分总 token 预算、短期历史预算和每次 flush/压缩大小。
- 对上下文组件做 priority 管理：
  - system prompt、current user message、tool schema、core memory 属于高优先级。
  - 旧对话和旧工具结果属于可裁剪/可摘要区域。
- 可以引入 `ContextComponentUsage`，记录每个组件 token 估算、是否保留、是否裁剪、裁剪原因。

不建议直接使用：

- LlamaIndex Memory 不应该替代本项目的 `agent_rollout_event`。
- 本项目的短期事实源已经是 PostgreSQL event stream，且有 Java 兼容 trace 契约。
- LlamaIndex 更适合作为 RAG 编排和部分 memory block 思路来源，而不是单 Agent Harness 的主会话状态存储。

## 6. Semantic Kernel ChatHistoryReducer

### 6.1 能力

Semantic Kernel 提供 ChatHistoryReducer 抽象，并内置：

- `ChatHistoryTruncationReducer`
- `ChatHistorySummarizationReducer`

Microsoft Learn 文档说明，TruncationReducer 在历史长度超过限制时截断；SummarizationReducer 会总结被移除消息并把摘要作为单条消息放回历史；两者都会保留 system message。

来源：

- Microsoft Learn Chat History：<https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/chat-history>
- ChatHistoryReducer API：<https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.contents.history_reducer.chat_history_reducer.chathistoryreducer>

### 6.2 对本项目的借鉴

可借鉴：

- reducer 接口抽象：

```text
input: messages + budget + policy
output: reduced messages + reduction report
```

- `target_count + threshold_count` 思想：
  - 不要刚超过一点就压缩。
  - 超过触发阈值后，一次性压到目标水位以下。

不建议直接使用：

- Semantic Kernel 的 reducer 偏消息数量和通用聊天历史。
- 本项目更需要 token 预算、工具证据链、trace event 和多租户上下文。

## 7. AutoGen Model Context

### 7.1 能力

AutoGen 的 Model Context 是一个“传给模型的消息视图”抽象，常见实现包括：

- `BufferedChatCompletionContext`：只保留最近 N 条消息。
- `TokenLimitedChatCompletionContext`：按 token 限制裁剪。
- `HeadAndTailChatCompletionContext`：保留开头和结尾，中间用 placeholder 表示跳过。

来源：

- AutoGen Model Context 文档：<https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/model-context.html>
- AutoGen model_context API：<https://microsoft.github.io/autogen/stable/reference/python/autogen_core.model_context.html>

### 7.2 对本项目的借鉴

可借鉴：

- 把“完整历史”和“模型可见历史”分开。
- `HeadAndTail + skipped placeholder` 可用于某些调试场景，告诉模型中间有历史被省略。

不建议直接使用：

- TokenLimited 从中间删除消息的策略对排障 Agent 风险较高。
- AIOps 场景中，中间某个工具结果、错误码、traceId 可能就是关键证据。
- AutoGen 对 tool call / tool result 的完整 ID 配对保护不如我们需要的严格。

## 8. Letta

### 8.1 能力

Letta 是长期状态和记忆管理做得比较系统的 Agent 项目。它的核心思想：

- 所有状态，包括 memories、user messages、reasoning、tool calls 都持久化，不因为被移出上下文窗口而丢失。
- Core memory / memory blocks 注入上下文。
- Archival memory 是语义可搜索数据库，按需通过工具查询。
- 当消息历史过长时做 message compaction。

来源：

- Letta Stateful Agents：<https://docs.letta.com/guides/core-concepts/stateful-agents/>
- Letta Compaction：<https://docs.letta.com/guides/core-concepts/messages/compaction/>
- Letta Archival Memory：<https://docs.letta.com/guides/core-concepts/memory/archival-memory/>
- Letta compaction 源码：<https://github.com/letta-ai/letta/blob/main/letta/services/summarizer/compact.py>
- Letta sliding window 源码：<https://github.com/letta-ai/letta/blob/main/letta/services/summarizer/summarizer_sliding_window.py>
- Letta summarizer prompt 源码：<https://github.com/letta-ai/letta/blob/main/letta/prompts/summarizer_prompt.py>

### 8.2 Compaction 实现要点

Letta 默认 compaction mode 是 `sliding_window`：

- 默认 `sliding_window_percentage=0.3`。
- 含义是总结较旧约 30% 消息，保留较新约 70% 消息。
- 如果压缩后仍然太大，可以每次增加约 10% 的被总结比例，直到 fits。
- 也支持 `all`：总结全部历史。
- 还支持 `self_compact_sliding_window` 和 `self_compact_all`：用 agent 自身 system prompt 和 tools 进行 self compaction，有助于 cache compatibility。

它的 sliding window 源码还体现了几个工程点：

- 区分 agent model 的 context window 和 summarizer model。
- token counting 会对近似估算加 safety margin。
- cutoff 会尽量落在 assistant/approval 消息边界。
- 压缩后会重新估算 token，如果仍超过 trigger threshold，尝试 fallback。
- summary 会作为专门 summary message 放入上下文。

### 8.3 对本项目的借鉴

可借鉴：

- 事件流/数据库保存完整事实源，进入模型的是 compacted view。
- 不固定“只保留最近 4 轮”，可以使用 `recent_keep_ratio` 或 `sliding_window_percentage`。
- 初始保留较多 recent context；压不下来再加大压缩比例。
- 摘要 prompt 要保留高层目标、发生了什么、重要细节、错误和修复、当前状态、下一步、lookup hints。
- 可以支持 fake/deterministic compactor 做本地测试，真实 LLM compactor 做 provider adapter。

不建议直接引入 Letta：

- Letta 是完整 Agent 平台，会带入另一套 agent runtime、memory tools、API、权限模型。
- 本项目已经有单 Agent Harness、ToolGateway、RolloutEventStore、Core/Archival Memory。
- 直接引入会造成边界混乱。

## 9. OpenAI Agents SDK

### 9.1 能力

OpenAI Agents SDK 的 Sessions 用于跨多次 agent run 自动维护会话历史。官方 cookbook 专门讨论了两类上下文管理技术：

- trimming：只保留最近若干 turn。
- compression：把旧 turn 压缩为摘要。

同时，OpenAI Agents SDK 文档中也提供 Sessions 机制，用于自动维护多轮上下文。

来源：

- OpenAI Agents SDK Sessions：<https://openai.github.io/openai-agents-python/sessions/>
- OpenAI Cookbook Session Memory：<https://developers.openai.com/cookbook/examples/agents_sdk/session_memory>

### 9.2 对本项目的借鉴

可借鉴：

- session 是会话历史接口，不等于长期记忆。
- trim 通常按 turn 边界处理，而不是按裸 message 数硬切。
- summarize session 可以把旧 turn 替换成 synthetic summary，再保留 recent turns。

不建议直接依赖：

- OpenAI Responses compaction session 与 OpenAI Responses API 绑定较深。
- 本项目需要兼容 Qwen/OpenAI-compatible/DashScope，并保留自己的 trace event。

## 10. Anthropic Context Management

### 10.1 能力

Anthropic 提供三类相关能力：

- Server-side compaction：达到 input token 阈值后自动生成 compaction block，后续请求把 compaction block 传回，API 会忽略它之前的旧内容。
- Context editing：服务端在 prompt 到达模型前清理旧 tool results 或 thinking blocks。
- Prompt caching：缓存 prompt 前缀以降低成本和延迟。

来源：

- Anthropic Compaction：<https://platform.claude.com/docs/en/build-with-claude/compaction>
- Anthropic Context Editing：<https://platform.claude.com/docs/en/build-with-claude/context-editing>
- Anthropic Prompt Caching：<https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- Anthropic Token Counting：<https://platform.claude.com/docs/en/build-with-claude/token-counting>
- Anthropic engineering blog：<https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>

### 10.2 对本项目的借鉴

可借鉴：

- compaction 和 context editing 是两种不同能力：
  - compaction 用摘要替代旧上下文。
  - context editing 清旧工具结果，但客户端仍保留完整历史。
- tool result clearing 应该保留 placeholder，让模型知道旧工具结果被清理过。
- token counting 是单独能力，应作为 provider capability。

注意：

- Prompt caching 不能解决 context overflow，它只是降低重复前缀的成本和延迟。
- Anthropic 的 server-side compaction 是 provider 能力，不能作为 Qwen/OpenAI-compatible 的必需能力。

## 11. LLMLingua / LLMLingua-2

### 11.1 能力

LLMLingua 是 Microsoft 提出的 prompt compression 项目，主要通过小模型/分类器压缩 prompt token，减少推理成本和延迟。

来源：

- LLMLingua GitHub：<https://github.com/microsoft/LLMLingua>
- LLMLingua paper：<https://arxiv.org/abs/2310.05736>
- LLMLingua-2 paper：<https://arxiv.org/abs/2403.12968>

### 11.2 对本项目的判断

不建议作为 10D 主路径。

原因：

- 它更适合压缩 RAG 长文档、prompt 冗余内容、非结构化背景材料。
- AIOps 排障上下文里，错误码、traceId、服务名、时间窗口、日志证据可能是短 token 但高价值，抽取式压缩有误删风险。
- 它不能替代事件流恢复、工具结果脱敏、摘要质量验证。

可以作为未来增强：

- 对 RAG 检索片段做压缩。
- 对超长日志工具结果做非关键字段压缩。
- 但必须在保留 evidence id/sourceRef 的前提下使用。

## 12. Headroom

### 12.1 能力

Headroom 的定位是 Agent 的上下文压缩层，放在应用和 LLM provider 之间，对即将进入模型的内容块做压缩。它更偏“内容级压缩”，尤其适合：

- 大工具输出。
- 日志和日志片段。
- JSON / 表格 / 搜索结果。
- RAG chunk。
- 文件内容和代码片段。

来源：

- Headroom GitHub：<https://github.com/headroomlabs-ai/headroom>
- context management 文档：<https://github.com/headroomlabs-ai/headroom/blob/main/docs/content/docs/context-management.mdx>
- compression 文档：<https://github.com/headroomlabs-ai/headroom/blob/main/docs/content/docs/how-compression-works.mdx>
- CCR 文档：<https://github.com/headroomlabs-ai/headroom/blob/main/docs/content/docs/ccr.mdx>

### 12.2 对本项目的判断

Headroom 适合进入 10D，但边界必须清楚：

```text
Headroom = 内容级压缩 backend
本项目 ContextManager = 会话级上下文治理
```

Headroom 当前更适合做 live-zone 内容压缩。它不会负责删除旧 message，也不会负责 session summary、历史淘汰、长期记忆和 RAG 检索。因此它不能解决“长会话一直增长，最终压缩后的短 message 也填满窗口”的问题。

对本项目更合适的用法是：

- `ToolResultReducer` 判断单个工具输出是否超过阈值。
- 原始工具结果先写入本项目 rollout event / artifact store。
- 超过阈值后调用 `HeadroomContentCompressionBackend` 压缩。
- 压缩结果进入上下文，并携带 `raw_ref`。
- 如需原文，由受控原文取回工具按 tenant/run/session/tool_call 权限校验后读取。

### 12.3 P0 取舍

P0 不做多级阈值，采用单一规则：

```text
单个工具输出 > 1000 tokens：触发 Headroom 内容级压缩。
单个工具输出 <= 1000 tokens：不压缩，直接进入上下文。
```

不依赖 Headroom CCR 本地存储作为事实源。CCR 可以作为可逆压缩思想参考，但本项目必须用自己的事件流或 artifact store 保存原文，保证多租户隔离、trace、审计和可恢复性。

### 12.4 与 LLMLingua 的关系

Headroom 更适合 P0 的大工具输出压缩，因为它面向 Agent 内容块，覆盖日志、JSON、搜索结果和工具输出。

LLMLingua 更适合作为未来实验，用在 RAG 长文档、普通长文本或 prompt 冗余压缩上。对于 AIOps 日志、错误码和 traceId 这类短但高价值的信息，LLMLingua 的抽取式压缩需要更严格评测后再引入。

## 13. 本项目当前基线

当前 Python 项目已经具备：

- `ConversationRuntime.start_run()`：请求开始时从 event store 读取 session 事件。
- `recover_active_history()`：从 rollout events 恢复 user、assistant、tool call、tool result。
- `_active_history_by_run`：一次 run 内内存维护 active history，run 结束释放。
- `ContextAssembler`：组装 system prompt、core memory、memory index、active history、current user message。
- `RolloutEventType` 已有 `HISTORY_TRIMMED` 和 `MODEL_CALL_CONTEXT_OVERFLOW`。

当前风险：

- 只有字符数估算，没有 token budget。
- `recover_active_history()` 会把 tool result JSON 恢复进历史，日志/告警/RAG 大结果容易撑爆上下文。
- 没有 latest summary 恢复逻辑。
- 没有 tool pair safe cutoff。
- 没有 context component usage trace。
- 10D 原草案把很多逻辑放在“压缩摘要”里，需要拆成 ContextManager / reducer / tool-result editing。

## 14. 推荐设计方向

### 14.1 架构定位

10D 应新增一个项目内的上下文治理层：

```text
ConversationRuntime
  -> recover_active_history(events)
  -> ContextManager.prepare(...)
       -> ContextBudgeter
       -> ToolResultReducer
       -> HistoryReducer / Compactor
       -> ContextComponentUsage report
  -> ContextAssembler.assemble(prepared_context)
  -> ModelGateway.complete(...)
```

这里 `ContextAssembler` 不再承担全部治理逻辑，它只负责把已治理好的组件组装成模型消息。

### 14.2 核心对象

建议引入：

```text
ContextBudget
ContextComponentUsage
PreparedContext
ContextManager
TokenEstimator
ToolResultReducer
ContentCompressionBackend
HistoryCompactor
CompactionResult
```

### 14.3 策略选择

P0 推荐组合：

1. 近似 token estimator。
2. Context budget 配置。
3. 工具结果先做 context editing 式清理和内容级压缩：
   - 单个工具输出超过 1000 tokens 时，用 Headroom backend 压缩。
   - 原文先持久化并生成 raw_ref。
   - 旧工具结果替换为 placeholder。
   - 保留 tool name/status/error_type/key fields/raw_ref。
4. 历史摘要使用 sliding-window 思路：
   - 默认总结旧 30% 到 50% 历史，保留最近 50% 到 70%。
   - 如果仍超预算，再加大压缩比例。
   - P0 也可以先用固定 recent turns 作为简化实现，但文档必须说明这是实现简化，不是最终设计上限。
5. `HISTORY_TRIMMED` 写入事件流。
6. 恢复时使用最新有效 `HISTORY_TRIMMED` 作为 summary 边界，只恢复其后的 recent events。

### 14.4 不做

10D 不做：

- 不启用 Letta / LlamaIndex Memory 替换事件流。
- 不依赖 Anthropic server-side compaction。
- 不依赖 OpenAI Responses compaction session。
- 不把 LLMLingua 作为主 history compactor。
- 不把 Headroom 作为完整会话级 ContextManager。
- 不依赖 Headroom CCR 本地存储作为原文事实源。
- 不把压缩摘要写入长期记忆。
- 不改变 Core / Archival Memory 设计。

## 15. 对原 10D 草案的修订要求

原 `docs/10D-context-window-compaction-plan.md` 应修订：

1. 状态从“内部初版草案”改为“已完成前置调研后的正式实施计划”。
2. 增加“调研依据”章节。
3. 把核心设计从“ContextAssembler 内做压缩”改为“ContextManager / Reducer 生成 PreparedContext，ContextAssembler 只组装”。
4. 增加 `ContextComponentUsage` 和上下文组件预算报告。
5. 增加 tool result context editing 作为独立步骤。
6. 明确 Headroom 只作为大工具输出内容级压缩 backend，单个工具输出超过 1000 tokens 才触发。
7. 压缩对象选择从固定 recent 4 轮调整为可配置 sliding-window 策略，P0 可保留 recent turns 简化，但要有 ratio 扩展点。
8. `HISTORY_TRIMMED` payload 应记录 compaction mode、trigger reason、component usage、from/to sequence、estimated before/after。
9. 恢复逻辑必须明确 latest summary 边界，避免重复恢复旧事件。
10. 测试必须覆盖 tool call/result 配对保护、placeholder、summary 恢复、预算 trace、raw_ref 和 Headroom 降级路径。

## 16. 最终建议

推荐路线：

```text
不直接引入外部 context/memory 框架；
以 LangChain/LangGraph + Letta + Anthropic 的会话级治理做法为主要参考；
用 Headroom 作为大工具输出的内容级压缩 backend；
在项目内实现一个轻量 ContextManager / Reducer 层；
保留事件流事实源；
把进入模型的内容视为可压缩、可追踪、可审计的上下文视图。
```

这样既符合“成熟框架优先”的原则，也不会牺牲本项目最重要的工程价值：

- Java 兼容 trace。
- 多租户上下文。
- ToolGateway 治理。
- 长期记忆边界。
- RAG/日志/告警证据链。
- 可本地 deterministic 测试。
