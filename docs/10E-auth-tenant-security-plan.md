# 10E 生产级多租户鉴权与安全策略实施计划

> 状态：已完成 P0 实现并通过主验收。本文同时保留 10E 的边界、设计和验收标准，用于后续回归和审计。

## 1. 阶段定位

10E 是第 10 步高级能力中的安全与生产级多租户子阶段。

本阶段要解决的问题不是“给表加一个 tenant_id 字段”，而是把当前开发态链路：

```text
前端 / 测试请求直接传 X-Tenant-Id、X-User-Id、X-Agent-Id
```

升级为生产态可信链路：

```text
认证入口
  -> 后端解析可信身份
  -> 生成 AuthenticatedPrincipal
  -> 生成 AgentRequestContext / TenantContext
  -> API / ToolGateway / Memory / RAG / Trace 全链路按权限和 tenant 访问
```

核心原则：

```text
模型只生成业务参数；
tenant_id、user_id、agent_id、session_id、run_id、tool_call_id、raw_ref 等运行时安全上下文由后端生成或注入；
工具、记忆、RAG、trace 查询都不能相信模型或前端随意传入的安全上下文字段。
```

## 2. 当前基线

当前已具备：

- `AgentRequestContext` 已包含：
  - `tenant_id`
  - `user_id`
  - `agent_id`
  - `session_id`
  - `request_id`
  - `roles`
  - `permissions`
- API 当前通过 `resolve_request_context()` 从 headers 读取：
  - `X-Tenant-Id`
  - `X-User-Id`
  - `X-Agent-Id`
  - `X-Request-Id`
- `AgentRequestContext.__post_init__()` 会给缺失字段填默认值。
- `agent_rollout_event` 查询已按 `tenant_id/user_id/agent_id/session_id` 过滤。
- `list_by_run` 已要求 `tenant_id + run_id`。
- Memory store 和 memory tools 已按 `tenant_id/user_id/agent_id` 隔离。
- ToolGateway 已有：
  - tool policy allow/deny
  - 参数校验
  - 错误分类
  - retry / timeout
  - trace 脱敏
- Memory 写入策略已拒绝 secret 和 raw dump。
- 10D 已生成 `raw_ref`，但还没有实现受控原文取回工具。

当前不足：

- headers 仍是开发态 trusted headers，生产环境不能直接相信。
- 没有统一的 `AuthenticatedPrincipal` / `AuthContextResolver`。
- 没有 `auth_mode` 区分 local/dev/prod。
- 没有 API 层权限校验，例如 chat、clear session、session read。
- 没有统一权限命名和权限判定函数。
- ToolGateway 还没有按用户权限限制工具调用。
- Memory tools 主要依赖 context 存在和内容 policy，还没有显式检查 `memory:write` / `memory:read` 权限。
- `raw_ref` 只有引用格式，还没有定义读取时的校验规则。
- Secret 管理仍主要依赖环境变量和局部脱敏，没有统一 secret redaction contract。
- PostgreSQL RLS 未启用。

## 3. 10E 目标

审核后收敛：

```text
10E P0 必做 dev_headers + trusted_gateway；
JWT 只定义 claims contract 和扩展口，不作为 P0 必须验签实现。
```

原因：

- 当前项目没有 JWT 依赖，也没有真实 SSO/JWKS 环境。
- 直接把 dev headers、trusted gateway、JWT 三条链路都做成生产能力，会让 P0 范围过大。
- trusted gateway 更符合当前工程阶段：由上游网关完成认证，本服务只校验可信来源并解析身份。
- JWT 可在 10E 后续小阶段实现，不阻塞本阶段把 tenant context 可信化。

P0 目标：

1. 明确认证入口，P0 实现 dev headers 和 trusted gateway headers，预留 JWT claims contract。
2. 新增统一身份对象：`AuthenticatedPrincipal`。
3. 新增统一认证解析层：`AuthContextResolver`。
4. 由后端认证上下文生成 `AgentRequestContext`，而不是在生产环境直接信任业务 headers。
5. 新增权限模型和权限校验：
   - `chat:invoke`
   - `session:read`
   - `session:clear`
   - `memory:read`
   - `memory:write`
   - `tool:execute`
   - `tool:execute:{tool_name}`
   - `raw_ref:read`
6. API 层做权限校验：
   - `/api/chat`
   - `/api/chat_stream`
   - `/api/chat/clear`
   - `/api/chat/session/{session_id}`
7. ToolGateway 执行前检查工具权限。
8. Memory tools 执行前能依赖 `run_context.request_context.permissions` 做读写权限判断。
9. 明确 raw_ref 读取安全规则，即使 P0 不实现取回工具，也要固定 contract。
10. 补齐安全测试，验证生产环境下缺少可信认证会拒绝请求。

## 4. 不做清单

10E P0 不做：

- 不自研完整 IAM 系统。
- 不做用户、租户、角色管理后台。
- 不接真实企业 SSO 页面。
- 不把 JWT 验签作为 P0 必做项；只保留接口和 claims 约定。
- 不强制启用 PostgreSQL RLS。
- 不实现完整 secret manager，只定义接口和使用边界。
- 不实现 RAG / Milvus 真实多租户管线，RAG 真实隔离进入 10F。
- 不实现完整 raw_ref 原文取回工具，除非作为很小的安全 contract 测试桩。
- 不把 `tenant_id/user_id/run_id/raw_ref` 暴露成模型可自由填写的工具参数。
- 不改变 Java 兼容 API response 外层结构，除非认证失败必须返回 HTTP 401/403。

## 5. 认证模式设计

建议新增配置：

```text
auth_mode = dev_headers | trusted_gateway | jwt
auth_dev_headers_enabled = true
auth_trusted_gateway_secret = optional
auth_permission_enforcement = true
auth_jwt_issuer = optional
auth_jwt_audience = optional
auth_jwt_secret = optional
auth_jwt_jwks_url = optional
```

P0 配置边界：

- P0 可实际启用的运行模式只有 `dev_headers` 和 `trusted_gateway`。
- `jwt` 只作为配置枚举、claims contract 和后续扩展口保留；如果 P0 阶段配置为 `auth_mode=jwt`，应明确返回“JWT auth not implemented in P0”的认证错误，而不是静默放行。
- `auth_mode=dev_headers` 只允许在 `local/dev/test` 环境使用。
- `auth_permission_enforcement=false` 只允许在 `local/test` 环境用于兼容老测试或临时排障，生产环境启动或请求时必须拒绝。
- `trusted_gateway` 模式必须配置 `auth_trusted_gateway_secret`，否则不能进入生产请求处理。

### 5.1 dev_headers

仅允许 `app_env in ('local', 'dev', 'test')` 使用。

读取：

```text
X-Tenant-Id
X-User-Id
X-Agent-Id
X-Request-Id
X-Roles
X-Permissions
```

用途：

- 本地开发。
- 单元测试。
- eval runner。

生产环境如果 `auth_mode=dev_headers`，启动或请求时必须拒绝。

实现约束：

- dev headers 默认权限只能由 `AuthContextResolver` 在 `local/dev/test` 模式下补齐。
- 不要在 `AgentRequestContext.__post_init__()` 里无条件授予默认权限，否则生产路径一旦绕过 resolver 会被误放行。
- 现有直接构造 `AgentRequestContext` 的单元测试，需要改成显式传 local permissions，或使用测试 helper 创建 dev context。

### 5.2 trusted_gateway

适用于 API Gateway / Ingress / 内部网关已经完成认证的场景。

读取可信 headers，例如：

```text
X-Auth-Tenant-Id
X-Auth-User-Id
X-Auth-Agent-Id
X-Auth-Roles
X-Auth-Permissions
X-Auth-Gateway-Secret
```

要求：

- 只有来自可信网关的请求才允许使用。
- P0 校验共享 secret，后续可升级为签名。
- `auth_trusted_gateway_secret` 为空时不能启用该模式。
- secret 比较必须使用常量时间比较。
- 网关 secret 不进入 prompt、工具返回或 trace。
- 不能读取普通 `X-Tenant-Id` 作为生产身份来源。
- 如果同一个请求同时带普通 `X-Tenant-Id` 和可信 `X-Auth-Tenant-Id`，只使用可信身份；普通 header 不能覆盖 trusted identity。

### 5.3 jwt

适用于 bearer token 模式。

从 JWT claims 解析：

```text
tenant_id
sub / user_id
agent_id
roles
permissions
```

P0 只定义 claims contract 和扩展口，不强制实现 JWT 验签。

后续实现 JWT 时再选择：

- HS256 测试模式。
- RS256 + JWKS 生产模式。
- 与企业 SSO / API Gateway 的 claims 映射。

## 6. 核心对象设计

建议新增：

```text
src/superbiz_agent/security/auth.py
src/superbiz_agent/security/permissions.py
src/superbiz_agent/security/redaction.py
tests/test_auth_tenant_security.py
```

### 6.1 AuthenticatedPrincipal

字段：

```text
tenant_id
user_id
agent_id
roles
permissions
auth_mode
subject
issuer
request_id
```

约束：

- `tenant_id/user_id/agent_id` 必须非空。
- 生产环境不能落到默认 tenant/user。
- roles / permissions 必须来自可信认证上下文或测试 fixture。
- `auth_mode` 必须记录身份来源，便于 trace 和排查。

### 6.2 AuthContextResolver

职责：

```text
Request
  -> 按 auth_mode 解析认证信息
  -> 校验签名 / token / 环境限制
  -> 返回 AuthenticatedPrincipal
```

错误：

- 缺少认证：401。
- 认证无效：401。
- 权限不足：403。
- dev headers 出现在 prod：403 或启动失败。

### 6.3 PermissionChecker

接口：

```python
def require_permission(context: AgentRequestContext, permission: str) -> None:
    ...
```

规则：

- `admin:*` 可覆盖所有权限。
- `tool:execute` 可执行普通工具。
- `tool:execute:{tool_name}` 可执行指定工具。
- `memory:write` 才能执行 `updateCoreMemory`、`saveArchivalMemory`。
- `memory:read` 才能执行 `searchMemory`、`listMemoryTopics`。
- `raw_ref:read` 只表示具备读取受控原文引用的资格；真正读取时仍必须校验 raw_ref 归属当前 tenant/user/agent/session。

兼容性规则：

- 权限校验必须是显式的，不能因为 permissions 为空就默认允许。
- local/dev 的默认权限由 resolver 或测试 helper 注入。
- 直接绕过 resolver 构造的 context 如果没有权限，应在需要权限的入口被拒绝；测试需要显式声明权限。
- 即使提供 `auth_permission_enforcement=false`，也只能在 `local/test` 生效；生产环境不能通过该开关关闭 API 或 ToolGateway 权限检查。

## 7. 请求上下文流转

目标流转：

```text
FastAPI request
  -> AuthContextResolver.resolve(request)
  -> AuthenticatedPrincipal
  -> AgentRequestContext(
       tenant_id=principal.tenant_id,
       user_id=principal.user_id,
       agent_id=principal.agent_id,
       session_id=payload.Id 或 path session_id,
       request_id=principal.request_id,
       roles=principal.roles,
       permissions=principal.permissions
     )
  -> AgentHarnessService
  -> ConversationRuntime
  -> ToolGateway / Memory / Trace / RAG
```

注意：

- `session_id` 可以来自请求体或 path，但必须只在当前 `tenant_id/user_id/agent_id` 下有效。
- `tenant_id/user_id/agent_id` 不来自模型，也不来自生产态普通业务 headers。
- `run_id` 继续由 `ConversationRuntime` 后端生成。
- `tool_call_id` 由模型生成的 tool call id 可以作为 LLM 协议 id，但 trace/权限不能只信它；后端仍以 run context + tool call event 绑定。

## 8. API 权限矩阵

| API | 权限 | 说明 |
|---|---|---|
| `POST /api/chat` | `chat:invoke` | 发起普通 Agent run |
| `POST /api/chat_stream` | `chat:invoke` | 发起流式 Agent run |
| `POST /api/chat/clear` | `session:clear` | 清空当前用户当前 agent 的指定会话 |
| `GET /api/chat/session/{session_id}` | `session:read` | 读取当前会话信息 |

P0 推荐默认权限：

```text
dev_headers 模式：如果未显式传 X-Permissions，则给 local 默认权限集合。
trusted_gateway 模式：必须来自可信网关认证信息，不能自动补全高权限。
后续 jwt 模式：必须来自已验签 token claims，不能自动补全高权限。
```

local 默认权限：

```text
chat:invoke
session:read
session:clear
memory:read
memory:write
tool:execute
raw_ref:read
```

## 9. 工具权限策略

ToolGateway 执行前新增权限检查：

```text
未知工具 -> TOOL_BLOCKED
policy deny -> TOOL_BLOCKED
权限不足 -> TOOL_BLOCKED / permission_denied
参数校验 -> PARAM_VALIDATION_FAILED
handler 执行
```

权限检查推荐集中在 ToolGateway：

```text
ToolGateway
  -> resolve required permission by tool name
  -> require_permission(run_context.request_context, permission)
  -> pass only business args to handler
```

不要在每个 tool handler 中重复实现租户和权限判断。handler 只从 `RunContext.request_context` 读取后端注入的 tenant/user/agent。

工具权限规则：

| 工具 | 权限 |
|---|---|
| `getCurrentDateTime` | `tool:execute` 或 `tool:execute:getCurrentDateTime` |
| `queryPrometheusAlerts` | `tool:execute` 或 `tool:execute:queryPrometheusAlerts` |
| `getAvailableLogTopics` | `tool:execute` 或 `tool:execute:getAvailableLogTopics` |
| `queryLogs` | `tool:execute` 或 `tool:execute:queryLogs` |
| `queryInternalDocs` | `tool:execute` 或 `tool:execute:queryInternalDocs` |
| `searchMemory` | `memory:read` |
| `listMemoryTopics` | `memory:read` |
| `updateCoreMemory` | `memory:write` |
| `saveArchivalMemory` | `memory:write` |

注意：

- 模型不能通过工具参数传 `tenant_id/user_id` 来绕过权限。
- 工具 handler 如果需要 tenant 信息，只能从 `RunContext.request_context` 取。
- 权限不足应返回结构化工具错误，让模型换路径或友好降级。

## 10. 数据隔离策略

### 10.1 Rollout Event / Trace

当前已经较好：

- insert 要求 tenant/user/agent/session 非空。
- session 查询按 tenant/user/agent/session 过滤。
- run 查询按 tenant/run 过滤。
- clear 按 tenant/user/agent/session 删除。

10E 补强：

- API 层保证 context 来自认证。
- 测试生产模式下不能伪造普通 `X-Tenant-Id` 读取他人 session。
- `list_by_run` 后续如暴露 API，应校验 run 属于当前 tenant，必要时再加 user/agent。

### 10.2 Memory

当前已按 tenant/user/agent 隔离。

10E 补强：

- memory tools 调用前检查 `memory:read` / `memory:write`。
- memory policy 继续拒绝 secret/raw dump。
- Memory context 注入只组装当前 authenticated tenant/user/agent 的 core memory 和 memory index。
- 权限检查优先放在 ToolGateway 的工具权限映射中，MemoryToolHandlers 保持业务逻辑简单。

### 10.3 RAG

当前 `queryInternalDocs` 仍是 fixture/tool contract，不是真实 Milvus 管线。

10E 只定义安全 contract：

- RAG 检索必须从后端 context 注入 `tenant_id`。
- 模型不能传 `tenant_id`。
- 真实 RAG / Milvus tenant filter 在 10F 实现。

### 10.4 raw_ref

当前 `raw_ref` 形式类似：

```text
rollout://tenant/{tenant_id}/session/{session_id}/run/{run_id}/tool_call/{tool_call_id}/sequence/{sequence}
```

读取规则：

- 模型可以把 `raw_ref` 作为受控引用传给后端工具。
- 后端必须解析并校验：
  - raw_ref tenant == 当前 context tenant
  - session/run/tool_call/sequence 存在
  - event 属于当前 tenant，必要时属于当前 user/agent/session
  - 当前用户有 `raw_ref:read`
  - 原文读取前再次脱敏
- 不允许模型拼接任意 DB key 直接读数据。

P0 可不实现取回工具，但必须把上述规则写入文档和测试计划。

验收口径：

- P0 如果不实现原文取回工具，则测试“没有任何 API 或工具可以仅凭 raw_ref 任意读取原文”。
- 后续实现取回工具时，必须先满足上述校验规则。

## 11. Secret 与脱敏策略

原则：

- `model_api_key` 只来自环境变量或后续 secret manager。
- API key / token / password 不进入 prompt。
- API key / token / password 不进入工具返回。
- API key / token / password 不进入 trace artifact。
- 错误栈、文件路径、authorization header 需要脱敏。

当前已有：

- Tool error translator 会脱敏异常文本。
- Memory policy 拒绝 secret。
- ToolResultReducer 会脱敏工具结果。

10E 补强：

- 抽出公共 `redaction.py`，统一 secret/path/authorization 脱敏。
- Tool error 和 ToolResultReducer 在 P0 优先复用同一套 redactor。
- Memory policy 可保留当前独立实现；如果改动过大，后续再逐步迁移，不能为了抽 redactor 破坏现有 memory 测试。
- 本阶段至少补一组认证和 trace secret 不泄露测试。

## 12. 建议实现步骤

### Step 1：配置和权限常量

新增配置：

```text
auth_mode
auth_dev_headers_enabled
auth_trusted_gateway_secret
auth_permission_enforcement
auth_jwt_issuer
auth_jwt_audience
auth_jwt_secret
auth_jwt_jwks_url
```

新增权限常量和 checker。

### Step 2：认证解析层

新增：

```text
AuthenticatedPrincipal
AuthContextResolver
AuthError
PermissionDenied
```

实现：

- dev headers resolver。
- trusted gateway headers resolver。
- JWT resolver 只保留接口和 claims contract，P0 不要求可用验签；如果被选中，应返回明确的未实现认证错误。

### Step 3：API 接入

修改 `resolve_request_context()`：

- local/dev 仍兼容旧 headers。
- production 根据 `auth_mode` 解析可信身份。
- production 下禁止 `auth_mode=dev_headers`。
- production 下禁止 `auth_permission_enforcement=false`。
- API endpoint 调用 `require_permission()`。
- 认证失败使用 HTTP 401。
- 权限不足使用 HTTP 403。
- 普通业务错误继续保持现有 Java 兼容 `ApiResponse` 结构。

### Step 4：ToolGateway 权限接入

执行 handler 前检查：

```text
tool:execute
tool:execute:{tool_name}
memory:read / memory:write
```

权限不足返回结构化 `TOOL_BLOCKED`。

注意：

- `auth_permission_enforcement=false` 只允许 local/test 用于兼容老测试或排障，不允许 production 使用。
- 默认应保持 `auth_permission_enforcement=true`，测试需要通过 resolver/helper 注入权限。

### Step 5：安全测试

新增 `tests/test_auth_tenant_security.py`。

覆盖：

- local/dev headers 仍可用。
- prod 下 dev headers 不可信。
- prod 下 `auth_permission_enforcement=false` 被拒绝。
- `auth_mode=jwt` 在 P0 返回明确未实现认证错误。
- trusted gateway 缺少 secret 拒绝。
- trusted gateway 配置缺少 `auth_trusted_gateway_secret` 时拒绝。
- trusted gateway 有效时生成正确 context。
- 普通 header 不能覆盖 trusted identity。
- 直接构造无权限 context 调工具会被拒绝，local helper 注入权限后才允许。
- chat 权限不足拒绝。
- clear session 权限不足拒绝。
- memory write 权限不足时工具被 blocked。
- tool execute 权限不足时工具被 blocked。
- tenant A 不能清理 / 读取 tenant B 的 session。
- secret 不进入 trace。

## 13. 允许写入范围

允许：

```text
src/superbiz_agent/security/
src/superbiz_agent/security/__init__.py
src/superbiz_agent/api/routes_chat.py
src/superbiz_agent/api/app.py
src/superbiz_agent/harness/context.py
src/superbiz_agent/tools/gateway.py
src/superbiz_agent/tools/error_translator.py
src/superbiz_agent/tools/errors.py
src/superbiz_agent/config.py
tests/test_auth_tenant_security.py
tests/test_skeleton.py
tests/test_chat_stream.py
tests/test_tool_gateway_reliability.py
tests/conftest.py
docs/04-python-migration-roadmap.md
docs/10-advanced-capabilities-plan.md
```

原则上不写：

```text
src/superbiz_agent/rag/
src/superbiz_agent/memory/store.py
src/superbiz_agent/persistence/models.py
alembic/versions/
```

除非测试发现必须做极小适配。

## 14. 验收命令

```bash
python3 -m pytest
PYTHONPATH=src python3 -m superbiz_agent.evals.runner
PYTHONPATH=src python3 -m compileall -q src tests
```

如果安装了 ruff：

```bash
python3 -m ruff check src tests
```

## 15. 阶段通过标准

10E 通过标准：

1. local/dev 模式兼容现有测试 headers。
2. production 模式不信任普通 `X-Tenant-Id/X-User-Id/X-Agent-Id`。
3. 认证失败返回 401。
4. 权限不足返回 403 或结构化 tool blocked。
5. `AgentRequestContext` 的 tenant/user/roles/permissions 来自后端认证上下文。
6. API endpoint 有明确权限要求。
7. ToolGateway 有工具权限检查。
8. Memory tools 有读写权限边界。
9. 模型不能通过工具参数伪造 tenant/user/run/raw_ref；P0 不提供任意 raw_ref 读取入口。
10. trace 和工具错误不泄露 secret。
11. `auth_mode=jwt` 在 P0 不会静默放行。
12. production 环境不能启用 `dev_headers`，也不能关闭 `auth_permission_enforcement`。
13. 事件流和 memory 的现有 tenant 过滤测试不回退。
14. 全量 pytest、eval runner、compileall 通过。

## 16. 自审清单

| 检查项 | 结论 |
|---|---|
| 是否直接实现代码 | 否，本文只做计划 |
| 是否把多租户隔离等同于 tenant_id 字段 | 否 |
| 是否区分 dev headers 和生产认证 | 是 |
| 是否禁止模型生成 tenant/user/run/raw_ref | 是 |
| 是否避免自研完整 IAM | 是 |
| 是否避免提前实现真实 RAG 多租户管线 | 是 |
| 是否保留当前 Harness 主链路 | 是 |
| 是否定义 API 权限矩阵 | 是 |
| 是否定义 ToolGateway 权限边界 | 是 |
| 是否定义 raw_ref 读取安全规则 | 是 |
| 是否将 JWT 从 P0 必做中移出 | 是 |
| 是否避免空权限 context 在生产误放行 | 是 |
| 是否定义验收测试 | 是 |
