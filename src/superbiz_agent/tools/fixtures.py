from __future__ import annotations

from copy import deepcopy
from typing import Any


AVAILABLE_LOG_REGIONS = [
    "ap-guangzhou",
    "ap-shanghai",
    "ap-beijing",
    "ap-chengdu",
]
DEFAULT_LOG_REGION = "ap-guangzhou"


LOG_TOPICS: list[dict[str, Any]] = [
    {
        "topicName": "system-metrics",
        "description": "系统指标日志，包含 CPU、内存、磁盘使用率等系统资源监控数据",
        "exampleQueries": [
            "cpu_usage:>80",
            "memory_usage:>85",
            "disk_usage:>90",
            "level:WARN AND service:payment-service",
        ],
        "relatedAlerts": ["HighCPUUsage", "HighMemoryUsage", "HighDiskUsage"],
    },
    {
        "topicName": "application-logs",
        "description": "应用日志，包含错误日志、警告日志、慢请求日志、下游依赖调用日志等",
        "exampleQueries": [
            "level:ERROR",
            "level:FATAL",
            "http_status:500",
            "response_time:>3000",
            "slow",
            "downstream OR redis OR database OR mq",
        ],
        "relatedAlerts": ["ServiceUnavailable", "SlowResponse", "HighMemoryUsage"],
    },
    {
        "topicName": "database-slow-query",
        "description": "数据库慢查询日志，包含执行时间较长的 SQL 查询",
        "exampleQueries": [
            "query_time:>2",
            "table:orders",
            "query_type:SELECT",
            "*",
        ],
        "relatedAlerts": ["SlowResponse", "ServiceUnavailable"],
    },
    {
        "topicName": "system-events",
        "description": "系统事件日志，包含 Kubernetes Pod 重启、OOM Kill、容器崩溃等事件",
        "exampleQueries": [
            "restart OR crash",
            "oom_kill",
            "event_type:PodRestart",
            "reason:OOMKilled",
        ],
        "relatedAlerts": ["ServiceUnavailable", "HighMemoryUsage"],
    },
]


PROMETHEUS_ALERTS: list[dict[str, Any]] = [
    {
        "alertName": "HighCPUUsage",
        "description": (
            "服务 payment-service 的 CPU 使用率持续超过 80%，当前值为 92%。"
            "实例: pod-payment-service-7d8f9c6b5-x2k4m，命名空间: production"
        ),
        "state": "firing",
        "activeAt": "2026-01-02T02:05:00Z",
        "duration": "25m0s",
    },
    {
        "alertName": "HighMemoryUsage",
        "description": (
            "服务 order-service 的内存使用率持续超过 85%，当前值为 91%。"
            "JVM 堆内存使用: 3.8GB/4GB，可能存在内存泄漏风险。"
            "实例: pod-order-service-5c7d8e9f1-m3n2p，命名空间: production"
        ),
        "state": "firing",
        "activeAt": "2026-01-02T02:15:00Z",
        "duration": "15m0s",
    },
    {
        "alertName": "SlowResponse",
        "description": (
            "服务 user-service 的 P99 响应时间持续超过 3 秒，当前值为 4.2 秒。"
            "受影响接口: /api/v1/users/profile, /api/v1/users/orders。"
            "可能原因：数据库慢查询或下游服务延迟"
        ),
        "state": "firing",
        "activeAt": "2026-01-02T02:20:00Z",
        "duration": "10m0s",
    },
]


def get_available_log_topics_result() -> dict[str, Any]:
    topics = deepcopy(LOG_TOPICS)
    return {
        "success": True,
        "topics": topics,
        "availableRegions": list(AVAILABLE_LOG_REGIONS),
        "defaultRegion": DEFAULT_LOG_REGION,
        "message": (
            f"共有 {len(topics)} 个可用的日志主题。建议使用默认地域 "
            f"'{DEFAULT_LOG_REGION}' 或省略 region 参数"
        ),
    }


def query_prometheus_alerts_result() -> dict[str, Any]:
    alerts = deepcopy(PROMETHEUS_ALERTS)
    return {
        "success": True,
        "alerts": alerts,
        "message": f"成功检索到 {len(alerts)} 个活动告警",
    }


def query_logs_result(
    *,
    region: str,
    log_topic: str,
    query: str | None,
    limit: int,
) -> dict[str, Any]:
    safe_query = (query or "").strip()
    logs = _select_logs(log_topic, safe_query)[:limit]
    return {
        "success": True,
        "region": region,
        "logTopic": log_topic,
        "query": safe_query or "DEFAULT_QUERY",
        "logs": logs,
        "total": len(logs),
        "message": f"成功查询到 {len(logs)} 条日志",
    }


def query_internal_docs_result(query: str) -> dict[str, Any]:
    normalized = query.lower()
    chunks: list[dict[str, Any]]
    if _contains_any(normalized, ["pod", "重启", "oom", "容器崩溃"]):
        chunks = _pod_restart_chunks()
    elif _contains_any(normalized, ["慢查询", "slow query", "数据库"]):
        chunks = _database_chunks()
    elif _contains_any(normalized, ["告警", "alert", "处理手册"]):
        chunks = _alert_chunks()
    elif _contains_any(normalized, ["流程", "最佳实践", "操作步骤", "排查指南"]):
        chunks = _operations_chunks()
    else:
        return {
            "status": "no_results",
            "message": "No relevant documents found in the knowledge base.",
        }

    return {
        "status": "ok",
        "count": len(chunks),
        "chunks": chunks,
    }


def _select_logs(log_topic: str, query: str) -> list[dict[str, Any]]:
    topic = log_topic.lower()
    normalized_query = query.lower()

    if topic == "system-metrics":
        return _system_metric_logs(normalized_query)
    if topic == "application-logs":
        return _application_logs(normalized_query)
    if topic == "database-slow-query":
        return _database_slow_query_logs()
    if topic == "system-events":
        return _system_event_logs(normalized_query)
    return _generic_logs(log_topic, normalized_query)


def _system_metric_logs(query: str) -> list[dict[str, Any]]:
    if _contains_any(query, ["memory", "内存", ">85", "oom"]):
        return [
            _log(
                "2026-01-02 10:15:00",
                "WARN",
                "order-service",
                "pod-order-service-5c7d8e9f1-m3n2p",
                "内存使用率过高: 91.0%, JVM堆内存: 3.8GB/4GB, GC次数: 128",
                {
                    "memory_usage": "91.0",
                    "jvm_heap_used": "3.8GB",
                    "jvm_heap_max": "4GB",
                    "gc_count": "128",
                },
            ),
            _log(
                "2026-01-02 10:07:00",
                "WARN",
                "order-service",
                "pod-order-service-5c7d8e9f1-m3n2p",
                "频繁 Full GC 警告: 过去10分钟内发生 15 次 Full GC",
                {"full_gc_count": "15", "avg_gc_time_ms": "850"},
            ),
        ]

    return [
        _log(
            "2026-01-02 10:25:00",
            "WARN",
            "payment-service",
            "pod-payment-service-7d8f9c6b5-x2k4m",
            "CPU使用率过高: 92.0%, 进程: java (PID: 1), 线程数: 245",
            {
                "cpu_usage": "92.0",
                "cpu_cores": "4",
                "load_average_1m": "3.82",
                "top_process": "java",
            },
        ),
        _log(
            "2026-01-02 10:23:00",
            "WARN",
            "payment-service",
            "pod-payment-service-7d8f9c6b5-x2k4m",
            "CPU使用率过高: 90.5%, 进程: java (PID: 1), 线程数: 241",
            {"cpu_usage": "90.5", "cpu_cores": "4", "load_average_1m": "3.74"},
        ),
    ]


def _application_logs(query: str) -> list[dict[str, Any]]:
    if _contains_any(query, ["slow", "response_time", ">3000", "慢"]):
        return [
            _log(
                "2026-01-02 10:29:00",
                "WARN",
                "user-service",
                "pod-user-service-8e9f0a1b2-k5j6h",
                "慢请求警告: /api/v1/users/profile, 响应时间: 4200ms, 阈值: 3000ms",
                {
                    "uri": "/api/v1/users/profile",
                    "response_time_ms": "4200",
                    "db_time_ms": "3800",
                },
            ),
            _log(
                "2026-01-02 10:27:00",
                "WARN",
                "user-service",
                "pod-user-service-8e9f0a1b2-k5j6h",
                "慢请求警告: /api/v1/users/orders, 响应时间: 4050ms, 阈值: 3000ms",
                {
                    "uri": "/api/v1/users/orders",
                    "response_time_ms": "4050",
                    "db_time_ms": "3650",
                },
            ),
        ]

    if _contains_any(query, ["downstream", "redis", "database", "mq"]):
        return [
            _log(
                "2026-01-02 10:23:00",
                "ERROR",
                "payment-service",
                "pod-payment-service-7d8f9c6b5-x2k4m",
                "Redis 连接超时: 无法连接到 Redis 集群, 节点: redis-cluster-01:6379",
                {"dependency": "redis", "timeout_ms": "3000", "retry_count": "3"},
            ),
            _log(
                "2026-01-02 10:21:00",
                "WARN",
                "order-service",
                "pod-order-service-5c7d8e9f1-m3n2p",
                "消息队列积压警告: 队列 order-process-queue 积压消息数: 15823",
                {"dependency": "rabbitmq", "queue": "order-process-queue"},
            ),
        ]

    return [
        _log(
            "2026-01-02 10:25:00",
            "ERROR",
            "order-service",
            "pod-order-service-5c7d8e9f1-m3n2p",
            "数据库连接池耗尽: active: 50/50, waiting: 23, timeout: 30000ms",
            {
                "error_type": "ConnectionPoolExhaustedException",
                "pool_active": "50",
                "waiting_threads": "23",
            },
        ),
        _log(
            "2026-01-02 10:18:00",
            "FATAL",
            "order-service",
            "pod-order-service-5c7d8e9f1-m3n2p",
            "java.lang.OutOfMemoryError: Java heap space at OrderService.processLargeOrder",
            {"error_type": "OutOfMemoryError", "heap_used": "3.9GB"},
        ),
        _log(
            "2026-01-02 10:16:00",
            "ERROR",
            "user-service",
            "pod-user-service-8e9f0a1b2-k5j6h",
            "HTTP 500 Internal Server Error: /api/v1/users/profile, 耗时: 5200ms",
            {"http_status": "500", "duration_ms": "5200"},
        ),
    ]


def _database_slow_query_logs() -> list[dict[str, Any]]:
    return [
        _log(
            "2026-01-02 10:27:00",
            "WARN",
            "mysql",
            "mysql-primary-01",
            "慢查询: SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC",
            {
                "query_time_sec": "3.2",
                "rows_examined": "1245678",
                "table": "orders",
                "query_type": "SELECT",
            },
        ),
        _log(
            "2026-01-02 10:24:00",
            "WARN",
            "mysql",
            "mysql-primary-01",
            "慢查询: UPDATE orders SET status = ? WHERE created_at < ?",
            {
                "query_time_sec": "4.5",
                "lock_time_sec": "2.1",
                "table": "orders",
                "query_type": "UPDATE",
            },
        ),
    ]


def _system_event_logs(query: str) -> list[dict[str, Any]]:
    if _contains_any(query, ["restart", "crash", "oom", "重启"]):
        return [
            _log(
                "2026-01-02 10:15:00",
                "WARN",
                "kubernetes",
                "kube-controller-manager",
                "Pod 重启事件: pod-order-service-5c7d8e9f1-m3n2p, 原因: OOMKilled",
                {
                    "event_type": "PodRestart",
                    "reason": "OOMKilled",
                    "restart_count": "3",
                    "namespace": "production",
                },
            ),
            _log(
                "2026-01-02 10:14:00",
                "ERROR",
                "kernel",
                "node-worker-02",
                "OOM Killer 触发: 进程 java (PID: 12345) 被杀死",
                {"event_type": "OOMKill", "memory_used": "3.9GB", "memory_limit": "4GB"},
            ),
        ]

    return _generic_logs("system-events", query)


def _generic_logs(log_topic: str, query: str) -> list[dict[str, Any]]:
    return [
        _log(
            "2026-01-02 10:30:00",
            "INFO",
            "generic-service",
            "instance-0",
            f"日志消息 #0, topic: {log_topic}, 查询条件: {query or 'DEFAULT_QUERY'}",
            {},
        ),
        _log(
            "2026-01-02 10:29:00",
            "WARN",
            "generic-service",
            "instance-1",
            f"日志消息 #1, topic: {log_topic}, 查询条件: {query or 'DEFAULT_QUERY'}",
            {},
        ),
    ]


def _pod_restart_chunks() -> list[dict[str, Any]]:
    return [
        _chunk(
            "doc-pod-restart-1",
            1,
            "runbooks/pod-restart.md",
            "确认 Pod 重启次数、lastState、exitCode 和 reason，优先区分 OOMKilled 与探针失败。",
            0.91,
        ),
        _chunk(
            "doc-pod-restart-2",
            2,
            "runbooks/oom-kill.md",
            "若 reason 为 OOMKilled，检查容器 memory limit、JVM heap、Full GC 和近期流量峰值。",
            0.88,
        ),
        _chunk(
            "doc-pod-restart-3",
            3,
            "runbooks/kubernetes-events.md",
            "结合 system-events 日志与 kube event，确认节点驱逐、镜像拉取失败或依赖超时。",
            0.83,
        ),
    ]


def _database_chunks() -> list[dict[str, Any]]:
    return [
        _chunk(
            "doc-db-slow-1",
            1,
            "runbooks/database-slow-query.md",
            "先定位慢 SQL、执行计划和 rows_examined，再确认是否缺少联合索引或存在锁等待。",
            0.89,
        ),
        _chunk(
            "doc-db-slow-2",
            2,
            "best-practices/mysql-indexing.md",
            "高频查询应避免深分页和无选择性条件，优先使用覆盖索引与稳定排序键。",
            0.82,
        ),
    ]


def _alert_chunks() -> list[dict[str, Any]]:
    return [
        _chunk(
            "doc-alert-1",
            1,
            "runbooks/alert-triage.md",
            "告警处理先确认影响面、持续时间、最近发布和相关日志，再决定升级或回滚。",
            0.87,
        ),
        _chunk(
            "doc-alert-2",
            2,
            "runbooks/high-cpu.md",
            "HighCPUUsage 需要检查实例维度 CPU、线程栈、GC 和热点接口。",
            0.8,
        ),
    ]


def _operations_chunks() -> list[dict[str, Any]]:
    return [
        _chunk(
            "doc-ops-1",
            1,
            "best-practices/incident-response.md",
            "排查指南建议按告警、日志、指标、变更四条线并行收敛，记录每一步证据。",
            0.84,
        ),
        _chunk(
            "doc-ops-2",
            2,
            "runbooks/service-degradation.md",
            "服务降级操作步骤包括冻结发布、确认错误率、切换限流策略和通知值班负责人。",
            0.79,
        ),
    ]


def _log(
    timestamp: str,
    level: str,
    service: str,
    instance: str,
    message: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "level": level,
        "service": service,
        "instance": instance,
        "message": message,
        "fields": fields,
    }


def _chunk(
    chunk_id: str,
    ref: int,
    source: str,
    content: str,
    score: float,
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "ref": ref,
        "source": source,
        "content": content,
        "score": score,
    }


def _contains_any(value: str, keywords: list[str]) -> bool:
    return any(keyword in value for keyword in keywords)
