from __future__ import annotations


class MemoryPersistenceError(RuntimeError):
    """Safe persistence error whose message may be exposed through ToolGateway."""

    code = "memory_store_error"
    status_code = 500
    retryable = False
    retryable_by_model = False
    allowed_next_actions = ("degrade_with_user_friendly_error",)

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MemoryStoreUnavailableError(MemoryPersistenceError):
    code = "memory_store_unavailable"
    status_code = 503
    retryable = True
    retryable_by_model = False

    def __init__(self) -> None:
        super().__init__("Long-term memory storage is temporarily unavailable.")


class MemoryStoreContractError(MemoryPersistenceError):
    code = "memory_store_contract_error"

    def __init__(self) -> None:
        super().__init__("Long-term memory storage does not satisfy the required schema contract.")


class MemoryStoreIsolationError(MemoryPersistenceError):
    code = "memory_store_isolation_error"

    def __init__(self) -> None:
        super().__init__("Long-term memory identity scope validation failed.")


class MemoryExactConflictUnresolvedError(MemoryPersistenceError):
    code = "memory_exact_conflict_unresolved"

    def __init__(self) -> None:
        super().__init__("The exact memory write conflict could not be resolved safely.")


class CoreMemoryContractError(MemoryPersistenceError):
    code = "core_memory_contract_error"

    def __init__(self) -> None:
        super().__init__("Core memory blocks do not satisfy the required contract.")

