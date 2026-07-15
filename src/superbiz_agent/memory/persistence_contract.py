from __future__ import annotations


ACTIVE_EXACT_INDEX = "uq_long_term_memory_active_exact"
ACTIVE_CONTENT_HASH_CHECK = "chk_long_term_memory_active_content_hash"
SCOPE_SERVICE_NONBLANK_CHECK = "chk_long_term_memory_scope_service_nonblank"
SCOPE_ENV_NONBLANK_CHECK = "chk_long_term_memory_scope_env_nonblank"
TAGS_ARRAY_CHECK = "chk_long_term_memory_tags_array"
CORE_BLOCK_KEY_CHECK = "chk_core_memory_block_key"
CORE_VERSION_POSITIVE_CHECK = "chk_core_memory_version_positive"
CORE_MAX_TOKENS_POSITIVE_CHECK = "chk_core_memory_max_tokens_positive"
CORE_CONTENT_HASH_FORMAT_CHECK = "chk_core_memory_content_hash_format"
CORE_UNIQUE_CONSTRAINT = "uq_core_memory_block"

ACTIVE_EXACT_PREDICATE_SQL = "status = 'active'"

M_P2_REQUIRED_CHECKS = frozenset(
    {
        ACTIVE_CONTENT_HASH_CHECK,
        SCOPE_SERVICE_NONBLANK_CHECK,
        SCOPE_ENV_NONBLANK_CHECK,
        TAGS_ARRAY_CHECK,
        CORE_BLOCK_KEY_CHECK,
        CORE_VERSION_POSITIVE_CHECK,
        CORE_MAX_TOKENS_POSITIVE_CHECK,
        CORE_CONTENT_HASH_FORMAT_CHECK,
    }
)
