from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


RAG_TOP_K_MAX = 1000


class Settings(BaseSettings):
    app_name: str = "SuperBizAgent Python"
    app_env: str = "local"
    host: str = "0.0.0.0"
    port: int = 9900

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/super_biz_agent"
    rollout_store_backend: str = "memory"
    memory_enabled: bool = True
    memory_store_backend: str = "memory"
    memory_search_top_k: int = 3
    memory_search_min_similarity: float = 0.5
    # Compatibility-only: M-P0 archival writes no longer use embedding similarity.
    memory_duplicate_similarity: float = 0.92
    memory_embedding_dimension: int = 64
    memory_embedding_provider: str = "local-deterministic"
    memory_embedding_model: str = "local-deterministic"
    memory_embedding_version: Optional[str] = "phase4-local"
    memory_embedding_batch_size: int = Field(default=10, ge=1)
    memory_embedding_base_url: Optional[str] = None
    memory_embedding_api_key: Optional[str] = Field(default=None, repr=False)
    memory_embedding_timeout_ms: int = Field(default=30_000, ge=1)

    prompt_dir: Path = Path("prompts")
    prompt_version: str = "ops-agent-system-v3"
    tool_schema_version: str = "ops-tools-v3"

    model_provider: str = "stub"
    model_base_url: Optional[str] = None
    model_api_key: Optional[str] = Field(default=None, repr=False)
    model_name: str = "qwen-plus"
    model_timeout_ms: int = 180_000
    model_max_retries: int = 1
    agent_max_tool_rounds: int = Field(default=4, ge=1)
    agent_max_tool_calls_per_run: int = Field(default=8, ge=1)

    context_max_tokens: int = 32_000
    context_reserved_output_tokens: int = 4_000
    context_compaction_trigger_ratio: float = 0.85
    context_compaction_target_ratio: float = 0.65
    context_recent_turns_to_keep: int = 4
    context_recent_keep_ratio: float = 0.70
    context_tool_result_compress_threshold_tokens: int = 1_000
    context_content_compression_backend: str = "headroom"
    context_tool_results_to_keep: int = 3
    context_compaction_prompt_version: str = "context-compaction-v1"

    tenant_default: str = "default-tenant"
    user_default: str = "default-user"
    agent_default: str = "ops-agent"

    auth_mode: str = "dev_headers"
    auth_dev_headers_enabled: bool = True
    auth_trusted_gateway_secret: Optional[str] = None
    auth_permission_enforcement: bool = True
    auth_jwt_issuer: Optional[str] = None
    auth_jwt_audience: Optional[str] = None
    auth_jwt_secret: Optional[str] = None
    auth_jwt_jwks_url: Optional[str] = None

    milvus_enabled: bool = False
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    rag_enabled: bool = False
    rag_fixture_mode: bool | None = None
    rag_milvus_uri: str = "http://localhost:19530"
    rag_milvus_token: Optional[str] = Field(default=None, repr=False)
    rag_milvus_collection: str = "superbiz_rag_chunks"
    rag_embedding_provider: str = "dashscope-openai-compatible"
    rag_embedding_model: str = "text-embedding-v4"
    rag_embedding_version: Optional[str] = None
    rag_embedding_dimension: int = Field(default=1024, ge=1)
    rag_embedding_batch_size: int = Field(default=32, ge=1)
    rag_embedding_base_url: Optional[str] = None
    rag_embedding_api_key: Optional[str] = Field(default=None, repr=False)
    rag_chunk_size_tokens: int = Field(default=1024, ge=1)
    rag_chunk_overlap_tokens: int = Field(default=100, ge=0)
    rag_ingestion_stale_after_seconds: int = Field(default=900, ge=1)
    rag_hybrid_top_k: int = Field(default=10, ge=1, le=RAG_TOP_K_MAX)
    rag_final_top_k: int = Field(default=3, ge=1, le=RAG_TOP_K_MAX)
    rag_rerank_enabled: bool = False
    rag_rerank_provider: str = "dashscope"
    rag_rerank_model: str = "qwen3-rerank"
    rag_rerank_base_url: Optional[str] = None
    rag_rerank_api_key: Optional[str] = Field(default=None, repr=False)
    rag_external_parser_enabled: bool = False
    rag_external_parser_provider: Optional[str] = None
    rag_external_parser_api_base_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def resolve_prompt_dir(self, project_root: Optional[Path] = None) -> Path:
        if self.prompt_dir.is_absolute():
            return self.prompt_dir
        base = project_root or Path.cwd()
        return base / self.prompt_dir

    @model_validator(mode="after")
    def validate_memory_backend(self) -> "Settings":
        if self.memory_store_backend not in {"memory", "postgres"}:
            raise ValueError(
                "memory_store_backend must be 'memory' or 'postgres', "
                f"got {self.memory_store_backend!r}"
            )
        if self.memory_store_backend == "postgres" and not self.memory_enabled:
            raise ValueError(
                "memory_store_backend cannot be 'postgres' when memory_enabled is false"
            )
        provider = self.memory_embedding_provider.strip().lower()
        if provider not in {
            "local-deterministic",
            "openai-compatible",
            "dashscope-openai-compatible",
        }:
            raise ValueError("memory_embedding_provider is not supported")
        object.__setattr__(self, "memory_embedding_provider", provider)
        if not self.memory_embedding_model.strip():
            raise ValueError("memory_embedding_model must not be blank")
        if self.memory_embedding_version is None:
            object.__setattr__(self, "memory_embedding_version", self.memory_embedding_model)
        elif not self.memory_embedding_version.strip():
            raise ValueError("memory_embedding_version must not be blank")
        if provider == "local-deterministic":
            if (
                self.memory_store_backend == "postgres"
                and self.app_env.strip().lower() not in {"local", "test"}
            ):
                raise ValueError(
                    "local-deterministic memory embedding is not allowed for production postgres"
                )
        else:
            if self.memory_embedding_dimension != 1024:
                raise ValueError("real memory embedding dimension must be 1024")
            if provider == "dashscope-openai-compatible" and self.memory_embedding_batch_size > 10:
                raise ValueError("DashScope memory embedding batch size cannot exceed 10")
        return self

    @model_validator(mode="after")
    def validate_rag_settings(self) -> "Settings":
        app_env = self.app_env.strip().lower()
        if self.rag_fixture_mode is None:
            object.__setattr__(self, "rag_fixture_mode", app_env in {"local", "test"})
        if self.rag_enabled and self.rag_fixture_mode:
            raise ValueError("rag_enabled and rag_fixture_mode cannot both be true.")
        if self.rag_fixture_mode and app_env not in {"local", "test"}:
            raise ValueError("rag_fixture_mode is only allowed in local/test environments.")
        if self.rag_final_top_k > self.rag_hybrid_top_k:
            raise ValueError("rag_final_top_k cannot exceed rag_hybrid_top_k.")
        if self.rag_chunk_overlap_tokens >= self.rag_chunk_size_tokens:
            raise ValueError("rag_chunk_overlap_tokens must be smaller than rag_chunk_size_tokens.")
        if self.rag_embedding_version is None:
            object.__setattr__(self, "rag_embedding_version", self.rag_embedding_model)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
