# RAG F0 Dev Baseline Report

Status: **infrastructure pending / pending independent dataset acceptance**.

## Execution

- Dataset: `rag-f0-v1`
- Dataset SHA-256: `0627e66b0b5b40cb1fbe326f2dfb980be2f43440d264056b5ebc774e157d26ec`
- Split planned: dev only; the 12-query holdout was not read or run.
- Planned: 48
- Executed: 0
- Skipped: 48
- Query failures: 0
- Infrastructure failure count: 48 planned queries blocked before execution
- Entrypoint exit status: `2`
- Failure: `rag_embedding_credentials_unavailable`

The sole real entrypoint, `scripts/run_rag_f0_baseline.py`, was invoked once with explicit dedicated
database and collection expectations. Its static fail-closed preflight found that the runtime
`Settings` had no RAG-specific embedding API key or RAG-specific embedding base URL. The F0 entrypoint
does not fall back to model-gateway credentials, so it exited before creating or connecting to
PostgreSQL, Milvus, or an embedding client.

## Configured Identity And Available Versions

The non-secret configured RAG identity was:

- Provider: `dashscope-openai-compatible`
- Model: `text-embedding-v4`
- Version: `text-embedding-v4`
- Dimension: `1024`
- RAG-specific API key configured: no
- RAG-specific base URL configured: no

The following locally available versions were verified, but PostgreSQL and pgvector were not used by
a baseline run:

- PostgreSQL binary: `16.14`
- pgvector extension files: `0.8.5`
- Milvus Lite: `3.0`
- PyMilvus: `2.6.16`
- LlamaIndex Core: `0.14.23`
- LlamaIndex Milvus adapter: `1.1.0`
- OpenAI SDK: `2.45.0`

## Metrics

No retrieval call ran, so there are no Recall@10, HitRate@3, MRR@10, binary nDCG@3,
binary nDCG@10, Precision@3, AveragePrecision@10, no-answer false-positive, p50/p95 latency,
bootstrap interval, slice, ranking, score, or per-query failure-class results. In particular, no
binary nDCG value is represented as graded nDCG, and no successful subset is reported as a baseline.

## Error Analysis

Hard-negative, symptom-only, multi-evidence, and no-answer retrieval behavior cannot be analyzed
because all dev queries were skipped before retrieval. No Dataset, query, evidence, qrel, chunking,
top-k, RRF, embedding model, or rerank setting was changed in response to this infrastructure result.

## Cleanup Evidence

- Embedding API calls: 0
- PostgreSQL service/database/role created by this attempt: no
- PostgreSQL F0 rows created: 0
- Milvus Lite file/collection created: no
- Matching temporary Milvus Lite files after the attempt: 0
- Cleanup failures: 0

Because the preflight stopped before resource construction, there were no ingestion store, embedding
client, retrieval runtime, database engine, PostgreSQL rows, database/role, service, collection, or
Milvus file to close or delete.

This artifact is not a quality baseline and must not be used to infer rerank value or production
latency. A future real dev run still requires RAG-specific embedding configuration and independent
quality acceptance. Remote Milvus TLS, authentication, reconnect, and timeout gates remain untested.
