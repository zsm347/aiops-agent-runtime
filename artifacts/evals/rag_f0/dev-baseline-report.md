# RAG F0 Real Dev Baseline Report

Status: **real dev baseline executed / pending independent quality acceptance**.

## Execution

- Dataset: `rag-f0-v1`
- Dataset SHA-256: `0627e66b0b5b40cb1fbe326f2dfb980be2f43440d264056b5ebc774e157d26ec`
- Split: dev only; the 12-query holdout was not read or run
- Planned / executed / skipped / failed: `48 / 48 / 0 / 0`
- Infrastructure failure count: `0`
- Retrieval: dense + native BM25 + RRF `k=60`, hybrid top 10, top 3 prefix
- Rerank: disabled
- Started / finished: `2026-07-17T13:23:40.738692Z` / `2026-07-17T13:23:52.033598Z`

The run used the sole protected entrypoint and the full production service chain:

```text
RagIngestionService -> real document embedding -> Milvus dense + native BM25 + RRF
-> default KB resolver -> real query embedding -> PostgreSQL active-document visibility
-> RagRetrievalService -> evidence mapping -> LlamaIndex retrieval metrics
```

## Runtime Identity

- Embedding provider: `dashscope-openai-compatible`
- Embedding model/version: `text-embedding-v4` / `text-embedding-v4`
- Embedding dimension: `1024`
- Embedding transport batch size: `10`
- Credential source: explicitly authorized model-gateway Settings pair
- PostgreSQL / pgvector: `16.14` / `0.8.5`
- Milvus Lite / PyMilvus: `3.0` / `2.6.16`
- LlamaIndex Core / Milvus adapter: `0.14.23` / `1.1.0`
- OpenAI SDK: `2.45.0`

No credential, endpoint, database URL, role, database name, collection name, or token is included in
the artifact.

## Overall Metrics

Metrics for answerable queries use 44 queries. No-answer false-positive rate uses four queries.
LlamaIndex `NDCG` is binary in this environment and is reported only as binary nDCG.

| Metric | Value | Bootstrap 95% CI | Queries |
|---|---:|---:|---:|
| Recall@10 | 0.9545 | [0.8864, 1.0000] | 44 |
| HitRate@3 | 0.8864 | [0.7955, 0.9773] | 44 |
| MRR@10 | 0.7320 | [0.6288, 0.8305] | 44 |
| binary nDCG@3 | 0.7426 | [0.6378, 0.8414] | 44 |
| binary nDCG@10 | 0.7823 | [0.7012, 0.8643] | 44 |
| Precision@3 | 0.3106 | [0.2727, 0.3485] | 44 |
| AveragePrecision@10 | 0.7203 | [0.6192, 0.8195] | 44 |
| No-answer false-positive rate | 1.0000 | [1.0000, 1.0000] | 4 |

Retrieval service latency on local Milvus Lite was p50 `216.74 ms` and p95 `349.33 ms`. These values
are a local dev pilot, not remote Milvus production latency or a capacity conclusion.

## Slice Summary

The JSON artifact contains every slice/metric bootstrap interval. The table below summarizes the
principal ranking metrics and query counts; `N/A` applies to no-answer-only rows.

| Slice | Answerable n | Recall@10 | Hit@3 | MRR@10 | binary nDCG@3 | No-answer n / FPR |
|---|---:|---:|---:|---:|---:|---:|
| alert_name | 10 | 1.0000 | 1.0000 | 0.7000 | 0.7762 | 0 / N/A |
| command | 8 | 1.0000 | 0.8750 | 0.8073 | 0.7641 | 0 / N/A |
| configuration | 9 | 1.0000 | 1.0000 | 0.8333 | 0.8499 | 1 / 1.0000 |
| error_code | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 / 1.0000 |
| exact_keyword | 10 | 1.0000 | 1.0000 | 0.7000 | 0.7762 | 0 / N/A |
| hard_negative | 20 | 0.9000 | 0.9000 | 0.7417 | 0.7705 | 2 / 1.0000 |
| multi_evidence | 5 | 1.0000 | 1.0000 | 0.8000 | 0.6774 | 0 / N/A |
| no_answer | 0 | N/A | N/A | N/A | N/A | 4 / 1.0000 |
| paraphrase | 5 | 0.8000 | 0.4000 | 0.4833 | 0.4000 | 0 / N/A |
| symptom_only | 8 | 0.8750 | 0.8750 | 0.6667 | 0.7202 | 1 / 1.0000 |

## Error Analysis

Two answerable queries missed all gold evidence at top 10:

- `q-dev-002` described repeated short-lived container restarts without naming CrashLoopBackOff. The
  result contained adjacent pod-log, waiting-state, and CrashLoop mitigation evidence, but not the
  gold section. This is primarily a paraphrase/section-ranking miss, with evidence granularity also
  contributing.
- `q-dev-006` combined missing Service traffic with a long-running init container. Retrieval split
  across PodNotReady, Service endpoint, and init-container sections but missed the gold evidence. The
  mixed symptom intent and close hard negatives weakened both dense and lexical signals.

Three additional queries had gold evidence in top 10 but outside top 3: job retry design at rank 4,
the NotReady node diagnosis command at rank 8, and node post-repair mitigation at rank 6. The correct
documents were present, so this is first-stage section ordering rather than a Recall@10 failure. It
is evidence that later ranking research may be useful, but this run does not establish that rerank is
effective.

All four no-answer queries returned 10 chunks. This follows the frozen unconditional top-10 contract:
there is no score threshold or retrieval gating. The 1.0 false-positive rate supports a later,
separately calibrated no-answer gate; it is not addressed by changing gold or tuning on this dev run.

Multi-evidence Recall@10 and Hit@3 were both 1.0, while binary nDCG@3 was 0.6774, indicating coverage
was strong but ordering of all required evidence was less consistent. Paraphrase was the weakest
answerable slice, with Recall@10 0.8 and Hit@3 0.4. Hard-negative and symptom-only Recall@10 were 0.9
and 0.875 respectively.

## Cleanup Evidence

- Runner cleanup: PostgreSQL F0 documents and knowledge base removed; Milvus collection removed
- Post-run PostgreSQL F0 tenant row checks: documents `0`, knowledge bases `0`
- Post-run Milvus collection check: absent
- One-time PostgreSQL database and role: deleted and independently checked absent
- Temporary PostgreSQL service: stopped
- Temporary Milvus Lite path and temporary service root: deleted
- Cleanup failures: `0`

This is a 48-query real dev pilot, not a production-capacity result. Dataset and quality remain pending
independent acceptance. No parameter was tuned from these results, holdout remained unopened, and no
claim is made about rerank effectiveness or remote Milvus TLS/auth/reconnect/timeout behavior.
