# 10F-F0 RAG Retrieval Dataset and Baseline

Status: `implemented / pending independent dataset acceptance`.

## Scope

F0 evaluates retrieval only. The frozen baseline is dense vector plus native Milvus BM25,
`RRFRanker(k=60)`, hybrid top 10, top 3 as the prefix of that same ranking, and no rerank. It does
not generate answers, run an LLM judge, enforce citations, or change production chunk/topK/RRF
defaults.

The evaluation scope is one fixed tenant, one active default knowledge base, and active documents
only. Cross-tenant, cross-KB, and inactive-document evaluation are explicitly `out_of_scope`; F0
does not report zero violations for cases it does not execute. Existing Batch D isolation and active
visibility tests remain mandatory regression gates.

## Dataset Protocol

The dataset follows a BEIR/TREC-style separation:

- `corpus.jsonl`: pinned source documents and source identity.
- `evidence.jsonl`: stable section evidence IDs, document IDs, heading paths, line spans, anchors,
  and original evidence text.
- `queries.jsonl`: reviewed query text, split, category, slices, answerability, and generation record.
- `qrels.jsonl`: query/evidence relevance with rationale and an original source quote.
- `manifest.json`: file hashes, aggregate dataset hash, fixed scope, source versions/licenses, and
  acceptance state.

Qrels bind stable evidence IDs rather than chunk IDs. Evidence IDs derive from document identity and
heading path. The runner maps each returned chunk back to evidence using `documentName` and the full
`headingPath`; a unique text-anchor fallback exists only for source formatting differences. Missing
or ambiguous mapping is a dataset/runner contract failure, never an implicit non-relevant result.

The first catalog contains 18 official documents, 140 evidence sections, 60 reviewed queries
(48 dev, 12 frozen holdout), and 62 qrels. It includes exact alert names, paraphrases, commands,
configuration keys, symptom-only cases, six multi-evidence cases, hard negatives within adjacent
Kubernetes failure families, and four no-answer dev cases. F0 runs dev only.

## Sources and Licensing

1. Kubernetes Website troubleshooting documents, commit
   `5e1d1bde0ca03efe09608d59c573d6ec87052c24`, licensed CC BY 4.0.
2. Prometheus Operator Kubernetes runbooks, commit
   `a685d14cf5128bb30e2bf935c3983decd772d885`, licensed Apache-2.0.

Every source entry records repository path, immutable URL, commit, license URL, fetch timestamp, and
SHA-256. `scripts/build_rag_f0_dataset.py --fetch` reproduces the vendored source and generated
JSONL files. The corpus remains operational source text; it is not a collection of test-only short
sentences.

All committed queries and qrels are human-authored and reviewed against pinned evidence. No LLM was
used to promote chunk rewrites directly to gold. The recorded authoring prompt describes coverage
requirements only and contains no credential.

## Runner Contract

`RagF0Runner` calls `RagRetrievalService.search()` with backend-owned fixed tenant scope. The real
entrypoint first ingests the pinned corpus through `RagIngestionService`, then invokes the production
chain:

```text
default KB resolver
  -> real query embedding
  -> Milvus dense + native BM25 + RRF
  -> PostgreSQL active-document visibility
  -> RagRetrievalResult
  -> evidence mapping and metrics
```

`scripts/run_rag_f0_baseline.py` is the only real F0 entrypoint; the eval module has no direct real
runtime CLI. The script requires explicit confirmation plus exact expected PostgreSQL
database and Milvus collection names. Database names must start with `superbiz_rag_f0_`; collection
names must start with `rag_f0_`. The configured database name, CLI expectation, and
`current_database()` must agree. The database must contain the operator-created marker from
`scripts/setup_rag_f0_database.sql`, bound to the current Dataset SHA, and must be at Alembic head
`20260714_01` with empty `rag_knowledge_base` and `rag_document` tables. The exact collection must
not exist before the run. These checks establish ownership before any evaluation row or chunk is
written; a Boolean confirmation alone is insufficient.

After a preflight succeeds, the runner always deletes the F0 tenant's documents before its knowledge
base and drops the collection that this invocation proved absent and then created. A failed run is
cleaned in the same `finally` path. A completed quality artifact is written only after runtime and
ingestion resources close and both PostgreSQL and Milvus cleanup succeed. Preflight, execution,
partial-query, or cleanup failures produce an `infrastructure_pending` artifact with no aggregate
quality metrics. A pre-existing collection is never claimed or dropped by the normal runner.
Cleanup re-queries PostgreSQL and Milvus and fails closed if owned resources remain.

The real-run artifact records planned/executed/skipped/failed query counts, the non-secret embedding
provider/model/version/dimension identity, and PostgreSQL, pgvector, Milvus Lite, LlamaIndex, adapter,
PyMilvus, and OpenAI package versions. F0 real execution is frozen to PostgreSQL 16.14 and pgvector
0.8.5; a version mismatch is an infrastructure failure rather than a quality result.

The entrypoint uses `Settings()` at runtime, does not inspect or print `.env`, and never logs tokens or
API keys. Missing credentials, disabled RAG, non-dedicated infrastructure, or unavailable services
produce an `infrastructure_pending` artifact with no quality claims. Deterministic embeddings and
fixtures are permitted only in unit tests, not in baseline artifacts.
The real entrypoint requires RAG-specific embedding credentials and endpoint configuration and does
not fall back to model-gateway credentials or endpoint settings.

Dense-only and BM25-only ablations are `not_available` in F0 because the current project service
contract exposes only scope-preserving hybrid retrieval. F0 does not bypass the service or add a
parallel Milvus query implementation to manufacture ablations.

## Metrics

LlamaIndex Core 0.14.23 provides `HitRate`, `MRR`, `Precision`, `Recall`, `AveragePrecision`, and
`NDCG` under `llama_index.core.evaluation.retrieval.metrics`. F0 uses these implementations rather
than handwritten retrieval formulas. The installed `NDCG` implementation accepts only expected IDs,
not graded qrel gains, so F0 reports it explicitly as binary nDCG. The Dataset retains graded
`relevance` values for rationale and later evaluators, but F0 treats every qrel with relevance greater
than zero as relevant. Empty answerable retrievals are represented by a guaranteed non-relevant
sentinel because the upstream metric classes reject empty retrieved ID lists.

Reported metrics are Recall@10, HitRate@3, MRR@10, binary nDCG@3, binary nDCG@10, Precision@3,
diagnostic AveragePrecision@10, no-answer false-positive rate, and p50/p95 service latency. The report
contains per-query rankings/scores, failure classes, macro and per-slice means, and deterministic
query-level bootstrap 95% intervals. If any query has an infrastructure failure, macro, per-slice,
and aggregate latency fields are empty; successful-query subsets are not presented as a baseline.
No-answer queries count any returned chunk as a false positive and are not passed through an LLM
judge.

## Interpretation

- Low Recall@10 should be investigated against evidence mapping, chunk boundaries, lexical BM25
  coverage, embedding behavior, and symptom-only query language.
- High Recall@10 with low HitRate@3 or MRR@10 is evidence that first-stage ranking may need later
  rerank evaluation; it is not permission to implement rerank in F0.
- Hard-negative or no-answer failures support later threshold/gating calibration. F0 does not select
  a threshold or change gold qrels to improve scores.

The dataset, report, and any real dev baseline remain
`pending independent dataset acceptance` until reviewed by the technical owner.

## 2026-07-17 Real-Run Preflight Evidence

The protected entrypoint was invoked once for the requested dev pilot. Runtime `Settings` exposed the
configured provider/model identity but no RAG-specific embedding API key or base URL. The entrypoint
therefore returned `infrastructure_pending: rag_embedding_credentials_unavailable` before creating
PostgreSQL or Milvus resources and before any embedding request. Execution was planned 48, executed
0, skipped 48, with no quality metrics. Holdout remained unopened and rerank remained disabled.

This preflight result is not a real dev baseline and does not change the frozen evaluation design.
Real execution remains blocked until dedicated RAG embedding configuration is available; model-gateway
credentials must not be substituted.
