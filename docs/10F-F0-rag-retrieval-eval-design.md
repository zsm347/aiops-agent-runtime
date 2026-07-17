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

`scripts/run_rag_f0_baseline.py` requires explicit confirmation that PostgreSQL is dedicated to the
evaluation. It uses `Settings()` at runtime, does not inspect or print `.env`, and never logs tokens or
API keys. Missing credentials, disabled RAG, non-dedicated PostgreSQL, or unavailable services produce
an `infrastructure_pending` artifact with no quality claims. Deterministic embeddings and fixtures are
permitted only in unit tests, not in baseline artifacts.

Dense-only and BM25-only ablations are `not_available` in F0 because the current project service
contract exposes only scope-preserving hybrid retrieval. F0 does not bypass the service or add a
parallel Milvus query implementation to manufacture ablations.

## Metrics

LlamaIndex Core 0.14.23 provides `HitRate`, `MRR`, `Precision`, `Recall`, `AveragePrecision`, and
`NDCG` under `llama_index.core.evaluation.retrieval.metrics`. F0 uses these implementations rather
than handwritten retrieval formulas. Empty answerable retrievals are represented by a guaranteed
non-relevant sentinel because the upstream metric classes reject empty retrieved ID lists.

Reported metrics are Recall@10, HitRate@3, MRR@10, nDCG@3, nDCG@10, Precision@3, diagnostic
AveragePrecision@10, no-answer false-positive rate, and p50/p95 service latency. The report contains
per-query rankings/scores, failure classes, macro and per-slice means, and deterministic query-level
bootstrap 95% intervals. No-answer queries count any returned chunk as a false positive and are not
passed through an LLM judge.

## Interpretation

- Low Recall@10 should be investigated against evidence mapping, chunk boundaries, lexical BM25
  coverage, embedding behavior, and symptom-only query language.
- High Recall@10 with low HitRate@3 or MRR@10 is evidence that first-stage ranking may need later
  rerank evaluation; it is not permission to implement rerank in F0.
- Hard-negative or no-answer failures support later threshold/gating calibration. F0 does not select
  a threshold or change gold qrels to improve scores.

The dataset, report, and any real dev baseline remain
`pending independent dataset acceptance` until reviewed by the technical owner.
