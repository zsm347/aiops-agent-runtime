# F0 Ablation Plan

Status: planned / pending execution / pending independent acceptance.

## Question

Measure whether frozen dense + native BM25 + RRF(k=60) improves dev retrieval quality over the
best scope-preserving single route under the same Dataset, chunking, embedding identity, top 10,
PostgreSQL active-document filter, and evidence mapping contract.

## Slices

| Mode | Milvus query mode | Query embedding used for ranking | Runs |
|---|---|---:|---:|
| Dense-only | DEFAULT | yes | one dev run |
| BM25-only | SPARSE | no | one dev run |
| Hybrid | HYBRID + RRFRanker(k=60) | yes | one dev run |

Each run uses all 48 dev queries and a fresh one-time PostgreSQL database/role and Milvus Lite path.
Holdout is not read. Ingestion, resolver, structured tenant/KB filters, active-document visibility,
result validation, evidence mapping, metric implementations, and cleanup are identical.

## Frozen Boundaries

- Dataset/qrels/evidence unchanged.
- Chunk size 1024, overlap 100.
- Embedding `text-embedding-v4`, dimension 1024, transport batch 10.
- Retrieval top 10; top 3 is the prefix of the same ranking.
- RRF k=60; no rerank or threshold/gating.
- No parameter selection from results and no repeated score runs.
- Model Gateway credential mapping is removed from project code. The operator supplies RAG-specific
  runtime settings; no credential value is persisted.

## Analysis

For each primary metric, select the better dense/BM25 macro value, then compute Hybrid minus that
single-route value. Use deterministic paired query-level bootstrap intervals for the delta. For
no-answer false-positive rate, lower is better and improvement is best-single minus Hybrid.

Report all null/negative deltas, mode latency as local Lite evidence only, per-slice comparisons,
ranking overlap, failure counts, and representative query-level wins/losses.
