# RAG F0 Dev Baseline Report

Status: **infrastructure pending / pending independent dataset acceptance**.

## Configuration

- Dataset: `rag-f0-v1`
- Dataset SHA-256: `0627e66b0b5b40cb1fbe326f2dfb980be2f43440d264056b5ebc774e157d26ec`
- Split: dev only, 48 queries; holdout was not run.
- Retrieval: dense + native BM25 + RRF `k=60`, hybrid top 10, top 3 prefix, rerank disabled.
- Required path: default KB resolver -> real query embedding -> Milvus -> PostgreSQL active filter.
- Dense-only diagnostic: `not_available` without a scope-preserving project service contract.
- BM25-only diagnostic: `not_available` without a scope-preserving project service contract.

## Infrastructure Fact

The isolated F0 worktree does not have an independently confirmed dedicated PostgreSQL evaluation
database. The runner therefore stopped at the mandatory preflight and wrote
`dev-baseline.json` with `infrastructure_pending: dedicated_postgres_not_confirmed`.

No query was executed, no real embedding request was made, and no deterministic or fixture result
was substituted. Consequently this report contains no Recall, HitRate, MRR, binary nDCG, Precision,
no-answer false-positive, latency, or per-slice quality claim.

The remediated entrypoint additionally requires an exact F0-prefixed database name, an exact
F0-prefixed collection name, a pre-created Dataset-SHA marker, the frozen Alembic head, empty RAG
tables, and an absent collection. It removes all resources owned by the run in `finally`; cleanup
failure prevents publication of completed quality metrics. These guards have been tested offline,
but no real baseline was run during remediation.

## Required Error Analysis After Infrastructure Is Available

1. If Recall@10 is low, inspect failed queries against chunk boundaries/evidence mapping, embedding
   semantics, BM25 lexical coverage, and symptom-only wording.
2. If Recall@10 is high while HitRate@3 or MRR@10 is low, treat that as evidence for a later rerank
   experiment; do not retrofit rerank into F0.
3. If hard-negative or no-answer false-positive rates are high, evaluate retrieval gating or a
   calibrated threshold in a later parameter-calibration stage. Do not modify gold qrels.

## Out of Scope

Tenant isolation, KB isolation, and inactive-document visibility are not measured in F0 and are not
reported as zero violations. Batch D regression tests remain the evidence for those contracts.
Answer generation, citation faithfulness, LLM judges, rerank, remote Milvus production deployment,
and holdout tuning are not part of this baseline.

The baseline remains **pending independent dataset acceptance** and **infrastructure pending**.
