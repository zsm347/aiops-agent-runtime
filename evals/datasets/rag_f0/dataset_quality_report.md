# RAG F0 Dataset Quality Report

Status: **pending independent dataset acceptance**.

- Dataset version: `rag-f0-v1`
- Dataset SHA-256: `0627e66b0b5b40cb1fbe326f2dfb980be2f43440d264056b5ebc774e157d26ec`
- Corpus documents: 18 (104,970 characters)
- Evidence sections: 140
- Queries: 60 (48 dev / 12 holdout)
- Qrels: 62
- Relevant evidence per answerable query: min=1, max=2, mean=1.11
- No-answer ratio: 6.7% (4/60)
- Hard-negative ratio: 45.0% (27/60)
- Query/gold lexical overlap: min=0.000, median=0.091, max=0.389
- Exact duplicate query count: 0
- Near-duplicate pairs at Jaccard >= 0.80: 0
- Evidence-to-chunk mapping: enforced by `tests/test_rag_eval.py` using the production chunker.
- Isolation and inactive-document evaluation: `out_of_scope` for F0; Batch D tests remain authoritative.

## Slice Counts

| Slice | Count |
| --- | ---: |
| `alert_name` | 10 |
| `command` | 12 |
| `configuration` | 14 |
| `error_code` | 6 |
| `exact_keyword` | 11 |
| `hard_negative` | 27 |
| `multi_evidence` | 6 |
| `no_answer` | 4 |
| `paraphrase` | 8 |
| `symptom_only` | 10 |

## Source and Gold Review

All query candidates and qrels were human-authored against pinned official source text. No LLM-generated query was promoted directly to gold. Every qrel contains a reviewer rationale and an exact source quote. Hard negatives intentionally pair adjacent Kubernetes failure modes such as CrashLooping vs NotReady, HPA saturation vs CPU throttling, and Service DNS vs EndpointSlice failures.

## Holdout Policy

The holdout split is frozen in `queries.jsonl`. F0 runner defaults to dev and rejects holdout execution unless an explicit future acceptance workflow opts in. F0 metrics and parameter recommendations must not use holdout.

## Acceptance

This report and any F0 baseline remain **pending independent dataset acceptance**. Dataset quality is not inferred from retrieval scores.
