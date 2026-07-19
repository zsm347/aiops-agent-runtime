# M-P1-R2 Dev Calibration Plan

- Branch: `feat/m-p1-real-embedding-retrieval`
- Tier: auxiliary/dev calibration
- Dataset: `memory_retrieval_v2.json`, dev split only
- Baseline: real `text-embedding-v4` (1024 dimensions) with PostgreSQL 16.14 and pgvector 0.8.5
- Run budget: one real embedding run after deterministic tests pass
- Superset query: topK 10, minimum similarity -1
- Calibration inputs: observed top1 similarity and top1/top2 margin only
- Outputs: redacted baseline report/manifest, dataset quality report, calibration report, Pareto frontier
- Stop condition: 48/48 dev cases executed with zero infrastructure failure, artifacts hashed, fixture cleanup verified
- Forbidden: holdout execution, production threshold changes, production gating, rerank, M-P3, frozen asset changes
- Fallback: report `infrastructure_pending` without fabricating baseline evidence

