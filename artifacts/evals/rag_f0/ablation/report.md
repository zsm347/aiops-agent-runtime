# RAG F0 Retrieval Ablation

Status: **completed / pending independent quality acceptance**.

Dataset SHA-256: `0627e66b0b5b40cb1fbe326f2dfb980be2f43440d264056b5ebc774e157d26ec`. Dev only; holdout not run; rerank disabled.

## Overall Comparison

| Metric | Dense | BM25 | Hybrid | Best single | Hybrid gain | Paired 95% CI |
|---|---:|---:|---:|---|---:|---:|
| recall_at_10 | 0.9773 | 0.7045 | 0.9773 | dense_only | +0.0000 | [+0.0000, +0.0000] |
| hit_rate_at_3 | 0.9091 | 0.6136 | 0.8864 | dense_only | -0.0227 | [-0.0909, +0.0455] |
| mrr_at_10 | 0.7678 | 0.5604 | 0.7233 | dense_only | -0.0445 | [-0.1312, +0.0428] |
| binary_ndcg_at_3 | 0.7761 | 0.5382 | 0.7342 | dense_only | -0.0419 | [-0.1315, +0.0449] |
| binary_ndcg_at_10 | 0.8148 | 0.5840 | 0.7806 | dense_only | -0.0342 | [-0.0986, +0.0328] |
| precision_at_3 | 0.3258 | 0.2273 | 0.3106 | dense_only | -0.0152 | [-0.0530, +0.0227] |
| average_precision_at_10 | 0.7545 | 0.5363 | 0.7112 | dense_only | -0.0433 | [-0.1287, +0.0445] |
| no_answer_false_positive_rate | 1.0000 | 0.7500 | 1.0000 | bm25_only | -0.2500 | [-0.7500, +0.0000] |

Positive gain means Hybrid is better. For no-answer false-positive rate, lower is better and the sign is normalized accordingly.

## Key Slices

| Slice | Metric | Dense | BM25 | Hybrid |
|---|---|---:|---:|---:|
| hard_negative | recall_at_10 | 0.9500 | 0.7250 | 0.9500 |
| hard_negative | hit_rate_at_3 | 0.8500 | 0.6500 | 0.9000 |
| hard_negative | mrr_at_10 | 0.7433 | 0.5604 | 0.7467 |
| hard_negative | no_answer_false_positive_rate | 1.0000 | 1.0000 | 1.0000 |
| symptom_only | recall_at_10 | 1.0000 | 0.6250 | 1.0000 |
| symptom_only | hit_rate_at_3 | 0.8750 | 0.5000 | 0.8750 |
| symptom_only | mrr_at_10 | 0.8333 | 0.4688 | 0.6792 |
| symptom_only | no_answer_false_positive_rate | 1.0000 | 0.0000 | 1.0000 |
| multi_evidence | recall_at_10 | 1.0000 | 0.6000 | 1.0000 |
| multi_evidence | hit_rate_at_3 | 1.0000 | 0.6000 | 1.0000 |
| multi_evidence | mrr_at_10 | 0.9000 | 0.5583 | 0.8000 |
| no_answer | no_answer_false_positive_rate | 1.0000 | 0.7500 | 1.0000 |
| paraphrase | recall_at_10 | 0.8000 | 0.2000 | 0.8000 |
| paraphrase | hit_rate_at_3 | 0.8000 | 0.2000 | 0.4000 |
| paraphrase | mrr_at_10 | 0.7000 | 0.2000 | 0.4833 |

Raw Milvus scores are mode-specific ranking values and are not compared across modes. Latency is local Milvus Lite dev evidence, not remote production capacity.

## Interpretation

Hybrid did not improve any reported macro metric over the best single route. Recall@10 tied Dense; all other normalized gains were negative, and no paired 95% interval was strictly positive. Dense was the best answerable route overall, while BM25 had the lowest no-answer false-positive rate.

Hybrid improved selected hard-negative/error-code rankings but degraded selected paraphrase, symptom-only, and multi-evidence rankings. This is a negative/null ablation result and does not support claiming Hybrid gain or proceeding to rerank from F0-Ablation alone.

The three modes were one-shot real API runs on separate fresh infrastructure. Query-level paired bootstrap captures Dataset-query variation, not possible embedding-provider run-to-run variation. No run was repeated to estimate that source of uncertainty.
