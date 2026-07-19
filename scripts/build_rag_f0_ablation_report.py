from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from superbiz_agent.evals.rag_runner import RagF0Report
from superbiz_agent.rag.models import RagRetrievalMode


MODE_ORDER = (
    RagRetrievalMode.DENSE,
    RagRetrievalMode.BM25,
    RagRetrievalMode.HYBRID,
)
PRIMARY_METRICS = (
    "recall_at_10",
    "hit_rate_at_3",
    "mrr_at_10",
    "binary_ndcg_at_3",
    "binary_ndcg_at_10",
    "precision_at_3",
    "average_precision_at_10",
    "no_answer_false_positive_rate",
)
LOWER_IS_BETTER = frozenset({"no_answer_false_positive_rate"})
KEY_SLICES = ("hard_negative", "symptom_only", "multi_evidence", "no_answer", "paraphrase")


def build_ablation_summary(
    reports: dict[RagRetrievalMode, RagF0Report],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260718,
) -> dict[str, Any]:
    _validate_reports(reports)
    metrics: dict[str, Any] = {}
    for index, metric_name in enumerate(PRIMARY_METRICS):
        values = {
            mode: reports[mode].metrics[metric_name].value
            for mode in MODE_ORDER
        }
        if any(value is None for value in values.values()):
            raise ValueError(f"metric has no macro value: {metric_name}")
        single_modes = (RagRetrievalMode.DENSE, RagRetrievalMode.BM25)
        if metric_name in LOWER_IS_BETTER:
            best_single = min(single_modes, key=lambda mode: float(values[mode]))
        else:
            best_single = max(single_modes, key=lambda mode: float(values[mode]))
        query_values = {
            mode: _per_query_metric(reports[mode], metric_name) for mode in MODE_ORDER
        }
        query_ids = tuple(sorted(query_values[RagRetrievalMode.HYBRID]))
        if any(tuple(sorted(query_values[mode])) != query_ids for mode in MODE_ORDER):
            raise ValueError(f"metric query sets differ across modes: {metric_name}")
        paired = []
        for query_id in query_ids:
            hybrid = query_values[RagRetrievalMode.HYBRID][query_id]
            single = query_values[best_single][query_id]
            paired.append(single - hybrid if metric_name in LOWER_IS_BETTER else hybrid - single)
        delta = _paired_summary(
            paired,
            samples=bootstrap_samples,
            seed=seed + index,
        )
        metrics[metric_name] = {
            "dense": values[RagRetrievalMode.DENSE],
            "bm25": values[RagRetrievalMode.BM25],
            "hybrid": values[RagRetrievalMode.HYBRID],
            "best_single_mode": best_single.value,
            "hybrid_gain_over_best_single": delta,
            "query_count": len(query_ids),
            "direction": "lower_is_better" if metric_name in LOWER_IS_BETTER else "higher_is_better",
        }

    slices: dict[str, Any] = {}
    all_slices = sorted(
        set.intersection(*(set(reports[mode].slice_metrics) for mode in MODE_ORDER))
    )
    for slice_name in all_slices:
        slices[slice_name] = {}
        metric_names = sorted(
            set.intersection(
                *(set(reports[mode].slice_metrics[slice_name]) for mode in MODE_ORDER)
            )
        )
        for metric_name in metric_names:
            slices[slice_name][metric_name] = {
                mode.value: reports[mode].slice_metrics[slice_name][metric_name].model_dump()
                for mode in MODE_ORDER
            }

    ranking_overlap = {}
    hybrid_rankings = _rankings(reports[RagRetrievalMode.HYBRID])
    for mode in (RagRetrievalMode.DENSE, RagRetrievalMode.BM25):
        single_rankings = _rankings(reports[mode])
        overlaps = [
            len(set(hybrid_rankings[query_id]) & set(single_rankings[query_id]))
            / max(1, len(set(hybrid_rankings[query_id]) | set(single_rankings[query_id])))
            for query_id in sorted(hybrid_rankings)
        ]
        ranking_overlap[mode.value] = {
            "mean_top10_evidence_jaccard": statistics.fmean(overlaps),
            "query_count": len(overlaps),
        }

    query_deltas = {}
    for metric_name in ("recall_at_10", "hit_rate_at_3", "mrr_at_10"):
        dense_values = _per_query_metric(reports[RagRetrievalMode.DENSE], metric_name)
        hybrid_values = _per_query_metric(reports[RagRetrievalMode.HYBRID], metric_name)
        improved = sorted(
            query_id
            for query_id in dense_values
            if hybrid_values[query_id] > dense_values[query_id]
        )
        degraded = sorted(
            query_id
            for query_id in dense_values
            if hybrid_values[query_id] < dense_values[query_id]
        )
        query_deltas[metric_name] = {
            "hybrid_improved_over_dense": improved,
            "hybrid_degraded_from_dense": degraded,
            "tied": len(dense_values) - len(improved) - len(degraded),
        }

    no_answer_results = {
        mode.value: {
            row.query_id: len(row.retrieved)
            for row in reports[mode].per_query
            if row.allow_no_answer
        }
        for mode in MODE_ORDER
    }

    observed_primary_gains = [
        metrics[name]["hybrid_gain_over_best_single"]["value"]
        for name in PRIMARY_METRICS
    ]

    return {
        "status": "completed",
        "acceptance": "pending_independent_quality_acceptance",
        "dataset_sha256": reports[RagRetrievalMode.HYBRID].dataset_sha256,
        "split": "dev",
        "execution": {
            mode.value: reports[mode].execution for mode in MODE_ORDER
        },
        "metrics": metrics,
        "slices": slices,
        "ranking_overlap": ranking_overlap,
        "query_deltas": query_deltas,
        "no_answer_retrieved_counts": no_answer_results,
        "latency_ms": {
            mode.value: reports[mode].latency_ms for mode in MODE_ORDER
        },
        "bootstrap": {
            "method": "paired_query_level_resampling_with_replacement",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "score_comparability": "raw scores are mode-specific ranking values and are not compared across modes",
        "conclusion": {
            "hybrid_has_observed_macro_gain_over_best_single": any(
                gain > 0 for gain in observed_primary_gains
            ),
            "hybrid_has_strict_positive_paired_ci": any(
                metrics[name]["hybrid_gain_over_best_single"]["ci95_low"] > 0
                for name in PRIMARY_METRICS
            ),
            "selection_policy": "no parameter changes or reruns based on dev results",
        },
        "holdout": "not_run",
        "rerank": "disabled",
    }


def _validate_reports(reports: dict[RagRetrievalMode, RagF0Report]) -> None:
    if set(reports) != set(MODE_ORDER):
        raise ValueError("all three retrieval modes are required")
    dataset_hashes = {report.dataset_sha256 for report in reports.values()}
    if len(dataset_hashes) != 1:
        raise ValueError("ablation reports use different Dataset hashes")
    for mode, report in reports.items():
        if report.status != "completed":
            raise ValueError(f"ablation report is not completed: {mode.value}")
        if report.execution != {"planned": 48, "executed": 48, "skipped": 0, "failed": 0}:
            raise ValueError(f"ablation execution is incomplete: {mode.value}")
        if report.infrastructure_failure_count != 0:
            raise ValueError(f"ablation has infrastructure failures: {mode.value}")
        if report.retrieval_config.get("retrieval_mode") != mode.value:
            raise ValueError(f"ablation report mode mismatch: {mode.value}")
        if report.retrieval_config.get("rrf_k") != 60:
            raise ValueError("RRF k changed during ablation")
        if report.retrieval_config.get("rerank_enabled") is not False:
            raise ValueError("rerank must remain disabled")
        if len(report.per_query) != 48:
            raise ValueError(f"per-query evidence is incomplete: {mode.value}")
        missing_metrics = set(PRIMARY_METRICS) - set(report.metrics)
        if missing_metrics:
            raise ValueError(f"ablation metrics are incomplete: {sorted(missing_metrics)}")


def _per_query_metric(report: RagF0Report, metric_name: str) -> dict[str, float]:
    return {
        row.query_id: row.metrics[metric_name]
        for row in report.per_query
        if metric_name in row.metrics
    }


def _rankings(report: RagF0Report) -> dict[str, tuple[str, ...]]:
    return {row.query_id: row.ranked_evidence_ids for row in report.per_query}


def _paired_summary(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    if not values:
        raise ValueError("paired bootstrap requires query values")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in range(len(values)))
        for _ in range(samples)
    )
    return {
        "value": statistics.fmean(values),
        "ci95_low": means[int(samples * 0.025)],
        "ci95_high": means[min(samples - 1, int(samples * 0.975))],
        "query_count": len(values),
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# RAG F0 Retrieval Ablation",
        "",
        "Status: **completed / pending independent quality acceptance**.",
        "",
        f"Dataset SHA-256: `{summary['dataset_sha256']}`. Dev only; holdout not run; rerank disabled.",
        "",
        "## Overall Comparison",
        "",
        "| Metric | Dense | BM25 | Hybrid | Best single | Hybrid gain | Paired 95% CI |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for metric_name in PRIMARY_METRICS:
        row = summary["metrics"][metric_name]
        gain = row["hybrid_gain_over_best_single"]
        lines.append(
            f"| {metric_name} | {row['dense']:.4f} | {row['bm25']:.4f} | "
            f"{row['hybrid']:.4f} | {row['best_single_mode']} | {gain['value']:+.4f} | "
            f"[{gain['ci95_low']:+.4f}, {gain['ci95_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "Positive gain means Hybrid is better. For no-answer false-positive rate, lower is better and the sign is normalized accordingly.",
        "",
        "## Key Slices",
        "",
        "| Slice | Metric | Dense | BM25 | Hybrid |",
        "|---|---|---:|---:|---:|",
    ])
    for slice_name in KEY_SLICES:
        for metric_name in ("recall_at_10", "hit_rate_at_3", "mrr_at_10", "no_answer_false_positive_rate"):
            row = summary["slices"].get(slice_name, {}).get(metric_name)
            if row is None:
                continue
            lines.append(
                f"| {slice_name} | {metric_name} | "
                f"{row[RagRetrievalMode.DENSE.value]['value']:.4f} | "
                f"{row[RagRetrievalMode.BM25.value]['value']:.4f} | "
                f"{row[RagRetrievalMode.HYBRID.value]['value']:.4f} |"
            )
    lines.extend([
        "",
        "Raw Milvus scores are mode-specific ranking values and are not compared across modes. Latency is local Milvus Lite dev evidence, not remote production capacity.",
        "",
        "## Interpretation",
        "",
        "Hybrid did not improve any reported macro metric over the best single route. Recall@10 tied Dense; all other normalized gains were negative, and no paired 95% interval was strictly positive. Dense was the best answerable route overall, while BM25 had the lowest no-answer false-positive rate.",
        "",
        "Hybrid improved selected hard-negative/error-code rankings but degraded selected paraphrase, symptom-only, and multi-evidence rankings. This is a negative/null ablation result and does not support claiming Hybrid gain or proceeding to rerank from F0-Ablation alone.",
        "",
        "The three modes were one-shot real API runs on separate fresh infrastructure. Query-level paired bootstrap captures Dataset-query variation, not possible embedding-provider run-to-run variation. No run was repeated to estimate that source of uncertainty.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    reports = {
        RagRetrievalMode.DENSE: RagF0Report.model_validate_json(args.dense.read_text("utf-8")),
        RagRetrievalMode.BM25: RagF0Report.model_validate_json(args.bm25.read_text("utf-8")),
        RagRetrievalMode.HYBRID: RagF0Report.model_validate_json(args.hybrid.read_text("utf-8")),
    }
    summary = build_ablation_summary(reports)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, args.output_markdown)


if __name__ == "__main__":
    main()
