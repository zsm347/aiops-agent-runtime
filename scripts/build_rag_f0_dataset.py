from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


KUBERNETES_COMMIT = "5e1d1bde0ca03efe09608d59c573d6ec87052c24"
RUNBOOKS_COMMIT = "a685d14cf5128bb30e2bf935c3983decd772d885"
FETCHED_AT = "2026-07-17T00:00:00Z"
DATASET_VERSION = "rag-f0-v1"
DATA_FILES = ("corpus.jsonl", "evidence.jsonl", "queries.jsonl", "qrels.jsonl")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9_./:-]+|[\u4e00-\u9fff]", re.IGNORECASE)


def _k8s_source(document_id: str, area: str, filename: str) -> dict[str, str]:
    path = f"content/en/docs/tasks/debug/{area}/{filename}.md"
    return {
        "document_id": document_id,
        "document_name": f"{document_id}.md",
        "repository": "kubernetes/website",
        "commit": KUBERNETES_COMMIT,
        "license": "CC-BY-4.0",
        "license_url": "https://github.com/kubernetes/website/blob/main/LICENSE",
        "source_path": path,
        "source_url": f"https://github.com/kubernetes/website/blob/{KUBERNETES_COMMIT}/{path}",
        "raw_url": f"https://raw.githubusercontent.com/kubernetes/website/{KUBERNETES_COMMIT}/{path}",
    }


def _runbook_source(document_id: str, filename: str) -> dict[str, str]:
    path = f"content/runbooks/kubernetes/{filename}.md"
    return {
        "document_id": document_id,
        "document_name": f"{document_id}.md",
        "repository": "prometheus-operator/runbooks",
        "commit": RUNBOOKS_COMMIT,
        "license": "Apache-2.0",
        "license_url": "https://github.com/prometheus-operator/runbooks/blob/main/LICENSE",
        "source_path": path,
        "source_url": f"https://github.com/prometheus-operator/runbooks/blob/{RUNBOOKS_COMMIT}/{path}",
        "raw_url": f"https://raw.githubusercontent.com/prometheus-operator/runbooks/{RUNBOOKS_COMMIT}/{path}",
    }


SOURCES = (
    _k8s_source("k8s-debug-pods", "debug-application", "debug-pods"),
    _k8s_source("k8s-debug-running-pod", "debug-application", "debug-running-pod"),
    _k8s_source("k8s-debug-service", "debug-application", "debug-service"),
    _k8s_source("k8s-debug-init-containers", "debug-application", "debug-init-containers"),
    _k8s_source("k8s-debug-statefulset", "debug-application", "debug-statefulset"),
    _k8s_source("k8s-pod-failure-reason", "debug-application", "determine-reason-pod-failure"),
    _k8s_source("k8s-resource-metrics", "debug-cluster", "resource-metrics-pipeline"),
    _k8s_source("k8s-troubleshoot-kubectl", "debug-cluster", "troubleshoot-kubectl"),
    _runbook_source("runbook-pod-crashloop", "KubePodCrashLooping"),
    _runbook_source("runbook-pod-not-ready", "KubePodNotReady"),
    _runbook_source("runbook-deployment-replicas", "KubeDeploymentReplicasMismatch"),
    _runbook_source("runbook-hpa-maxed", "KubeHpaMaxedOut"),
    _runbook_source("runbook-job-failed", "KubeJobFailed"),
    _runbook_source("runbook-cpu-throttling", "CPUThrottlingHigh"),
    _runbook_source("runbook-pv-filling", "KubePersistentVolumeFillingUp"),
    _runbook_source("runbook-node-not-ready", "KubeNodeNotReady"),
    _runbook_source("runbook-api-budget-burn", "KubeAPIErrorBudgetBurn"),
    _runbook_source("runbook-kubelet-too-many-pods", "KubeletTooManyPods"),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(f"{_canonical_json(row)}\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _clean_heading(value: str) -> str:
    value = re.sub(r"\s*\{#[^}]+\}\s*$", "", value).strip()
    value = value.replace('{{% heading "whatsnext" %}}', "What's next")
    value = value.replace('{{% heading "prerequisites" %}}', "Prerequisites")
    return value


def _frontmatter_title(lines: list[str], fallback: str) -> tuple[str, int]:
    if not lines or lines[0].strip() != "---":
        return fallback, 0
    for index, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            title = next(
                (
                    item.split(":", 1)[1].strip().strip("\"'")
                    for item in lines[1:index]
                    if item.lower().startswith("title:")
                ),
                fallback,
            )
            return title, index + 1
    raise ValueError(f"Unclosed frontmatter in {fallback}")


def _has_information(value: str) -> bool:
    return any(character.isalnum() for character in value)


def _anchor(body_lines: list[str]) -> str:
    for line in body_lines:
        candidate = line.strip().lstrip("-*+> ").strip()
        if (
            len(candidate) >= 12
            and _has_information(candidate)
            and not candidate.startswith(("```", "{{", "<!--"))
            and not set(candidate) <= {"-", "|", ":", " "}
        ):
            return candidate[:240]
    return " ".join(line.strip() for line in body_lines if line.strip())[:240]


def _sections(document_id: str, text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    _, content_start = _frontmatter_title(lines, document_id)
    stack: dict[int, str] = {}
    current_path: tuple[str, ...] = ()
    current_start = 1
    body: list[str] = lines[:content_start]
    sections: list[dict[str, Any]] = []

    def flush(end_line: int) -> None:
        value = "\n".join(body).strip()
        if not _has_information(value):
            return
        identity = _canonical_json([document_id, list(current_path)])
        sections.append(
            {
                "evidence_id": "ev-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "document_id": document_id,
                "heading_path": list(current_path),
                "line_start": current_start,
                "line_end": max(current_start, end_line),
                "text_anchor": _anchor(body),
                "text": value,
            }
        )

    for line_number, line in enumerate(lines[content_start:], content_start + 1):
        match = HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue
        flush(line_number - 1)
        level = len(match.group(1))
        stack = {key: value for key, value in stack.items() if key < level}
        stack[level] = _clean_heading(match.group(2))
        current_path = tuple(stack[key] for key in sorted(stack))
        current_start = line_number
        body = []
    flush(len(lines))
    return sections


def _source_quote(evidence: dict[str, Any], needle: str) -> str:
    for line in evidence["text"].splitlines():
        if needle.casefold() in line.casefold():
            return line.strip()
    for paragraph in re.split(r"\n\s*\n", evidence["text"]):
        normalized = " ".join(line.strip() for line in paragraph.splitlines())
        if needle.casefold() in normalized.casefold():
            return normalized
    raise ValueError(f"Quote needle {needle!r} is missing from {evidence['evidence_id']}")


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value)}


def _quality_report(
    root: Path,
    corpus: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    dataset_sha: str,
) -> None:
    qrels_by_query = Counter(row["query_id"] for row in qrels)
    slice_counts = Counter(item for row in queries for item in row["slices"])
    overlap = []
    corpus_tokens = {row["document_id"]: _tokens(f"{row['title']} {row['text']}") for row in corpus}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    qrels_index: dict[str, list[dict[str, Any]]] = {}
    for row in qrels:
        qrels_index.setdefault(row["query_id"], []).append(row)
    for query in queries:
        query_tokens = _tokens(query["text"])
        gold_docs = {
            evidence_by_id[row["evidence_id"]]["document_id"]
            for row in qrels_index.get(query["query_id"], [])
        }
        gold_tokens = (
            set().union(*(corpus_tokens[item] for item in gold_docs)) if gold_docs else set()
        )
        overlap.append(len(query_tokens & gold_tokens) / len(query_tokens) if query_tokens else 0.0)

    near_duplicates = []
    for index, left in enumerate(queries):
        left_tokens = _tokens(left["text"])
        for right in queries[index + 1 :]:
            right_tokens = _tokens(right["text"])
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if similarity >= 0.8:
                near_duplicates.append((left["query_id"], right["query_id"], similarity))

    answerable_counts = [
        qrels_by_query[row["query_id"]] for row in queries if not row["allow_no_answer"]
    ]
    no_answer_count = sum(row["allow_no_answer"] for row in queries)
    hard_negative_count = sum("hard_negative" in row["slices"] for row in queries)
    lines = [
        "# RAG F0 Dataset Quality Report",
        "",
        "Status: **pending independent dataset acceptance**.",
        "",
        f"- Dataset version: `{DATASET_VERSION}`",
        f"- Dataset SHA-256: `{dataset_sha}`",
        f"- Corpus documents: {len(corpus)} ({sum(len(row['text']) for row in corpus):,} characters)",
        f"- Evidence sections: {len(evidence)}",
        f"- Queries: {len(queries)} ({sum(row['split'] == 'dev' for row in queries)} dev / {sum(row['split'] == 'holdout' for row in queries)} holdout)",
        f"- Qrels: {len(qrels)}",
        f"- Relevant evidence per answerable query: min={min(answerable_counts)}, max={max(answerable_counts)}, mean={sum(answerable_counts) / len(answerable_counts):.2f}",
        f"- No-answer ratio: {no_answer_count / len(queries):.1%} ({no_answer_count}/{len(queries)})",
        f"- Hard-negative ratio: {hard_negative_count / len(queries):.1%} ({hard_negative_count}/{len(queries)})",
        f"- Query/gold lexical overlap: min={min(overlap):.3f}, median={sorted(overlap)[len(overlap) // 2]:.3f}, max={max(overlap):.3f}",
        f"- Exact duplicate query count: {len(queries) - len({row['text'].casefold() for row in queries})}",
        f"- Near-duplicate pairs at Jaccard >= 0.80: {len(near_duplicates)}",
        "- Evidence-to-chunk mapping: enforced by `tests/test_rag_eval.py` using the production chunker.",
        "- Isolation and inactive-document evaluation: `out_of_scope` for F0; Batch D tests remain authoritative.",
        "",
        "## Slice Counts",
        "",
        "| Slice | Count |",
        "| --- | ---: |",
        *(f"| `{name}` | {count} |" for name, count in sorted(slice_counts.items())),
        "",
        "## Source and Gold Review",
        "",
        "All query candidates and qrels were human-authored against pinned official source text. No LLM-generated query was promoted directly to gold. Every qrel contains a reviewer rationale and an exact source quote. Hard negatives intentionally pair adjacent Kubernetes failure modes such as CrashLooping vs NotReady, HPA saturation vs CPU throttling, and Service DNS vs EndpointSlice failures.",
        "",
        "## Holdout Policy",
        "",
        "The holdout split is frozen in `queries.jsonl`. F0 runner defaults to dev and rejects holdout execution unless an explicit future acceptance workflow opts in. F0 metrics and parameter recommendations must not use holdout.",
        "",
        "## Acceptance",
        "",
        "This report and any F0 baseline remain **pending independent dataset acceptance**. Dataset quality is not inferred from retrieval scores.",
    ]
    (root / "dataset_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: Path, *, fetch: bool) -> None:
    source_root = root / "corpus_sources"
    source_root.mkdir(parents=True, exist_ok=True)
    corpus: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    source_manifest = []

    for source in SOURCES:
        path = source_root / source["document_name"]
        if fetch:
            with urllib.request.urlopen(source["raw_url"], timeout=30) as response:
                path.write_bytes(response.read())
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus source: {path}; run with --fetch")
        raw = path.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        title, _ = _frontmatter_title(text.splitlines(), source["document_id"])
        digest = _sha256_bytes(text.encode("utf-8"))
        corpus.append(
            {
                "document_id": source["document_id"],
                "document_name": source["document_name"],
                "title": title,
                "text": text,
                "tenant_id": "rag-f0-tenant",
                "knowledge_base_id": "rag-f0-default-kb",
                "source_url": source["source_url"],
                "source_commit": source["commit"],
                "license": source["license"],
                "sha256": digest,
            }
        )
        evidence.extend(_sections(source["document_id"], text))
        source_manifest.append({**source, "fetched_at": FETCHED_AT, "sha256": digest})

    evidence_index = {(row["document_id"], tuple(row["heading_path"])): row for row in evidence}
    specs = _read_jsonl(root / "query_specs.jsonl")
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    for spec in specs:
        relevance_specs = spec.pop("relevances", [])
        queries.append(spec)
        for relevance in relevance_specs:
            key = (relevance["document_id"], tuple(relevance["heading_path"]))
            selected = evidence_index.get(key)
            if selected is None:
                raise ValueError(f"Unknown evidence section for {spec['query_id']}: {key}")
            qrels.append(
                {
                    "query_id": spec["query_id"],
                    "evidence_id": selected["evidence_id"],
                    "relevance": relevance["relevance"],
                    "rationale": relevance["rationale"],
                    "source_quote": _source_quote(selected, relevance["quote_contains"]),
                }
            )

    _write_jsonl(root / "corpus.jsonl", corpus)
    _write_jsonl(root / "evidence.jsonl", evidence)
    _write_jsonl(root / "queries.jsonl", queries)
    _write_jsonl(root / "qrels.jsonl", qrels)
    file_hashes = {name: _sha256_bytes((root / name).read_bytes()) for name in DATA_FILES}
    dataset_sha = _sha256_bytes(_canonical_json(file_hashes).encode("utf-8"))
    manifest = {
        "dataset_version": DATASET_VERSION,
        "status": "pending_independent_dataset_acceptance",
        "dataset_sha256": dataset_sha,
        "files": file_hashes,
        "fixed_scope": {
            "tenant_id": "rag-f0-tenant",
            "knowledge_base_id": "rag-f0-default-kb",
            "document_status": "active",
        },
        "default_split": "dev",
        "holdout_policy": "frozen_not_run_in_f0",
        "out_of_scope": ["tenant_isolation", "knowledge_base_isolation", "inactive_documents"],
        "sources": source_manifest,
        "query_generation": {
            "method": "human_authored_and_reviewed",
            "model": None,
            "batch": "rag-f0-human-review-20260717",
            "prompt": "Author operational queries from pinned evidence; include paraphrase, symptom-only, hard-negative, multi-evidence, and no-answer cases. Do not promote automatic chunk rewrites to gold.",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _quality_report(root, corpus, evidence, queries, qrels, dataset_sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("evals/datasets/rag_f0"))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    build(args.root, fetch=args.fetch)


if __name__ == "__main__":
    main()
