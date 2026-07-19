from __future__ import annotations

import hashlib
import json
from pathlib import Path

from superbiz_agent.evals.memory_retrieval_v2 import (
    analyze_memory_retrieval_v2_quality,
    load_memory_retrieval_v2_dataset,
    serialize_v2_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "memory_retrieval_v2.json"
MANIFEST = ROOT / "evals" / "datasets" / "memory_retrieval_v2.manifest.json"
OUTPUT = ROOT / "artifacts" / "evals" / "memory_retrieval_v2" / "dataset_quality.json"


def main() -> int:
    dataset, dataset_sha = load_memory_retrieval_v2_dataset(DATASET, MANIFEST)
    report = analyze_memory_retrieval_v2_quality(dataset)
    report.update({"dataset_sha256": dataset_sha, "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest()})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(serialize_v2_artifact(report))
    print(json.dumps({"status": report["status"], "dataset_sha256": dataset_sha, "output": str(OUTPUT.relative_to(ROOT)), "quality_gate": report["quality_gate"]}, sort_keys=True))
    return 0 if report["quality_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
