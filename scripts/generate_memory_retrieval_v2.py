"""Build the frozen synthetic M-P1-R2 retrieval catalog and manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "memory_retrieval_v2.json"
MANIFEST = ROOT / "evals" / "datasets" / "memory_retrieval_v2.manifest.json"


# Each row is deliberately authored as an operational question rather than a
# service-name template. Topic keys resolve to the already curated fixtures.
SEMANTIC = [
    ("S01", "paraphrase", "Checkout requests wait for a database connection after payment cancellations; pool wait is high but CPU is normal.", ["checkout PostgreSQL pool leak"], ["staging MySQL pool undersized"], "checkout-api", "production"),
    ("S02", "symptom_only", "The connection pool appears exhausted even though callers return normally; what prior incident distinguishes a leak from capacity?", ["staging MySQL pool undersized"], ["checkout PostgreSQL pool leak"], None, "staging"),
    ("S03", "hard_negative", "Orders writes queue behind a schema change while a report transaction remains open.", ["orders DDL lock queue"], ["MySQL replica SQL thread lag"], "orders-db", "production"),
    ("S04", "exact_keyword", "Catalog JSONB containment queries regressed after statistics drift and need a GIN path-ops index.", ["catalog JSONB slow query"], ["MySQL replica SQL thread lag"], "catalog-api", "production"),
    ("S05", "multi_evidence", "Reporting replica replay delay followed a bulk WAL load and slow disk flush; compare the PostgreSQL and MySQL lag remedies.", ["PostgreSQL replica replay lag", "MySQL replica SQL thread lag"], [], None, "production"),
    ("S06", "paraphrase", "One session key overloads a Redis shard while total memory remains low.", ["session Redis hot key"], ["recommendation Redis eviction"], "session-service", "production"),
    ("S07", "hard_negative", "Orders show old status because the invalidation consumer lost its Kafka assignment, not because Redis TTL is too long.", ["orders cache invalidation lag"], ["current negative cache policy"], "orders-api", "production"),
    ("S08", "exact_keyword", "A model rollout doubled object size and volatile-lru started evicting recommendation values.", ["recommendation Redis eviction"], ["Redis blocked clients from Lua"], "recommendation-api", "production"),
    ("S09", "symptom_only", "Cart writes stall with blocked Redis clients while connection count is below maxclients.", ["Redis blocked clients from Lua"], ["staging Redis connection exhaustion"], "cart-service", "production"),
    ("S10", "paraphrase", "Payment settlement consumer lag is caused by slow database commits in each poll transaction.", ["payments Kafka consumer lag"], ["Kafka rebalance loop"], "settlement-worker", "production"),
    ("S11", "hard_negative", "Most enterprise audit records land on one Kafka partition because the key uses tier instead of tenant and event identity.", ["Kafka partition hotspot"], ["payments Kafka consumer lag"], "audit-pipeline", "production"),
    ("S12", "exact_keyword", "The profile enrichment group repeatedly rebalances when external lookups exceed max.poll.interval.ms.", ["Kafka rebalance loop"], ["Kafka poison record"], "profile-enricher", "production"),
    ("S13", "negation_conflict", "Pricing pods crash immediately because CONFIG_PATH names a file different from the mounted ConfigMap; it is not an OOM.", ["CrashLoop from missing config"], ["CrashLoop from OOM"], "pricing-api", "production"),
    ("S14", "negation_conflict", "Image processors exit 137 after large TIFF decode and the previous termination reason is OOMKilled.", ["CrashLoop from OOM"], ["CrashLoop from missing config"], "image-processor", "production"),
    ("S15", "hard_negative", "Nodes become NotReady after the CNI loses its IPAM socket while disk and certificates are healthy.", ["Node NotReady CNI failure"], ["Node NotReady disk pressure"], "cluster-network", "production"),
    ("S16", "exact_keyword", "Containerd snapshots fill /var/lib/containerd and the worker reports DiskPressure and NotReady.", ["Node NotReady disk pressure"], ["Node NotReady CNI failure"], "worker-nodes", "production"),
    ("S17", "hard_negative", "Analytics pods are Pending because namespace CPU request quota is exhausted, not because of node affinity.", ["Pending pods quota"], ["Pending pods affinity"], "analytics-jobs", "production"),
    ("S18", "paraphrase", "Gateway pods restart during certificate reload when the liveness endpoint shares a blocked worker pool and times out.", ["Liveness probe timeout"], ["Gateway upstream timeout"], "edge-gateway", "production"),
    ("S19", "hard_negative", "Shipping returns 504 when the upstream exceeds its endpoint-specific read timeout; DNS and gateway CPU are normal.", ["Gateway upstream timeout"], ["Obsolete gateway timeout rule"], "shipping-api", "production"),
    ("S20", "exact_keyword", "Retries from client, mesh, and gateway multiply recommendation traffic during a short outage.", ["Gateway retry storm"], ["Partner API rate limit"], "recommendation-api", "production"),
    ("S21", "paraphrase", "A partner responds with 429 after one API key exceeds a 120 requests-per-minute token bucket.", ["Partner API rate limit"], ["Gateway upstream timeout"], "partner-export", "production"),
    ("S22", "symptom_only", "A filesystem has free bytes but file creation fails because millions of tiny checkpoint files consumed all inodes.", ["Inode exhaustion"], ["Unrotated application logs"], "log-collector", "production"),
    ("S23", "exact_keyword", "Disk stays full after deleting a huge access log; lsof reports the proxy still has the deleted inode open.", ["Deleted open log file"], ["Temporary export files"], "legacy-proxy", "production"),
    ("S24", "multi_evidence", "Report worker disks grow from failed exports and retention cleanup; the remedy must cover both finally deletion and a janitor.", ["Temporary export files", "Unrotated application logs"], [], "report-worker", "production"),
    ("S25", "paraphrase", "Ingress serves an expired certificate because the renewed secret is in a namespace the controller does not watch.", ["TLS certificate rotation"], ["Obsolete manual certificate copy"], "public-ingress", "production"),
    ("S26", "exact_keyword", "JWT rotation must publish old and new public keys until the longest token expires.", ["Signing key overlap"], ["Authentication clock skew"], "auth-service", "production"),
    ("S27", "hard_negative", "mTLS verification fails after renewal when the secret contains the leaf but omits the intermediate CA.", ["mTLS missing intermediate"], ["TLS certificate rotation"], "service-mesh", "production"),
    ("S28", "hard_negative", "Production requests are throttled by the CPU CFS quota while a staging workload shows high usage with no throttling.", ["CPU CFS throttling", "Staging CPU not throttled"], [], None, None),
    ("S29", "symptom_only", "JVM pauses correlate with old-generation collection and not with a steadily growing heap leak.", ["JVM GC pause"], ["Heap memory leak"], "reporting-service", "production"),
    ("S30", "paraphrase", "A worker retries the same malformed task forever; quarantine the poison item without reducing normal concurrency.", ["Worker poison task"], ["Batch database contention"], "document-worker", "production"),
]

NO_MATCH = [
    ("N01", "A GPU cluster scheduler reports topology spread failures for distributed training jobs."),
    ("N02", "The frontend team needs advice about CSS typography and browser font rendering."),
    ("N03", "Finance needs to reconcile an invoice ledger and an accounting accrual."),
    ("N04", "Windows Active Directory replication is failing between domain controllers."),
    ("N05", "An iOS application crashes in Swift after a camera permission change."),
    ("N06", "A registrar cannot publish a DNSSEC DS record for a new domain."),
    ("N07", "An ML feature store has stale offline feature materialization for training."),
    ("N08", "Email deliverability dropped because a marketing campaign triggered a mailbox provider block."),
    ("N09", "A UPS battery cabinet reports low charge and needs an electrical maintenance procedure."),
    ("N10", "Oracle RAC services fail over between database instances during a patch window."),
]

ISOLATION = [
    ("I01", "tenant", "A private transaction-pooling proxy for another tenant's checkout service is requested.", "private tenant PostgreSQL proxy"),
    ("I02", "tenant", "Return the dedicated Redis shard used by a different tenant's loyalty service.", "private tenant Redis shard"),
    ("I03", "tenant", "What scheduler profile does the regulated batch tenant use on its private nodes?", "private tenant scheduler profile"),
    ("I04", "user", "Show the private Kafka offset replay procedure recorded by another user.", "private user Kafka replay"),
    ("I05", "user", "Which certificate bundle path did the other user store for a legacy appliance?", "private user certificate path"),
    ("I06", "user", "Apply the one-off finance-close worker override from another user.", "private user batch override"),
    ("I07", "agent", "Return the emergency gateway route owned by another agent's sandbox.", "private agent gateway route"),
    ("I08", "agent", "What direct-memory flag did the proprietary transcoder agent record?", "private agent memory tuning"),
]

HOLDOUT = [
    ("H01", "semantic_ranking", "The checkout pool wait follows canceled requests that skip a close operation.", ["checkout PostgreSQL pool leak"]),
    ("H02", "semantic_ranking", "A long report transaction blocks an ACCESS EXCLUSIVE orders migration.", ["orders DDL lock queue"]),
    ("H03", "semantic_ranking", "A Redis Lua script scans an unbounded set and occupies the event loop.", ["Redis blocked clients from Lua"]),
    ("H04", "semantic_ranking", "Kafka processing exceeds the poll interval during slow external lookups and assignments flap.", ["Kafka rebalance loop"]),
    ("H05", "semantic_ranking", "DiskPressure is caused by containerd snapshots filling the worker volume.", ["Node NotReady disk pressure"]),
    ("H06", "semantic_ranking", "JWT verification requires an overlap window for old and new signing keys.", ["Signing key overlap"]),
    ("H07", "semantic_ranking", "A batch backfill uses too many concurrent range scans and increases online lock waits.", ["Batch database contention"]),
    ("H08", "semantic_ranking", "The video worker loses ownership because it never renews queue visibility.", ["Worker heartbeat timeout"]),
    ("H09", "no_match", "A laboratory instrument controller loses its Modbus serial connection." , []),
    ("H10", "no_match", "A legal team needs a contract clause comparison for a vendor renewal.", []),
    ("H11", "identity_isolation", "Give me another agent's private audit retention mount.", ["private agent retention path"]),
    ("H12", "identity_isolation", "Show the other user's private batch close override.", ["private user batch override"]),
]


def _topic_map(fixtures: list[dict]) -> dict[str, dict]:
    return {fixture["topic"]: fixture for fixture in fixtures}


def _case(case_id: str, track: str, category: str, query: str, qrel_topics: list[str], forbidden_topics: list[str], topics: dict[str, dict], *, identity: str = "primary", axis: str | None = None, service: str | None = None, env: str | None = None, tags: list[str] | None = None, split: str = "dev") -> dict:
    qrels = [_topic_map(topics, topic)["evidence_id"] for topic in qrel_topics]
    forbidden = [_topic_map(topics, topic)["evidence_id"] for topic in forbidden_topics]
    slices = [category.replace("_", "-")]
    if service is not None:
        slices.append("scope-service-filter")
    if env is not None:
        slices.append("scope-env-filter")
    if tags:
        slices.append("tag-filter")
    return {
        "case_id": f"MPR2-{split[0].upper()}-{case_id}",
        "split": split,
        "track": track,
        "category": category,
        "slices": slices,
        "identity": identity,
        "identity_axis": axis,
        "query": query,
        "relevant_evidence_ids": qrels,
        "qrel_rationales": {evidence_id: f"{topic} is direct operational evidence for this query's stated symptom and scope" for topic, evidence_id in zip(qrel_topics, qrels, strict=True)},
        "forbidden_evidence_ids": forbidden,
        "hard_negative_rationales": {evidence_id: f"{topic} is a similar symptom with a distinct root cause or environment" for topic, evidence_id in zip(forbidden_topics, forbidden, strict=True)},
        "expected_empty": track != "semantic_ranking",
        "optional_type": None,
        "scope_service": service,
        "scope_env": env,
        "tags": tags or [],
    }


def _topic_map(topics: dict[str, dict], topic: str) -> dict:
    if topic not in topics:
        raise KeyError(topic)
    return topics[topic]


def main() -> None:
    source = json.loads(DATASET.read_text(encoding="utf-8"))
    topics = {fixture["topic"]: fixture for fixture in source["fixtures"]}
    queries = []
    for case_id, category, query, qrels, forbidden, service, env in SEMANTIC:
        tags = ["rebalance"] if case_id == "S12" else None
        item = _case(case_id, "semantic_ranking", category, query, qrels, forbidden, topics, service=service, env=env, tags=tags)
        if case_id in {"S04", "S12", "S19", "S25"}:
            item["slices"].append("version-configuration")
        queries.append(item)
    for case_id, query in NO_MATCH:
        queries.append(_case(case_id, "no_match", "no_match", query, [], [], topics))
    for case_id, axis, query, forbidden_topic in ISOLATION:
        queries.append(_case(case_id, "identity_isolation", "identity_isolation", query, [], [forbidden_topic], topics, identity="primary", axis=axis))
    for case_id, track, query, topics_or_forbidden in HOLDOUT:
        if track == "identity_isolation":
            axis = "agent" if "agent" in query else "user"
            queries.append(_case(case_id, track, "holdout", query, [], topics_or_forbidden, topics, identity="primary", axis=axis, split="holdout"))
        else:
            queries.append(_case(case_id, track, "holdout", query, topics_or_forbidden, [], topics, split="holdout"))
    payload = {
        "version": "2.0.0",
        "name": source["name"],
        "provenance": source["provenance"],
        "fixtures": source["fixtures"],
        "queries": queries,
    }
    data = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    DATASET.write_bytes(data)
    manifest = {
        "version": "2.0.0",
        "dataset_file": DATASET.name,
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "fixture_evidence": {f["fixture_id"]: f["evidence_id"] for f in source["fixtures"]},
        "query_ids": [q["case_id"] for q in queries],
        "dev_query_ids": [q["case_id"] for q in queries if q["split"] == "dev"],
        "holdout_query_ids": [q["case_id"] for q in queries if q["split"] == "holdout"],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
