#!/usr/bin/env python3
"""Fetch pinned Loghub samples and materialize the CTX-P0A dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from superbiz_agent.harness.token_estimator import ApproxTokenEstimator  # noqa: E402


LOGHUB_REPOSITORY = "https://github.com/logpai/loghub"
LOGHUB_COMMIT = "dd61d0952749ee7963bde24220d1be5ede023033"
LOGHUB_RAW_ROOT = f"https://raw.githubusercontent.com/logpai/loghub/{LOGHUB_COMMIT}"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cache" / "headroom_log_poc" / "loghub" / LOGHUB_COMMIT
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "headroom_log_poc_v1.json"
COMPRESSION_THRESHOLD_TOKENS = 1_000


@dataclass(frozen=True)
class LogSource:
    name: str
    path: str
    sha256: str
    line_count: int
    byte_count: int

    @property
    def url(self) -> str:
        return f"{LOGHUB_RAW_ROOT}/{self.path}"


LOG_SOURCES = (
    LogSource(
        name="Apache",
        path="Apache/Apache_2k.log",
        sha256="c7efa3eb686e3a96bd2f8f4457b2a7887e9cf2f3649327f1b4e87af841363ce8",
        line_count=2_000,
        byte_count=171_239,
    ),
    LogSource(
        name="OpenStack",
        path="OpenStack/OpenStack_2k.log",
        sha256="025a1bc64ff5b2ef4a4bda6c4ad5c5c5f18478b71cd1ad2b0676e01625629f2f",
        line_count=2_000,
        byte_count=595_119,
    ),
    LogSource(
        name="HDFS",
        path="HDFS/HDFS_2k.log",
        sha256="7c967000980c086ed55fa6544ba4f05fe66d44622795e890c68caf8bbb635035",
        line_count=2_000,
        byte_count=287_848,
    ),
    LogSource(
        name="Zookeeper",
        path="Zookeeper/Zookeeper_2k.log",
        sha256="e40e0af5ef9eb6e4097200f260b9d1f626b3676f861a432e87977242e75543d8",
        line_count=2_000,
        byte_count=279_891,
    ),
    LogSource(
        name="Linux",
        path="Linux/Linux_2k.log",
        sha256="b3e20bc1afe732ab1bf3ed1de4bf9c809e4194e02f7dea911d918e5342e8e173",
        line_count=2_000,
        byte_count=216_485,
    ),
)
SOURCE_BY_NAME = {source.name: source for source in LOG_SOURCES}


@dataclass(frozen=True)
class ExpectedFact:
    kind: str
    value: str
    source_line: int | None = None


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    source_name: str
    start_line: int
    end_line: int
    user_query: str
    query: str
    critical_event_position: str
    critical_line: int | None
    expected_facts: tuple[ExpectedFact, ...]
    ab_candidate: bool = False
    negative_control: bool = False
    selection_summary: str | None = None
    forbidden_fault_conclusions: tuple[str, ...] = ()


CASE_SPECS = (
    CaseSpec(
        case_id="openstack_404_trace_instance_middle",
        source_name="OpenStack",
        start_line=20,
        end_line=110,
        user_query=(
            "Identify the HTTP failure in this OpenStack window. Return the status code, "
            "request trace ID, affected instance, timestamp, and request latency exactly."
        ),
        query="HTTP failure status trace instance latency",
        critical_event_position="middle",
        critical_line=60,
        expected_facts=(
            ExpectedFact("error_code", "404", 61),
            ExpectedFact("trace_id", "req-0b851395-2895-44b9-8265-a27d0bb52910", 60),
            ExpectedFact("instance", "b9000564-fe1a-409b-b8cc-1e88b294cd1d", 47),
            ExpectedFact("timestamp", "2017-05-16 00:00:21.067", 60),
            ExpectedFact("numeric_value", "0.0793190", 61),
            ExpectedFact("exception_type", "HTTP exception thrown", 60),
        ),
        ab_candidate=True,
        forbidden_fault_conclusions=("HTTP 500", "OutOfMemoryError", "service outage"),
    ),
    CaseSpec(
        case_id="openstack_404_trace_tail",
        source_name="OpenStack",
        start_line=1_170,
        end_line=1_254,
        user_query=(
            "Find the tail HTTP exception. Return its status code, request trace ID, instance ID, "
            "timestamp, and latency exactly."
        ),
        query="tail HTTP exception request instance latency",
        critical_event_position="tail",
        critical_line=1_253,
        expected_facts=(
            ExpectedFact("error_code", "404", 1_254),
            ExpectedFact("trace_id", "req-ea4d9b17-3021-441c-a951-975ae6253a8b", 1_253),
            ExpectedFact("instance", "70c1714b-c11b-4c88-b300-239afe1f5ff8", 1_248),
            ExpectedFact("timestamp", "2017-05-16 00:09:19.366", 1_253),
            ExpectedFact("numeric_value", "0.0890410", 1_254),
            ExpectedFact("exception_type", "No instances found for any event", 1_253),
        ),
    ),
    CaseSpec(
        case_id="hdfs_block_exception_head",
        source_name="HDFS",
        start_line=70,
        end_line=160,
        user_query=(
            "Report the first block-serving exception with its DataNode, block ID, destination, "
            "timestamp, and exception wording exactly."
        ),
        query="first block serving exception DataNode destination",
        critical_event_position="head",
        critical_line=78,
        expected_facts=(
            ExpectedFact("timestamp", "081109 214043", 78),
            ExpectedFact("instance", "10.251.30.85:50010", 78),
            ExpectedFact("block_id", "blk_-2918118818249673980", 78),
            ExpectedFact("host", "/10.251.90.64", 78),
            ExpectedFact("exception_type", "Got exception while serving", 78),
        ),
        ab_candidate=True,
        forbidden_fault_conclusions=("data loss", "disk failure", "OutOfMemoryError"),
    ),
    CaseSpec(
        case_id="hdfs_block_exception_tail",
        source_name="HDFS",
        start_line=250,
        end_line=325,
        user_query=(
            "Report the late block-serving exception with its DataNode, block ID, destination, "
            "timestamp, and exception wording exactly."
        ),
        query="late block serving exception DataNode destination",
        critical_event_position="tail",
        critical_line=319,
        expected_facts=(
            ExpectedFact("timestamp", "081110 072124", 319),
            ExpectedFact("instance", "10.251.42.246:50010", 319),
            ExpectedFact("block_id", "blk_-7658293778087733436", 319),
            ExpectedFact("host", "/10.251.30.179", 319),
            ExpectedFact("exception_type", "Got exception while serving", 319),
        ),
    ),
    CaseSpec(
        case_id="zookeeper_ioexception_middle",
        source_name="Zookeeper",
        start_line=580,
        end_line=650,
        user_query=(
            "Identify the session-closing exception. Return the timestamp, session ID, exception "
            "type, service state, and server endpoint exactly."
        ),
        query="session closing exception server not running",
        critical_event_position="middle",
        critical_line=624,
        expected_facts=(
            ExpectedFact("timestamp", "2015-08-20 17:14:11,414", 624),
            ExpectedFact("instance", "0.0.0.0:2181", 624),
            ExpectedFact("trace_id", "0x0", 624),
            ExpectedFact("exception_type", "java.io.IOException", 624),
            ExpectedFact("failure_message", "ZooKeeperServer not running", 624),
        ),
        ab_candidate=True,
        forbidden_fault_conclusions=("OutOfMemoryError", "cluster data loss", "HTTP 500"),
    ),
    CaseSpec(
        case_id="zookeeper_repeated_shutdown_head",
        source_name="Zookeeper",
        start_line=730,
        end_line=820,
        user_query=(
            "Identify the repeated LearnerHandler shutdown failure near the beginning. Return the "
            "first timestamp, peer endpoint, severity, and exact failure wording."
        ),
        query="repeated LearnerHandler shutdown exception first",
        critical_event_position="head",
        critical_line=755,
        expected_facts=(
            ExpectedFact("timestamp", "2015-07-29 19:03:35,413", 755),
            ExpectedFact("instance", "/10.10.34.11:52225", 755),
            ExpectedFact("error_code", "ERROR", 755),
            ExpectedFact(
                "exception_type", "Unexpected exception causing shutdown while sock still open", 755
            ),
        ),
    ),
    CaseSpec(
        case_id="apache_repeated_worker_errors_tail",
        source_name="Apache",
        start_line=500,
        end_line=599,
        user_query=(
            "Summarize the repeated worker failures and directory denial. Return worker state 10, "
            "the denied client, timestamp, path, and exact denial wording."
        ),
        query="worker error state 10 directory index forbidden client",
        critical_event_position="tail",
        critical_line=580,
        expected_facts=(
            ExpectedFact("error_code", "error state 10", 514),
            ExpectedFact("instance", "client 63.13.186.196", 580),
            ExpectedFact("timestamp", "Sun Dec 04 07:45:45 2005", 580),
            ExpectedFact("failure_message", "Directory index forbidden by rule", 580),
            ExpectedFact("host", "/var/www/html/", 580),
        ),
    ),
    CaseSpec(
        case_id="linux_auth_burst_and_logrotate_tail",
        source_name="Linux",
        start_line=1_810,
        end_line=1_904,
        user_query=(
            "Identify the repeated SSH authentication source and the later logrotate anomaly. "
            "Return the source IP, user, final anomaly timestamp, status value, and exact message."
        ),
        query="SSH authentication burst later logrotate anomaly",
        critical_event_position="tail",
        critical_line=1_904,
        expected_facts=(
            ExpectedFact("instance", "207.243.167.114", 1_879),
            ExpectedFact("failure_message", "authentication failure", 1_879),
            ExpectedFact("host", "user=root", 1_879),
            ExpectedFact("timestamp", "Jul 27 04:16:09", 1_904),
            ExpectedFact("error_code", "[1]", 1_904),
            ExpectedFact("exception_type", "ALERT exited abnormally", 1_904),
        ),
        ab_candidate=True,
        forbidden_fault_conclusions=("kernel panic", "HTTP 500", "filesystem corruption"),
    ),
    CaseSpec(
        case_id="hdfs_normal_below_threshold_control",
        source_name="HDFS",
        start_line=1,
        end_line=6,
        user_query=(
            "Does this HDFS window contain any WARN, ERROR, FATAL, or exception entry? Return the "
            "negative finding and one normal block ID."
        ),
        query="WARN ERROR FATAL exception",
        critical_event_position="none",
        critical_line=None,
        expected_facts=(
            ExpectedFact(
                "negative_fact",
                "No WARN, ERROR, FATAL, or exception entries are present in this selected window.",
            ),
            ExpectedFact("block_id", "blk_7128370237687728475", 3),
        ),
        negative_control=True,
        selection_summary=(
            "No WARN, ERROR, FATAL, or exception entries are present in this selected window."
        ),
        forbidden_fault_conclusions=("OutOfMemoryError", "data loss", "service outage"),
    ),
    CaseSpec(
        case_id="openstack_success_below_threshold_control",
        source_name="OpenStack",
        start_line=64,
        end_line=68,
        user_query=(
            "Does this OpenStack claim window contain ERROR/FATAL or an HTTP 4xx/5xx response? "
            "Return the negative finding, instance ID, and requested memory."
        ),
        query="ERROR FATAL HTTP 4xx 5xx claim",
        critical_event_position="none",
        critical_line=None,
        expected_facts=(
            ExpectedFact(
                "negative_fact",
                "No ERROR, FATAL, or HTTP 4xx/5xx response is present in this selected window.",
            ),
            ExpectedFact("instance", "96abccce-8d1f-4e07-b6d1-4b2ab87e23b4", 64),
            ExpectedFact("numeric_value", "2048 MB", 64),
        ),
        negative_control=True,
        selection_summary=(
            "No ERROR, FATAL, or HTTP 4xx/5xx response is present in this selected window."
        ),
        forbidden_fault_conclusions=("HTTP 404", "HTTP 500", "claim failed", "service outage"),
    ),
)


class DataIntegrityError(RuntimeError):
    """Raised when a pinned source does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_opener(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": "CTX-P0A/1.0"})
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310 - pinned HTTPS URL


def fetch_source(
    source: LogSource,
    cache_root: Path,
    *,
    opener: Callable[[str], BinaryIO] = _default_opener,
    attempts: int = 3,
) -> Path:
    destination = cache_root / source.path
    if destination.exists():
        _validate_source_file(source, destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with opener(source.url) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            _validate_source_file(source, partial)
            partial.replace(destination)
            return destination
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(attempt)
    raise DataIntegrityError(f"failed to fetch {source.path}: {last_error}") from last_error


def _validate_source_file(source: LogSource, path: Path) -> None:
    actual_sha = sha256_file(path)
    if actual_sha != source.sha256:
        raise DataIntegrityError(
            f"SHA-256 mismatch for {source.path}: expected {source.sha256}, got {actual_sha}"
        )
    actual_bytes = path.stat().st_size
    if actual_bytes != source.byte_count:
        raise DataIntegrityError(
            f"byte count mismatch for {source.path}: expected {source.byte_count}, got {actual_bytes}"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        actual_lines = sum(1 for _ in handle)
    if actual_lines != source.line_count:
        raise DataIntegrityError(
            f"line count mismatch for {source.path}: expected {source.line_count}, got {actual_lines}"
        )


def fetch_all(cache_root: Path) -> dict[str, Path]:
    return {source.name: fetch_source(source, cache_root) for source in LOG_SOURCES}


def build_dataset(cache_root: Path, output_path: Path) -> dict:
    source_paths = {source.name: cache_root / source.path for source in LOG_SOURCES}
    for source in LOG_SOURCES:
        _validate_source_file(source, source_paths[source.name])

    estimator = ApproxTokenEstimator()
    cases = [_build_case(spec, source_paths[spec.source_name], estimator) for spec in CASE_SPECS]
    dataset = {
        "schema_version": "1.0",
        "dataset_id": "headroom_log_poc_v1",
        "description": (
            "queryLogs-shaped cases derived from exact lines in pinned LogPAI Loghub 2k samples"
        ),
        "headroom_version": "0.31.0",
        "compression_threshold_estimated_tokens": COMPRESSION_THRESHOLD_TOKENS,
        "source_policy": {
            "repository": LOGHUB_REPOSITORY,
            "commit": LOGHUB_COMMIT,
            "paper": "https://arxiv.org/abs/2008.06448",
            "usage": (
                "Freely available for research or academic work; reference Loghub and cite the "
                "Loghub paper where applicable."
            ),
        },
        "sources": [
            {
                **asdict(source),
                "url": source.url,
            }
            for source in LOG_SOURCES
        ],
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return dataset


def _build_case(spec: CaseSpec, source_path: Path, estimator: ApproxTokenEstimator) -> dict:
    source = SOURCE_BY_NAME[spec.source_name]
    lines = source_path.read_text(encoding="utf-8").splitlines()
    selected = lines[spec.start_line - 1 : spec.end_line]
    if len(selected) != spec.end_line - spec.start_line + 1:
        raise DataIntegrityError(f"invalid line window for {spec.case_id}")

    logs = [
        {"lineNumber": line_number, "message": message}
        for line_number, message in enumerate(selected, start=spec.start_line)
    ]
    source_message_sha256 = [
        {
            "lineNumber": row["lineNumber"],
            "sha256": hashlib.sha256(row["message"].encode("utf-8")).hexdigest(),
        }
        for row in logs
    ]
    tool_args = {
        "region": "ap-shanghai",
        "logTopic": f"loghub/{spec.source_name.lower()}",
        "query": spec.query,
        "limit": len(logs),
    }
    raw_tool_result = {
        "success": True,
        "region": tool_args["region"],
        "logTopic": tool_args["logTopic"],
        "query": tool_args["query"],
        "logs": logs,
        "total": len(logs),
        "message": f"Selected {len(logs)} exact source lines from LogPAI Loghub.",
    }
    if spec.selection_summary:
        raw_tool_result["selectionSummary"] = spec.selection_summary

    rendered = json.dumps(raw_tool_result, ensure_ascii=False, sort_keys=True)
    estimated_tokens = estimator.estimate_text(rendered)
    return {
        "case_id": spec.case_id,
        "source": {
            "repository": LOGHUB_REPOSITORY,
            "commit": LOGHUB_COMMIT,
            "path": source.path,
            "url": source.url,
            "file_sha256": source.sha256,
            "start_line": spec.start_line,
            "end_line": spec.end_line,
        },
        "user_query": spec.user_query,
        "tool_args": tool_args,
        "raw_tool_result": raw_tool_result,
        "source_message_sha256": source_message_sha256,
        "expected_facts": [asdict(fact) for fact in spec.expected_facts],
        "forbidden_fault_conclusions": list(spec.forbidden_fault_conclusions),
        "size": {
            "line_count": len(logs),
            "characters": len(rendered),
            "utf8_bytes": len(rendered.encode("utf-8")),
            "estimated_tokens": estimated_tokens,
            "threshold_class": (
                "below_threshold_control"
                if spec.negative_control
                else "compression_candidate"
            ),
        },
        "critical_event_position": spec.critical_event_position,
        "critical_line": spec.critical_line,
        "negative_control": spec.negative_control,
        "ab_candidate": spec.ab_candidate,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = fetch_all(args.cache_root)
    dataset = build_dataset(args.cache_root, args.dataset_output)
    for source in LOG_SOURCES:
        print(f"verified {source.path} sha256={source.sha256} cache={paths[source.name]}")
    print(f"materialized {len(dataset['cases'])} cases at {args.dataset_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
