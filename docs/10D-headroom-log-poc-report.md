# CTX-P0A: Headroom Log Compression Lightweight PoC

Status: `completed`
Run ID: `20260716T061036Z`
Decision: **No-Go** for production `ToolResultReducer` integration in the tested configuration.

## 1. Scope and decision rule

This PoC evaluates `headroom-ai==0.31.0` through the Python library
`headroom.compress()` only. It does not change production code and does not use the Headroom
proxy, MCP, CCR retrieval, local Kompress model, or the `ml`/`evals`/`all` extras.

Correctness is the hard gate. A missing error code, trace ID, instance, or negative fact makes that
content type No-Go. Token reduction and latency are benefits, not correctness substitutes. The
Headroom official benchmark is reference material only and is not used as evidence for this
decision.

## 2. Implemented configuration

- Package: `headroom-ai==0.31.0` from the base distribution only.
- Headroom release tag: [`v0.31.0`](https://github.com/headroomlabs-ai/headroom/releases/tag/v0.31.0),
  commit `55efb1c77d5b67f7ad0620372c6256c8b0547591`.
- Library call: `headroom.compress(messages, model="qwen-plus", config=...)`.
- `CompressConfig`: `kompress_model="disabled"`, `protect_recent=0`,
  `min_tokens_to_compress=1000`, `compress_user_messages=False`,
  `compress_system_messages=False`.
- `CompressResult` consumption: `messages`, `tokens_before`, `tokens_after`,
  `compression_ratio`, and `transforms_applied` are read explicitly. The result is never converted
  with `str()`.
- Project token metric: `ApproxTokenEstimator` over the full message envelope.
- Headroom token metric: Headroom `EstimatingTokenCounter`, registered against the unchanged
  configured model name `qwen-plus`.

Headroom 0.31.0 maps `qwen*` to its HuggingFace tokenizer. In the base install the tokenizer class
imports successfully but later raises `ModuleNotFoundError: transformers` during counting instead
of taking the documented estimator fallback. Installing `transformers` would violate this PoC's
no-ML dependency boundary. The runner therefore registers Headroom's own no-ML estimator for
`qwen-plus` and records that choice in `run_manifest.json`. The initial failed smoke result is
retained locally under `smoke-20260716T061036Z`; the valid smoke and main run use the explicit
estimator registration.

## 3. Real dataset provenance

All log messages are exact physical lines from [LogPAI Loghub](https://github.com/logpai/loghub) at
commit [`dd61d0952749ee7963bde24220d1be5ede023033`](https://github.com/logpai/loghub/commit/dd61d0952749ee7963bde24220d1be5ede023033).
The committed dataset wraps lines in a `queryLogs` JSON envelope and adds line numbers; it does not
rewrite the source `message`. Per-line SHA-256 provenance is stored beside each case and is not sent
to Headroom, so integrity metadata does not distort the tool-result shape or token metrics.

| Source | Logical lines | Bytes | SHA-256 |
|---|---:|---:|---|
| `Apache/Apache_2k.log` | 2,000 | 171,239 | `c7efa3eb686e3a96bd2f8f4457b2a7887e9cf2f3649327f1b4e87af841363ce8` |
| `OpenStack/OpenStack_2k.log` | 2,000 | 595,119 | `025a1bc64ff5b2ef4a4bda6c4ad5c5c5f18478b71cd1ad2b0676e01625629f2f` |
| `HDFS/HDFS_2k.log` | 2,000 | 287,848 | `7c967000980c086ed55fa6544ba4f05fe66d44622795e890c68caf8bbb635035` |
| `Zookeeper/Zookeeper_2k.log` | 2,000 | 279,891 | `e40e0af5ef9eb6e4097200f260b9d1f626b3676f861a432e87977242e75543d8` |
| `Linux/Linux_2k.log` | 2,000 | 216,485 | `b3e20bc1afe732ab1bf3ed1de4bf9c809e4194e02f7dea911d918e5342e8e173` |

The source files have 2,000 logical lines; four omit a final newline, so `wc -l` reports 1,999 for
those files. The fetcher validates logical line count, byte count, and SHA-256.

Loghub states that the datasets are freely available for research or academic work, and asks users
to reference the repository and cite the paper where applicable. This PoC follows its
[`LICENSE`](https://github.com/logpai/loghub/blob/dd61d0952749ee7963bde24220d1be5ede023033/LICENSE)
and cites Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, and Michael R. Lyu,
"[Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics](https://arxiv.org/abs/2008.06448),"
ISSRE 2023. Raw downloads remain in ignored `cache/` storage and are not committed.

## 4. Dataset composition

The dataset contains 10 cases from all five sources:

- 8 compression candidates, each above 1,000 project-estimated tokens: 3,116 to 7,679 tokens.
- 2 below-threshold negative controls: 322 and 497 tokens.
- Critical events at head, middle, and tail positions.
- Repeated worker/authentication failures, block-serving exceptions, a ZooKeeper IOException,
  OpenStack HTTP exceptions, ordinary logs, and two explicit no-fault controls.
- Size across all cases: 5 to 100 log lines and 322 to 7,679 estimated tokens.

Dataset path: `evals/datasets/headroom_log_poc_v1.json` (294 KiB). Each case includes `source`,
`user_query`, `tool_args`, `raw_tool_result`, `expected_facts`, `size`, and
`critical_event_position`.

## 5. Compression results

Headroom's `compression_ratio` below is the fraction of input tokens removed. Latency is local
wall time for one synchronous `compress()` call; the first row includes cold pipeline startup.

| Case | Headroom before | after | removed | latency ms | transforms | fact check | contract |
|---|---:|---:|---:|---:|---|---|---|
| `openstack_404_trace_instance_middle` | 9,963 | 2,369 | 76.22% | 151.764 | `protected:user_message`, `mixed:0.13` | **failed** | passed |
| `openstack_404_trace_tail` | 9,177 | 2,491 | 72.86% | 66.046 | `protected:user_message`, `mixed:0.17` | **failed** | passed |
| `hdfs_block_exception_head` | 5,104 | 4,259 | 16.56% | 9.509 | `protected:user_message`, `mixed:0.73` | passed | passed |
| `hdfs_block_exception_tail` | 4,302 | 3,598 | 16.36% | 8.186 | `protected:user_message`, `mixed:0.73` | passed | passed |
| `zookeeper_ioexception_middle` | 4,432 | 1,142 | 74.23% | 14.755 | `protected:user_message`, `mixed:0.18` | passed | passed |
| `zookeeper_repeated_shutdown_head` | 5,125 | 1,437 | 71.96% | 9.980 | `protected:user_message`, `mixed:0.24` | passed | passed |
| `apache_repeated_worker_errors_tail` | 4,011 | 3,082 | 23.16% | 7.542 | `protected:user_message`, `mixed:0.74` | passed | passed |
| `linux_auth_burst_and_logrotate_tail` | 4,317 | 3,434 | 20.45% | 7.977 | `protected:user_message`, `mixed:0.76` | passed | passed |
| `hdfs_normal_below_threshold_control` | 505 | 505 | 0.00% | 0.259 | `protected:user_message` | passed | passed |
| `openstack_success_below_threshold_control` | 751 | 751 | 0.00% | 0.337 | `protected:user_message` | passed | passed |

For the 8 candidates, aggregate Headroom tokens were 46,431 before and 21,812 after: 24,619
tokens removed (53.02%). The median per-case removal was 47.56%; median latency was 9.745 ms.
The project estimator independently measured 37,092 before and 17,709 after, a reduction of
19,383 tokens (52.26%). These two estimators are reported separately and are not treated as
interchangeable.

All 10 cases preserved message count, role order, assistant tool call, `tool_call_id`, tool name,
parseable tool-result JSON, and original root JSON keys. Neither estimator observed token
inflation. Both below-threshold controls were byte-for-byte unchanged. No compressed output added
one of the case's forbidden fault conclusions.

## 6. Fact failures

Headroom replaced many OpenStack rows with `<<ccr:...>>` markers. CCR retrieval is explicitly out
of scope, so a marker is not equivalent to preserving a fact in the model-visible result.

| Case | Preserved | Lost |
|---|---|---|
| `openstack_404_trace_instance_middle` | error code `404`; instance `b9000564-fe1a-409b-b8cc-1e88b294cd1d` | trace ID `req-0b851395-2895-44b9-8265-a27d0bb52910`; timestamp `2017-05-16 00:00:21.067`; latency `0.0793190`; `HTTP exception thrown` |
| `openstack_404_trace_tail` | error code `404`; instance `70c1714b-c11b-4c88-b300-239afe1f5ff8` | trace ID `req-ea4d9b17-3021-441c-a951-975ae6253a8b`; timestamp `2017-05-16 00:09:19.366`; latency `0.0890410`; `No instances found for any event` |

The missing trace IDs are explicit hard-gate violations. The same loss appears with the critical
event in the middle and at the tail, so OpenStack-style high-cardinality JSON logs are **No-Go** in
this configuration.

All expected HDFS block IDs/DataNodes, ZooKeeper endpoints/session IDs/exception types, Apache
state/client/path facts, Linux source/user/status facts, timestamps, and both negative statements
were mechanically preserved.

## 7. qwen model A/B

The runner selected four mechanically passing representatives:

| Case | Raw call | Compressed call | Regression eligibility |
|---|---|---|---|
| `hdfs_block_exception_head` | pending | pending | not evaluated |
| `zookeeper_ioexception_middle` | pending | pending | not evaluated |
| `linux_auth_burst_and_logrotate_tail` | pending | pending | not evaluated |
| `hdfs_block_exception_tail` | pending | pending | not evaluated |

Project `Settings` resolved `model_provider="stub"` and `model_name="qwen-plus"`. Per the execution
rules, the runner did not change provider/model, did not read or print `.env` or API keys directly,
and made **0 of 8** planned calls. Model A/B is `pending`, not passed. No case can be used for a
model regression claim because no raw baseline answer exists.

## 8. Decision by content type

| Content type | Mechanical result | Model A/B | Decision |
|---|---|---|---|
| OpenStack high-cardinality HTTP/trace logs | trace IDs and event details lost in 2/2 | pending | **No-Go** |
| HDFS repeated block logs | 2/2 passed; 16.4% to 16.6% removed | pending | mechanical evidence only |
| ZooKeeper repeated/session logs | 2/2 passed; 72.0% to 74.2% removed | pending | mechanical evidence only |
| Apache repeated worker logs | 1/1 passed; 23.2% removed | pending | mechanical evidence only |
| Linux repeated authentication logs | 1/1 passed; 20.5% removed | pending | mechanical evidence only |
| Below-threshold controls | 2/2 unchanged | not selected | passed |

Overall production recommendation: **No-Go**. Token reduction is real, latency is small after cold
startup, and four log families passed mechanical checks, but the tested library-only result loses
critical trace facts in a representative OpenStack shape. Correctness is a hard gate, qwen A/B is
also still pending, and this PoC intentionally provides no raw retrieval or production fallback
that could compensate for the loss.

The next justified action is not production integration. A follow-up would first need a separately
scoped safety experiment that disables lossy/CCR row replacement for trace-heavy payloads or proves
protected-pattern retention, then repeats the qwen A/B with the project's configured real endpoint.

## 9. Reproduction and artifacts

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]' -c constraints/headroom-poc.txt headroom-ai
.venv/bin/python scripts/fetch_headroom_poc_logs.py
.venv/bin/python scripts/run_headroom_log_poc.py --run-id 20260716T061036Z
```

Ignored local artifacts are under
`artifacts/evals/context/headroom_log_poc/20260716T061036Z/`, including:

- `run_manifest.json`, `artifact_manifest.json`, and environment snapshot;
- `case_results.json` with compressed messages and all per-fact checks;
- `metrics.json`, `metrics.md`, `model_ab.json`, and `claim_validation.md`;
- `PLAN.md`, `CHECKLIST.md`, `summary.md`, and `runlog.summary.md`.

The official Headroom README reports broader benchmark results such as 60-95% token reduction for
JSON workloads. Those numbers describe Headroom's own workloads and are not substituted for the
53.02% aggregate reduction or the correctness failures measured here.
