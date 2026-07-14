# Memory Eval Judge v1

Status: optional semantic-evaluation contract. Default execution is disabled;
this file does not authorize a model call.

System instruction:

```text
You are an independent evaluator for an agent's long-term-memory behavior.
Evaluate only the supplied, redacted case, retrieved memories, tool trace, and
answer. Do not infer facts that are absent. Deterministic safety failures are
authoritative and must not be overridden. Return JSON only, conforming exactly
to the schema below.
```

Response schema:

```json
{
  "groundedness": "pass|fail|not_evaluated",
  "historical_evidence_compliance": "pass|fail|not_evaluated",
  "unsupported_claims": ["string"],
  "reasoning": "brief redacted explanation"
}
```

Inputs must include the prompt hash, judge model/version, and only redacted
message/tool/memory content. This contract requires prior human calibration;
without it, its output is advisory and cannot create a production-quality gate.
