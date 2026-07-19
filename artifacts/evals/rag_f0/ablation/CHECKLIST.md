# F0 Ablation Checklist

- [x] Bind campaign to accepted F0 real dev baseline and frozen Dataset SHA.
- [x] Verify LlamaIndex DEFAULT/SPARSE/HYBRID APIs and common MetadataFilters path.
- [x] Implement mode-aware Milvus store and retrieval service without a second adapter.
- [x] Prove BM25 skips query embedding and all modes retain tenant/KB plus active-document scope.
- [x] Remove Model Gateway credential mapping from the baseline entrypoint and shared factory.
- [x] Pass focused, RAG joint, full stub, eval, Ruff, compileall, pip, and diff gates.
- [x] Run Dense-only once on 48 dev queries and clean all resources.
- [x] Run BM25-only once on 48 dev queries and clean all resources.
- [x] Run Hybrid once on 48 dev queries and clean all resources.
- [x] Aggregate paired deltas, slices, latency, overlap, and error analysis.
- [x] Freeze artifact hashes, commit, push, and retain Draft PR status.
