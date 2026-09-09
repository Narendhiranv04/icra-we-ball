# Fresh three-domain joint-grounding gate

Exactly one fresh Qwen thinking request was made for each of K1, L1, and W3. Each request used the first three production views for its domain. There were no transport retries, semantic retries, high-level replans, or GT/reference inputs.

| Case | Strict V2 | Semantic coherence | Compiled | Grounding | A* | Final outcome |
|---|---|---|---|---|---:|---|
| K1 | PASS | PASS | PARTIAL | not admitted | 0 | GRAPH_COMPILATION_FAILURE |
| L1 | PASS | FAIL | no | not reached | 0 | TASK_SPECIFICATION_FAILURE |
| W3 | PASS | FAIL | no | not reached | 0 | TASK_SPECIFICATION_FAILURE |

The aggregate is three semantic FM calls, three transport attempts, zero retries, and zero A* calls. K1's raw response was replayed without an FM call after the final compiler-boundary assertion was added; its partial graph is now rejected before grounding. The precise compact diagnostics are in `summary.json`; raw responses, images, render caches, and scene artifacts are intentionally excluded.
