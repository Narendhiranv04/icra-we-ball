# Joint relational grounding replay

This is a zero-FM-call replay of all 32 archived three-view V2 contracts from `live_v2_thinking_3view_full32_20260909T105016IST`. The archived task contracts were run against their deterministic scene variants through canonical compilation, joint grounding, operation binding, and A* where a planning contract was available.

| Outcome | Kitchen | Living | Workshop | Overall |
|---|---:|---:|---:|---:|
| Success | 1 | 3 | 1 | 5 |
| Task Specification Failure | 0 | 2 | 0 | 2 |
| Graph Compilation Failure | 0 | 0 | 1 | 1 |
| Object Discovery Failure | 11 | 3 | 0 | 14 |
| Functional Assignment Failure | 0 | 2 | 8 | 10 |
| Planning Failure | 0 | 0 | 0 | 0 |

Graph compilation succeeded for 29/30 (96.67%) task-specification-valid contracts. The replay recorded 1 relation-assisted role recovery, 9 operation-assisted role recoveries, 34 relation-assisted scene bindings, and 1 semantic-UNKNOWN scene binding accepted only after all incident required physical relations verified TRUE.

K12 now compiles a FULL four-operation contract but remains `OBJECT_DISCOVERY_FAILURE`: the observed scene lacks acceptable coffee-source, water-source, and coffee-stirrer bindings. Its 12-action projected candidate is retained diagnostically and is not reported as task success. L1 is a complete 10-action success with two personal-support and one shared-control operation binding. W3 is a complete 5-action success with all three symbolic goals and an explicit `drive_fastener_group` binding.

No raw FM response, image, rendered media, detector cache, or per-run scene artifact is duplicated here. Exact per-case outcomes and remaining reasons are in `summary.json`.
