# Live-equivalent joint-grounding replay

This is a zero-FM-call replay of all 32 archived three-view V2 contracts from `live_v2_thinking_3view_full32_20260909T105016IST`. Every raw V2 document crossed the same safe-normalization and strict-validation boundary used by live inference before compilation.

| Outcome | Kitchen | Living | Workshop | Overall |
|---|---:|---:|---:|---:|
| Success | 0 | 2 | 1 | 3 |
| Task Specification Failure | 11 | 7 | 9 | 27 |
| Graph Compilation Failure | 0 | 0 | 0 | 0 |
| Object Discovery Failure | 1 | 1 | 0 | 2 |
| Functional Assignment Failure | 0 | 0 | 0 | 0 |
| Planning Failure | 0 | 0 | 0 | 0 |

Five contracts passed task-specification validation and all five compiled (100%). None of those five contained provisional roles, so the archived sample exercised no provisional type resolutions. Four cases invoked A*: K12, L1, L3, and W3.

The stage corrections are explicit: K1 changed from `SUCCESS` to `TASK_SPECIFICATION_FAILURE` (`INCONSISTENT_OPERATION_REUSE_CARDINALITY`); K2 changed from `OBJECT_DISCOVERY_FAILURE` to `TASK_SPECIFICATION_FAILURE` (undeclared `serving_surface`); W9 changed from `GRAPH_COMPILATION_FAILURE` to `TASK_SPECIFICATION_FAILURE` (`INCOMPLETE_OPERATION_PARTICIPANT_STRUCTURE`).

No raw FM response, image, rendered media, detector cache, or per-run scene artifact is duplicated here.
