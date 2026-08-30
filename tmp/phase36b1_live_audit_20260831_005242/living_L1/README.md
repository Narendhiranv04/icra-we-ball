# Case Audit Report: living_L1 (LIVING_ROOM L1)

- **Domain**: living_room
- **Paper Label**: L1
- **Internal Variant**: F0_ALL_OBJECTS_IN_STAGING
- **Task Instruction**: serve tea
- **Code SHA**: `a0149206eca005d5f7026986d499c9236d986031`
- **Model**: `qwen35-9b` (`http://127.0.0.1:18000/v1`)
- **VLM Calls**: 0

## Outcomes
- **Live Specification Status**: `VLM_SPEC_FAILED`
- **Live Failure Category**: `TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE`
- **Live Pipeline Terminal Result**: `VLM_SPEC_FAILED` (Exit code 1)
- **Provider Replay**: `NOT_RUN`
- **Random Replay Seed 0**: `NOT_RUN`
- **Random Replay Seed 1**: `NOT_RUN`
- **Specification SHA256**: `N/A`

## Reference G_F Evaluation
- **Reference Complete**: N/A (No Candidate G_F)
- **Exact Structural Match**: N/A

## Prompt Leakage Audit
- Zero leakage of GT object IDs, internal checker APIs, or simulator region catalogs.
