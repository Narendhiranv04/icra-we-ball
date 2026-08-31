# Case Audit Report: living_L1 (LIVING_ROOM L1)

- **Domain**: `living_room`
- **Paper Variant**: `L1`
- **Internal Variant**: `F0_ALL_OBJECTS_IN_STAGING`
- **Exact Task Instruction**: `Prepare the living room for two people watching television: put one cup and saucer on each personal support and the remote on a shared support.`
- **Frozen Runtime Code SHA**: `a0149206eca005d5f7026986d499c9236d986031`
- **Model**: `qwen35-9b` (`http://127.0.0.1:18000/v1`)
- **VLM Request Attempts**: 1
- **VLM Responses Received**: 1

## Outcomes
- **Specification Terminal Status**: `VLM_SPEC_FAILED`
- **Canonical G_F Produced**: `False`
- **Runtime Failure Category**: `MALFORMED_VLM_SPECIFICATION`
- **Failure Layer**: `MODEL_OUTPUT`
- **Failure Subtype**: `MALFORMED_VLM_SPECIFICATION`
- **Live Terminal Result**: `VLM_SPEC_FAILED` (Exit code 1)
- **Provider Replay**: `NOT_RUN`
- **Random Replay Seed 0**: `NOT_RUN`
- **Random Replay Seed 1**: `NOT_RUN`
- **Specification SHA256 Identity**: `N/A`
- **Prompt Leakage Audit**: `PASS`

## Reference G_F Evaluation
- **Reference Complete**: N/A
- **Exact Structural Match**: N/A
