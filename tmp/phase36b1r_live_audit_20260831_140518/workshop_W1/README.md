# Case Audit Report: workshop_W1 (WORKSHOP W1)

- **Domain**: `workshop`
- **Paper Variant**: `W1`
- **Internal Variant**: `F0_MANUAL_FIRST_ONE_REGION`
- **Exact Task Instruction**: `Find a compatible screw and driver, insert the screw tip-down into the workbench repair hole, drive it fully, and return the driver safely.`
- **Frozen Runtime Code SHA**: `a0149206eca005d5f7026986d499c9236d986031`
- **Model**: `qwen35-9b` (`http://127.0.0.1:18000/v1`)
- **VLM Request Attempts**: 1
- **VLM Responses Received**: 1

## Outcomes
- **Specification Terminal Status**: `VLM_SPEC_FAILED`
- **Canonical G_F Produced**: `False`
- **Runtime Failure Category**: `UNMAPPED_FUNCTIONAL_CONCEPT`
- **Failure Layer**: `MODEL_OUTPUT`
- **Failure Subtype**: `UNMAPPED_FUNCTIONAL_CONCEPT`
- **Live Terminal Result**: `VLM_SPEC_FAILED` (Exit code 1)
- **Provider Replay**: `NOT_RUN`
- **Random Replay Seed 0**: `NOT_RUN`
- **Random Replay Seed 1**: `NOT_RUN`
- **Specification SHA256 Identity**: `N/A`
- **Prompt Leakage Audit**: `PASS`

## Reference G_F Evaluation
- **Reference Complete**: N/A
- **Exact Structural Match**: N/A
