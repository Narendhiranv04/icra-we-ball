# Case Audit Report: kitchen_K2 (KITCHEN K2)

- **Domain**: kitchen
- **Paper Label**: K2
- **Internal Variant**: F1_HIDDEN_COFFEE_VESSEL
- **Task Instruction**: prepare coffee and soup
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
