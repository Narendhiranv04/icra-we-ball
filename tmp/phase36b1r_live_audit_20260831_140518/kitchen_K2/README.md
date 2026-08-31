# Case Audit Report: kitchen_K2 (KITCHEN K2)

- **Domain**: `kitchen`
- **Paper Variant**: `K2`
- **Internal Variant**: `F1_HIDDEN_COFFEE_VESSEL`
- **Exact Task Instruction**: `Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil. Search closed kitchen storage for anything still required.`
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
