# Pass 3.6B.1R Connected Representative Live VLM Audit

- **Run ID**: `phase36b1r_live_audit_20260831_140518`
- **Runtime Code SHA**: `a0149206eca005d5f7026986d499c9236d986031` (detached worktree at `/home/naren/RA_iiith/.experiment_worktrees/phase36b1r_runtime`)
- **Artifact Parent SHA**: `dc09270052404db2b51cb442f53df6fb27531c9f`
- **Model**: `qwen35-9b` (`http://127.0.0.1:18000/v1`)
- **Infrastructure Status**: ONLINE
- **Total Request Attempts**: 3
- **Total Responses Received**: 3

## Representative Cases
1. **Kitchen K2** (`F1_HIDDEN_COFFEE_VESSEL`): VLM_SPEC_FAILED (G_F: FAILED)
2. **Living Room L1** (`F0_ALL_OBJECTS_IN_STAGING`): VLM_SPEC_FAILED (G_F: FAILED)
3. **Workshop W1** (`F0_MANUAL_FIRST_ONE_REGION`): VLM_SPEC_FAILED (G_F: FAILED)

## Summary of Findings
- **Frozen Runtime Integrity**: Zero code modifications to runtime, grounding, search, canonicalization, prompts, schemas, or evaluators.
- **Detached Execution**: All pipeline runs executed from detached worktree at frozen commit `a0149206eca005d5f7026986d499c9236d986031`.
- **Input Image Fidelity**: Exact RGB images persisted under `inputs/` and supplied via `--observation-image`.
- **Runtime Output Tree**: Complete runner output tree preserved under `live/runtime_output/`.
- **Evaluation Integrity**: Offline evaluator executed via `evaluate_gf_against_reference()`; non-evaluated cases recorded strictly as `null` (never fabricated zeroes).

## Artifact Directory Structure
```
phase36b1r_live_audit_20260831_140518/
├── environment/
├── case_registry.json
├── kitchen_K2/
│   ├── inputs/
│   ├── live/
│   │   ├── fm_diagnostics/
│   │   └── runtime_output/
│   ├── reference/
│   ├── provider_replay/
│   └── random_replays/
├── living_L1/
├── workshop_W1/
├── summary.json
├── summary.csv
├── vlm_call_audit.json
├── SHA256SUMS.txt
└── README.md
```
