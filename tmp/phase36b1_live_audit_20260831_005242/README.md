# Pass 3.6B.1 Representative Live VLM Audit

- **Run ID**: `phase36b1_live_audit_20260831_005242`
- **Code SHA**: `a0149206eca005d5f7026986d499c9236d986031`
- **Artifact Parent SHA**: `a0149206eca005d5f7026986d499c9236d986031`
- **Branch**: `naren/pipeline_check`
- **Model**: `qwen35-9b` (`http://127.0.0.1:18000/v1`)
- **Model Infrastructure**: `OFFLINE (Connection refused)`
- **Total Live VLM Calls**: 0 (Target <= 3)

## Representative Cases
1. **Kitchen K2** (`F1_HIDDEN_COFFEE_VESSEL`): VLM_SPEC_FAILED
2. **Living Room L1** (`F0_ALL_OBJECTS_IN_STAGING`): VLM_SPEC_FAILED
3. **Workshop W1** (`F0_MANUAL_FIRST_ONE_REGION`): VLM_SPEC_FAILED

## Summary of Findings
- **Frozen Runtime Integrity**: Zero modifications to runtime, grounding, search, canonicalization, prompts, schemas, or evaluators during or after audit.
- **Input RGB Images**: Preserved in full resolution (1280x960 for Kitchen/Living, 3 camera views for Workshop) under `inputs/` with sha256 checksums.
- **GT Reference G_F**: Generated offline for all 3 representative domains using GTSpecProvider.
- **Infrastructure Status**: Recorded in `environment/model_connectivity.json`.

## Artifact Directory Structure
```
phase36b1_live_audit_20260831_005242/
├── environment/
├── case_registry.json
├── kitchen_K2/
│   ├── inputs/
│   ├── live/
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
