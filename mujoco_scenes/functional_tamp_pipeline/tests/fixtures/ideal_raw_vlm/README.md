# Ideal Raw VLM Fixtures

This directory contains manually curated "perfect VLM response" fixtures for:
- `kitchen_K1.json` (conforms to `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA`)
- `living_room_L1.json` (conforms to `RESPONSE_SCHEMA`)
- `workshop_W1.json` (conforms to `RESPONSE_SCHEMA`)

## Design Principles & Anti-Leak Rules
1. **Model-Facing Schema Compliance**: Exactly adheres to the production schemas without relaxations.
2. **Pure Open-Vocabulary Semantics**: Uses natural language capability/function phrases, open-vocabulary candidate categories, and natural relations.
3. **Zero IR / Benchmark Identifier Leakage**:
   - No canonical role IDs (`coffee_container`, `PERSONAL_CUP_SAUCER_REGION`, `driver`, etc.) as raw role IDs.
   - No canonical predicate tokens (`INSERTABLE_IN`, `FITS_SET_ON`, `CAN_DRIVE_SCREW`, etc.) in relation names.
   - No simulator backend entity handles (`workshop_power_driver`, etc.).
   - No benchmark variant or internal oracle labels (`F0`, `K1`, `object_0001`, `region_0001`, `LEFT_DRAWER`, etc.).
4. **Diagnostic Control Role**: Serves as a deterministic semantic control to isolate model capacity failures from downstream canonicalization and interface failures.
