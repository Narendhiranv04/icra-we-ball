# Ideal Raw VLM Fixtures (P3-C & P3-C.1)

This directory contains manually curated "perfect VLM response" semantic control fixtures for:
- `kitchen_K1.json` (conforms to `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA`)
- `living_room_L1.json` (conforms to `RESPONSE_SCHEMA`)
- `workshop_W1.json` (conforms to `RESPONSE_SCHEMA`)

## Design Principles & Anti-Leak Invariants
1. **Model-Facing Schema Compliance**: Exactly adheres to production schemas without relaxations (`validate_kitchen_functional_specification` and `validate_requirement_response`).
2. **Neutral Model-Local Identifiers**: All raw IDs are neutral (`role_1`..`role_6`, `group_1`, `search_1`..`search_3`) so that canonicalizers must recover semantic meaning exclusively from natural language fields (`function`, `description`, `candidate_categories`, `required_properties`, and `relations`).
3. **Zero IR / Benchmark Identifier Leakage**:
   - Zero canonical role IDs (`coffee_container`, `PERSONAL_CUP_SAUCER_REGION`, `driver`, `fastener`, etc.) anywhere in raw role IDs.
   - Zero canonical predicate tokens (`INSERTABLE_IN`, `FITS_SET_ON`, `NEAR_SEAT`, `CAN_DRIVE_SCREW`, `COMPATIBLE_WITH`, etc.) in relation names or interaction group relation lists.
   - Zero simulator backend entity handles (`workshop_power_driver`, `workshop_medium_phillips_screw`, etc.).
   - Zero benchmark variant or oracle tokens (`F0`, `K1`, `K2`, `L1`, `W1`, `object_0001`, `region_0001`, `LEFT_DRAWER`, `RIGHT_DRAWER`, `TOOL_CABINET`, `D1`, `D2`, `C1`, `C2`, `B1`).
4. **Semantically Grounded Physical Properties**: Unary shape/property constraints are included only where physically justified by task requirements and verifier capabilities (e.g. vessel cavity for liquids, elongated tool for stirring/eating), avoiding unnecessary shape assertions on supply sources or composite groups.
5. **Diagnostic Control Role**: Serves as a deterministic semantic control to isolate VLM generation failures from downstream canonicalization and interface failures without live network or model calls.
