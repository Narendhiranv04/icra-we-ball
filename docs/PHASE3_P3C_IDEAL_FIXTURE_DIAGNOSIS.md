# Phase 3 Pass P3-C: Ideal Raw VLM Fixture Framework & Canonicalization Loss Diagnosis

**Execution Date**: 2026-08-31  
**Target Branch**: `naren/pipeline_check`  
**Status**: **COMPLETE — ZERO-NETWORK DIAGNOSTIC CONTROL ESTABLISHED**

---

## 1. Executive Summary & Diagnostic Control Purpose

Pass P3-C establishes an offline, deterministic semantic control to answer:
> *"If the VLM had produced a semantically excellent, schema-valid raw response using natural open-vocabulary language, what would the current deterministic canonicalization software do with it?"*

This cleanly decouples **Model Capacity Failures** from **Canonicalizer / Interface Failures** across all three benchmark domains.

All three ideal raw fixtures:
1. Adhere strictly to the production model-facing schemas (`KITCHEN_FUNCTIONAL_GRAPH_SCHEMA` and `RESPONSE_SCHEMA`) without relaxations.
2. Comply with strict anti-leakage invariants: zero internal canonical role IDs as raw IDs, zero canonical predicate tokens (`INSERTABLE_IN`, `FITS_SET_ON`, `CAN_DRIVE_SCREW`, etc.), zero simulator backend handles (`workshop_power_driver`), and zero benchmark oracle tokens (`F0`, `K1`, `object_0001`, `region_0001`, `LEFT_DRAWER`, etc.).
3. Execute through the **real, unmodified production canonicalizers** with **zero live VLM/network calls** via dependency injection.

---

## 2. Canonicalization Diagnostic Summary Table

| Domain | Fixture File | Schema | Semantic Completeness | Current Canonicalizer Outcome | Role Recall | Relation Recall | First Failing Layer |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **Kitchen** | `kitchen_K1.json` | `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA` | Complete (6 roles, 4 rels, 2 groups) | `CANONICALIZED` | 1.0 (6/6) | 1.0 (4/4) | None on ideal fixture (P3-E addresses noisy/clipped VLM raw variations) |
| **Living Room** | `living_room_L1.json` | `RESPONSE_SCHEMA` | Complete (6 roles, 4 rels, 1 group) | `CANONICALIZATION_FAILED` | N/A | N/A | **Layer B (Canonicalizer)**: `environment_vlm_requirements.py:172` (`map_living_room_relation`) |
| **Workshop** | `workshop_W1.json` | `RESPONSE_SCHEMA` | Complete (3 roles, 3 rels, 1 group, 3 regions) | `CANONICALIZED` | 1.0 (3/3) | 1.0 (3/3) | None on ideal fixture (P3-G addresses duplicate role collisions & context relations) |

---

## 3. Detailed Per-Domain Diagnostics & Concept Loss Traces

### 3.1 Kitchen (K1 Ideal Fixture)
- **Raw Fixture Semantic Summary**:
  - Expresses 6 distinct functional roles using raw local IDs: `role_coffee_cup` (count 2), `role_soup_bowl` (count 2), `role_coffee_stirrer` (count 1, `REUSABLE`), `role_soup_utensil` (count 2, `DISTINCT`), `role_water_source` (count 1), `role_coffee_source` (count 1).
  - Expresses 4 natural binary relations: `fits inside`, `reaches the bottom` across tool/target pairs.
  - Expresses 2 interaction groups: `group_stir_coffee` (`SEQUENTIAL_REUSE_ALLOWED`, count 2) and `group_serve_soup` (`DEDICATED_PER_TARGET`, count 2).
- **Current Canonicalizer Outcome**: `CANONICALIZED` via `kitchen_vlm_functional_graph.py::compile_vlm_functional_graph`.
- **Concept Preservation Classification**:
  - `role_coffee_cup` $\to$ `coffee_container` (`PRESERVED`, count 2)
  - `role_soup_bowl` $\to$ `soup_container` (`PRESERVED`, count 2)
  - `role_coffee_stirrer` $\to$ `coffee_stirrer` (`PRESERVED`, count 1)
  - `role_soup_utensil` $\to$ `soup_eating_utensil` (`PRESERVED`, count 2)
  - `role_water_source` $\to$ `water_source` (`PRESERVED`, count 1)
  - `role_coffee_source` $\to$ `coffee_source` (`PRESERVED`, count 1)
  - `fits inside`, `reaches the bottom` $\to$ `INSERTABLE_IN`, `REACHES_BOTTOM` (`PRESERVED`)
  - Interaction groups $\to$ `coffee_stirring`, `soup_serving` (`PRESERVED`)
- **Offline Reference Comparison**:
  - `role_precision`: 1.0, `role_recall`: 1.0
  - `relation_precision`: 1.0, `relation_recall`: 1.0
- **Identified Failure Surface for P3-E**:
  - Although the ideal fixture succeeds, the canonicalizer contains rigid hardcoding (e.g. only 2 specific tool-target pair patterns accepted in operation group compiler; unmapped relations silently fallback to `INSERTABLE_IN`). Pass P3-E will harden these pathways against raw concept loss.

---

### 3.2 Living Room (L1 Ideal Fixture)
- **Raw Fixture Semantic Summary**:
  - Expresses 6 functional roles: `role_personal_table` (REGION count 2), `role_shared_table` (REGION count 1, `SHARED`), `role_drinkware_set` (OBJECT count 2), `role_tv_remote` (OBJECT count 1), `role_viewer_seat` (FIXED_TARGET count 2), `role_seating_pair` (FIXED_TARGET count 1).
  - Expresses 4 natural relations: `can hold drinkware set`, `near seat`, `can hold remote`, `accessible from both seats`.
  - Expresses 1 interaction group: `group_personal_support` for pairing side tables with drinkware and seating context.
- **Current Canonicalizer Outcome**: `CANONICALIZATION_FAILED`.
- **Exact Failure Details**:
  - **Exception Type**: `UnmappedFunctionalConceptError`
  - **Error Category**: `UNMAPPED_FUNCTIONAL_CONCEPT`
  - **First Failing Module**: `environment_vlm_requirements.py:172` in `map_living_room_relation()`
  - **Error Message**: `VLM living room relation 'can hold drinkware set' cannot be mapped to any reviewed relation`
- **Concept Loss & Diagnosis**:
  - **Lost Concepts**: The entire graph fails to compile because `map_living_room_relation` fails closed on natural placement verb phrases like `"can hold drinkware set"`.
  - **Root Cause**: The Living Room relation alias matcher only permits a narrow dictionary of rigid phrases (`"placed on"`, `"supports"`, `"holds"` without noun suffixes) and lacks flexible semantic relation normalization.
  - **Target Pass for Repair**: `P3-F` (Living Room Canonicalizer Repair).

---

### 3.3 Workshop (W1 Ideal Fixture)
- **Raw Fixture Semantic Summary**:
  - Expresses 3 functional roles: `role_driver_tool` (OBJECT count 1), `role_threaded_fastener` (OBJECT count 1), `role_joint_target` (FIXED_TARGET count 1).
  - Expresses 3 natural relations: `compatible with`, `reaches target`, `compatible with target`.
  - Expresses 1 interaction group: `group_fasten_repair`.
  - Proposes 3 natural candidate inspectable regions: `storage_drawer_left`, `storage_drawer_right`, `tall_tool_cabinet`.
- **Current Canonicalizer Outcome**: `CANONICALIZED` via `workshop_phase1/requirements.py::FMRequirementProvider`.
- **Concept Preservation Classification**:
  - `role_driver_tool` $\to$ `driver` (`PRESERVED`, count 1)
  - `role_threaded_fastener` $\to$ `fastener` (`PRESERVED`, count 1)
  - `role_joint_target` $\to$ `repair_target` (`PRESERVED`, count 1)
  - `compatible with`, `reaches target`, `compatible with target` $\to$ `COMPATIBLE_WITH`, `REACHES_TARGET`, `COMPATIBLE_WITH_TARGET` (`PRESERVED`)
  - `group_fasten_repair` $\to$ `group_fasten_repair` (`PRESERVED`)
  - Natural region proposals $\to$ `('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET')` (`PRESERVED` via natural region resolution).
- **Offline Reference Comparison**:
  - `role_precision`: 1.0, `role_recall`: 1.0
  - `relation_precision`: 1.0, `relation_recall`: 1.0
- **Identified Failure Surface for P3-G**:
  - The ideal fixture canonicalizes cleanly; however, live VLM outputs frequently trigger duplicate `driver` role collisions and undeclared role references. Pass P3-G will repair these edge cases.

---

## 4. Architectural Rules Formally Validated

1. **No Oracle Leakage**: Fixtures rely solely on open-vocabulary descriptions and local role IDs.
2. **Zero Network Requirement**: All diagnostic tests run completely offline with mocked transports.
3. **Deterministic Failure Localization**: Failures are isolated to exact source code lines and functions without guessing or live model variance.
