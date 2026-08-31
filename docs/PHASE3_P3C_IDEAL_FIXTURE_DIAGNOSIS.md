# Phase 3 Pass P3-C / P3-C.1: Ideal Raw VLM Fixture Framework & Canonicalization Loss Diagnosis

**Execution Date**: 2026-09-01  
**Target Branch**: `naren/pipeline_check`  
**Status**: **P3-C / P3-C.1 COMPLETE — EVIDENCE-BASED DIAGNOSTIC CONTROL ESTABLISHED**

---

## 1. Executive Summary & Scientific Purpose

Pass P3-C / P3-C.1 establishes a deterministic, evidence-based offline semantic control to answer:
> *"If the VLM had produced a semantically excellent, schema-valid raw response using neutral model-local identifiers and natural open-vocabulary language, what would the current deterministic software do with it?"*

This cleanly separates:
1. **Model Capacity Failures** (which did not occur here since zero live model calls were made) from
2. **Canonicalizer / Interface Failures** (deterministic software defects in `kitchen_vlm_functional_graph.py`, `environment_vlm_requirements.py`, and `workshop_phase1/requirements.py`).

### Anti-Leak Invariants & Neutral Identifiers
- **Neutral Model-Local IDs**: All fixtures use neutral identifiers (`role_1`..`role_6`, `group_1`..`group_2`, `search_1`..`search_3`). No semantic hints are conveyed through ID strings.
- **Zero Predicate Leakage**: All relation-bearing fields (`functional_relations[].relation`, `interaction_groups[].required_relations[]`, `interaction_groups[].context_relations[]`) are verified to contain zero internal canonical predicate tokens (`INSERTABLE_IN`, `FITS_SET_ON`, `NEAR_SEAT`, `CAN_DRIVE_SCREW`, `COMPATIBLE_WITH`, etc.).
- **Zero Oracle Tokens**: Zero simulator backend handles (`workshop_power_driver`, `workshop_medium_phillips_screw`) and zero benchmark oracle region/object IDs (`LEFT_DRAWER`, `RIGHT_DRAWER`, `TOOL_CABINET`, `D1`, `D2`, `C1`, `C2`, `B1`, `object_0001`, `region_0001`, `K1`, `L1`, `W1`).

---

## 2. Canonicalization Diagnostic & Reference Summary

| Domain | Fixture File | Production Schema | Raw Semantics | Current Canonicalizer Outcome | Role Identity Recall | Relation Recall | OpGroup Identity Recall | Reference Complete | Exact Structural Match | First Failing Layer / Notes |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Kitchen** | `kitchen_K1.json` | `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA` | Complete (6 roles, 4 rels, 2 groups) | **`CANONICALIZED`** | **1.0** (6/6) | **1.0** (4/4) | **1.0** (2/2) | **True** | False | No failure on ideal fixture. Exact match is False due to minor `selection_preference` string representation nuance (`deterministic_rank` vs `""`). |
| **Living Room** | `living_room_L1.json` | `RESPONSE_SCHEMA` | Complete (6 roles, 4 rels, 1 group) | **`CANONICALIZATION_FAILED`** | N/A | N/A | N/A | N/A | N/A | **Layer B (Canonicalizer)**: `environment_vlm_requirements.py:172` in `map_living_room_relation()` (`UnmappedFunctionalConceptError`). |
| **Workshop** | `workshop_W1.json` | `RESPONSE_SCHEMA` | Complete (3 roles, 3 rels, 1 group, 3 regions) | **`CANONICALIZED`** | **1.0** (3/3) | **1.0** (3/3) | 1.0 (0 missing) | **True** | False | No failure on ideal fixture. Exact match is False because candidate declares semantically valid `group_1` (`extra_operation_groups`), which is absent in static GT reference. |

---

## 3. Evidence-Based Concept Preservation & Loss Breakdown

### 3.1 Kitchen (K1 Ideal Fixture)
- **Trace Evidence Source**: `gf_k.raw_requirements[0]["roles"]` (`raw_vlm_role_id`) and `gf_k.raw_requirements[0]["operation_groups"]` (`raw_vlm_group_id`).
- **Concept Trace**:
  - `role:role_1` (contain coffee) $\to$ `PRESERVED -> coffee_container` (count: 2, `DISTINCT`, `OPEN_CAVITY`)
  - `role:role_2` (contain soup) $\to$ `PRESERVED -> soup_container` (count: 2, `DISTINCT`, `OPEN_CAVITY`)
  - `role:role_3` (stir beverage) $\to$ `PRESERVED -> coffee_stirrer` (count: 1, `REUSABLE`, `ELONGATED_OBJECT`)
  - `role:role_4` (provide eating utensil) $\to$ `PRESERVED -> soup_eating_utensil` (count: 2, `DISTINCT`, `ELONGATED_OBJECT`)
  - `role:role_5` (source of water) $\to$ `PRESERVED -> water_source` (count: 1, `DISTINCT`, `SEMANTIC_ONLY`)
  - `role:role_6` (source of coffee) $\to$ `PRESERVED -> coffee_source` (count: 1, `DISTINCT`, `SEMANTIC_ONLY`)
  - `rel:role_3->role_1` ("fits inside", "reaches the bottom") $\to$ `PRESERVED -> INSERTABLE_IN, REACHES_BOTTOM`
  - `rel:role_4->role_2` ("fits inside", "reaches the bottom") $\to$ `PRESERVED -> INSERTABLE_IN, REACHES_BOTTOM`
  - `group:group_1` $\to$ `PRESERVED -> coffee_stirring` (`SEQUENTIAL_REUSE_ALLOWED`)
  - `group:group_2` $\to$ `PRESERVED -> soup_serving` (`DEDICATED_PER_TARGET`)
- **Full GT Evaluator Metrics**:
  - `role_identity_recall`: 1.0, `role_identity_precision`: 1.0
  - `role_exact_recall`: 0.833, `role_exact_precision`: 0.833 (due to slight differences in candidate category lists)
  - `relation_recall`: 1.0, `relation_precision`: 1.0
  - `operation_group_identity_recall`: 1.0, `operation_group_identity_precision`: 1.0
  - `reference_complete`: True, `exact_structural_match`: False
  - `missing_roles`: `[]`, `extra_roles`: `[]`, `missing_relations`: `[]`, `extra_relations`: `[]`

---

### 3.2 Living Room (L1 Ideal Fixture)
- **End-to-End Failure**:
  - **Exception Type**: `UnmappedFunctionalConceptError`
  - **Category**: `UNMAPPED_FUNCTIONAL_CONCEPT`
  - **Module**: `environment_vlm_requirements.py:172` in `map_living_room_relation()`
  - **Message**: `VLM living room relation 'can hold drinkware set' cannot be mapped to any reviewed relation`
- **Granular Sub-Concept Mapping Breakdown**:
  Because end-to-end execution halts at the first unmapped relation, each raw component was evaluated independently through production mapping functions:

| Raw Component | Raw Type & Text | Mapping Function | Outcome | Target Canonical Concept / Error |
|---|---|---|:---:|---|
| `role_1` | `REGION`: "hold items for viewer" | `map_living_room_role_function` | **PRESERVED/MAPPABLE** | `PERSONAL_CUP_SAUCER_REGION` |
| `role_2` | `REGION`: "hold items for viewers" | `map_living_room_role_function` | **PRESERVED/MAPPABLE** | `SHARED_REMOTE_REGION` |
| `role_3` | `OBJECT`: "contain hot beverage and saucer" | `map_living_room_object_payload_role` | **PRESERVED/MAPPABLE** | `CUP_SAUCER_SET` |
| `role_4` | `OBJECT`: "control television" | `map_living_room_object_payload_role` | **PRESERVED/MAPPABLE** | `REMOTE` |
| `role_5` | `FIXED_TARGET`: "viewer seating position" | `map_living_room_fixed_target_role` | **SYSTEM_CONTEXT_COMPILED** | `SEATING_POSITION` |
| `role_6` | `FIXED_TARGET`: "paired viewer seating area" | `map_living_room_fixed_target_role` | **SYSTEM_CONTEXT_COMPILED** | `SEATING_PAIR` |
| `rel:role_1->role_3` | "can hold drinkware set" | `map_living_room_relation` | **REJECTED** | `UnmappedFunctionalConceptError` (modal phrase "can hold" missing in alias dictionary) |
| `rel:role_1->role_5` | "near seat" | `map_living_room_relation` | **PRESERVED/MAPPABLE** | `NEAR_SEAT` |
| `rel:role_2->role_4` | "can hold remote" | `map_living_room_relation` | **REJECTED** | `UnmappedFunctionalConceptError` (modal phrase "can hold" missing in alias dictionary) |
| `rel:role_2->role_6` | "accessible from both seats" | `map_living_room_relation` | **PRESERVED/MAPPABLE** | `ACCESSIBLE_FROM_BOTH_SEATS` |
| `group_1` | Support drinkware beside seat | `_compile_operation_groups` | **NOT_REACHED_DUE_TO_PRIOR_FAILURE** | Blocked by unmapped `rel:role_1->role_3` |

- **Root Cause & Repair Target**:
  Current `map_living_room_relation` recognizes exact verbs `"supports"`, `"holds"`, `"placed on"`, `"rest on"`, but fails on `"can hold"`. This is a clear canonicalizer defect scheduled for repair in `P3-F`.

---

### 3.3 Workshop (W1 Ideal Fixture)
- **Trace Evidence Source**: `FMRequirementProvider.normalized_roles`, `normalized_relations`, `normalized_operation_groups`, and `candidate_regions`.
- **Concept Trace**:
  - `role:role_1` (drive screw) $\to$ `PRESERVED -> driver` (count: 1, `DISTINCT`)
  - `role:role_2` (fasten joint) $\to$ `PRESERVED -> fastener` (count: 1, `DISTINCT`)
  - `role:role_3` (frame joint repair target) $\to$ `PRESERVED -> repair_target` (`SYSTEM_OWNED_FIXED_TARGET_REPRESENTATION`)
  - `rel:role_1->role_2` ("compatible with") $\to$ `PRESERVED -> COMPATIBLE_WITH`
  - `rel:role_1->role_3` ("reaches target") $\to$ `PRESERVED -> REACHES_TARGET`
  - `rel:role_2->role_3` ("compatible with target") $\to$ `PRESERVED -> COMPATIBLE_WITH_TARGET`
  - `group:group_1` $\to$ `PRESERVED -> group_1` (tool: `driver`, target: `fastener`, context: `repair_target`)
  - `search_1`, `search_2`, `search_3` $\to$ `PRESERVED -> resolved as ('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET')`
- **Full GT Evaluator Metrics & Disclosure**:
  - `role_identity_recall`: 1.0, `role_identity_precision`: 1.0
  - `role_exact_recall`: 1.0, `role_exact_precision`: 1.0
  - `relation_recall`: 1.0, `relation_precision`: 1.0
  - `operation_group_identity_recall`: 1.0, `operation_group_identity_precision`: 0.0 (`extra_operation_groups: ['group_1']`)
  - `reference_complete`: True, `exact_structural_match`: False
  - **Structural Difference Disclosure**: The ideal raw fixture expresses a semantically valid interaction group (`group_1` for driving the fastener into the target joint). The offline static GT reference does not declare an operation group (`operation_groups: ()`). This is preserved as an extra representational structure rather than pruned to artificially force an exact structural match.

---

## 4. Summary of Hardened Contracts

1. **Zero Tautological Labels**: Preservation classifications are directly tied to transformation trace metadata and mapping function outputs.
2. **Zero Network Dependence**: All diagnostic executions use `MockFMAdapter` with zero HTTP or socket calls.
3. **P3-F Repair Readiness**: The test-side `MockFMAdapter` implements all attributes required for full post-repair Living Room canonicalization without test suite breakage.
