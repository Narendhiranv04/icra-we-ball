# Phase 3 Pass P3-G: Workshop Canonicalizer Repair & Context Boundaries

> **Pass**: P3-G — Workshop Canonicalizer Repair  
> **Date**: 2026-09-01  
> **Branch**: `naren/pipeline_check`  
> **Status**: COMPLETE (FROZEN)  
> **Version**: `phase3_p3g_v1`  

---

## 1. Executive Summary

In Pass P3-G, the Workshop VLM-to-Canonical-$G_F$ semantic compiler (`mujoco_scenes/workshop_phase1/requirements.py`) was comprehensively refactored and hardened to produce deterministic, inspectable, strictly fail-closed canonical requirement graphs without heuristics, candidate-category role manufacture, silent role loss, unsupported predicate emission, schema defaults, arbitrary duplicate-role merging, endpoint fabrication, raw group pass-through, or planner-context contamination of $\phi^*$.

The repaired Workshop compiler was verified against the curated ideal raw fixture `mujoco_scenes/functional_tamp_pipeline/tests/fixtures/ideal_raw_vlm/workshop_W1.json`, achieving:
- **100% Role Identity Recall (3/3)** and **100% Role Identity Precision (3/3)**
- **100% Role Exact Recall (3/3)** and **100% Role Exact Precision (3/3)**
- **100% Relation Recall (3/3)** and **100% Relation Precision (3/3)**
- **100% Operation Group Identity Recall (0/0)** and **100% Operation Group Identity Precision (0/0)**
- **`extra_operation_groups: []`** (0 runtime operation groups emitted in active $G_F$; raw `group_1` validated and recorded as redundant wrapper)
- **`reference_complete: True`** and **`exact_structural_match: True`** against ground truth reference graph $G_F$
- **Full validation** under `graph.validate()` and `validate_runtime_gf(gf)`
- **End-to-end downstream execution**: `status = ACTION_SEQUENCE_READY`, 5 plan action steps (6 combined), 0 audit violations, independent plan replay `VALID`, access path `VALID`.

---

## 2. Core Architectural & Semantic Invariants

### 2.1 Forward-Only Phrase Matching & Strict Role Semantic Authority
- Role classification is determined strictly and exclusively from `function` and `description` via `map_workshop_role_function`, `map_workshop_fixed_target_role`, and `map_workshop_context_region_role`.
- `candidate_categories` are strictly isolated from role classification and cannot manufacture or distort functional roles (e.g. nonsense functions like `"paint wall"` with candidate categories `["screwdriver"]` fail closed with `UnmappedFunctionalConceptError`).

### 2.2 Strict Fixed Target & Context Region Authority
- Fixed target roles map strictly to `"repair_target"`. Any unmapped or unknown fixed target raises `UnmappedFunctionalConceptError` (eliminated silent fallback to `"repair_target"`).
- Workbench support REGION roles map to `MAIN_WORKBENCH_ZONE` and are recorded in concept accounting with `status: "ABSORBED_INTO_PLANNER_CONTEXT"`. They are **never** emitted as selectable `FunctionalRole` nodes in active $G_F$.
- Any unmapped or unknown region role raises `UnmappedFunctionalConceptError`.

### 2.3 Strict Signature Direction Normalization & Self-Relation Rejection
Relations are mapped into the frozen Workshop predicate signatures:
1. `(driver, COMPATIBLE_WITH, fastener)`
2. `(driver, REACHES_TARGET, repair_target)`
3. `(fastener, COMPATIBLE_WITH_TARGET, repair_target)`

- **Forward Relations**: Preserved directly with `direction_status: "PRESERVED"` and `structural_destination: "GRAPH_RELATION"`.
- **Reverse / Passive Phrasing**: (e.g. `fastener -- "driven by" --> driver` or `repair_target -- "reached by" --> driver`) normalized to canonical signatures with `direction_status: "NORMALIZED_TO_CANONICAL_SIGNATURE"`.
- **Planner Context Absorption**: `repair_target LOCATED_ON MAIN_WORKBENCH_ZONE` is absorbed into planner context (`structural_destination: "ABSORBED_INTO_PLANNER_CONTEXT"`).
- **Rejection of Self-Relations**: Self-relations raise `MalformedVLMSpecificationError`.
- **Unsupported Functional LOCATED_ON**: Functional roles (`driver`, `fastener`) with `LOCATED_ON` raise `UnsupportedCheckerCapabilityError`.

### 2.4 Unary Property Policy (Zero Runtime Unary Predicates)
- Active canonical Workshop $G_F$ has **zero** active unary predicates.
- Legacy capability markers (`CAN_DRIVE_SCREW`, `CAN_FASTEN`) are completely eliminated from runtime $G_F$.
- Unmapped unary properties raise `UnmappedFunctionalConceptError`.
- Unsupported unary properties on functional roles (`PLANAR_SUPPORT`, `OPEN_CAVITY`, `ELONGATED_OBJECT`) fail closed with `UnsupportedCheckerCapabilityError`.
- `PLANAR_SUPPORT` on workbench context is absorbed into planner context (`ABSORBED_INTO_PLANNER_CONTEXT`).

### 2.5 Operation Group Redundancy & Zero Runtime Groups
- Raw interaction groups are validated for semantic consistency against graph relations (`COMPATIBLE_WITH` and `REACHES_TARGET`).
- Since Workshop runtime semantics and grounding are completely defined by the 3 canonical relations without requiring group-level scheduling, active Workshop $G_F$ emits `operation_groups = ()`.
- Raw interaction groups are accounted for in concept accounting with `status: "MERGED_BY_EXPLICIT_RULE"` and `structural_destination: "REDUNDANT_WITH_CANONICAL_GRAPH_RELATIONS"`.

### 2.6 Duplicate-Role Authority & Alternative Merging
- Duplicate driver or fastener roles fail closed with `AmbiguousCanonicalizationError` unless explicit alternative evidence is provided in function/description (e.g. `"alternative driver option"` vs `"interchangeable tool candidate"`).
- When explicit alternative evidence is present, candidates are merged into a single distinct canonical role with combined categories and recorded as `status: "MERGED_BY_EXPLICIT_ALTERNATIVE_RULE"`.

### 2.7 Schema Validation Without Fallback Defaults
- Strict validation of all schema fields on roles (`id`, `entity_kind`, `function`, `description`, `required_count`, `binding_policy`, `candidate_categories`, `visible_candidates`, `required_properties`), relations (`subject_role`, `relation`, `object_role`), and groups (`id`, `function`, `tool_role`, `target_role`, `required_target_count`, `usage_policy`, `required_relations`). Missing required fields fail closed with `MalformedVLMSpecificationError`.

---

## 3. Empirical Verification & Test Matrix

| Test Suite | Command | Result |
|---|---|---|
| Workshop Dedicated Suite | `pytest mujoco_scenes/tests/test_workshop_vlm_requirements.py` | 23/23 PASSED (100%) |
| Ideal Fixtures Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_ideal_fixtures.py` | 4/4 PASSED (100%) |
| Boundary & Interface Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_vlm_interface_boundary.py` | 6/6 PASSED (100%) |
| Architecture Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_architecture.py` | 57/57 PASSED (100%) |
| Executable Grounding IR | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_executable_grounding_ir.py` | 5/5 PASSED (100%) |
| Reference Evaluator | `pytest mujoco_scenes/tests/test_phase3_6b0_reference_evaluator.py` | 24/24 PASSED (100%) |
| Kitchen VLM Graph (Frozen) | `pytest mujoco_scenes/tests/test_kitchen_vlm_functional_graph.py` | 36/36 PASSED (100%) |
| Living Room VLM (Frozen) | `pytest mujoco_scenes/tests/test_living_room_vlm_canonicalization.py` | 11/11 PASSED (100%) |
| Downstream Execution (W1 GT) | `scripts/evaluate_functional_tamp_variants.py --domains workshop --variants W1 --mode gt` | ACTION_SEQUENCE_READY (Combined: 6, Match: YES, Valid: 1/1) |
| Cross-Domain GT Check (K1, L1) | `scripts/evaluate_functional_tamp_variants.py --domains kitchen,living_room --variants K1,L1 --mode gt` | ACTION_SEQUENCE_READY (Match: 2/2, Valid: 2/2) |

---

## 4. Canonicalization Version Provenance

The canonicalization version for Workshop is tracked as:
```python
WORKSHOP_VLM_CANONICALIZATION_VERSION = "phase3_p3g_v1"
```
and is recorded in `metadata["vlm_canonicalization_version"]` and `metadata["canonicalization_trace"]["vlm_canonicalization_version"]`.
