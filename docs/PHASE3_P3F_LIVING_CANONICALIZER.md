# Phase 3 Pass P3-F: Living Room Canonicalizer Repair

> **Pass**: P3-F — Living Room Semantic Compiler Repair  
> **Date**: 2026-09-01  
> **Branch**: `naren/pipeline_check`  
> **Status**: COMPLETE  
> **Version**: `phase3_p3f_v1`  

---

## 1. Executive Summary

In Pass P3-F, the Living Room VLM-to-Canonical-$G_F$ semantic compiler was comprehensively repaired to produce deterministic, inspectable, fail-closed canonical requirement graphs without heuristics, silent role/relation loss, predicate fabrication, or cardinality distortion.

The repaired compiler was verified against the curated ideal raw fixture `fixtures/ideal_raw_vlm/living_room_L1.json`, achieving:
- **100% Role Identity Recall (6/6)** and **100% Role Identity Precision (6/6)**
- **100% Role Exact Recall (6/6)**
- **100% Relation Recall (4/4)** and **100% Relation Precision (4/4)**
- **100% Operation Group Identity Recall (1/1)** and **100% Operation Group Identity Precision (1/1)**
- **`reference_complete: True`** against ground truth reference graph $G_F$
- **Full validation** under `graph.validate()` and `validate_runtime_gf(gf)`

---

## 2. Core Architectural & Semantic Improvements

### 2.1 Forward-Only Phrase Matching & Role Semantic Authority
- Replaced substring heuristics with forward-only phrase containment (`_contains_phrase(norm, a_norm)`).
- Enforced strict **role semantic authority** based exclusively on `function` and `description`. Open-vocabulary `candidate_categories` are strictly isolated from role classification and cannot manufacture or distort functional roles.
- Nonsense function phrases (e.g. `"paint wall"`, `"decorate room"`) fail closed with `UnmappedFunctionalConceptError`.

### 2.2 Composite Cup-Saucer Bundle Semantics & Cardinality Accounting
The composite role `CUP_SAUCER_SET` requires deterministic cardinality resolution:
- **Case A (Aggregate Composite Role)**: Single raw role describing cup+saucer bundle with count $N \longrightarrow$ canonical count $N$ with status `PRESERVED`.
- **Case B (Component Decomposition)**: Separate cup component ($N$) and saucer component ($N$) $\longrightarrow$ canonical composite count $N$ with status `COMPOSED_FROM_COMPONENT_ROLES` (preventing count inflation from $2+2=4$ to 2).
- **Case C (Mismatched Component Counts)**: Cup count $N \neq$ Saucer count $M \longrightarrow$ fails closed with `MalformedVLMSpecificationError`.
- **Case D (Ambiguous Multiple Component Roles)**: Multiple disjoint cup roles or saucer roles $\longrightarrow$ fails closed with `AmbiguousCanonicalizationError`.
- **Case E (Missing Component Role)**: Only cup component without matching saucer component $\longrightarrow$ fails closed with `AmbiguousCanonicalizationError`.
- **Case F (Conflicting Binding Policies)**: Component policy mismatch (e.g. `DISTINCT` vs `SHARED`) $\longrightarrow$ fails closed with `MalformedVLMSpecificationError`.

### 2.3 Strict Signature Direction Normalization
Relations are mapped into the frozen Living Room predicate signatures:
1. `(PERSONAL_CUP_SAUCER_REGION, FITS_SET_ON, CUP_SAUCER_SET)`
2. `(PERSONAL_CUP_SAUCER_REGION, NEAR_SEAT, SEATING_POSITION)`
3. `(SHARED_REMOTE_REGION, FITS_ON, REMOTE)`
4. `(SHARED_REMOTE_REGION, ACCESSIBLE_FROM_BOTH_SEATS, SEATING_PAIR)`

- **Forward Relations**: Preserved directly with `direction_status: "PRESERVED"`.
- **Passive / Reverse Placement Phrasing**: (e.g. `CUP_SAUCER_SET -- "placed on" --> PERSONAL_CUP_SAUCER_REGION`) normalized to canonical subject/object signature with `direction_status: "NORMALIZED_TO_CANONICAL_SIGNATURE"`.
- **Contextual Self-Referential Phrasing**: Relations on region roles targeting the region itself for proximity to seat normalized to the fixed anchor with `direction_status: "NORMALIZED_TO_CANONICAL_SIGNATURE"`.
- **Incompatible Endpoints & Ambiguous Fragments**: Generic fragments (`"on"`, `"holds"`, `"near"`) or incompatible endpoints raise `UnmappedFunctionalConceptError` / `MalformedVLMSpecificationError`.

### 2.4 Unary Property Fail-Closed Verification
- `PLANAR_SUPPORT` on `REGION` roles: `PRESERVED` (or `MERGED_BY_EXPLICIT_RULE` on duplicate mentions).
- `PLANAR_SUPPORT` on `OBJECT` roles: raises `MalformedVLMSpecificationError`.
- Unsupported properties (`OPEN_CAVITY`, `ELONGATED_OBJECT`): raises `UnsupportedCheckerCapabilityError`.
- Unmapped properties: raises `UnmappedFunctionalConceptError`.

### 2.5 Structural Operation Group Distribution & Validation
- Canonical operation group ID: `personal_support_group` with canonical function `SUPPORT_DRINKWARE`.
- Function validated against `LIVING_INTERACTION_GROUP_ALIASES`.
- Group relations (`FITS_SET_ON`, `NEAR_SEAT`) are encapsulated inside the `OperationGroup` object without redundant top-level graph duplication.
- Top-level `gf.relations` retains only top-level un-grouped relations (`SHARED_REMOTE_REGION FITS_ON REMOTE` and `SHARED_REMOTE_REGION ACCESSIBLE_FROM_BOTH_SEATS SEATING_PAIR`), exactly matching GT reference structure.

### 2.6 Comprehensive Concept Accounting Trace
Every canonicalized graph includes detailed metadata in `canonicalization_trace["concept_accounting"]`:
- `roles`: per-role source, canonical count, binding policy, and mapping status.
- `properties`: raw phrase, canonical predicate, and preservation/merging status.
- `relations`: raw phrase, canonical signature, direction status, and structural destination (`OPERATION_REQUIRED_RELATION`, `OPERATION_CONTEXT_RELATION`, or `GRAPH_RELATION`).
- `operation_groups`: raw ID, canonical ID, function mapping status, endpoints, and relations.

---

## 3. Empirical Verification & Test Matrix

| Test Suite | Command | Result |
|---|---|---|
| Living Room Dedicated Tests | `pytest mujoco_scenes/tests/test_living_room_vlm_canonicalization.py` | 6/6 PASSED (100%) |
| Ideal Fixtures Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_ideal_fixtures.py` | 4/4 PASSED (100%) |
| Executable Grounding IR & Boundary | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_executable_grounding_ir.py mujoco_scenes/functional_tamp_pipeline/tests/test_vlm_interface_boundary.py` | 11/11 PASSED (100%) |
| Environment VLM Requirements | `pytest mujoco_scenes/tests/test_environment_vlm_requirements.py` | 16/16 PASSED (100%) |
| Full Pipeline Regression Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/` | 255/255 PASSED (100%) |

---

## 4. Canonicalization Version Bump

The canonicalization version for Living Room has been set to:
```python
LIVING_ROOM_VLM_CANONICALIZATION_VERSION = "phase3_p3f_v1"
```
and is recorded in `metadata["vlm_canonicalization_version"]` and `metadata["canonicalization_trace"]["version"]`.
