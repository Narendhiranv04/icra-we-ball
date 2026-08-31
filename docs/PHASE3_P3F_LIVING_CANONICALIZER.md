# Phase 3 Pass P3-F, P3-F.1 & P3-F.2: Living Room Canonicalizer Repair & Per-Role Property Evidence Closure

> **Pass**: P3-F — Living Room Semantic Compiler Repair  
> **Pass**: P3-F.1 — Living Room Fail-Closed Authority Closure  
> **Pass**: P3-F.2 — Living Room Per-Raw-Role Property Evidence Closure  
> **Date**: 2026-09-01  
> **Branch**: `naren/pipeline_check`  
> **Status**: COMPLETE (FROZEN)  
> **Version**: `phase3_p3f_2_v1`  

---

## 1. Executive Summary

In Passes P3-F, P3-F.1, and P3-F.2, the Living Room VLM-to-Canonical-$G_F$ semantic compiler was comprehensively repaired, sealed, and hardened to produce deterministic, inspectable, strictly fail-closed canonical requirement graphs without heuristics, silent role/relation loss, predicate fabrication, cardinality distortion, or cross-role property leakage.

The repaired compiler was verified against the curated ideal raw fixture `fixtures/ideal_raw_vlm/living_room_L1.json`, achieving:
- **100% Role Identity Recall (6/6)** and **100% Role Identity Precision (6/6)**
- **100% Role Exact Recall (6/6)**
- **100% Relation Recall (4/4)** and **100% Relation Precision (4/4)**
- **100% Operation Group Identity Recall (1/1)** and **100% Operation Group Identity Precision (1/1)**
- **`reference_complete: True`** against ground truth reference graph $G_F$
- **Full validation** under `graph.validate()` and `validate_runtime_gf(gf)`
- **End-to-end downstream execution**: `status = ACTION_SEQUENCE_READY`, 10 action steps, 0 audit violations, independent plan replay `VALID`.

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

### 2.3 Strict Signature Direction Normalization & Self-Relation Rejection
Relations are mapped into the frozen Living Room predicate signatures:
1. `(PERSONAL_CUP_SAUCER_REGION, FITS_SET_ON, CUP_SAUCER_SET)`
2. `(PERSONAL_CUP_SAUCER_REGION, NEAR_SEAT, SEATING_POSITION)`
3. `(SHARED_REMOTE_REGION, FITS_ON, REMOTE)`
4. `(SHARED_REMOTE_REGION, ACCESSIBLE_FROM_BOTH_SEATS, SEATING_PAIR)`

- **Forward Relations**: Preserved directly with `direction_status: "PRESERVED"`.
- **Passive / Reverse Placement Phrasing**: (e.g. `CUP_SAUCER_SET -- "placed on" --> PERSONAL_CUP_SAUCER_REGION`) normalized to canonical subject/object signature with `direction_status: "NORMALIZED_TO_CANONICAL_SIGNATURE"`.
- **Rejection of Self-Relation Endpoint Fabrication (P3-F.1)**: Self-relations (e.g. `PERSONAL_REGION -- "near seat" --> PERSONAL_REGION`) are never silently rewritten to fixed anchors; endpoint substitution is strictly forbidden and raises `MalformedVLMSpecificationError`.
- **Incompatible Endpoints & Ambiguous Fragments**: Generic fragments (`"on"`, `"holds"`, `"near"`) or incompatible endpoints raise `UnmappedFunctionalConceptError` / `MalformedVLMSpecificationError`.

### 2.4 Removal of Fixed-Target Synthesis & Strict Raw Fixed-Target Requirement (P3-F.1)
- Removed all `system_context_seating_position` / `system_context_seating_pair` synthesis logic.
- If relations or operation groups reference fixed anchors (`SEATING_POSITION` or `SEATING_PAIR`) and the corresponding `FIXED_TARGET` roles were not explicitly declared in raw requirements: **fails closed** with `MalformedVLMSpecificationError`.
- Concept accounting records explicit fixed targets as `PRESERVED` (no synthetic `SYSTEM_CONTEXT_COMPILED` labels).

### 2.5 Disjoint Slot Composition Rules (P3-F.1)
- Implemented `_extract_disjoint_slot_identity(function_text, description_text)` to determine disjoint slot identity (`VIEWER_1` vs `VIEWER_2`) from `function + description` ONLY.
- Multiple roles mapping to `PERSONAL_CUP_SAUCER_REGION` or `SEATING_POSITION` may ONLY compose if each has explicit distinct slot identities, counts == 1, and `DISTINCT` binding policy.
- Generic duplicate roles without disjoint slot evidence fail closed with `AmbiguousCanonicalizationError`.

### 2.6 Per-Raw-Role Property Evidence Closure (P3-F.2)
- Replaced canonical-union property checking with strict per-raw-role tracking via `raw_role_properties_map[raw_id]`.
- **Core Invariant**: Every independently required physical support raw role contributing to `PERSONAL_CUP_SAUCER_REGION` or `SHARED_REMOTE_REGION` must independently provide its own raw VLM evidence mapping to `PLANAR_SUPPORT` ($\text{PLANAR\_SUPPORT} \in \text{canonicalize}(r_i.\text{required\_properties})$).
- A contributing raw support role cannot borrow `PLANAR_SUPPORT` evidence from another raw role. If any contributing role omits it, canonicalization fails closed with `MalformedVLMSpecificationError` identifying the offending raw role ID.
- Role concept accounting accurately reflects unary predicates supplied by that specific raw role. Property concept accounting records distinct raw roles' properties independently as `PRESERVED` (no false cross-role duplicate merging).
- Synonymous property aliases on the *same* raw role merge with `MERGED_BY_EXPLICIT_RULE`.

### 2.7 Schema-Required Field Validation Without Fallback Defaults (P3-F.1)
- Strict validation of all schema-required fields on raw roles (`id`, `entity_kind`, `function`, `required_count`, `binding_policy`, `candidate_categories`, `visible_candidates`, `required_properties`), interaction groups (`id`, `function`, `tool_role`, `target_role`, `required_target_count`, `usage_policy`, `required_relations`), and functional relations (`subject_role`, `relation`, `object_role`). Missing fields fail closed immediately.

---

## 3. Empirical Verification & Test Matrix

| Test Suite | Command | Result |
|---|---|---|
| Living Room Dedicated & Regression Tests | `pytest mujoco_scenes/tests/test_living_room_vlm_canonicalization.py` | 11/11 PASSED (100%) |
| Ideal Fixtures Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_ideal_fixtures.py` | 4/4 PASSED (100%) |
| Executable Grounding IR & Boundary | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_executable_grounding_ir.py mujoco_scenes/functional_tamp_pipeline/tests/test_vlm_interface_boundary.py` | 11/11 PASSED (100%) |
| Environment VLM Requirements | `pytest mujoco_scenes/tests/test_environment_vlm_requirements.py` | 16/16 PASSED (100%) |
| Kitchen Regression Suite (Frozen) | `pytest mujoco_scenes/tests/test_kitchen_vlm_functional_graph.py` | 36/36 PASSED (100%) |
| Full Pipeline Regression Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/` | 255/255 PASSED (100%) |
| Combined Test Suite | `pytest ...` | 318/318 PASSED (100%) |
| Downstream Execution (L1 GT) | `scripts/evaluate_functional_tamp_variants.py --domains living_room --variants L1 --mode gt` | ACTION_SEQUENCE_READY (10 actions, Audit VALID, Replay VALID) |

---

## 4. Canonicalization Version Provenance

The canonicalization version for Living Room is tracked as:
```python
LIVING_ROOM_VLM_CANONICALIZATION_VERSION = "phase3_p3f_2_v1"
```
and is recorded in `metadata["vlm_canonicalization_version"]` and `metadata["canonicalization_trace"]["version"]`.
