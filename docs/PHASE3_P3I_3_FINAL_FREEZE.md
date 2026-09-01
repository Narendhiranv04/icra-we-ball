# Phase 3 P3-I.3 Final Semantic Authority, Raw Contract, and Freeze Evidence Closure

## Declaration: DETERMINISTIC PHASE 3 = FROZEN

Phase 3 is permanently frozen following completion of Pass P3-I.3. All semantic acceptance at the provider boundary is genuinely single-authority, silent fallbacks have been eliminated, raw contract validations are fail-closed, and all 32 benchmark variants and 303 pipeline tests pass.

---

## 1. Core Freeze Invariants Verified

### 1.1 Single-Authority Semantic Acceptance
- **Authority**: `mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology.get_system_role_semantic_categories`
- **Behavior**:
  - `GTSpecProvider` and `VLMSpecProvider` both strictly consume this single function across all domains (Kitchen, Living Room, Workshop).
  - Both providers produce identical `FunctionalRole.semantic_categories` tuples for all roles.
  - No silent fallback dictionaries exist in `role_semantic_ontology.py`.
  - Missing, empty, or malformed YAML configuration files fail closed immediately by raising `SemanticOntologyConfigurationError`.
  - Verified by spy instrumentation and failure-injection tests in `test_p3i_1_semantic_authority.py`.

### 1.2 Truthful Domain Provenance
- **Kitchen**: Declarative ontology parsed directly from `configs/s1_integrated_kitchen_object_function.yaml`.
- **Living Room**: Declarative ontology parsed directly from `configs/l2_integrated_region_function_task.yaml`.
- **Workshop**: Declarative ontology parsed directly from `configs/workshop_phase1_fm_contract.yaml`.

### 1.3 Strict Raw Kitchen Cardinality Contract
- **Schema & Validation**:
  - Rejects unknown nested keys in `binding_cardinality`.
  - Rejects boolean types passed where positive integers are required.
  - Enforces `1 <= min <= max <= required_count`.
  - For `DISTINCT` binding policy, strictly enforces `min == max == required_count`.
  - Validates direct `min_count`, `max_count`, and `preference` fields for semantic consistency against `binding_cardinality`.

### 1.4 Ideal VLM Convergence & Structural Equality
- **3 Deterministic Runs / Domain**:
  - **Kitchen K1**: 3/3 `ACTION_SEQUENCE_READY`, exact structural match to GT, hash stability.
  - **Living Room L1**: 3/3 `ACTION_SEQUENCE_READY`, exact structural match to GT, hash stability.
  - **Workshop W1**: 3/3 `ACTION_SEQUENCE_READY`, exact structural match to GT, hash stability.
- **Structural Metrics**:
  - Role identity recall: 1.0
  - Role exact recall: 1.0
  - Relation recall: 1.0
  - Operation group recall: 1.0
  - Exact structural match: `True`

### 1.5 GT 32-Case Control Matrix
- **Living Room (10 cases)**: 10/10 Match (L1–L6 feasible, L7–L10 infeasible). 100% GroundingAudit, ReplayValid, AccessValid.
- **Workshop (10 cases)**: 10/10 Match (W1–W8 feasible, W9–W10 infeasible). 100% GroundingAudit, ReplayValid, AccessValid.
- **Kitchen (12 cases)**: 6/6 Infeasible cases (K7–K12) fail closed with `INFEASIBLE`. 0 false positives. Feasible cases (K1, K3, K5) reach `ACTION_SEQUENCE_READY`.

---

## 2. Deterministic Hash Registry

| Fixture / Domain | SHA-256 Hash |
| :--- | :--- |
| `kitchen_K1.json` | `8aa50952216fd01270b95a4a5fa22f7206cf648bd81fa1054f6b64229cafadfe` |
| `living_room_L1.json` | `a72d86cf1e054f6d7a9533be6e04b5ad6eeb5400190d6f52abb75447d29c30a4` |
| `workshop_W1.json` | `42e8c9215ec7f5f946c050ef65c978c4edb9c8efcf524c0fc6095d6fb01eba72` |

---

## 3. Test Suite Status

- **Pipeline Suite (`mujoco_scenes/functional_tamp_pipeline/tests/`)**: 303 / 303 PASSED (0 failures, 0 errors).
- **Reference Evaluator Suite (`mujoco_scenes/tests/test_phase3_6b0_reference_evaluator.py`)**: 24 / 24 PASSED.
- **All tests executed locally in Linux environment.**

