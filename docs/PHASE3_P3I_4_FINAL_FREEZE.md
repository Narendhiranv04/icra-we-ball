# Phase 3 Deterministic Pipeline Final Freeze (Pass P3-I.4)

**Freeze Date**: September 2, 2026  
**Status**: `DETERMINISTIC_PHASE_3_FROZEN`  
**Ontology Version**: `phase3_p3i_4_semantic_ontology_v1`  
**Search Policy Version**: `phase3_p3h_1_v1`  
**Tested Code Commit SHA**: `38c1bdf7e7f22bed5e2ad0906c80b46bef0c2de0`  
**Starting Remote SHA**: `ac638d893042f0ee82ee77e8d89684a385d19584`  

---

## 1. Executive Summary

Pass **P3-I.4** represents the final deterministic Phase-3 repair and verification freeze for the Functional TAMP Pipeline across all three benchmark domains (**Kitchen**, **Living Room**, and **Workshop**).

All architectural requirements, strict semantic authority contracts, fail-closed schema validators, multi-camera active perception trackers, and symbolic planners are fully converged and verified against the committed tree:
- **32 / 32 benchmark variants** matched expected outcomes in the Ground Truth (GT) matrix:
  - **20 / 20 feasible cases** reached `ACTION_SEQUENCE_READY` with valid grounding audits, replay validation, and accessibility satisfaction.
  - **12 / 12 infeasible cases** failed closed cleanly to `INFEASIBLE` with 0 false positives.
- **3-run deterministic convergence** verified on ideal fixtures for $K_1$, $L_1$, and $W_1$ (100% bitwise structural match of $\mathcal{G}_F$, search contract, $\mathcal{G}_O$, $\phi^*$, and generated symbolic plans).
- **306 / 306 pytest tests** passed in `mujoco_scenes/functional_tamp_pipeline/tests/`.
- **Zero Phase-4 modifications**: Phase-4 execution boundaries remain untouched.

---

## 2. P3-I.4 Key Repairs & Architectural Tightening

1. **Workshop Declarative Semantic Authority (§3, §4)**:
   - Added explicit reviewed `functional_roles` block to `mujoco_scenes/configs/workshop_phase1_fm_contract.yaml` specifying accepted categories for `driver`, `fastener`, and `repair_target`.
2. **Fail-Closed Declarative Semantic Authority (§5, §6)**:
   - Replaced all ad-hoc fallback derivations in `role_semantic_ontology.py` with strict declarative loading from config files.
   - Normalized all missing/malformed errors to `SemanticOntologyConfigurationError`.
   - Bumped ontology version to `phase3_p3i_4_semantic_ontology_v1`.
3. **Generic Schema Cardinality Mismatch Fixed (§7)**:
   - Removed kitchen-specific cardinality fields from the generic `RESPONSE_SCHEMA` in `mujoco_scenes/workshop_phase1/fm_adapter.py`.
   - Maintained strict separation: generic Living Room/Workshop schema forbids cardinality fields; kitchen schema explicitly validates `binding_cardinality`.
4. **Cardinality Preference Conflict Closed (§8)**:
   - Enforced equality between `preferred` and `preference` in both `validate_kitchen_functional_specification` and `compile_vlm_functional_graph`, raising validation errors on conflicting definitions.
5. **Cross-Group Tool Distinctness Bug Fixed (§9)**:
   - In `mujoco_scenes/task_witness.py` (lines 2108–2122), fixed the function-aware distinctness check to evaluate disjointness across operation groups rather than within groups, unblocking sequential single-tool reuse (e.g., 1 spoon stirring 2 coffees).
6. **Task Vocabulary Alias Integrity (§10)**:
   - Added reviewed spoon aliases (`dessert spoon`, `serving spoon`, `silver spoon`) to `mujoco_scenes/configs/semantic_vocabulary.yaml`.
   - Cleaned powder terms from `coffee_source` to prevent false positive detections of open cup rims as coffee jars.

---

## 3. 32-Case GT Control Matrix

Full evaluation executed using `scripts/evaluate_functional_tamp_variants.py --mode gt --output-root /tmp/p3i4_gt_matrix_final`:

| Domain | Variant | Expected Status | Actual Status | Match | Inspections | Plan Steps | Combined Steps | Grounding Audit | Replay Valid | Access Valid |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **kitchen** | K1 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 0 | 24 | 24 | VALID | VALID | VALID |
| **kitchen** | K2 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 3 | 26 | 29 | VALID | VALID | VALID |
| **kitchen** | K3 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 4 | 24 | 28 | VALID | VALID | VALID |
| **kitchen** | K4 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 4 | 26 | 30 | VALID | VALID | VALID |
| **kitchen** | K5 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 2 | 24 | 26 | VALID | VALID | VALID |
| **kitchen** | K6 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 5 | 26 | 31 | VALID | VALID | VALID |
| **kitchen** | K7 | INFEASIBLE | INFEASIBLE | **YES** | 5 | 0 | 5 | N/A | N/A | N/A |
| **kitchen** | K8 | INFEASIBLE | INFEASIBLE | **YES** | 5 | 0 | 5 | N/A | N/A | N/A |
| **kitchen** | K9 | INFEASIBLE | INFEASIBLE | **YES** | 5 | 0 | 5 | N/A | N/A | N/A |
| **kitchen** | K10 | INFEASIBLE | INFEASIBLE | **YES** | 5 | 0 | 5 | N/A | N/A | N/A |
| **kitchen** | K11 | INFEASIBLE | INFEASIBLE | **YES** | 5 | 0 | 5 | N/A | N/A | N/A |
| **kitchen** | K12 | INFEASIBLE | INFEASIBLE | **YES** | 5 | 0 | 5 | N/A | N/A | N/A |
| **living_room** | L1 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 0 | 10 | 10 | VALID | VALID | VALID |
| **living_room** | L2 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 0 | 8 | 8 | VALID | VALID | VALID |
| **living_room** | L3 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 0 | 10 | 10 | VALID | VALID | VALID |
| **living_room** | L4 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 0 | 10 | 10 | VALID | VALID | VALID |
| **living_room** | L5 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 0 | 8 | 8 | VALID | VALID | VALID |
| **living_room** | L6 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 0 | 10 | 10 | VALID | VALID | VALID |
| **living_room** | L7 | INFEASIBLE | INFEASIBLE | **YES** | 0 | 0 | 0 | N/A | N/A | N/A |
| **living_room** | L8 | INFEASIBLE | INFEASIBLE | **YES** | 0 | 0 | 0 | N/A | N/A | N/A |
| **living_room** | L9 | INFEASIBLE | INFEASIBLE | **YES** | 0 | 0 | 0 | N/A | N/A | N/A |
| **living_room** | L10 | INFEASIBLE | INFEASIBLE | **YES** | 0 | 0 | 0 | N/A | N/A | N/A |
| **workshop** | W1 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 1 | 5 | 6 | VALID | VALID | VALID |
| **workshop** | W2 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 1 | 5 | 6 | VALID | VALID | VALID |
| **workshop** | W3 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 2 | 5 | 7 | VALID | VALID | VALID |
| **workshop** | W4 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 2 | 5 | 7 | VALID | VALID | VALID |
| **workshop** | W5 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 3 | 5 | 8 | VALID | VALID | VALID |
| **workshop** | W6 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 3 | 5 | 8 | VALID | VALID | VALID |
| **workshop** | W7 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 3 | 5 | 8 | VALID | VALID | VALID |
| **workshop** | W8 | ACTION_SEQUENCE_READY | ACTION_SEQUENCE_READY | **YES** | 3 | 5 | 8 | VALID | VALID | VALID |
| **workshop** | W9 | INFEASIBLE | INFEASIBLE | **YES** | 3 | 0 | 3 | N/A | N/A | N/A |
| **workshop** | W10 | INFEASIBLE | INFEASIBLE | **YES** | 3 | 0 | 3 | N/A | N/A | N/A |

**Summary**:
- **Total Attempted**: 32 / 32 (100%)
- **Exact Matches**: 32 / 32 (100%)
- **Feasible Cases ($20/20$)**: 20 reached `ACTION_SEQUENCE_READY`
- **Infeasible Cases ($12/12$)**: 12 correctly failed closed to `INFEASIBLE`

---

## 4. Deterministic Convergence & Artifact Hashes

Across 3 independent runs on ideal offline VLM fixtures:

| Case | Fixture SHA-256 | $\mathcal{G}_F$ Projection SHA-256 | Search Contract SHA-256 | Final $\mathcal{G}_O$ SHA-256 | $\phi^*$ SHA-256 | Plan SHA-256 | Converged |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Kitchen K1** | `8aa50952216fd01270b95a4a5fa22f72...` | `af9261fd9eac6d2216d09fd8066ef745...` | `d110d2efad4ca269f98101540d190251...` | `cf696bf0632e0a0b0fa5225b375f8edf...` | `57e60530123427976c48c5de55791bd3...` | `570ee9bf88d14aec5266a2599674f69c...` | **3 / 3** |
| **Living Room L1** | `a72d86cf1e054f6d7a9533be6e04b5ad...` | `66527516d39f07e40bed67b898ef5b6b...` | `092a8ab7aaf930d400544fc46eac2ee0...` | `db04ba9ead53b6596ada4714ef4c43a9...` | `6ee23f00ec43aa723cccf86272fcb691...` | `358c7f285910d8001950f7f85c416140...` | **3 / 3** |
| **Workshop W1** | `42e8c9215ec7f5f946c050ef65c978c4...` | `adec449b43019f80466c53bed72fc19a...` | `e85b9357f45438e9d73370e6f6211074...` | `a664ff1134c82d1adcb835f5bc15107c...` | `3cac91aac1e88e0f5f5b239a9e4d6dad...` | `1a5db75ee18e2e079bcbe01a4fe2c531...` | **3 / 3** |

---

## 5. Test Suite Verification

- `pytest mujoco_scenes/functional_tamp_pipeline/tests/`: **306 passed, 0 failed** in 350.96s.
- `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_p3i_full_ideal_convergence.py`: **6 passed, 0 failed** in 283.54s.

---

## 6. Verification Environment

All tests and matrix evaluations were executed locally on Linux:
- Python: `3.13.11` (`/home/naren/miniconda3/bin/python`)
- Pytest: `9.1.1` (`/home/naren/miniconda3/bin/pytest`)
- Working Directory: `/home/naren/RA_iiith`
- Target Branch: `naren/pipeline_check`
- Explicit note: Local environment verification; no remote GitHub CI status checks attached.
