# Phase 3 Pass P3-G, P3-G.1 & P3-G.2: Workshop Canonicalizer Repair & Closure

> **Passes**: P3-G, P3-G.1 & P3-G.2 — Workshop Canonicalizer Repair & Full Symmetry Closure  
> **Date**: 2026-09-01  
> **Branch**: `naren/pipeline_check`  
> **Status**: COMPLETE (FROZEN)  
> **Version**: `phase3_p3g_2_v1`  

---

## 1. Executive Summary

In Pass **P3-G**, **P3-G.1**, and **P3-G.2**, the Workshop VLM-to-Canonical-$G_F$ semantic compiler (`mujoco_scenes/workshop_phase1/requirements.py`) was comprehensively refactored and hardened into a deterministic, strictly fail-closed pipeline:
$$\text{Raw VLM Spec} \longrightarrow \text{Deterministic Semantic Compiler} \longrightarrow \text{Canonical } G_F \longrightarrow \text{validate\_runtime\_gf()} \longrightarrow \text{ground\_graph()}$$

The compiler operates without heuristics, candidate-category role manufacture, silent role loss, unsupported predicate emission, schema defaults, arbitrary duplicate-role merging, endpoint fabrication, raw group pass-through, or planner-context contamination of $\phi^*$.

The compiler was verified against the curated ideal raw fixture `mujoco_scenes/functional_tamp_pipeline/tests/fixtures/ideal_raw_vlm/workshop_W1.json`, achieving:
- **100% Role Identity Recall (3/3)** and **100% Role Identity Precision (3/3)**
- **100% Role Exact Recall (3/3)** and **100% Role Exact Precision (3/3)**
- **100% Relation Recall (3/3)** and **100% Relation Precision (3/3)**
- **100% Operation Group Identity Recall (0/0)** and **100% Operation Group Identity Precision (0/0)**
- **`extra_operation_groups: []`** (0 runtime operation groups emitted in active $G_F$; raw `group_1` validated and recorded as redundant wrapper)
- **`reference_complete: True`** and **`exact_structural_match: True`** against ground truth reference graph $G_F$
- **Full validation** under `graph.validate()` and `validate_runtime_gf(gf)`
- **End-to-end downstream dry-run validation**: `status = ACTION_SEQUENCE_READY`, 5 plan action steps (6 combined), 0 audit violations, independent plan replay `VALID`, access path `VALID`, `Scientifically Valid: 1 / 1`.

---

## 2. Core Architectural & Semantic Invariants

### 2.1 Forward-Only Phrase Matching & Strict Role Semantic Authority
- Role classification is determined strictly and exclusively from `function` and `description` via `map_workshop_role_function`, `map_workshop_fixed_target_role`, and `map_workshop_context_region_role`.
- `candidate_categories` are strictly isolated from role classification and cannot manufacture or distort functional roles (e.g. nonsense functions like `"paint wall"` with candidate categories `["screwdriver"]` fail closed with `UnmappedFunctionalConceptError`).

### 2.2 Strict Fixed Target & Context Region Authority
- Fixed target roles map strictly to `"repair_target"`. Any unmapped or unknown fixed target raises `UnmappedFunctionalConceptError`.
- Workbench support REGION roles map to `MAIN_WORKBENCH_ZONE` and are recorded in concept accounting with `status: "ABSORBED_INTO_PLANNER_CONTEXT"`. They are **never** emitted as selectable `FunctionalRole` nodes in active $G_F$.
- Any unmapped or unknown region role raises `UnmappedFunctionalConceptError`.

### 2.3 Fully Symmetric Active/Passive Relation Direction Grammar (P3-G.2)
Relations are mapped into the frozen Workshop predicate signatures:
1. `(driver, COMPATIBLE_WITH, fastener)`
2. `(driver, REACHES_TARGET, repair_target)`
3. `(fastener, COMPATIBLE_WITH_TARGET, repair_target)`

- **Canonical Endpoint Order (`driver -> fastener`, `driver -> repair_target`, `fastener -> repair_target`)**:
  - Accepts ONLY active grammar (`"compatible with"`, `"reaches target"`, `"threads into"`, `"torque to screw"`, etc.).
  - Rejects ALL passive grammar (`"is driven by"`, `"driven by"`, `"is reached by"`, `"reached by"`, `"receives fastener"`, `"is threaded by"`, `"fastened by"`), raising `MalformedVLMSpecificationError`.
  - Yields `direction_status: "PRESERVED"` and `structural_destination: "GRAPH_RELATION"`.
- **Reversed Endpoint Order (`fastener -> driver`, `repair_target -> driver`, `repair_target -> fastener`)**:
  - Accepts ONLY explicit reviewed passive/reverse phrases (`"is driven by"`, `"driven by"`, `"is reached by"`, `"reached by"`, `"receives fastener"`, `"threaded by"`, etc.).
  - Rejects active phrases with reversed endpoints, raising `MalformedVLMSpecificationError`.
  - Yields `direction_status: "NORMALIZED_TO_CANONICAL_SIGNATURE"`.
- **Planner Context Absorption**: `repair_target LOCATED_ON MAIN_WORKBENCH_ZONE` is absorbed into planner context (`structural_destination: "ABSORBED_INTO_PLANNER_CONTEXT"`).
- **Rejection of Self-Relations**: Self-relations raise `MalformedVLMSpecificationError`.
- **Unsupported Functional LOCATED_ON**: Functional roles (`driver`, `fastener`) with `LOCATED_ON` raise `UnsupportedCheckerCapabilityError`.

### 2.4 Unary Property Policy (Zero Runtime Unary Predicates)
- Active canonical Workshop $G_F$ has **zero** active unary predicates.
- Legacy capability markers (`CAN_DRIVE_SCREW`, `CAN_FASTEN`) are completely eliminated from runtime $G_F$.
- Unmapped unary properties raise `UnmappedFunctionalConceptError`.
- Unsupported unary properties on functional roles (`PLANAR_SUPPORT`, `OPEN_CAVITY`, `ELONGATED_OBJECT`) fail closed with `UnsupportedCheckerCapabilityError`.
- `PLANAR_SUPPORT` on workbench context is absorbed into planner context (`ABSORBED_INTO_PLANNER_CONTEXT`).

### 2.5 Operation Group Cardinality, Redundancy Proof & Provenance (P3-G.2)
- **Exact Group Relation Cardinality (P3-G.2)**: `required_relations` and `context_relations` must each contain **exactly 1** relation phrase. Lists with 0 or $>1$ phrases raise `MalformedVLMSpecificationError`.
- **Mandatory Usage Policy**: Workshop interaction group requires `usage_policy = "DEDICATED_PER_TARGET"`. Any other policy (e.g. `SEQUENTIAL_REUSE_ALLOWED`) raises `MalformedVLMSpecificationError`.
- **Mandatory Context Target**: Interaction group must explicitly include `context_role` mapping to `repair_target` and non-empty `context_relations` mapping to `REACHES_TARGET`.
- **Redundancy Proof Rule**: Before recording a group as redundant, the compiler proves that the corresponding canonical top-level graph triples exist:
  $$\{(\text{"driver"}, \text{"COMPATIBLE\_WITH"}, \text{"fastener"}), (\text{"driver"}, \text{"REACHES\_TARGET"}, \text{"repair\_target"})\} \subseteq \text{seen\_canonical\_triples}$$
  If any top-level graph relation is missing, the compiler raises `MalformedVLMSpecificationError`. The group does NOT synthesize or repair missing graph relations.
- **Trace Provenance (P3-G.2)**: Concept accounting preserves exact raw phrases (`raw_required_relation`, `raw_context_relation`) alongside canonical mapping and represented triples. Active Workshop $G_F$ emits `operation_groups = ()`.

### 2.6 Raw Metadata Counts Truthfulness
`VLMSpecProvider._workshop()` reports true raw counts from canonicalization trace:
- `gf.metadata["raw_roles_count"]`: 3 (raw requirements count)
- `gf.metadata["raw_relations_count"]`: 3 (raw relations count)
- `gf.metadata["raw_operation_groups_count"]`: 1 (raw interaction groups count)
- `len(gf.operation_groups)`: 0 (canonical runtime groups count)

### 2.7 Canonicalization Version Provenance
The canonicalization version for Workshop is bumped to:
```python
WORKSHOP_VLM_CANONICALIZATION_VERSION = "phase3_p3g_2_v1"
```
and is recorded in `metadata["vlm_canonicalization_version"]` and `metadata["canonicalization_trace"]["vlm_canonicalization_version"]`.

---

## 3. Empirical Verification & Test Matrix

| Test Suite | Command | Result |
|---|---|---|
| Workshop Dedicated Suite | `pytest mujoco_scenes/tests/test_workshop_vlm_requirements.py` | 32/32 PASSED (100%) |
| Ideal Fixtures Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_ideal_fixtures.py` | 4/4 PASSED (100%) |
| Boundary & Interface Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_vlm_interface_boundary.py` | 6/6 PASSED (100%) |
| Architecture Suite | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_architecture.py` | 57/57 PASSED (100%) |
| Executable Grounding IR | `pytest mujoco_scenes/functional_tamp_pipeline/tests/test_executable_grounding_ir.py` | 5/5 PASSED (100%) |
| Reference Evaluator | `pytest mujoco_scenes/tests/test_phase3_6b0_reference_evaluator.py` | 24/24 PASSED (100%) |
| Kitchen VLM Graph (Frozen) | `pytest mujoco_scenes/tests/test_kitchen_vlm_functional_graph.py` | 36/36 PASSED (100%) |
| Living Room VLM (Frozen) | `pytest mujoco_scenes/tests/test_living_room_vlm_canonicalization.py` | 11/11 PASSED (100%) |
| Workshop W1 GT Dry-Run Evaluation | `scripts/evaluate_functional_tamp_variants.py --domains workshop --variants W1 --mode gt` | ACTION_SEQUENCE_READY (Combined: 6, Match: YES, Valid: 1/1) |

---

## 4. Fresh W1 GT Reproducibility Record (P3-G.2 Tested Code SHA: `5a337261`)

Three independent trials of `evaluate_functional_tamp_variants.py --domains workshop --variants W1 --mode gt` were executed into isolated output roots on clean code commit `5a337261`:

| Trial | Output Root | Terminal Status | Inspected Regions | PlanLen (Comb) | Grounding | Audit | Replay | Access | Spec SHA256 |
|---|---|---|---|---|---|---|---|---|---|
| Trial 1 | `/tmp/w1_gt_repro_run1_20260901_162815` | `ACTION_SEQUENCE_READY` | `["LEFT_DRAWER"]` (1) | 5 (6) | `COMPLETE` | `VALID` | `VALID` | `VALID` | `c92a080e...` |
| Trial 2 | `/tmp/w1_gt_repro_run2_20260901_162815` | `ACTION_SEQUENCE_READY` | `["LEFT_DRAWER"]` (1) | 5 (6) | `COMPLETE` | `VALID` | `VALID` | `VALID` | `c92a080e...` |
| Trial 3 | `/tmp/w1_gt_repro_run3_20260901_162815` | `ACTION_SEQUENCE_READY` | `["LEFT_DRAWER"]` (1) | 5 (6) | `COMPLETE` | `VALID` | `VALID` | `VALID` | `c92a080e...` |

### Classification: `W1_GT_REPRODUCIBLE_READY_ON_P3G2_SHA`
All 3 runs identically yielded `ACTION_SEQUENCE_READY` with 1 inspection (`LEFT_DRAWER`), plan length 5, and full validation across grounding audit, replay, and access check.



