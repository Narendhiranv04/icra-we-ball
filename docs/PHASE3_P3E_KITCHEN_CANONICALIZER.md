# Pass 3.7G (P3-E): Kitchen Canonicalizer Repair

**Pass ID**: `P3-E`  
**Commit Type**: `Pass 3.7G (P3-E): Make Kitchen canonicalization fail-closed and lossless`  
**Branch**: `naren/pipeline_check`  
**Date**: 2026-09-01  
**Scope**: `mujoco_scenes/kitchen_vlm_functional_graph.py`, `mujoco_scenes/tests/test_kitchen_vlm_functional_graph.py`, `mujoco_scenes/functional_tamp_pipeline/tests/`

---

## 1. Architectural Contract & Objectives

The Kitchen functional canonicalizer serves as the deterministic semantic compiler between the raw VLM functional specification and the canonical executable graph $G_F$:
$$\text{Raw Kitchen VLM Document} \longrightarrow \text{Deterministic Semantic Compiler} \longrightarrow \text{Canonical Executable } G_F \longrightarrow \text{validate\_runtime\_gf()} \longrightarrow \text{ground\_graph()}$$

Under the frozen P3-D/P3-D.1 predicate signatures and system context contract:
- Every task-required raw concept must be exactly one of:
  1. **PRESERVED**: Mapped 1:1 to active canonical roles, predicates, or operation groups.
  2. **EXPLICITLY MERGED**: Merged according to an explicit, verified domain rule with full diagnostic evidence.
  3. **EXPLICITLY REJECTED**: Failed closed with structured, domain-scoped diagnostic exceptions.
- There is **ZERO SILENT LOSS** and **ZERO PREDICATE FABRICATION**.

---

## 2. Defects Identified & Repaired

| Defect ID | Prior Defective Behaviour | Repaired Fail-Closed Implementation |
|---|---|---|
| **A. Silent Raw Role Dropping** | Unmapped role functions silently dropped with `continue`, causing downstream `INCOMPLETE`/`INFEASIBLE`. | Raises `UnmappedFunctionalConceptError` with complete function, description, and candidate category diagnostics. |
| **B. Role Collision Heuristic** | Colliding raw roles mapping to same canonical role resolved by `max(existing, new)` heuristic. | Raises `AmbiguousCanonicalizationError` (duplicate canonical roles fail closed). |
| **C. Unknown Relation Fabrication** | Unmapped relation strings defaulted to `INSERTABLE_IN` (`if mapped_rel is None: mapped_rel = 'INSERTABLE_IN'`). | Removed fallback; unmapped relations raise `UnmappedFunctionalConceptError`. |
| **D. Broad Keyword Fallback** | `map_binary_relation()` mapped broad phrases like `'reach'`, `'require'`, `'stir with'`, `'eat with'` to `INSERTABLE_IN`. | Removed keyword substring heuristics; strict reviewed dictionary matching with ambiguity detection. |
| **E. Silent Property Dropping** | Unmapped required properties silently ignored if not in dictionary. | Unmapped items in `required_properties` raise `UnmappedFunctionalConceptError`. |
| **F. Operation Group Fabrication** | Unmapped operation relations defaulted to `INSERTABLE_IN`; empty relations defaulted to `['INSERTABLE_IN']`. | Empty `required_relations` or unmapped relations raise `MalformedVLMSpecificationError` / `UnmappedFunctionalConceptError`. |
| **G. Target Count Clipping** | `required_target_count` clipped to `min(req, target_role_count)` silently modifying task requirements. | If `req_target_count > target_role_count`, raises `MalformedVLMSpecificationError`. |
| **H. Invalid Value Self-Repair** | Non-positive counts converted to `1`; invalid policies converted to `DISTINCT`. | Invalid raw values raise `MalformedVLMSpecificationError` without self-repair. |

---

## 3. Ideal Fixture Diagnostics (Kitchen K1)

The ideal raw Kitchen fixture (`kitchen_K1.json`) was evaluated with zero network calls via `MockFMAdapter`:

```
============================================================
IDEAL FIXTURE EVALUATION: Kitchen K1
============================================================
G_F Validation:
  - Canonical Nodes: 6 ('coffee_container', 'soup_container', 'coffee_stirrer', 'soup_eating_utensil', 'coffee_source', 'water_source')
  - Canonical Relations: 4 (INSERTABLE_IN and REACHES_BOTTOM on both pairs)
  - Operation Groups: 2 ('coffee_stirring', 'soup_serving')
  - validate_runtime_gf(): PASSED

Reference Evaluation against Ground Truth (GTSpecProvider):
  - role_identity_recall: 1.0 (6/6)
  - role_identity_precision: 1.0 (6/6)
  - relation_recall: 1.0 (4/4)
  - relation_precision: 1.0 (4/4)
  - operation_group_identity_recall: 1.0 (2/2)
  - operation_group_identity_precision: 1.0 (2/2)
  - reference_complete: True
============================================================
```

---

## 4. Concept Accounting & Invariant Trace

Canonical compilation produces an explicit `concept_accounting` trace in the compilation metadata:
- **Roles**: All raw roles recorded with their canonical role, count, binding policy, and `status: "PRESERVED"`.
- **Properties**: All raw required properties recorded with mapped canonical predicate and status (`PRESERVED` or `MERGED_BY_EXPLICIT_RULE` for duplicate aliases).
- **Relations**: All raw relation tuples recorded with canonical predicates and endpoints.
- **Operation Groups**: All operation groups recorded with canonical group ID, tools, targets, target counts, usage policies, and required relations.

---

## 5. Source-Audit Phi Preference Separation Invariant

Confirmed invariant in `grounding.py`:
- `check_semantic_role_compatibility()` uses candidate category lists purely as a compatibility filter / gate (`TRUE`/`FALSE`/`UNKNOWN`).
- Semantic candidate category ordering in $G_F$ informs detector vocabulary generation, but **category ranking is NOT an optimization objective** used to choose $\phi^*$ among already feasible assignments.
- Grounding selection remains purely constraint-satisfaction based on spatial and physical predicates.

---

## 6. Ground-Truth Control Path Verification

Kitchen K1 GT control path was re-verified end-to-end:
- **Pipeline Status**: `ACTION_SEQUENCE_READY`
- **Plan Grounding Audit**: `plan_replay_valid: true`, `violations: []`
- **Action Sequence**: 24 deterministic STRIPS actions covering object retrieval, pouring, stirring, and placement to serving area.

---

## 7. Comprehensive Test Suite Results

- `mujoco_scenes/tests/test_kitchen_vlm_functional_graph.py`: **25 / 25 PASSED**
- `mujoco_scenes/functional_tamp_pipeline/tests/`: **255 / 255 PASSED**
- Total Test Suite: **280 / 280 PASSED (100%)**
