# Phase 3 Master Plan

> **Version**: 1.1 (Corrected Post-Audit)
> **Created**: 2026-08-31
> **Branch**: `naren/pipeline_check`
> **HEAD at creation**: `5551f67c`

---

## Instructions for Coding Agents

1. **Read this entire plan before editing code.**
2. Inspect the CURRENT repository state rather than assuming old notes are true.
3. Work only on **CURRENT NEXT PASS** unless explicitly instructed otherwise.
4. Do not modify frozen downstream components without evidence that the failure belongs there.
5. **Diagnose layer before fixing symptom** — use the Diagnostic Decision Tree (§5).
6. Run the pass-specific tests listed in each pass description.
7. Do not claim success without artifact/test evidence.
8. Update the pass status and factual notes at the end of each session.
9. Stop after completing the requested pass unless explicitly asked to continue.
10. Preserve the GT/VLM → same canonical G_F architecture.

---

## 0. Mission & Non-Negotiable Architecture

### Mission
Phase 3 produces a **complete functional grounding** pipeline that, given a task instruction and RGB-D observations, deterministically yields either:
- `ACTION_SEQUENCE_READY` with a valid symbolic action plan, or
- `INFEASIBLE` with a correct causal explanation.

### Architecture Invariants — DO NOT VIOLATE

1. `G_F` represents WHAT functional roles, properties, cardinalities, and relations are required.
2. `G_O` represents WHAT has been observed/verified in the physical scene.
3. `ground_graph(G_F, G_O)` is the **sole authority** for selecting functional role assignments (φ*).
4. φ* is immutable downstream — the symbolic compiler consumes it, never reassigns roles.
5. No FM/VLM call after G_F has been constructed.
6. GT and VLM paths **must converge** onto the same canonical `FunctionalRequirementGraph` interface.
7. Once a valid canonical G_F exists, downstream behaviour is agnostic to whether it came from GT or VLM.
8. Search updates G_O and re-runs the same deterministic grounding — it does not modify G_F.
9. FM semantic preferences inform discovery vocabulary, not post-feasibility φ* ranking.
10. No GT oracle in the VLM runtime path. No hidden simulator state as semantic evidence.

### Pipeline Flow
```
Task instruction + initial RGB
        |
        v
   GT or VLM  ──────────────┐
        |                    |
        v                    |
   Raw Specification         |
        |                    |
        v                    |
   Semantic Canonicalizer    |
        |                    |
        v                    |
   Canonical G_F             |
        |                    |
  RGB-D / perception         |
        |                    |
        v                    |
   Observed Scene Graph G_O  |
        |                    |
        +────────────────────+
        |
        v
   ground_graph(G_F, G_O) → φ*
        |
        v
   Symbolic task compiler
        |
        v
   A* planner → Action Sequence
```

---

## 1. Current Verified State

### VERIFIED ✓
| Component | Evidence |
|---|---|
| `FunctionalRequirementGraph` data model | `models.py` — clean frozen dataclass with `validate()` |
| `ObservedSceneGraph` data model | `scene_graph.py` — keyed relation storage, stage tracking |
| `GraphGroundingResult` model | `models.py` — complete/incomplete/infeasible tri-state |
| `ground_graph()` constraint solver | `grounding.py` — combinatorial role assignment with relation checking |
| `GTSpecProvider` for all 3 domains | Produces valid G_F; 32/32 GT regression passes |
| GT downstream: K1-K12, L1-L10, W1-W10 | All 20 feasible → `ACTION_SEQUENCE_READY`, all 12 infeasible → `INFEASIBLE` |
| `validate_runtime_gf()` structural validator | `task_interface_validator.py` |
| `VLMSpecProvider.provide()` structural validation & error wrapping | `vlm_spec_provider.py` — complete method body with try/except wrapping in `MalformedVLMSpecificationError`, and `@staticmethod` on `_workshop` (prior truncation diagnosis was a tooling/display artifact) |
| Kitchen domain adapter (symbolic compiler) | `domains/kitchen.py` — `KitchenPlanningCompiler`, `build_kitchen_observed_scene_graph` |
| Living Room domain adapter | `domains/living_room.py` — `compile_living_room_task_from_graph`, `build_living_room_observed_scene_graph` |
| Workshop domain adapter | `domains/workshop.py` — `build_workshop_observed_scene_graph` |
| Search loop | `search.py` — `search_until_satisfied()` protocol |
| A* planner | `planning.py` + `symbolic_planning_core.py` |
| Reference evaluator | `gf_reference_evaluator.py` — structural comparison metrics |
| Runtime isolation (no GT imports in VLM path) | `test_executable_grounding_ir.py::test_static_runtime_isolation` |
| VLM prompt leak protection & deterministic audit | `audit.py`, `test_vlm_interface_boundary.py`, `test_provenance_and_audit.py` |
| Provenance fingerprinting & stale resume prevention | `audit.py::compute_provenance_fingerprint`, `scripts/run_phase36b2_matrix.py` |
| FM adapter retry/reconnect | `fm_adapter.py` — 3-attempt retry with socket error handling |
| Ideal raw VLM fixture suite & MockFMAdapter | `test_ideal_fixtures.py`, `fixtures/ideal_raw_vlm/` |
| Concept-exact canonicalization preservation diagnostics | `docs/PHASE3_P3C_IDEAL_FIXTURE_DIAGNOSIS.md`, `test_ideal_fixtures.py` |
| Frozen Predicate Signature Registry | `predicate_registry.py` — immutable domain-scoped predicate signatures and direction rules |
| System Context & Planner Constants Registry | `system_context_registry.py` — formal separation of selectable assets, fixed anchors, planner constants, and search regions |
| 255 unit/integration tests pass | `pytest mujoco_scenes/functional_tamp_pipeline/tests/` |

### PARTIALLY VERIFIED ~
| Component | Status |
|---|---|
| Kitchen VLM canonicalizer | Compiles valid G_F for some inputs; role recall ~67%; unmapped relations default to `INSERTABLE_IN`; operation groups only accepted for 2 hard-coded tool↔target pairs |
| Living Room VLM canonicalizer | Entity-kind hard gate works; but compositional cup+saucer cardinality rules untested across diverse raw outputs; context nodes require systematic contract |
| Workshop VLM canonicalizer | Role normalization exists; but 10/10 live cases failed at spec stage due to multiple distinct VLM raw patterns |
| Region resolution (Kitchen) | Maps natural language to D1/D2/C2/B1/C1 via fuzzy matching; but VLM-emitted local IDs not always resolvable |

### BROKEN / UNRESOLVED ISSUES ✗
| Component | Evidence |
|---|---|
| VLM → ACTION_SEQUENCE_READY | 0/20 feasible variants succeeded in Pass 3.6B.2 |
| VLM → correct INFEASIBLE | 4/12 infeasible variants matched terminal label (K7, K9, L8, L10), but had low role recall (0.50-0.67) and zero exact structural match — terminal label correctness did not reflect verified causal correctness of the inferred deficiency |
| Kitchen: canonicalizer unmapped relations | `map_binary_relation()` falls back to `INSERTABLE_IN` rather than failing closed or mapping explicitly |
| Kitchen: operation groups hard-gated | Lines 420-425: only `coffee_stirrer→coffee_container` and `soup_eating_utensil→soup_container` are accepted; other valid pairings are dropped |
| Kitchen: concept dropping vs raw absence | In some cases raw VLM omits secondary roles (e.g. `coffee_source`), while in other cases canonicalization drops or clips counts (`required_target_count`) |
| Workshop: live VLM failure modes | 10/10 live cases failed at spec validation, with 4 distinct failure modes: (1) duplicate `driver` role collisions, (2) unmapped capability phrases, (3) undeclared role references in relations/groups, (4) malformed operation group context relations |

### NOT YET TESTED
| Component | Note |
|---|---|
| 9B vs 27B raw specification comparison | Not performed |
| Clean benchmark with single frozen commit | Pass 3.6B.2 mixed artifacts across commits |

---

## 2. Current Failure Map

### Kitchen
| Layer | Failure | Severity |
|---|---|---|
| **Raw VLM / Model** | VLM frequently omits secondary source roles (`coffee_source`, `soup_container`) in raw response | HIGH — causes downstream INCOMPLETE/INFEASIBLE |
| **Canonicalizer** | Unmapped binary relations default to `INSERTABLE_IN` (line 394, line 232) instead of failing closed | MEDIUM — masks wrong relations |
| **Canonicalizer** | Operation groups hard-gated to exactly 2 canonical patterns (lines 420-425) | MEDIUM — any VLM variation in tool↔target naming gets dropped |
| **Canonicalizer** | `required_target_count` silently clipped (line 460) | LOW — violates semantic preservation |
| **G_F↔G_O** | If VLM omits required sources, symbolic planner cannot find contents → caught as `INFEASIBLE` | HIGH |
| **Downstream** | `KitchenPlanningCompiler` builds STRIPS from `compile_observed_symbolic_state()` which hardcodes `symbolic_task` defaults (coffee, water, soup contents) | LOW |

### Living Room
| Layer | Failure | Severity |
|---|---|---|
| **Canonicalizer** | Raw VLM emitting separate cup (2) and saucer (2) roles can lead to count distortion if naive summation or naive max is applied | HIGH |
| **Canonicalizer** | Context targets (`SEATING_POSITION`, `SEATING_PAIR`) must be integrated systematically without penalizing VLM for omitting scene fixtures | HIGH |
| **G_F↔G_O** | G_O stores relations as `REGION --FITS_SET_ON--> CUP_SAUCER_SET` (subject=region), while VLM describes `cup placed_on table` (subject=cup). Canonicalizer disambiguates via role type but fragile for novel phrasings | MEDIUM |
| **VLM raw** | 5/10 variants → VLM_SPEC_FAILED (L4-L7, L9) — needs layer diagnosis | MEDIUM |

### Workshop
| Layer | Failure | Severity |
|---|---|---|
| **VLM raw & Canonicalizer** | Duplicate canonical `driver` role collisions (W1, W2, W6, W9) when VLM emits multiple alternative tools | HIGH |
| **Canonicalizer** | Unmapped natural language capability phrases for fastener/screw (W5, W7, W10) | HIGH |
| **VLM raw** | Relations/groups referencing undeclared or raw role IDs (W3, W4) | MEDIUM |
| **VLM raw** | Malformed operation group specifications with duplicate context relations (W8) | MEDIUM |

---

## 3. Canonical Phase-3 Contracts

### Raw VLM Specification (per domain)
The VLM emits a JSON document conforming to `RESPONSE_SCHEMA` or `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA` in `fm_adapter.py`. Contains:
- `status`: "SUPPORTED" | "UNSUPPORTED"
- `functional_roles[]`: id, entity_kind, function, required_count, binding_policy, candidate_categories, visible_candidates, required_properties
- `functional_relations[]`: subject_role, object_role, relation
- `interaction_groups[]`: id, tool_role, target_role, required_target_count, usage_policy, required_relations, function
- `inspectable_regions[]`, `inspection_order[]`

**Owner**: VLM (one-shot generation at task start). Retained as `raw_vlm_response` in metadata.

### Canonical G_F (`FunctionalRequirementGraph`)
Defined in `models.py`. Contains:
- `nodes: dict[str, FunctionalRole]` — name, entity_kind, count, semantic_categories, unary_predicates, binding_policy, verification_mode
- `relations: tuple[FunctionalRelation]` — subject_role, predicate, object_role
- `operation_groups: tuple[OperationGroup]` — tool_role, target_role, required_target_count, usage_policy, required_relations
- `candidate_regions`, `region_ranking`, `detector_vocabulary`
- `source`: "GT_FUNCTIONAL_SPEC_ONLY" | "VLM_CANONICAL_G_F" | "VLM_FUNCTIONAL_SPEC"

**Owner**: Semantic canonicalizer (deterministic, inspectable, fail-closed). Must pass `validate()` and `validate_runtime_gf()`.

### G_O (`ObservedSceneGraph`)
Defined in `scene_graph.py`. Contains:
- `nodes: dict[str, ObservedNode]` — instance_id, entity_kind, canonical_category, semantic_labels, unary_predicates, geometry
- `relations: dict[(pred, subj, obj), ObservedRelation]` — subject_id, predicate, object_id, status ∈ {TRUE, FALSE, UNKNOWN}
- `inspected_regions`, `stage_index`

**Owner**: Perception pipeline (domain-specific builders: `build_kitchen_observed_scene_graph`, etc.)

### φ* (`GraphGroundingResult`)
Defined in `models.py`. Contains:
- `status`: "COMPLETE" | "INCOMPLETE" | "INFEASIBLE"
- `assignment: dict[role_name → instance_id(s)]`
- `operation_bindings: dict[group_id → [{tool_id, target_id}]]`
- `missing_roles`, `unsatisfied_relations`

**Owner**: `ground_graph()` in `grounding.py` — sole assignment authority.

### Action Sequence (`PipelineResult`)
Defined in `models.py`. Terminal status ∈ {`ACTION_SEQUENCE_READY`, `INCOMPLETE`, `INFEASIBLE`, `VLM_SPEC_FAILED`}.

---

## 4. Predicate Signature Registry

### Kitchen Domain
| Predicate | Subject Kind | Object Kind | G_F Direction | G_O Direction | Checker |
|---|---|---|---|---|---|
| `INSERTABLE_IN` | OBJECT (tool) | OBJECT (container) | tool → container | tool → container | `pairwise_relation_evaluation` |
| `REACHES_BOTTOM` | OBJECT (tool) | OBJECT (container) | tool → container | tool → container | `pairwise_relation_evaluation` |
| `OPEN_CAVITY` | OBJECT (unary) | — | — | — | geometric predicate |
| `ELONGATED_OBJECT` | OBJECT (unary) | — | — | — | geometric predicate |

### Living Room Domain
| Predicate | Subject Kind | Object Kind | G_F Direction | G_O Direction | Checker |
|---|---|---|---|---|---|
| `FITS_SET_ON` | REGION | OBJECT (CUP_SAUCER_SET) | REGION → CUP_SAUCER_SET | REGION → CUP_SAUCER_SET | geometric fit evaluation |
| `FITS_ON` | REGION | OBJECT (REMOTE) | REGION → REMOTE | REGION → REMOTE | geometric fit evaluation |
| `NEAR_SEAT` | REGION | FIXED_TARGET (SEATING_POSITION) | REGION → SEATING_POSITION | REGION → SEATING_POSITION | distance threshold |
| `ACCESSIBLE_FROM_BOTH_SEATS` | REGION | FIXED_TARGET (SEATING_PAIR) | REGION → SEATING_PAIR | REGION → SEATING_PAIR | distance threshold |
| `PLANAR_SUPPORT` | REGION (unary) | — | — | — | geometric normal/planarity |

### Workshop Domain
| Predicate | Subject Kind | Object Kind | G_F Direction | G_O Direction | Checker |
|---|---|---|---|---|---|
| `COMPATIBLE_WITH` | OBJECT (driver) | OBJECT (fastener) | driver → fastener | driver → fastener | `evaluate_compatible_with` |
| `REACHES_TARGET` | OBJECT (driver) | FIXED_TARGET (repair_target) | driver → repair_target | driver → repair_target | `evaluate_reaches_target` |
| `COMPATIBLE_WITH_TARGET` | OBJECT (fastener) | FIXED_TARGET (repair_target) | fastener → repair_target | fastener → repair_target | `evaluate_compatible_with_target` |
| `CAN_DRIVE_SCREW` | OBJECT (unary) | — | — | — | geometric predicate |
| `CAN_FASTEN` | OBJECT (unary) | — | — | — | geometric predicate |

**Key rule**: The canonicalizer must convert natural grammatical direction into these fixed signatures. Both "spoon fits inside cup" and "cup accepts spoon" must compile to `spoon --INSERTABLE_IN--> cup`.

---

## 5. Diagnostic Decision Tree

When a case fails, apply this tree **in order**:

```
1. Does GT G_F → G_O → ground_graph → compiler → A* succeed?
   ├─ NO → Downstream bug (grounding / compiler / planner). Fix before testing VLM.
   └─ YES →
       2. Does the manually correct "ideal raw" fixture → canonicalizer → G_F → downstream succeed?
          ├─ NO → Canonicalizer or G_F↔G_O contract bug.
          └─ YES →
              3. Does the live VLM raw response contain all required semantic concepts?
                 ├─ NO → Model capacity issue (try prompt improvement, then 27B).
                 └─ YES →
                     4. Does the canonicalized G_F preserve those concepts?
                        ├─ NO → Canonicalizer bug (concept dropped/mangled).
                        └─ YES →
                            5. Does grounding find a valid φ*?
                               ├─ NO → G_F↔G_O representation mismatch or search issue.
                               └─ YES →
                                   6. Does the compiler/planner produce a valid plan?
                                      ├─ NO → Symbolic compilation bug.
                                      └─ YES → ✓ Success
```

**Layer codes for diagnostic tagging:**
- `A` = Model capacity / raw VLM understanding
- `B` = Canonicalizer / VLM→G_F adapter
- `C` = G_F↔G_O representational mismatch
- `D` = Graph grounding
- `E` = Symbolic compilation / A*
- `F` = Search / region resolution
- `X` = Infrastructure (network, syntax, provenance)

---

## 6. Ordered Implementation Passes

### P3-A: Benchmark Provenance & Diagnostic Infrastructure
**Status**: `[x] COMPLETE`

**Objective**: Establish immutable configuration fingerprinting, prevent silent stale-case reuse, harden runtime accounting, ensure raw VLM failure artifact retention, and implement deterministic prompt leakage auditing.

**Scope**: `mujoco_scenes/functional_tamp_pipeline/audit.py`, `scripts/run_phase36b2_matrix.py`, `mujoco_scenes/functional_tamp_pipeline/tests/test_provenance_and_audit.py`

**Accomplished Changes**:
1. Added `compute_provenance_fingerprint()` to `audit.py` providing immutable SHA-256 configuration hashes covering git commit, git dirty state, source tree diff hash (`git_dirty_source_hash`), model identifier, FM endpoint, canonicalization version, prompt/schema hash, task instruction hash, and search order mode.
2. Updated `scripts/run_phase36b2_matrix.py` to require exact fingerprint match for case reuse, supported `--no-resume` / `--fresh` CLI flags, and eliminated silent stale case reuse.
3. Added `clean_case_directory()` ensuring that when `--fresh` or a fingerprint mismatch occurs, old case artifacts (such as stale `fm_call_*.json` logs) are wiped before execution.
4. Implemented request-only `audit_prompt_leakage()` in `audit.py` strictly scanning outgoing model requests while excluding generated model completions from false leakage flagging.
5. Removed synthetic `fm_call_001.json` creation from canonical G_F fallback, ensuring raw diagnostic unavailability is truthfully recorded.
6. Updated provider replay matrix aggregation to use the strong `validate_vlm_replay()` validator boolean (`provider_replay_validation_pass_count` / `provider_replay_validation_pass_rate`).
7. Hardened benchmark accounting: distinct tracking for `invocation_wall_time_seconds`, `summed_live_case_runtime_seconds`, `summed_newly_executed_runtime_seconds`, `number_of_cases_executed_this_invocation`, `number_of_cases_reused`, and `total_trial_records`.
8. Explicitly clarified provider replay terminology as deterministic downstream solver replay of saved G_F (not stochastic VLM generation reproducibility).
9. Ensured `TAMP_FM_DIAGNOSTICS_DIR` is set for all live runs so raw model responses are retained on disk even when validation or canonicalization fails.
10. Added comprehensive unit tests in `test_provenance_and_audit.py` (232/232 tests pass).

**Acceptance**: All P3-A and P3-A.1 acceptance criteria verified.

---

### P3-B: K1/L1/W1 GT Downstream Controls & Canonical phi* Identity Integrity
**Status**: `[x] COMPLETE`

**Objective**: Verify that GT G_F → actual G_O → ground_graph → compiler → A* works for K1, L1, W1 individually with full artifact preservation and strictly verified canonical $\phi^*$ instance identity integrity.

**Accomplished Changes & Empirical Findings**:
1. **Identified & Fixed Workshop Post-Grounding Identity Rewrite**: Diagnosed that `WorkshopDomainAdapter.evaluate_satisfaction()` was rewriting generic $G_O$ instance tracks (`object_0003`, `object_0002`) into simulator backend names (`workshop_power_driver`, `workshop_medium_phillips_screw`) via semantic labels (`_physical_handle()`). Replaced this with direct canonical preservation of $G_O$ instance IDs in $\phi^*$, supplying source inspection regions and target fixtures via `planning_context()` without mutating $\phi^*$.
2. **Kitchen K1 GT Control**: Successfully executed to `ACTION_SEQUENCE_READY` (24 STRIPS actions, 18 $G_O$ nodes, 0 audit violations, independent replay `VALID`).
3. **Living Room L1 GT Control**: Successfully executed to `ACTION_SEQUENCE_READY` (10 STRIPS actions, 14 $G_O$ nodes, 0 audit violations, independent replay `VALID`).
4. **Workshop W1 GT Control**: Successfully executed to `ACTION_SEQUENCE_READY` (5 STRIPS actions, 7 $G_O$ nodes, 0 audit violations, independent replay `VALID`).
5. **Plan Grounding Audit Hardening**: Integrated `audit_plan_grounding` systematically across all three domain runners, confirming `all_assignment_nodes_observed: true`, `plan_uses_only_grounded_task_objects: true`, and `plan_replay_valid: true`.
6. **Unit & Regression Testing**: Added `test_phi_identity_integrity.py` testing that canonical $\phi^*$ never projects into simulator body handles (233/233 test suite passing).
7. **Documentation**: Full control evidence and exact W1 identity trace table documented in `docs/PHASE3_P3B_GT_CONTROLS.md`.

**Acceptance**: All three controls yield `ACTION_SEQUENCE_READY` with 0 audit violations and complete canonical identity preservation.

---

### P3-B.1: Canonical phi* Output Contract & Grounding Audit Hardening
**Status**: `[x] COMPLETE`

**Objective**: Enforce the invariant that `PipelineResult.assignment == GraphGroundingResult.assignment` on successful runs, separate domain-specific compiler projections into explicit non-$\phi^*$ artifacts, harden `audit_plan_grounding` to fail closed without heuristic string prefixes, and purge unused semantic backend handle helpers.

**Accomplished Changes & Empirical Findings**:
1. **Canonical Output Normalization**: Normalized `PipelineResult.assignment` in Living Room to return `ground_result.assignment` directly. The slot $\to$ region support mapping is preserved separately as a compiler projection in `planner_projection.json`, `region_assignments.json`, and `functional_region_witness.json`, ensuring `result.json.assignment == graph_grounding_result.json.assignment` across all 3 domains.
2. **Grounding Audit Hardening**: Eliminated permissive string-prefix allowlists (`region_*`, `seat_*`, `pos_*`, `slot_*`) from `audit_plan_grounding()`. Audit now strictly checks that every plan argument is an assigned $\phi^*$ object, an observed $G_O$ node (`REGION` or `FIXED_TARGET`), or an explicitly passed domain constant via `allowed_context_ids`.
3. **Dead Code Elimination**: Removed dead `WorkshopDomainAdapter._physical_handle()` and `SEMANTIC_ENTITY_HANDLES` from `domains/workshop.py` to prevent latent reintroduction of simulator backend handles on canonical Phase-3 paths.
4. **Regression & Unit Tests**: Added negative regression tests for unobserved region/seat/object arguments and verified that `PipelineResult.assignment == GraphGroundingResult.assignment` across Kitchen K1, Living Room L1, and Workshop W1 (236/236 test suite passing).

**Acceptance**: All three domain controls pass with 0 violations under the hardened audit, and `PipelineResult.assignment` strictly equals canonical $\phi^*$.

---

### P3-C: Ideal Raw VLM Fixture Framework & Canonicalization Loss Diagnosis
**Status**: `[x] COMPLETE`

**Objective**: Create manually-crafted "perfect VLM response" semantic fixtures per domain using natural open-vocabulary language without internal identifiers or oracle strings, and diagnose concept preservation and failure layers across the real current canonicalizers with zero model calls.

**Scope**: `mujoco_scenes/functional_tamp_pipeline/tests/fixtures/ideal_raw_vlm/` (`kitchen_K1.json`, `living_room_L1.json`, `workshop_W1.json`, `README.md`), `mujoco_scenes/functional_tamp_pipeline/tests/test_ideal_fixtures.py`, `docs/PHASE3_P3C_IDEAL_FIXTURE_DIAGNOSIS.md`.

**Accomplished Changes & Empirical Findings**:
1. **Curated Ideal Raw Fixtures**: Authored `kitchen_K1.json` (conforming to `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA`), `living_room_L1.json` (conforming to `RESPONSE_SCHEMA`), and `workshop_W1.json` (conforming to `RESPONSE_SCHEMA`).
2. **Strict Anti-Leak Invariants Enforced**: Proved that all fixtures contain zero canonical role IDs (`coffee_container`, `driver`, `CUP_SAUCER_SET`), zero canonical predicate tokens (`INSERTABLE_IN`, `FITS_SET_ON`), zero backend handles (`workshop_power_driver`), and zero oracle strings (`F0`, `K1`, `object_0001`, `LEFT_DRAWER`).
3. **Zero-Network Diagnostic Control Executed**:
   - **Kitchen K1**: `CANONICALIZED` via `compile_vlm_functional_graph` with 100% role recall (6/6) and 100% relation recall (4/4). Passes `graph.validate()` and `validate_runtime_gf()`.
   - **Living Room L1**: `CANONICALIZATION_FAILED` with `UnmappedFunctionalConceptError: VLM living room relation 'can hold drinkware set' cannot be mapped to any reviewed relation` localized to `environment_vlm_requirements.py:172` in `map_living_room_relation()` (Layer B failure).
   - **Workshop W1**: `CANONICALIZED` via `FMRequirementProvider` with 100% role recall (3/3), 100% relation recall (3/3), and candidate regions resolved from natural phrases to `('LEFT_DRAWER', 'RIGHT_DRAWER', 'TOOL_CABINET')`. Passes `graph.validate()` and `validate_runtime_gf()`.
4. **Comprehensive Diagnostic Report**: Documented all traces and layer localizations in `docs/PHASE3_P3C_IDEAL_FIXTURE_DIAGNOSIS.md`.

**Acceptance**: Semantically correct ideal raw fixtures exist, pass schema and anti-leak validation, and are deterministically evaluated against production canonicalizers without live VLM calls, precisely localizing interface/canonicalizer failure layers for downstream repair in P3-E/F/G.

---

### P3-C.1: Ideal Fixture & Diagnostic Validity Hardening
**Status**: `[x] COMPLETE`

**Objective**: Neutralize all raw fixture local identifiers, prune semantically unjustified unary property assertions, harden anti-leakage validation across all relation and group fields, replace tautological preservation labels with evidence-based traces, make the test-side FM adapter future-proof for post-repair Living Room canonicalization, and disclose full GT reference comparison metrics.

**Accomplished Changes & Empirical Findings**:
1. **Neutral Model-Local Identifiers**: Refactored all raw fixtures to use neutral local identifiers (`role_1`..`role_6`, `group_1`..`group_2`, `search_1`..`search_3`), ensuring canonicalizers deduce meaning exclusively from natural language semantic fields without ID leakage.
2. **Pruned Unjustified Unary Properties**: Removed unnecessary shape constraints from Kitchen source roles (`water_source`, `coffee_source`), Kitchen soup spoons (`open cavity`), Living Room composite drinkware sets (`open cavity`), and Workshop screws (`slotted head`).
3. **Hardened Anti-Leak Assertions**: Verified all relation-bearing fields (`functional_relations[].relation`, `interaction_groups[].required_relations[]`, `interaction_groups[].context_relations[]`), all raw IDs, and all serialized text against canonical predicates, simulator backend handles, and benchmark oracle region/object tokens.
4. **Evidence-Derived Preservation Tracing**: Replaced automatic `PRESERVED` labels with granular trace inspections across `raw_vlm_role_id`, `raw_vlm_group_id`, `normalized_roles`, `normalized_relations`, `normalized_operation_groups`, and individual Living Room mapping functions.
5. **Future-Proof Living Room Test Adapter**: Equipped `MockFMAdapter` with complete mock state (`last_raw_response`, `last_raw_requirement_response`, `last_raw_inspection_response`, `metrics`, `last_observation_images`), ensuring tests remain green when P3-F implements Living Room relation repair.
6. **Full GT Reference Metrics & Structural Disclosure**: Evaluated canonicalized graphs with `evaluate_gf_against_reference()`, disclosing that Workshop candidate G_F declares a valid interaction group not present in the static GT reference (`extra_operation_groups: ['group_1']`, `exact_structural_match: False`).

**Acceptance**: All 3 fixtures are schema-compliant and neutralized, anti-leak checks pass comprehensively, preservation traces are evidence-derived, and full reference metrics are recorded without altering production canonicalizers.

---

### P3-C.2: Concept-Exact Diagnostic Closure
**Status**: `[x] COMPLETE`

**Objective**: Establish concept-exact diagnostic accounting across all three domains: verify exact predicate triples on Kitchen and Workshop relations using unique deterministic keys; derive Kitchen and Living Room property mappings evidence-based from production mapper functions; contextualize Living Room relation mapping with canonical subject/object arguments; resolve Workshop regions 1-to-1; preserve complete evaluator diagnostic structures; and correct documentation regarding Kitchen cardinality interval metrics and Workshop empty-reference recall.

**Accomplished Changes & Empirical Findings**:
1. **Concept-Exact Relation Keys**: Replaced endpoint-only dictionary keys with unique deterministic keys (`rel:{idx}:{s}:{r}:{o}`), preventing overwriting of multiple distinct relations sharing the same subject/object endpoints.
2. **Predicate-Specific Relation Verification**: Verified that mapped predicate triples `(s_canon, mapped_pred, o_canon)` explicitly exist in $G_F$ for Kitchen (via `map_binary_relation`) and Workshop (via `provider.normalized_relations`).
3. **Contextual Living Room Sub-Diagnostic**: Aligned the test-side Living Room relation mapping sub-check with production by passing mapped `subject_role` and `object_role` to `map_living_room_relation`.
4. **Evidence-Derived Property Traces**: Evaluated Kitchen properties with `map_unary_property` (classifying subsequent identical predicates on the same role as `MERGED_BY_EXPLICIT_RULE`) and Living Room properties with `_map_properties` (`PLANAR_SUPPORT`), while explicitly recording 0 unary property cases for Workshop.
5. **1-to-1 Workshop Region Resolution**: Resolved each raw proposal (`search_1`, `search_2`, `search_3`) individually to `LEFT_DRAWER`, `RIGHT_DRAWER`, and `TOOL_CABINET` via `resolve_workshop_region_proposal`.
6. **Corrected Evaluator Explanations**: Documented that Kitchen `role_exact_recall = 0.833` and `exact_structural_match = False` arise strictly from `role_cardinality_diagnostics['coffee_stirrer']` (`candidate_range: [1, 1]` vs `reference_range: [1, 2]`), and that Workshop `operation_group_identity_recall = 1.0` is a vacuous metric over an empty reference.

**Acceptance**: Concept-exact preservation keys, predicate verifications, property traces, 1-to-1 region resolutions, and accurate metric explanations are validated with all Phase-3 tests passing and zero changes to production canonicalizers.

---

### P3-D: Predicate & System-Context Freeze
**Status**: `[x] COMPLETE`

**Objective**: Freeze the predicate signature registry (§4) in code and establish the formal contract distinguishing task-functional G_F roles from system-owned scene context.

**Scope**:
1. Created `mujoco_scenes/functional_tamp_pipeline/predicate_registry.py` defining immutable domain-scoped predicate signatures, arities, role families, directional constraints, checker ownership, and active/legacy status.
2. Created `mujoco_scenes/functional_tamp_pipeline/system_context_registry.py` formalizing the four system context categories: `SELECTABLE_FUNCTIONAL_ASSET`, `SYSTEM_FIXED_FUNCTIONAL_ANCHOR`, `PLANNER_CONTEXT_CONSTANT`, and `SEARCH_REGION`.
3. Resolved Workshop `CAN_DRIVE_SCREW` / `CAN_FASTEN` as legacy observation-diagnostic capability markers and normalized GTSpecProvider Workshop G_F to `unary_predicates=()`, establishing unified GT/VLM canonical predicate contract.
4. Updated `task_interface_validator.py` (`validate_runtime_gf`) to strictly enforce frozen predicate signatures and directions across unary predicates, binary relations, and operation groups.
5. Hardened `audit_plan_grounding` by eliminating cross-domain global allowlists and generic Kitchen fallback.
6. Corrected Living Room planner-projection documentation comment in `domains/living_room.py`.
7. Re-verified K1, L1, W1 GT downstream controls to `ACTION_SEQUENCE_READY` with zero audit violations.

**Tests**: `test_predicate_and_context_registry.py`, `test_executable_grounding_ir.py`, `test_ideal_fixtures.py`, `test_live_visualizer.py`.

**Acceptance**: Predicate registry and system context registries are frozen code artifacts. All 248 Phase-3 tests pass.

**Do not touch**: `ground_graph()` internals.

---

### P3-D.1: Context Authority & Registry Closure
**Status**: `[x] COMPLETE`

**Objective**: Close remaining context authority bypasses, enforce read-only registry immutability, make domain role-ownership binding in runtime validation, and source-audit Workshop emittable predicates.

**Scope**:
1. **Removed `home_region` unconditional audit bypass**: Removed `if arg == home_region: continue` in `audit_plan_grounding`. All plan arguments (including home regions) must be valid under `is_valid_planner_argument`.
2. **Closed `allowed_context_ids` OBJECT bypass**: On standard domains, `allowed_context_ids` cannot authorize unassigned `OBJECT` nodes. Only registered planner constants or actual $G_O$ `REGION`/`FIXED_TARGET` nodes are authorized.
3. **Removed false Workshop constant `work_surface`**: Verified from source that `work_surface` was only a compiler dictionary key. Workshop planner constants frozen as `MAIN_WORKBENCH_ZONE` and `workshop_frame_joint`.
4. **Enforced strict role ownership in `validate_runtime_gf`**:
   - `SELECTABLE_FUNCTIONAL_ASSET`: allowed as ordinary functional $G_F$ roles.
   - `SYSTEM_FIXED_FUNCTIONAL_ANCHOR`: must have `entity_kind == 'FIXED_TARGET'`.
   - `PLANNER_CONTEXT_CONSTANT`: forbidden from appearing as $G_F$ roles.
   - `SEARCH_REGION`: forbidden from appearing as selectable $G_F$ roles.
   - Unauthorized roles (e.g. `workbench_surface` in Workshop) fail closed.
5. **Enforced search region contract**: Candidate regions must belong to domain search ontology and `set(region_ranking) == set(candidate_regions)`.
6. **Classified Workshop unsupported emittables**: `LOCATED_ON`, `PLANAR_SUPPORT`, `OPEN_CAVITY`, `ELONGATED_OBJECT` classified as `CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED` in `predicate_registry.py` and rejected by runtime validation; production disposition carried to P3-G.
7. **Read-only registry immutability**: Wrapped exported mappings in `types.MappingProxyType` to guarantee runtime immutability.
8. **Self-contained tests**: 15 comprehensive unit/regression tests committed in `test_predicate_and_context_registry.py`.

**Tests**: `test_predicate_and_context_registry.py`, `test_executable_grounding_ir.py`, `test_phase3_vlm_replay_validator.py`, full suite (255 passed).

---

### P3-E / P3-E.1 / P3-E.2: Kitchen Canonicalizer Repair, Lexical Precision & Role Semantic Authority
**Status**: `[x] COMPLETE (FROZEN)`

**Objective**: Repair the Kitchen VLM canonicalizer to achieve 100% concept preservation on the ideal fixture without silent fallbacks, enforce lexical precision, validate group function semantics, isolate role semantic authority from candidate categories, and ensure provenance closure.

**Accomplished Changes & Empirical Findings**:
1. **Eliminated Silent Role Dropping**: Roles with unmapped functions or candidate categories raise `UnmappedFunctionalConceptError` rather than silently continuing.
2. **Eliminated Canonical Role Collision Heuristics**: Colliding raw roles mapping to the same canonical role fail closed with `AmbiguousCanonicalizationError` (zero `max()`/`sum()` heuristics).
3. **Fail-Closed Binary and Unary Property Mapping**: Removed `INSERTABLE_IN` default fallback in `map_binary_relation()` and broad keyword substring heuristics. Unmapped relations and required properties raise `UnmappedFunctionalConceptError`.
4. **Eliminated Reverse Short-Fragment Containment (P3-E.1)**: Enforced forward-only phrase/exact matching for binary relations, unary properties, and roles (`a_norm == norm or _contains_phrase(norm, a_norm)`). Reverse fragments (`"fit"`, `"inside"`, `"bottom"`, `"reach"`, `"open"`, `"shape"`) strictly return `None`.
5. **Interaction Group Function Semantic Validation (P3-E.1)**: Validates raw interaction group function phrases against `KITCHEN_INTERACTION_GROUP_ALIASES` and verifies semantic consistency with tool/target endpoint pairs. Contradictory/unmapped group functions fail closed.
6. **Role Semantic Authority Decoupled from Candidate Categories (P3-E.2)**: `map_kitchen_role_function()` determines role identity strictly and exclusively from `function` + `description`. Nonsense functions (e.g. `"hammer a nail"`) fail closed with `UnmappedFunctionalConceptError` regardless of candidate categories; contradictory categories (e.g. `["bowl"]` on coffee stirrer) do not distort the role identity.
7. **Strict Operation Group Accounting & Cardinality**: Validates tool/target pairs against declared roles, rejects duplicate group collisions with `AmbiguousCanonicalizationError`, and rejects unmapped/empty operation relations. Removed target count clipping (`min(req, target_count)`).
8. **Removed Dead Self-Repair Fallbacks (P3-E.1)**: Consumes schema-required fields directly without fallback defaults.
9. **Concept Accounting Trace & Provenance (P3-E.2)**: Generates comprehensive `concept_accounting` metadata recording factual counts on ideal K1 (6/6 raw roles with `role_semantic_source: "FUNCTION_AND_DESCRIPTION"`, 7/7 raw property phrases with 4 PRESERVED and 3 MERGED_BY_EXPLICIT_RULE, 4/4 relations, and 2/2 operation groups).
10. **Provenance & Version Isolation (P3-E.1)**: Bumped Kitchen canonicalization version to `phase3_p3e_1_v1` across compiler and `VLMSpecProvider._kitchen`, keeping other domains isolated.
11. **Ideal Kitchen K1 Fixture Evaluation**: Yields 100% role recall (6/6), 100% role precision (6/6), 100% relation recall (4/4), 100% relation precision (4/4), and `reference_complete = True` against GT reference.
12. **Regression & Adversarial Tests**: Full unit/negative regression suite in `test_kitchen_vlm_functional_graph.py` (36/36 passed; full Phase-3 suite: 291/291 passed).
13. **Documentation**: Full evidence documented in `docs/PHASE3_P3E_KITCHEN_CANONICALIZER.md`.

**Acceptance**: All P3-E, P3-E.1, and P3-E.2 acceptance criteria verified. P3-E / P3-E.1 / P3-E.2 FROZEN.

---

### P3-F / P3-F.1: Living Room Canonicalizer Repair & Fail-Closed Authority Closure
**Status**: `[x] COMPLETE (FROZEN)`

**Objective**: Repair the Living Room VLM canonicalizer to correctly handle composite role cardinalities, relation directions, unary property validation, operation group distribution, concept accounting traces, and enforce strict fail-closed authority closure (no endpoint fabrication, no synthetic fixed targets, disjoint slot composition rules, property fail-closed enforcement).

**Accomplished Changes & Empirical Findings**:
1. **Forward-Only Phrase Matching & Role Semantic Authority**: Implemented `map_living_room_role_function`, `map_living_room_object_payload_role`, and `map_living_room_fixed_target_role` with forward-only phrase matching and strict role semantic authority based exclusively on `function` and `description` (excluding candidate categories).
2. **Composite Role Semantics & Cardinality Accounting**: Implemented aggregate composite bundle handling ($N \to N$, status `PRESERVED`) and component decomposition handling (cup $N$ + saucer $N \to N$, status `COMPOSED_FROM_COMPONENT_ROLES`), with fail-closed rejection of mismatched counts ($N \neq M$), ambiguous multiple components, and conflicting binding policies.
3. **Relation Canonicalization & Rejection of Endpoint Fabrication (P3-F.1)**: Implemented `canonicalize_living_room_relation` mapping relations to the 4 canonical Living Room signatures with direction normalization for passive/reverse phrasing (`direction_status: "NORMALIZED_TO_CANONICAL_SIGNATURE"`). Self-relations on regions are strictly rejected with `MalformedVLMSpecificationError` (zero endpoint fabrication).
4. **Removal of Fixed-Target Synthesis (P3-F.1)**: Deleted synthetic fixed-target compilation fallback. If relations or groups reference fixed anchors without explicit raw declarations, the compiler fails closed with `MalformedVLMSpecificationError`. Explicit raw fixed targets are accurately labeled `PRESERVED`.
5. **Disjoint Slot Composition (P3-F.1)**: Implemented `_extract_disjoint_slot_identity()` for `PERSONAL_CUP_SAUCER_REGION` and `SEATING_POSITION`. Multi-role synthesis requires explicit disjoint slot identity (e.g. viewer 1 / viewer 2 or left / right), count=1, and `DISTINCT` binding policy. Generic duplicates fail closed with `AmbiguousCanonicalizationError`.
6. **Unary Property Fail-Closed Enforcement (P3-F.1)**: Support REGION roles omitting `PLANAR_SUPPORT` fail closed with `MalformedVLMSpecificationError` (zero default dictionary injection). Duplicate synonymous planar properties merge with `MERGED_BY_EXPLICIT_RULE`. Unsupported properties (`OPEN_CAVITY`, `ELONGATED_OBJECT`) raise `UnsupportedCheckerCapabilityError`.
7. **Schema-Required Field Validation (P3-F.1)**: Validates all schema-required fields before compilation; missing fields fail closed immediately without silent fallback defaults.
8. **Structural Operation Group Distribution**: Canonical operation group `personal_support_group` encapsulates `FITS_SET_ON` and `NEAR_SEAT`, leaving top-level `gf.relations` containing only un-grouped relations (`SHARED_REMOTE_REGION FITS_ON REMOTE` and `SHARED_REMOTE_REGION ACCESSIBLE_FROM_BOTH_SEATS SEATING_PAIR`), achieving 100% precision/recall against GT reference graph.
9. **Concept Accounting Trace & Version Bump**: Added full concept accounting trace across roles, properties, relations, and operation groups; version set to `phase3_p3f_v1`.
10. **Comprehensive Testing & Downstream Verification**: Ideal fixture `living_room_L1` canonicalizes with 100% role/relation/group recall and precision and `reference_complete = True`. Full 282-test suite passes (100%). Downstream evaluation on Living Room L1 GT yields `ACTION_SEQUENCE_READY`, 10 action steps, 0 audit violations, and independent replay `VALID`.
11. **Documentation**: Full architecture and empirical results documented in `docs/PHASE3_P3F_LIVING_CANONICALIZER.md`.

**Acceptance**: All P3-F and P3-F.1 acceptance criteria verified. P3-F / P3-F.1 FROZEN.

---

### P3-G: Workshop Canonicalizer Repair
**Status**: `[ ] NOT STARTED`

**Objective**: Repair the Workshop VLM canonicalizer to handle alternative role proposals, unmapped capability phrases, and operation group validation.

**Scope of Hypotheses and Tests** (in `workshop_phase1/requirements.py` and `vlm_spec_provider.py`):
1. **Duplicate / alternative role resolution**: Classify duplicate canonical role emissions:
   - Equivalent alternatives (e.g. manual screwdriver vs cordless drill) → merge with explicit rule.
   - Genuinely distinct requirements → preserve separately.
   - Ambiguous proposals → reject explicitly.
2. **Capability vocabulary**: Expand ontology phrase mappings for fastener driving and holding capabilities.
3. **Operation group validation**: Validate group cardinalities and context relations consistently against declared roles.
4. **P3-D.1 Canonicalizer Obligations**:
   - Resolve `workbench_surface` role emission (must not be emitted as a selectable G_F role; belongs to system/planner context).
   - Resolve `LOCATED_ON` relation emission (remove or convert to system context; reject in canonical G_F).
   - Resolve generic unary properties (`PLANAR_SUPPORT`, `OPEN_CAVITY`, `ELONGATED_OBJECT`) emitted by `map_workshop_unary_property` (must not pollute canonical Workshop G_F).

**Tests**: `test_ideal_fixtures.py` must pass for `workshop_W1`. Run `pytest mujoco_scenes/tests/test_workshop_vlm_requirements.py`.

**Acceptance**: Ideal workshop fixture → canonical G_F matching GT structure with zero unauthorized roles or predicates.

---

### P3-H: Region Resolution & Search Contract
**Status**: `[ ] NOT STARTED`

**Objective**: Establish a strict region proposal resolution contract across all domains without leaking unpermitted variant information.

**Scope**: Kitchen region resolution (D1-C1), Workshop region resolution (LEFT_DRAWER, RIGHT_DRAWER, TOOL_CABINET).

**Key Rules**:
1. Clarify what candidate regions are observable/discoverable per domain.
2. Ensure natural language region aliases resolve deterministically.
3. If VLM proposes invalid/empty search regions, handle via explicit domain fallback policy without accessing hidden variant state.

**Tests**: Extend `test_vlm_interface_boundary.py` with region resolution edge cases.

**Acceptance**: All valid region proposals resolve deterministically.

---

### P3-I: K1/L1/W1 Full Ideal-Fixture Convergence
**Status**: `[ ] NOT STARTED`

**Objective**: End-to-end validation that ideal raw fixtures → canonicalizer → G_F → G_O → φ* → action sequence for K1/L1/W1.

**Scope**: Integration test across all 3 domains.

**Inputs**: Ideal fixtures from P3-C, canonicalizers repaired in P3-E/F/G.

**Tests**: Run each ideal fixture through the full pipeline test harness.

**Acceptance**: All three → `ACTION_SEQUENCE_READY`. Proves the software interface is fully functional.

**Do not touch**: VLM prompt, FM adapter, model configuration.

---

### P3-J: Live 9B Causal Diagnosis
**Status**: `[ ] NOT STARTED`

**Objective**: Run Qwen3.5-9B once on K1/L1/W1 and perform causal layer diagnosis using the decision tree (§5).

**Scope**: Live VLM call + raw response analysis. No code changes.

**Protocol**:
1. Save raw VLM response BEFORE canonicalization.
2. Audit raw concepts: CORRECT / MISSING / EXTRA / WRONG_CARDINALITY / WRONG_BINDING_POLICY / WRONG_RELATION / WRONG_ENTITY_KIND.
3. Attempt canonicalization and record any gaps.
4. If canonicalization succeeds, run downstream and identify failure layer (if any).

**Artifacts**: Save per-case `{raw_response.json, concept_audit.json, canonical_gf.json, layer_diagnosis.txt}`.

**Acceptance**: Clear per-case layer diagnosis for K1/L1/W1.

---

### P3-K: 9B vs 27B Raw Specification Comparison
**Status**: `[ ] NOT STARTED`

**Objective**: If P3-J reveals model capacity limitations, compare raw specification quality between Qwen3.5-9B and Qwen3.5-27B on K1/L1/W1.

**Metrics** (evaluated at RAW specification level):
- Schema-valid rate
- Role identity recall / precision
- Role cardinality accuracy
- Binding policy accuracy
- Relation recall / precision
- Operation group recall
- Region proposal correctness
- Malformed specification rate

**Decision Rule**: If 9B raw is correct but canonicalizer breaks → fix canonicalizer (software issue). If 9B raw is fundamentally missing required concepts and 27B provides them → adopt 27B.

**Acceptance**: Documented comparison with decision on model selection.

---

### P3-L: Full 32-Variant Deterministic Fixture Matrix
**Status**: `[ ] NOT STARTED`

**Objective**: Validate all 32 variants using reviewed deterministic fixtures.

**Gate**:
- 20/20 feasible → `ACTION_SEQUENCE_READY`
- 12/12 infeasible → `INFEASIBLE` with correct causal deficiency

**Acceptance**: 32/32 correct terminal outcomes from deterministic fixtures.

---

### P3-M: Clean Live VLM Benchmark
**Status**: `[ ] NOT STARTED`

**Objective**: Run the complete 32-variant matrix against the selected live model on a single frozen commit with hardened provenance and accounting.

**Protocol**:
1. Single frozen git commit; clean worktree.
2. Exact model identifier, prompt version, canonicalizer version recorded.
3. Exact configuration fingerprint checked for every case.
4. Raw VLM response retained for every case (including `VLM_SPEC_FAILED`).
5. Full artifact chain retained per case.
6. Distinct accounting for invocation wall time vs summed case runtime.

**Artifacts**: `tmp/p3m_clean_benchmark_YYYYMMDD_HHMMSS/` with `summary.json`, `results.csv`, per-case directories.

**Acceptance**: Results are scientifically reportable with layer diagnoses for all failures.

---

### P3-N: Phase-3 Freeze & Phase-4 Handoff
**Status**: `[ ] NOT STARTED`

**Objective**: Freeze Phase 3 and produce the Phase 4 interface contract.

**Acceptance**: All items in §10 freeze checklist are TRUE.

---

## 7. Test Matrix

### Smoke Gate (K1/L1/W1)
| Stage | K1 | L1 | W1 |
|---|---|---|---|
| GT downstream control | P3-B | P3-B | P3-B |
| Ideal fixture compiles | P3-C | P3-C | P3-C |
| Ideal fixture end-to-end | P3-I | P3-I | P3-I |
| Live 9B diagnosis | P3-J | P3-J | P3-J |

### Domain Regression Gates
| Domain | Pass | Gate |
|---|---|---|
| Kitchen K1-K12 | P3-L | 12/12 correct (6 feasible, 6 infeasible) |
| Living Room L1-L10 | P3-L | 10/10 correct (6 feasible, 4 infeasible) |
| Workshop W1-W10 | P3-L | 10/10 correct (8 feasible, 2 infeasible) |

### Final Gates
| Gate | Pass | Criterion |
|---|---|---|
| Deterministic fixture matrix | P3-L | 32/32 correct terminal outcomes |
| Live VLM matrix | P3-M | Documented with layer-separated metrics |
| Phase-3 freeze | P3-N | All checklist items TRUE |

---

## 8. Artifact Contract

Per pipeline run, retain:

| Artifact | Purpose | Required |
|---|---|---|
| `raw_vlm_response.json` | Original VLM output before any processing | YES (VLM mode) |
| `validated_vlm_specification.json` | Schema-validated VLM output | YES (VLM mode) |
| `canonical_gf.json` | Canonical G_F after canonicalization | YES |
| `observed_scene_graph.json` | G_O state at grounding time | YES |
| `graph_grounding_result.json` | φ* or failure diagnosis | YES |
| `canonical_grounding_witness.json` | Witness payload for compiler | YES (if COMPLETE) |
| `action_plan.json` | Final action sequence | YES (if ACTION_SEQUENCE_READY) |
| `plan_grounding_audit.json` | Audit that plan respects φ* | YES (if plan exists) |
| `canonicalization_trace.json` | Mapping from raw to canonical concepts | YES (VLM mode) |
| `layer_diagnosis.txt` | Which layer caused failure | YES (if failed) |
| `prompt_leakage_audit.json` | Deterministic prompt/payload leak verification | YES (VLM mode) |
| `case_manifest.json` | Includes immutable provenance fingerprint | YES |

---

## 9. Model Evaluation Protocol

1. **Repair software first**: Do not switch models to work around canonicalizer bugs.
2. **Baseline on 9B**: Qwen3.5-9B is the current model. All software validation uses 9B.
3. **Evaluate raw quality**: Compare 9B raw specification quality before and after canonicalizer repair.
4. **Conditional 27B**: Only if P3-J shows that 9B raw responses are fundamentally missing required concepts, run 27B on K1/L1/W1 for comparison.
5. **Compare at RAW level**: The 9B vs 27B comparison is made at the raw specification level, not terminal task success.
6. **Decision rule**: If 9B becomes nearly as good as 27B after compiler repair, retain 9B (scientifically stronger result). If 27B consistently provides concepts that 9B cannot, adopt 27B.

---

## 10. Phase-3 Freeze Checklist

```
[x] Architecture: one canonical G_F representation shared by GT and VLM
[x] Architecture: one canonical G_O representation
[x] Architecture: ground_graph is sole assignment authority
[x] Architecture: φ* is immutable downstream
[x] Architecture: no hidden FM call after specification
[x] Architecture: no GT oracle in VLM runtime path
[x] Architecture: no semantic role reassignment in compiler
[x] Software: VLMSpecProvider.provide() correctly validates and wraps errors
[x] Infrastructure: Benchmark artifacts have explicit reproducibility fingerprints and stale resume protection
[x] Infrastructure: Prompt leakage audit is deterministically verified over payloads
[x] Software: K1, L1, W1 GT controls verified to ACTION_SEQUENCE_READY with 0 audit violations
[ ] Software: all ideal raw fixtures compile to correct G_F
[ ] Software: 20/20 feasible fixtures → ACTION_SEQUENCE_READY
[ ] Software: 12/12 infeasible fixtures → correct causal INFEASIBLE
[ ] Software: deterministic replays match exactly
[ ] Evaluation: live VLM comparison completed
[ ] Evaluation: per-case layer diagnosis documented
[ ] Evaluation: 9B vs 27B comparison documented (if warranted)
[ ] Provenance: frozen commit, clean worktree
[ ] Provenance: every phase transition has retained artifacts
[ ] Handoff: PipelineResult schema stable for Phase 4
[ ] Handoff: Phase 4 consumes plan without knowing GT vs VLM origin
```

---

## 11. CURRENT NEXT PASS
 
### **CURRENT NEXT PASS: P3-F**
 
**Exact Objective**: Repair the Living Room VLM canonicalizer to achieve 100% concept preservation on the ideal fixture (L1) without silent fallbacks, correctly handling composite role cardinalities, relation directions, and system context integration.
 
**Prerequisites**: P3-A, P3-B, P3-B.1, P3-C, P3-C.1, P3-C.2, P3-D, P3-D.1, P3-E, P3-E.1, and P3-E.2 are complete and frozen.
 
**What to do**:
1. Establish negative regression tests for unmapped Living Room relations, ambiguous role mappings, and composite role counting.
2. Repair `map_living_room_relation()` in `environment_vlm_requirements.py` to correctly map valid spatial phrasing without fabricating unmapped predicates.
3. Formulate strict composite role cardinality handling for multi-part items (`CUP_SAUCER_SET`).
4. Validate relation directions and context targets against frozen predicate signatures.
 
**Acceptance Criteria**:
- `test_ideal_fixtures.py` passes for `living_room_L1` with `role_recall = 1.0`, `relation_recall = 1.0`.
- All Phase-3 unit and regression tests pass.
 
**Expected next pass**: P3-G (Workshop Canonicalizer Repair)
