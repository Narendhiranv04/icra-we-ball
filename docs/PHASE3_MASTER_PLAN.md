# Phase 3 Master Plan

> **Version**: 1.0
> **Created**: 2026-08-31
> **Branch**: `naren/pipeline_check`
> **HEAD at creation**: `3df35c34` (Pass 3.6B.2)

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
| Kitchen domain adapter (symbolic compiler) | `domains/kitchen.py` — `KitchenPlanningCompiler`, `build_kitchen_observed_scene_graph` |
| Living Room domain adapter | `domains/living_room.py` — `compile_living_room_task_from_graph`, `build_living_room_observed_scene_graph` |
| Workshop domain adapter | `domains/workshop.py` — `build_workshop_observed_scene_graph` |
| Search loop | `search.py` — `search_until_satisfied()` protocol |
| A* planner | `planning.py` + `symbolic_planning_core.py` |
| Reference evaluator | `gf_reference_evaluator.py` — structural comparison metrics |
| Runtime isolation (no GT imports in VLM path) | `test_executable_grounding_ir.py::test_static_runtime_isolation` |
| VLM prompt leak protection | `test_vlm_interface_boundary.py::test_zero_leakage_in_kitchen_and_workshop_payloads` |
| FM adapter retry/reconnect | `fm_adapter.py` — 3-attempt retry with socket error handling |
| 223 unit tests pass | `pytest mujoco_scenes/functional_tamp_pipeline/tests/` |

### PARTIALLY VERIFIED ~
| Component | Status |
|---|---|
| Kitchen VLM canonicalizer | Compiles valid G_F for some inputs; role recall ~67%, missing `water_source`/`coffee_source` for many VLM outputs; unmapped relations default to `INSERTABLE_IN`; operation groups only accepted for 2 hard-coded tool↔target pairs |
| Living Room VLM canonicalizer | Entity-kind hard gate works; but CUP_SAUCER_SET count merging untested with diverse VLM outputs; FIXED_TARGET context nodes (SEATING_POSITION, SEATING_PAIR) may not be emitted by VLM |
| Workshop VLM canonicalizer | Role normalization exists; but 10/10 live cases → VLM_SPEC_FAILED (all Qwen outputs had structural problems or duplicate role collisions) |
| Region resolution (Kitchen) | Maps natural language to D1/D2/C2/B1/C1 via fuzzy matching; but VLM-emitted local IDs not always resolvable |

### BROKEN ✗
| Component | Evidence |
|---|---|
| `VLMSpecProvider.provide()` method body | Lines 34-36: `try: graph.validate(); from .task_interface_validator import validate_runtime_gf` — **truncated**, no except block, no return. The `_workshop` def at line 37 overwrites the incomplete try block. This is a **syntax-level bug** that means the validation/error-wrapping in `provide()` is dead code. |
| VLM → ACTION_SEQUENCE_READY | 0/20 feasible variants succeeded in Pass 3.6B.2 |
| VLM → correct INFEASIBLE | Only 4/12 infeasible variants matched (K7, K9, L8, L10) — but these were accidental via VLM_SPEC_FAILED |
| Kitchen: canonicalizer drops `water_source` and `coffee_source` | Role recall 0.50-0.67; these roles consistently present in GT but missing from VLM G_F |
| Kitchen: unmapped binary relation defaults to `INSERTABLE_IN` | `map_binary_relation()` line 232-233 and line 394 — any unrecognized relation text silently becomes INSERTABLE_IN |
| Kitchen: operation groups hard-gated to 2 patterns | Lines 420-425: only `coffee_stirrer→coffee_container` and `soup_eating_utensil→soup_container` are accepted; all others silently dropped |
| Kitchen: `required_target_count` clipped to role count | Line 460: `min(req_target_count, int(roles[target_role]["count"]))` |
| Living Room: system-owned context nodes not auto-injected | `SEATING_POSITION` and `SEATING_PAIR` must come from VLM; if VLM omits them → validation failure |
| Workshop: all live VLM cases VLM_SPEC_FAILED | Qwen emits `required_target_count: 2` with `fastener max_count: 1`, or non-compliant category types |

### NOT YET TESTED
| Component | Note |
|---|---|
| Ideal raw VLM fixture framework | No manually-crafted "perfect VLM response" fixtures exist |
| Canonicalization preservation metrics | No quantitative measurement of what raw concepts survive canonicalization |
| 9B vs 27B raw specification comparison | Not performed |
| Clean benchmark with single frozen commit | Pass 3.6B.2 mixed artifacts across commits |

---

## 2. Current Failure Map

### Kitchen
| Layer | Failure | Severity |
|---|---|---|
| **Canonicalizer** | `water_source` and `coffee_source` roles silently dropped when VLM emits them with generic function text | HIGH — causes INCOMPLETE/INFEASIBLE even when VLM understood task |
| **Canonicalizer** | Unmapped binary relations default to `INSERTABLE_IN` (line 394, line 232) instead of failing closed | MEDIUM — masks wrong relations |
| **Canonicalizer** | Operation groups hard-gated to exactly 2 canonical patterns (lines 420-425) | MEDIUM — any VLM variation in tool↔target naming gets dropped |
| **Canonicalizer** | `required_target_count` silently clipped (line 460) | LOW — but violates semantic preservation |
| **G_F↔G_O** | If VLM omits `coffee_source`/`water_source`, symbolic planner can't find contents → compile exception → caught as `INFEASIBLE` | HIGH |
| **Downstream** | `KitchenPlanningCompiler` builds STRIPS from `compile_observed_symbolic_state()` which hardcodes `symbolic_task` defaults (coffee, water, soup contents) — legitimate fixed domain knowledge but boundary not explicit | LOW |

### Living Room
| Layer | Failure | Severity |
|---|---|---|
| **Canonicalizer** | 2 cups + 2 saucers can merge into `total_count=4` for `CUP_SAUCER_SET` if VLM emits separate cup and saucer roles that both map to `CUP_SAUCER_SET` | HIGH |
| **Canonicalizer** | `SEATING_POSITION` and `SEATING_PAIR` context nodes must be VLM-emitted; if missing, operation groups referencing them fail validation | HIGH |
| **G_F↔G_O** | G_O stores relations as `REGION --FITS_SET_ON--> CUP_SAUCER_SET` (subject=region), while VLM may describe `cup placed_on table` (subject=cup). Canonicalizer disambiguates via role type but fragile for novel phrasings | MEDIUM |
| **VLM raw** | 5/10 variants → VLM_SPEC_FAILED (L4-L7, L9) — needs diagnosis | UNKNOWN |

### Workshop
| Layer | Failure | Severity |
|---|---|---|
| **VLM raw** | Qwen emits `required_target_count: 2` for operation groups while `fastener.max_count = 1` → graph validation rejects | HIGH — 10/10 VLM_SPEC_FAILED |
| **Canonicalizer** | Duplicate raw role descriptions for "driver" functionality can collide during normalization | MEDIUM |
| **System context** | `repair_target` and `workbench_surface` are deterministic scene fixtures but VLM must currently emit them | MEDIUM |

### Infrastructure
| Layer | Failure | Severity |
|---|---|---|
| **`VLMSpecProvider.provide()`** | Incomplete try block (lines 34-37) — `validate()` and `validate_runtime_gf()` calls are dead code | HIGH — error wrapping in `MalformedVLMSpecificationError` does not function |
| **Benchmark provenance** | Pass 3.6B.2 resumed old cases from different commits; `total_runtime_seconds: 0.45` is wrapper overhead, not execution time | MEDIUM |

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

### P3-A: Fix VLMSpecProvider & Benchmark Infrastructure
**Status**: `[ ] NOT STARTED`

**Objective**: Fix the broken `VLMSpecProvider.provide()` method and establish clean benchmark infrastructure.

**Scope**: `vlm_spec_provider.py`, `scripts/run_phase36b2_matrix.py`

**Required Changes**:
1. Fix `VLMSpecProvider.provide()` — complete the try/except block at lines 34-37 so that `validate()` and `validate_runtime_gf()` are called, and validation errors are wrapped in `MalformedVLMSpecificationError`.
2. Verify all three `_kitchen()`, `_living_room()`, `_workshop()` static methods have `@staticmethod` decorators consistently.
3. Add provenance fingerprint (git commit, canonicalizer version, model, prompt hash) to benchmark output.

**Tests**: `pytest mujoco_scenes/functional_tamp_pipeline/tests/ -v`

**Acceptance**: `VLMSpecProvider.provide()` correctly validates and wraps errors for all three domains. All existing tests pass.

**Do not touch**: `ground_graph()`, domain adapters, GT path.

---

### P3-B: K1/L1/W1 GT Downstream Controls
**Status**: `[ ] NOT STARTED`

**Objective**: Verify that GT G_F → actual G_O → ground_graph → compiler → A* works for K1, L1, W1 individually with full artifact preservation.

**Scope**: Run only. No code changes expected unless a downstream bug is found.

**Inputs**: GT spec provider, actual scene files for K1 (F0_ALL_VISIBLE), L1 (F0_ALL_OBJECTS_IN_STAGING), W1 (F0_MANUAL_FIRST_ONE_REGION).

**Tests**:
```bash
PYTHONPATH=. python -m mujoco_scenes.functional_tamp_pipeline.run --domain kitchen --variant K1 --mode gt
PYTHONPATH=. python -m mujoco_scenes.functional_tamp_pipeline.run --domain living_room --variant L1 --mode gt
PYTHONPATH=. python -m mujoco_scenes.functional_tamp_pipeline.run --domain workshop --variant W1 --mode gt
```

**Artifacts**: Save G_F, G_O, φ*, action plan, and grounding audit per case to `tmp/p3b_gt_controls/`.

**Acceptance**: All three → `ACTION_SEQUENCE_READY` with valid plans.

**Do not touch**: VLM canonicalizers.

---

### P3-C: Ideal Raw VLM Fixture Framework
**Status**: `[ ] NOT STARTED`

**Objective**: Create one manually-crafted "perfect VLM response" fixture per domain that uses natural semantic language (not internal identifiers) and is schema-valid.

**Scope**: Create `tests/fixtures/ideal_raw_vlm/` with `kitchen_K1.json`, `living_room_L1.json`, `workshop_W1.json`.

**Key constraints**:
- Must conform to `RESPONSE_SCHEMA` / `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA`
- Must use semantic language: "stir coffee", "contain soup", "hold liquid", "placed on personal table near seat"
- Must NOT use internal identifiers: no `INSERTABLE_IN`, `OPEN_CAVITY`, `D1`, `C2` in the fixture
- Must include all task-required concepts (all 6 Kitchen roles, Living Room regions + payloads + context, Workshop driver + fastener + targets)

**Tests**: Write `test_ideal_fixtures.py` that feeds each fixture through the real canonicalizer and verifies:
1. Schema validation passes
2. Canonicalization produces a valid G_F
3. `validate_runtime_gf()` passes
4. All expected canonical roles are present (preservation check)

**Acceptance**: All 3 fixtures canonicalize without error and produce G_F with 100% role recall vs GT reference.

**Do not touch**: Canonicalizer code — this pass is diagnostic only. If fixtures fail, record the failure for P3-D/E/F/G.

---

### P3-D: Predicate & System-Context Freeze
**Status**: `[ ] NOT STARTED`

**Objective**: Freeze the predicate signature registry (§4) in code and resolve the system-owned context node question.

**Scope**: Create `functional_tamp_pipeline/predicate_registry.py` with the canonical predicate table. Decide and implement which nodes are system-owned:

**Likely system-owned** (deterministic scene fixtures):
- `SEATING_POSITION` — always present in living room G_O
- `SEATING_PAIR` — always present in living room G_O
- `repair_target` — always present in workshop scene
- `workbench_surface` — always present in workshop scene

**Design decision**: System-owned context nodes should be injected by the canonicalizer during G_F construction if the VLM does not emit them, with explicit `source: "SYSTEM_OWNED_DOMAIN_CONTEXT"` provenance. The VLM should NOT be penalized for omitting them.

**Tests**: Add assertions to `test_executable_grounding_ir.py` that system-owned nodes are present in every valid G_F regardless of source.

**Acceptance**: Predicate registry is a frozen code artifact. System-owned injection is explicit and traced. All existing tests still pass.

**Do not touch**: `ground_graph()` internals.

---

### P3-E: Kitchen Canonicalizer Repair
**Status**: `[ ] NOT STARTED`

**Objective**: Fix the Kitchen VLM canonicalizer to achieve 100% concept preservation on the ideal fixture.

**Required Changes** (in `kitchen_vlm_functional_graph.py`):
1. **Extend `KITCHEN_ROLE_REGISTRY`** to cover `water_source` and `coffee_source` with broader natural language aliases (kettle, water pitcher, coffee jar, coffee grounds, etc.)
2. **Remove the `INSERTABLE_IN` default fallback** in `map_binary_relation()` (line 232-233) and in the relation compilation (line 394). Unmapped relations should either be explicitly mapped or raise `UnmappedFunctionalConceptError`.
3. **Widen operation group acceptance** (lines 420-425): instead of hard-coding only `coffee_stirrer→coffee_container` and `soup_eating_utensil→soup_container`, accept any operation group whose tool_role and target_role are both present in the mapped roles, and assign a deterministic canonical group ID based on the tool→target pair.
4. **Remove `required_target_count` clipping** (line 460): if the VLM says `required_target_count: 3` but the target role has `count: 2`, either preserve the VLM's count and let validation reject it, or explicitly log a diagnostic.

**Tests**: `test_ideal_fixtures.py` must pass for kitchen_K1. Run `pytest mujoco_scenes/tests/test_kitchen_vlm_functional_graph.py`.

**Acceptance**: Ideal kitchen fixture → canonical G_F with role recall = 1.0, relation recall = 1.0 vs GT reference.

**Do not touch**: Living Room or Workshop canonicalizers; GT path; `ground_graph()`.

---

### P3-F: Living Room Canonicalizer Repair
**Status**: `[ ] NOT STARTED`

**Objective**: Fix the Living Room VLM canonicalizer for correct count merging and system-owned context injection.

**Required Changes** (in `environment_vlm_requirements.py`):
1. **Fix CUP_SAUCER_SET count merging**: If VLM emits separate "cup" (count=2) and "saucer" (count=2) roles that both map to `CUP_SAUCER_SET`, do NOT sum to 4. The count should be `max(cup_count, saucer_count)` = 2, representing 2 cup+saucer bundles.
2. **Inject system-owned SEATING_POSITION and SEATING_PAIR** if not emitted by VLM, with explicit provenance.
3. **Verify relation direction**: Ensure all VLM-emitted "placed_on" / "on" relations compile to `REGION --FITS_SET_ON--> CUP_SAUCER_SET` (subject=region) regardless of grammatical direction.

**Tests**: `test_ideal_fixtures.py` must pass for living_room_L1. Run existing living room tests.

**Acceptance**: Ideal living room fixture → canonical G_F matching GT structure.

**Do not touch**: Kitchen or Workshop canonicalizers.

---

### P3-G: Workshop Canonicalizer Repair
**Status**: `[ ] NOT STARTED`

**Objective**: Fix the Workshop VLM canonicalizer for deterministic duplicate merging and context injection.

**Required Changes** (in `workshop_phase1/requirements.py` and `vlm_spec_provider.py`):
1. **Merge semantically duplicate driver roles**: If VLM emits "manual_screwdriver" and "power_screwdriver" as separate roles both mapping to `driver`, merge deterministically (max count, union of categories).
2. **Inject system-owned `repair_target` and `workbench_surface`** if not emitted by VLM.
3. **Validate `required_target_count <= target_role.max_count`** before graph construction, or clip with explicit diagnostic.
4. **Ensure operation group validation** handles the common Qwen failure pattern (required_target_count=2 with fastener.max_count=1).

**Tests**: `test_ideal_fixtures.py` must pass for workshop_W1. Run `pytest mujoco_scenes/tests/test_workshop_vlm_requirements.py`.

**Acceptance**: Ideal workshop fixture → canonical G_F matching GT structure.

---

### P3-H: Region Resolution & Search Contract
**Status**: `[ ] NOT STARTED`

**Objective**: Ensure VLM-proposed inspectable regions resolve correctly to physical scene regions across all domains.

**Scope**: Kitchen region resolution (D1-C1), Workshop region resolution (LEFT_DRAWER, RIGHT_DRAWER, TOOL_CABINET). Living Room has no articulated search regions.

**Required Changes**:
1. Document which regions are discoverable vs fixed in each domain.
2. Ensure region resolution handles common VLM phrasings without silent failures.
3. If VLM proposes no resolvable regions, fall back to the domain's complete known region set (with explicit provenance).

**Tests**: Extend `test_vlm_interface_boundary.py` with additional region resolution edge cases.

**Acceptance**: All known region phrasings from historical VLM outputs resolve correctly.

---

### P3-I: K1/L1/W1 Full Ideal-Fixture Convergence
**Status**: `[ ] NOT STARTED`

**Objective**: End-to-end validation that ideal raw fixtures → canonicalizer → G_F → G_O → φ* → action sequence for K1/L1/W1.

**Scope**: Integration test. No new code changes unless issues are found.

**Inputs**: Ideal fixtures from P3-C, canonicalizers repaired in P3-E/F/G.

**Tests**: Run each ideal fixture through the full pipeline using a test harness that:
1. Feeds the fixture JSON as if it were a VLM response
2. Canonicalizes to G_F
3. Builds G_O from the actual scene
4. Runs `ground_graph()`
5. Compiles symbolic problem
6. Runs A* planner

**Acceptance**: All three → `ACTION_SEQUENCE_READY`. This proves the software interface is correct.

**Do not touch**: VLM prompt, FM adapter, model configuration.

---

### P3-J: Live 9B Causal Diagnosis
**Status**: `[ ] NOT STARTED`

**Objective**: Run Qwen3.5-9B once on K1/L1/W1 and perform causal layer diagnosis.

**Scope**: Live VLM call + raw response analysis. No code changes.

**Protocol**:
1. Save raw VLM response BEFORE canonicalization.
2. For each expected semantic concept, classify: CORRECT / MISSING / EXTRA / WRONG_CARDINALITY / WRONG_BINDING_POLICY / WRONG_RELATION / WRONG_ENTITY_KIND
3. Attempt canonicalization. If it fails, identify the canonicalizer gap.
4. If canonicalization succeeds, run downstream. If downstream fails, identify the layer.
5. Record diagnosis per §5 decision tree.

**Artifacts**: Save per-case `{raw_response.json, concept_audit.json, canonical_gf.json, layer_diagnosis.txt}`.

**Acceptance**: Clear per-case layer diagnosis for K1/L1/W1.

---

### P3-K: 9B vs 27B Raw Specification Comparison
**Status**: `[ ] NOT STARTED`

**Objective**: If P3-J reveals model capacity limitations, compare raw specification quality between Qwen3.5-9B and Qwen3.5-27B on K1/L1/W1.

**Metrics** (evaluated at RAW specification level, not terminal outcome):
- Schema-valid rate
- Role identity recall / precision
- Role cardinality accuracy
- Binding policy accuracy
- Relation recall / precision
- Operation group recall
- Context-role correctness
- Region proposal correctness
- Malformed specification rate

**Decision**: If 9B raw is correct but canonicalizer breaks → fix canonicalizer (not a model problem). If 9B raw is fundamentally wrong and 27B fixes it → adopt 27B.

**Acceptance**: Documented comparison with decision on model selection.

---

### P3-L: Full 32-Variant Deterministic Fixture Matrix
**Status**: `[ ] NOT STARTED`

**Objective**: Validate all 32 variants using reviewed deterministic fixtures (not live VLM).

**Scope**: Create fixtures for all 32 variants or verify GT-generated G_F works for all.

**Gate**:
- 20/20 feasible → `ACTION_SEQUENCE_READY`
- 12/12 infeasible → `INFEASIBLE` with correct causal deficiency

This tests the SOFTWARE INTERFACE independently of live FM variability.

**Acceptance**: 32/32 correct terminal outcomes from deterministic fixtures.

---

### P3-M: Clean Live VLM Benchmark
**Status**: `[ ] NOT STARTED`

**Objective**: Run the complete 32-variant matrix against the selected live model on a single frozen commit.

**Protocol**:
1. Single frozen git commit; clean worktree (`git status` shows no changes).
2. Exact model identifier, prompt version, canonicalizer version recorded.
3. No mixing artifacts from earlier commits.
4. No stale-case resume unless fingerprint proves exact configuration match.
5. Raw VLM response retained even when canonicalization fails.
6. Full artifact chain retained per case: raw → validated → canonical G_F → G_O → φ* → plan.
7. Per-case provenance fingerprint: `{git_commit, prompt_hash, model, canonicalizer_version}`.
8. Total actual runtime recorded (not wrapper overhead).

**Artifacts**: `tmp/p3m_clean_benchmark_YYYYMMDD_HHMMSS/` with `summary.json`, `results.csv`, per-case directories.

**Acceptance**: Results are scientifically reportable. Each failure has a layer diagnosis.

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
| `provenance.json` | git commit, model, prompt hash, config hashes | YES |

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
[ ] Architecture: one canonical G_F representation shared by GT and VLM
[ ] Architecture: one canonical G_O representation
[ ] Architecture: ground_graph is sole assignment authority
[ ] Architecture: φ* is immutable downstream
[ ] Architecture: no hidden FM call after specification
[ ] Architecture: no GT oracle in VLM runtime path
[ ] Architecture: no semantic role reassignment in compiler
[ ] Software: VLMSpecProvider.provide() correctly validates and wraps errors
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

### **CURRENT NEXT PASS: P3-A**

**Exact Objective**: Fix the broken `VLMSpecProvider.provide()` method and establish clean benchmark infrastructure.

**Prerequisites**: None — this is the first pass.

**What to do**:
1. Open `mujoco_scenes/functional_tamp_pipeline/vlm_spec_provider.py`
2. Fix lines 34-37: the `provide()` method's try block is truncated — the `_workshop` method definition starts inside it. Complete the try/except/return logic matching the `GTSpecProvider.provide()` pattern (which correctly calls `validate()` + `validate_runtime_gf()` and returns the graph).
3. Verify `_workshop` has `@staticmethod` decorator (currently missing at line 37).
4. Run: `pytest mujoco_scenes/functional_tamp_pipeline/tests/ -v`
5. Run: `python -c "from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider; print('import OK')"` to verify no syntax errors.

**Acceptance Criteria**:
- `VLMSpecProvider.provide()` has a complete method body that calls `validate()`, calls `validate_runtime_gf()`, wraps validation errors in `MalformedVLMSpecificationError`, and returns the validated graph.
- All 223+ existing tests pass.
- Import succeeds without syntax error.

**Expected next pass**: P3-B
