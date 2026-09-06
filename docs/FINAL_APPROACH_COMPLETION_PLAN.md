# FM-Guided Functional-TAMP: Final Approach Completion Plan

> **Purpose**: Single authoritative engineering plan for bringing the FM/VLM-guided
> Functional-TAMP pipeline to its strongest scientifically valid end-to-end performance.
>
> **Constraint**: NO fixes are implemented in this document. NO benchmark runs are launched.
> This is the blueprint another coding session executes stage-by-stage.

---

## §1 Repository Provenance

| Field | Value |
|---|---|
| Repository | `Narendhiranv04/icra-we-ball` |
| Branch | `vlm-testing-pipeline` |
| HEAD at audit time | `d0282997837c89d45a4a2c32c5f0c1eba954b1e1` |
| Latest evaluation artifacts | `benchmark_reports/final_vlm_evaluation_v3_rerun5/` |
| Evaluation artifact commits | `849dfc55` (24 preserved) + `eba26916` (8 resumed) |
| Model | `qwen35-9b` via vLLM at `gvlab2.iiit.ac.in:8000` |
| Evaluation invariants | 32 variants, 1 VLM call each, 0 replans, `COMPLETED_RESUMED` |

---

## Implementation Progress

| Phase / Stage | Description | Status | Commit / State | Result / Evidence |
|---|---|---|---|---|
| **Phase 0** | Repository audit, provenance verification, raw response archive check | **COMPLETED** | `d0282997` (clean + untracked plan) | 32/32 raw FM responses verified in v3_rerun5 |
| **Phase A.1** | True Raw-FM Replay Harness (`--spec-source raw-replay`) | **COMPLETED** | Working tree (run.py, evaluate_vlm_functional_tamp.py) | Ingests `fm_call_001.json` content -> sanitizer -> compiler; 0 VLM requests |
| **Phase A.2** | Evaluator Metric Corrections & Failure Taxonomy (5 paper-level categories) | **COMPLETED** | Working tree (evaluation_metrics.py, evaluate_vlm_functional_tamp.py) | Separated role recall vs complete spec, any vs complete grounding, non-empty plan rate |
| **Phase A.3** | Metric & Replay Unit/Regression Tests | **COMPLETED** | Working tree (test_raw_replay_and_evaluator_metrics.py) | 6/6 tests passing (e2e raw replay, invariants, taxonomy, grounding metrics) |
| **Phase B.1** | Kitchen Semantic Compiler & Causal Disambiguation | **PENDING** | - | Generic relation mappings, source/container collision resolution |
| **Phase B.2** | Living Room Structural Semantic Compilation | **PENDING** | - | Generic personal/shared region inference, relation mappings |
| **Phase B.3** | Workshop Compiler & W3 Instrument Misclassification Fix | **PENDING** | - | Precedence fix: implement/tool vs fastener action, compound relation parsing |
| **Phase C.1** | Kitchen K1 Deep Grounding & Search-Stop Diagnosis | **PENDING** | - | Complete visible grounding stops search, no unnecessary inspection |
| **Phase C.2** | Living Room L1 End-to-End Grounding | **PENDING** | - | Refreshment payload & seating position verification |
| **Phase C.3** | Workshop Grounding & Ranked Search | **PENDING** | - | Fastener & driver grounding, search recovery |
| **Phase D** | Candidate Planning & Symbolic Problem Verification | **PENDING** | - | Valid candidate plan generation, user-level goal evaluation |
| **Phase E** | Full Frozen Raw-Replay (32 variants × 0 VLM calls) | **PENDING** | - | Matrix generated in `benchmark_reports/raw_replay_after_final_interface_v1/` |
| **Phase F** | Pre-Final Freeze, Anti-Leakage Audit, vLLM Server Check | **PENDING** | - | Clean tree, single git commit, frozen hashes, server sanity |
| **Phase G** | Final 32 × 1 Clean Live Experiment & Offline Evaluation | **PENDING** | - | 32 variants, 1 request each, 0 replans, publication tables |

---

## §2 Anti-Leakage Contract

Every fix in this plan MUST satisfy ALL of these constraints:

1. **Zero GT/benchmark leakage**: No runtime code path may read GT role names, GT assignments, GT action sequences, or reference specification data.
2. **Exactly 1 FM call per variant**: No repair, critic, or retry calls.
3. **No per-variant aliases**: e.g., "K3 phrase X means role Y" is forbidden. All mapping logic must be generic.
4. **Exploration is automatic**: Task instructions NEVER mention drawers, cabinets, search, or inspection. Search emerges from the pipeline's observation-gap detection.
5. **Workshop instruction is material-agnostic**: NEVER mentions screw, driver, bolt, nut, or any specific fastener type.
6. **Partial plans ≠ full-task success**: `full_task_satisfied` can ONLY be true when ALL GT final-state goals are met. `PARTIAL_ACTION_SEQUENCE_READY` must never be counted as `outcome_correct=True` for feasible variants.
7. **Environment-projected context anchors**: Allowed ONLY with strict provenance (system fixed-anchor registry, never from GT).

---

## §3 Frozen Task Instructions

| Domain | Canonical Instruction |
|---|---|
| Kitchen | Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil. |
| Living Room | Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people. |
| Workshop | Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench. |

---

## §4 Current Baseline (v3_rerun5)

### §4.1 Primary Metrics

| Metric | Value |
|---|---|
| Outcome Correct | 0.0% |
| Feasible-task Success | 0.0% |
| Feasibility Recovery | 0.0% |
| Goal Coverage | 0.0% |
| False Completion | 0.0% |
| VLM Requests | 1.00 |
| High-level Replans | 0.00 |

### §4.2 Pipeline Diagnostic Table

| Metric | Kitchen | Living | Workshop | Overall |
|---|---:|---:|---:|---:|
| Raw VLM role F1 | 94.4% | 43.9% | 49.5% | 64.6% |
| Raw VLM relation F1 | 18.5% | 5.7% | 0.0% | 8.7% |
| Raw VLM group F1 | 50.0% | 0.0% | N/A | 27.3% |
| Sanitization success | 100.0% | 100.0% | 100.0% | 100.0% |
| Full canonicalization | 0.0% | 0.0% | 0.0% | 0.0% |
| Partial canonicalization | 100.0% | 100.0% | 100.0% | 100.0% |
| Runtime contract coverage | 94.4% | 40.0% | 53.3% | 64.6% |
| Search eligible | 100.0% | 0.0% | 100.0% | 68.8% |
| Grounding success / eligible | 75.0% | 90.0% | 90.0% | 84.4% |
| Candidate planning success | 100.0% | N/A | 0.0% | 50.0% |
| Partial-plan rate | 8.3% | 0.0% | 50.0% | 18.8% |
| Candidate goal coverage | 93.7% | N/A | 37.0% | 65.3% |
| Full-task success | 0.0% | 0.0% | 0.0% | 0.0% |
| Mean regions inspected | 5.00 | 0.00 | 2.40 | 2.62 |

### §4.3 Failure Distribution

| Failure Category | Count |
|---|---:|
| CANONICALIZATION_AMBIGUITY | 16 |
| CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC | 14 |
| FM_SEMANTIC_OMISSION | 2 |

### §4.4 Interpretation

**Kitchen (94.4% role F1, 0% full-task success)**: The FM correctly identifies roles but the downstream pipeline has two problems: (a) some role pairs collide to the same canonical name → `CANONICALIZATION_AMBIGUITY`, and (b) unresolved required properties/relations block planning despite correct grounding. K1 status is `ACTION_SEQUENCE_READY, cpl=0` which means the "plan" produces zero actions — the planning compiler fails to translate the grounded assignment into actual symbolic actions.

**Living Room (43.9% role F1, 0% everything)**: The FM under-specifies region roles (uses generic "support refreshment" instead of "personal support" vs "shared support"), causing the semantic compiler to reject them as `UNRESOLVED_SEMANTIC`. Zero search eligibility (0 regions inspected) because living room has no closed storage. The primary blocker is role mapping failure.

**Workshop (49.5% role F1, 50% partial)**: The FM uses phrases like "fastening implement" or "fastening tool" which the workshop mapper cannot resolve to `driver`. W1/W2/W6/W8/W10 produce 2-action partial plans (typically just the driver-return actions), never the full 3-goal sequence.

---

## §5 Architecture Overview

```
Task Instruction + RGB Images
         │
         ▼
   ┌─────────────┐
   │  FM Call     │  fm_adapter.py — SYSTEM_PROMPT + RESPONSE_SCHEMA
   │  (1 call)    │  → raw JSON functional graph
   └─────┬───────┘
         │ raw dict
         ▼
   ┌─────────────────────┐
   │ Structural Sanitizer │  structural_sanitizer.py
   │ (Stage A)            │  normalize IDs, validate schema, resolve count ambiguity
   └─────┬───────────────┘
         │ SanitizationResult
         ▼
   ┌─────────────────────┐
   │ Semantic Compiler    │  semantic_compiler.py
   │ (Stage B)            │  _map_role() → canonical name, _relation() → canonical predicate
   │                      │  causal_position() for source/dest/instrument disambiguation
   │                      │  → FunctionalRequirementGraph
   └─────┬───────────────┘
         │ G_F (candidate graph)
         ▼
   ┌─────────────────────┐
   │ Executability        │  executability.py (Stage C)
   │ Analysis             │  identify UNINSTANTIABLE roles/groups
   └─────┬───────────────┘
         │
         ▼
   ┌─────────────────────┐
   │ Perception + Search  │  domain adapter (kitchen.py / living_room.py / workshop.py)
   │                      │  → ObservedSceneGraph (G_O)
   └─────┬───────────────┘
         │
         ▼
   ┌─────────────────────┐
   │ Grounding φ*         │  grounding.py — ground_graph(G_F, G_O)
   │                      │  → GraphGroundingResult (assignment)
   │                      │  If VLM mode + incomplete: ground_verified_candidate_subgraph()
   └─────┬───────────────┘
         │
         ▼
   ┌─────────────────────┐
   │ A* Planning          │  symbolic_planning_core.py — deterministic_astar()
   │                      │  → action sequence
   └─────┬───────────────┘
         │
         ▼
   ┌─────────────────────┐
   │ Evaluation           │  evaluate_vlm_functional_tamp.py + evaluation_metrics.py
   │                      │  → Primary 7 metrics + diagnostic table
   └─────────────────────┘
```

---

## §6 Failure Taxonomy with Root Cause Analysis

### §6.1 CANONICALIZATION_AMBIGUITY (16 variants)

**Root cause**: `_map_role()` in `semantic_compiler.py` calls domain-specific mappers that return `None` for FM phrases that don't match any alias. When a role maps to `None` and is NOT a context-only anchor, it goes to `unresolved_roles[]`. When multiple raw roles map to the same canonical name, the collision logic in `can_merge_roles()` either merges (if identical causal position) or rejects as ambiguous.

**Kitchen examples** (K2, K5, K11, K12):
- FM produces roles like "contain coffee" and "contain coffee powder" — both map to `coffee_container` via `map_kitchen_role_function()`, causing collision. The `can_merge_roles()` function detects different causal positions (one is source, other is destination) and rejects.

**Living Room examples** (L3-L6, L8-L9):
- FM produces "support refreshment" or "support entertainment control" for REGION roles. `map_living_room_role_function()` only recognizes "personal" or "shared" variants. Generic "support" functions return `None` → `UNRESOLVED_SEMANTIC`.
- The semantic compiler's living room fallback (lines 83-94) tries to infer personal vs shared from binding_policy and relation spatial keywords, but many FM outputs lack these specific signals.

**Workshop examples** (W3-W5, W6-W7, W9):
- FM produces "fastening implement" or "fastening tool" for the driver role. `map_workshop_role_function()` looks for specific tool keywords. The causal-position fallback in semantic_compiler.py (lines 102-109) checks for `instrument` + `group_tool` position AND matching candidate_categories against the system ontology. If the FM uses generic categories instead of matching the ontology, this fails.

### §6.2 CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC (14 variants)

**Root cause**: The semantic compiler successfully maps all roles but some relations/groups remain unresolved. The `_relation()` function calls domain-specific relation mappers that return `None` for unrecognized phrases. When an interaction group's `required_relations` all map to `None`, the group is disabled.

**Kitchen examples** (K1, K3, K6, K4, K7-K10):
- FM group `required_relations` contain phrases like "fits inside", "placed on", "stir with" — the kitchen relation mapper `map_binary_relation()` doesn't recognize some of these.
- K1: Status `ACTION_SEQUENCE_READY` but `cpl=0` means 0 actions produced. The planning compiler (`run_to_plan()` in kitchen.py) fails to generate any actions despite having a grounded assignment.

**Living Room** (L1, L10):
- Groups use "support payload" as required_relation — unrecognized by `canonicalize_living_room_relation()`.

**Workshop** (W1, W2, W8, W10):
- Groups have unresolved context_relations or the relation phrases don't match workshop relation canonicalizer.

### §6.3 FM_SEMANTIC_OMISSION (2 variants: L2, L7)

**Root cause**: The FM output is structurally valid and canonicalization succeeds, but the raw VLM specification is incomplete — it omits roles that are present in the GT reference. For L2 and L7, the FM apparently fails to produce some required living room roles entirely.

---

## §7 Stage-by-Stage Fix Plan

> **CRITICAL RULE**: Do NOT modify the VLM prompt first. First determine what
> performance is achievable by replaying the CURRENT saved 32 raw VLM outputs
> through corrected deterministic downstream code.

### Stage 0: Replay Infrastructure Validation

**Goal**: Confirm the replay harness faithfully reproduces v3_rerun5 results.

1. Run `scripts/evaluate_vlm_functional_tamp.py --mode vlm --spec-source replay --specification-root benchmark_reports/final_vlm_evaluation_v3_rerun5 --output-root /tmp/replay_baseline`
2. Verify all 32 per-variant records match v3_rerun5 exactly (same failure_category, same candidate_plan_length, same grounded_roles).
3. If any discrepancy: fix the replay harness BEFORE proceeding.

**Files**: [evaluate_vlm_functional_tamp.py](file:///home/naren/RA_iiith/scripts/evaluate_vlm_functional_tamp.py)

---

### Stage 1: Kitchen Semantic Compiler Fixes

**Goal**: Resolve the 8 CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC and 4 CANONICALIZATION_AMBIGUITY failures in Kitchen.

#### §7.1.1 Fix Kitchen Relation Mapping

**Problem**: FM uses relation phrases like "fits inside", "placed on", "stir with", "supplies material to" that `map_binary_relation()` doesn't recognize.

**Fix** (GENERIC, no per-variant aliases):
- In [kitchen_vlm_functional_graph.py](file:///home/naren/RA_iiith/mujoco_scenes/kitchen_vlm_functional_graph.py), extend `map_binary_relation()` to recognize:
  - "fits inside" / "fits in" / "placed in" / "goes into" → `INSERTABLE_IN`
  - "placed on" / "set on" / "rests on" / "sits on" → `FITS_SET_ON` or `ON_SURFACE` (depending on subject/object kinds)
  - "stir with" / "stirred by" / "used to stir" → `REACHES_BOTTOM` (for stirring implements)
  - "supplies" / "transfers to" / "pours into" / "provides material" → `SUPPLIES_MATERIAL_TO`
- Each mapping must be purely generic phrase-matching (regex), not per-variant.

**Verification**: After this fix, replay all 12 Kitchen variants. Expect fewer disabled groups and more candidate plans.

#### §7.1.2 Fix Kitchen Role Collision (coffee_container ambiguity)

**Problem**: FM distinguishes "contain coffee" (destination: receives brewed coffee) from "contain coffee powder" (source: raw material). Both map to `coffee_container` via `map_kitchen_role_function()`. The `causal_position()` function correctly identifies them as having different causal roles (source vs destination), so `can_merge_roles()` rejects the merge → ambiguity.

**Fix**: The `_map_role()` function in `semantic_compiler.py` already has a source-provider disambiguation path (lines 67-75) that maps source-position roles to `coffee_source` or `water_source`. The issue is that this path requires the text to contain both "source"/"provider"/"ingredient" AND a material keyword. Some FM outputs describe the source role as "contain coffee powder" without using the word "source".

**Generic fix**: In `causal_position()` (semantic_compiler.py line 19-44), extend source detection to include "powder", "granules", "raw material", "ingredient" in the function text. Then the source-provider path at line 67-75 will engage for these roles, mapping them to `coffee_source` instead of `coffee_container`.

Alternatively, in `_map_role()` for kitchen (line 64-76), after the existing source-provider check, add a fallback: if `map_kitchen_role_function()` returns a canonical name that's already in the `nodes` dict, AND the new role has `source` in its causal position while the existing one has `destination`, map the new one to `{material}_source`.

#### §7.1.3 Fix Kitchen Planning Compiler (cpl=0 problem)

**Problem**: K1 has status `ACTION_SEQUENCE_READY` but `candidate_plan_length=0`. This means the kitchen planning compiler (`run_to_plan()` in kitchen.py) returns an empty action sequence despite having grounded roles.

**Root cause investigation needed**: Check what `run_to_plan()` actually does when it has a partial grounding (e.g., only `coffee_stirrer`, `soup_eating_utensil`, `water_source` grounded but missing `coffee_container`, `soup_container`). The symbolic planner likely cannot generate any meaningful actions because the missing containers are precondition-providers for all useful actions.

**Fix**: This may be a fundamental limitation — if the FM doesn't produce `coffee_container` and `soup_container` roles at all (or they get filtered), no preparation actions are possible. The fix is in Stages 1.1-1.2 above: once role collisions and relation mapping are fixed, more roles should survive canonicalization, enabling actual plans.

**Files**:
- [kitchen_vlm_functional_graph.py](file:///home/naren/RA_iiith/mujoco_scenes/kitchen_vlm_functional_graph.py) — `map_binary_relation()`, `map_kitchen_role_function()`
- [semantic_compiler.py](file:///home/naren/RA_iiith/mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py) — `causal_position()`, `_map_role()`
- [kitchen.py](file:///home/naren/RA_iiith/mujoco_scenes/functional_tamp_pipeline/domains/kitchen.py) — `run_to_plan()`

---

### Stage 2: Living Room Semantic Compiler Fixes

**Goal**: Resolve the 7 CANONICALIZATION_AMBIGUITY and 1 CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC failures in Living Room. Enable search eligibility.

#### §7.2.1 Fix Living Room Region Role Mapping

**Problem**: FM produces "support refreshment" / "support refreshment setting" / "support entertainment control" for REGION roles. `map_living_room_role_function()` only matches when explicit "personal"/"shared"/"central"/"side" keywords are present.

**Fix** (GENERIC): In [environment_vlm_requirements.py](file:///home/naren/RA_iiith/mujoco_scenes/environment_vlm_requirements.py) `map_living_room_role_function()`:
- Currently requires explicit personal/shared keywords. The semantic_compiler.py fallback (lines 83-94) tries to infer from binding_policy and relation spatial keywords.
- Strengthen the fallback: if a REGION role's `binding_policy` is `DISTINCT` and `required_count` >= 2, it's personal. If `binding_policy` is `SHARED` and `required_count` == 1, it's shared.
- For "support entertainment control" with SHARED policy: map to `SHARED_REMOTE_REGION`.
- For "support refreshment setting" with DISTINCT policy and count=2: map to `PERSONAL_CUP_SAUCER_REGION`.
- These inferences use ONLY the FM's own structural annotations (binding_policy, count), not GT.

#### §7.2.2 Fix Living Room Relation Mapping

**Problem**: FM groups use "support payload", "placed on", "compatible with", "near" as required_relations. `canonicalize_living_room_relation()` only recognizes specific canonical signatures:
1. `(PERSONAL_CUP_SAUCER_REGION, FITS_SET_ON, CUP_SAUCER_SET)`
2. `(PERSONAL_CUP_SAUCER_REGION, NEAR_SEAT, SEATING_POSITION)`
3. `(SHARED_REMOTE_REGION, FITS_ON, REMOTE)`
4. `(SHARED_REMOTE_REGION, ACCESSIBLE_FROM_BOTH_SEATS, SEATING_PAIR)`

**Fix** (GENERIC): Extend `canonicalize_living_room_relation()` phrase matching:
- "support payload" / "supports" / "placed on" / "set on" / "rests on" → `FITS_SET_ON` (when object is payload) or `FITS_ON` (when object is REMOTE)
- "near" / "near seat" / "nearby" / "beside" / "adjacent to" → `NEAR_SEAT` (when object is SEATING_POSITION)
- "compatible with" → infer from subject/object canonical roles: if (REGION, OBJECT) → `FITS_SET_ON` or `FITS_ON`; if (REGION, FIXED_TARGET) → `NEAR_SEAT` or `ACCESSIBLE_FROM_BOTH_SEATS`
- "accessible to both" / "accessible from both" / "shared access" → `ACCESSIBLE_FROM_BOTH_SEATS`

#### §7.2.3 Fix Living Room Group Function Mapping

**Problem**: FM group functions like "refreshment setting", "refreshment setting for person 1", "entertainment control placement" are not recognized by `map_living_room_operation_group_function()`.

**Fix** (GENERIC): In `LIVING_INTERACTION_GROUP_ALIASES`, add:
- `personal_support_group`: ["refreshment setting", "refreshment setting for person", "provide refreshment setting"]
- `remote_placement_group`: ["entertainment control placement", "entertainment control", "place entertainment control"]
- These are generic task-semantic phrases, not per-variant.

#### §7.2.4 Fix Living Room Search Eligibility

**Problem**: Living room has 0 search eligibility because `search_eligible` requires both `graph_dict` (canonicalization succeeded) AND `regions_available` (non-empty). Living room has no closed storage regions, so `regions_available = []`.

**Analysis**: This is CORRECT behavior — living room has open scenes with all objects visible. Search eligibility should be 0. The issue is that even without search, the grounding should work on the initial observation. The blocker is upstream (role mapping failure), not search.

**Files**:
- [environment_vlm_requirements.py](file:///home/naren/RA_iiith/mujoco_scenes/environment_vlm_requirements.py) — all living room mappers
- [semantic_compiler.py](file:///home/naren/RA_iiith/mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py) — living room fallback path

---

### Stage 3: Workshop Semantic Compiler Fixes

**Goal**: Resolve the 6 CANONICALIZATION_AMBIGUITY and 4 CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC failures in Workshop. Upgrade 50% partial to higher coverage.

#### §7.3.1 Fix Workshop Role Mapping (driver misclassification)

**Problem**: FM produces "fastening implement" / "fastening tool" for the driver role. `map_workshop_role_function()` at line 500 has `has_fastener_action` check for the word "fastening", causing these implement/tool roles to be classified as `CAN_FASTEN` (fastener) instead of `CAN_DRIVE_SCREW` (driver). This creates a collision with the actual fastener role → `CANONICALIZATION_AMBIGUITY`.

**Root cause confirmed**: `map_workshop_role_function({"function": "fastening implement"})` returns `CAN_FASTEN`. The semantic_compiler's causal-position fallback (line 102-109) only fires when `mapped is None`, never correcting the wrong classification. Then `{'CAN_FASTEN': 'fastener'}` mapping at line 109 collides with the actual fastener role.

**Fix** (GENERIC, two-part):
1. In `map_workshop_role_function()` (requirements.py ~line 496-507): Before the `has_fastener_action` check, add an explicit guard: if the function text contains BOTH a fastener-action word ("fastening") AND an instrument word ("implement", "tool", "equipment"), prioritize the instrument classification. Specifically:
   ```python
   # Tool/implement semantics override generic "fastening" action
   has_implement = any(w in words for w in ("implement", "tool", "equipment", "device"))
   if has_fastener_action and has_implement and not is_fastener_target:
       return "CAN_DRIVE_SCREW"
   ```
2. In `semantic_compiler.py` `_map_role()` workshop path (line 102-109): Strengthen the fallback to also fire when `mapped` is `CAN_FASTEN` but the causal position includes `instrument` + `group_tool` — override to `driver` if candidate_categories match the driver ontology.

#### §7.3.2 Fix Workshop Relation Mapping

**Problem**: Workshop interaction groups use relation phrases that `canonicalize_workshop_relation()` doesn't recognize.

**Fix** (GENERIC): Extend workshop relation canonicalizer:
- "drives into" / "fastens into" / "screws into" / "inserts into" → `INSERTABLE_IN`
- "compatible with" / "interface compatible" / "fits with" → `INSERTABLE_IN` (when subject is tool, object is fastener/target)
- "reaches" / "reaches target" / "reaches bottom" → `REACHES_BOTTOM`

#### §7.3.3 Fix Workshop Partial → Full Planning

**Problem**: W1/W2/W8 produce 2-action partial plans. The full Workshop task has 3 goals: (1) insert fastener into target, (2) repair the target, (3) return driver to workbench. Partial plans typically achieve only goal 3 (return driver).

**Root cause**: When grounding is partial (missing `fastener` role), the symbolic planner can only plan actions for grounded roles. The driver-return action is trivially plannable. The insert/repair actions require `fastener` which is ungrounded.

**Fix pathway**: If the role mapping fix (§7.3.1) resolves the `fastener` role as well, partial plans become full plans. The `fastener` role is typically expressed by the FM as "component", "screw", or "bolt" — check if `map_workshop_role_function()` recognizes these.

**Files**:
- [requirements.py](file:///home/naren/RA_iiith/mujoco_scenes/workshop_phase1/requirements.py) — workshop mappers
- [semantic_compiler.py](file:///home/naren/RA_iiith/mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py) — workshop fallback
- [workshop.py](file:///home/naren/RA_iiith/mujoco_scenes/functional_tamp_pipeline/domains/workshop.py) — planning

---

### Stage 4: Evaluation Metric Corrections

**Goal**: Ensure all metrics are computed correctly before the final run.

#### §7.4.1 Fix `raw_vlm_role_recall` in Evaluator Diagnostic Table

**Problem**: The evaluator's diagnostic table computes `raw_vlm_role_recall` as `sum(r["raw_vlm_spec_complete"]) / d_total` (line 348 of evaluate_vlm_functional_tamp.py). This is a BINARY complete/incomplete flag, not the actual recall percentage. The pipeline_diagnostic_table.md from evaluation_metrics.py correctly computes per-variant role F1 from `raw_semantic_evaluation.py` — these two tables disagree.

**Analysis**: Two separate diagnostic tables exist:
1. `evaluation_metrics.py:write_detailed_report()` (line 203-259): Computes per-domain averages of `raw_role_f1`, `raw_relation_f1`, etc. — this is CORRECT.
2. `evaluate_vlm_functional_tamp.py` (line 348-373): Computes `raw_vlm_role_recall` as binary spec-complete flag — this is a DIFFERENT metric (not a recall percentage).

**Fix**: The evaluator's `raw_vlm_role_recall` in the evaluation_summary.json is actually "fraction of variants where the raw FM output was complete" — rename to `raw_vlm_spec_complete_rate` for clarity. The actual per-domain F1 scores are correctly computed in `pipeline_diagnostic_table.md` by `evaluation_metrics.py`. Both should be present in the final output.

#### §7.4.2 Verify `candidate_goal_coverage` Computation

**Problem**: K1 shows `candidate_goal_coverage=1.00` despite having `cpl=0`. This means the candidate_search_statistics report `total_goals=N, satisfied_goals=N` even when no actions are produced.

**Root cause**: The `candidate_search_statistics` may be populated from the A* search's initial state goal check, which for Kitchen may report goals satisfied even before any actions (e.g., if the initial state already satisfies some goal atoms that happen to match the partial specification).

**Fix**: `candidate_goal_coverage` should be conditioned on `candidate_plan_length > 0` OR explicitly define it as "coverage of goals that the grounded subgraph could theoretically achieve". Document the semantics clearly.

#### §7.4.3 Fix `outcome_correct` for Infeasible Variants

**Problem**: In `evaluation_metrics.py` line 173-174, `outcome_correct` is forced to `False` when `not gt_feasible and not runtime_contract_complete`. This means infeasible variants with incomplete FM outputs can never be `outcome_correct=True` even if the pipeline correctly concludes infeasibility.

**Analysis**: This is CORRECT behavior — if the runtime doesn't have a complete contract, any infeasibility conclusion is unreliable because the FM might have omitted the very role that makes the variant infeasible. Keep this logic.

---

### Stage 5: Full-Task Goal Coverage Fix

**Goal**: Enable non-zero `full_task_goal_coverage` once planning produces actual actions.

#### §7.5.1 Kitchen full_task_coverage()

**Location**: [evaluation_metrics.py](file:///home/naren/RA_iiith/mujoco_scenes/functional_tamp_pipeline/evaluation_metrics.py) lines 47-113.

**Analysis**: The Kitchen full_task_coverage checks:
1. 2× coffee settings: `(at, c, dining_table) ∧ (contains, c, coffee) ∧ (contains, c, water) ∧ (stirred, c)` — requires `coffee_container` role grounded to actual cup IDs
2. 2× soup settings: `(at, bowl, dining_table) ∧ (contains, bowl, soup)` + `(at, utensil, bowl)` — requires `soup_container` + `soup_eating_utensil`

**Dependency**: Full-task success requires ALL of: coffee_container, water_source, coffee_source, coffee_stirrer, soup_container, soup_eating_utensil to be grounded AND the planner to generate the complete action sequence.

#### §7.5.2 Workshop full_task_coverage()

**Analysis**: Workshop checks 3 goals:
1. `(inserted, fastener, workshop_frame_joint)` — requires `fastener` grounded
2. `(repaired, workshop_frame_joint)` — requires insert + repair sequence
3. `(at, driver, MAIN_WORKBENCH_ZONE) ∧ (hand_empty,)` — requires `driver` grounded

**Dependency**: Goals 1-2 require `fastener` role to be grounded. Goal 3 requires `driver`. Currently only `driver` is sometimes grounded (leading to partial plans achieving goal 3 only).

#### §7.5.3 Living Room full_task_coverage()

**Analysis**: Living room checks 3 goals (lines 83-111):
1-2. Personal refreshment settings (2×): Each needs verified `FITS_SET_ON`, `NEAR_SEAT`, payload parts placed on support
3. Remote placement: `(on, remote, dest)` + `FITS_ON` + `ACCESSIBLE_FROM_BOTH_SEATS`

**Dependency**: Requires `PERSONAL_CUP_SAUCER_REGION`, `CUP_SAUCER_SET`, `SEATING_POSITION`, `SHARED_REMOTE_REGION`, `REMOTE`, `SEATING_PAIR` all grounded.

---

### Stage 6: Replay Validation After Fixes

**Goal**: Before ANY live FM call, replay the EXISTING 32 saved FM outputs through the FIXED deterministic downstream code.

**Protocol**:
```bash
python scripts/evaluate_vlm_functional_tamp.py \
    --mode vlm \
    --spec-source replay \
    --specification-root benchmark_reports/final_vlm_evaluation_v3_rerun5 \
    --output-root benchmark_reports/replay_after_fixes_v1
```

**Expected improvements after Stages 1-3**:
- Kitchen: Many CANONICALIZATION_AMBIGUITY → resolved. More variants with actual action sequences.
- Living Room: Some CANONICALIZATION_AMBIGUITY → resolved via stronger role inference from structural signals.
- Workshop: "fastening implement" → `driver` mapping. More full plans instead of partial.

**This replay determines the ceiling of performance achievable without changing the FM prompt.**

---

### Stage 7: FM Prompt Refinement (ONLY if replay ceiling is insufficient)

> **Do NOT execute this stage until Stage 6 replay results are analyzed.**

If the replay shows that the FM's raw outputs fundamentally lack required information (e.g., FM never produces "personal" in region role descriptions), then GENERIC prompt refinements are allowed:

1. **Strengthen role-function guidance**: Add to SYSTEM_PROMPT section A a clarification that support surfaces serving individual users should use "personal" or "individual" in their function description, while shared surfaces should use "shared" or "common".
2. **Strengthen relation guidance**: Add examples of atomic relation phrases the robot's verifiers can actually check (already partially present in the "Robot Verifier Capabilities" section).
3. **Do NOT add domain-specific vocabulary**: The prompt must remain domain-agnostic.

---

### Stage 8: Final 32-Variant Live Evaluation

**Protocol**:
```bash
# Ensure clean working tree
git stash  # or commit
git status --short  # must be empty

# Run
python scripts/evaluate_vlm_functional_tamp.py \
    --mode vlm \
    --spec-source live \
    --output-root benchmark_reports/final_vlm_evaluation_v4 \
    --dry-run
```

**Invariants to verify**:
- 32 variants total, 20 feasible, 12 infeasible, 13 recovery
- Exactly 1 VLM request per variant
- 0 high-level replans
- Single git commit, clean working tree
- Single model (`qwen35-9b`), single prompt hash

---

## §8 Per-Variant Detailed Failure Traces (Representative)

### K1 (Kitchen, feasible)

**Raw FM output** (6 roles):
```
role_1: OBJECT "contain coffee powder"   cnt=1 DISTINCT   → coffee_source (disambiguated via causal source position ✓)
role_2: OBJECT "contain water"           cnt=1 DISTINCT   → water_source (disambiguated ✓)
role_3: OBJECT "contain coffee"          cnt=2 DISTINCT   → coffee_container ✓
role_4: OBJECT "contain soup"            cnt=2 DISTINCT   → soup_container ✓
role_5: OBJECT "stir coffee"             cnt=2 REUSABLE   → coffee_stirrer ✓
role_6: OBJECT "eat soup"                cnt=2 DISTINCT   → soup_eating_utensil ✓
```

**Raw FM relations**:
```
role_1 --[compatible with]--> role_3    → UNRESOLVED (no "compatible with" mapper)
role_2 --[compatible with]--> role_3    → UNRESOLVED
role_5 --[fits inside]--> role_3        → UNRESOLVED ("fits inside" not in kitchen relation mapper)
role_6 --[fits inside]--> role_4        → UNRESOLVED
```

**Raw FM groups**:
```
group_1: "prepare coffee" tool=role_5 target=role_3 cnt=2 rels=["fits inside"] → DISABLED (relation unmapped)
group_2: "serve soup"     tool=role_6 target=role_4 cnt=2 rels=["fits inside"] → DISABLED (relation unmapped)
```

**Result**: All 6 roles mapped correctly, but ALL relations and groups fail because "fits inside" and "compatible with" are not recognized by `map_binary_relation()`. Groups are disabled → no operation constraints → planner has grounded roles but no operation structure → `ACTION_SEQUENCE_READY` with 0 actions.

**Fix needed**: Add "fits inside" → `INSERTABLE_IN` and "compatible with" → `COMPATIBLE_SOURCE` in `map_binary_relation()`.

### L1 (Living Room, feasible)

**Raw FM output** (5 roles):
```
role_1: REGION "support refreshment setting"       cnt=2 DISTINCT  → PERSONAL_CUP_SAUCER_REGION (via structural: cnt=2, DISTINCT)
role_2: OBJECT "contain refreshment item"           cnt=2 DISTINCT  → CUP_SAUCER_SET? or None (generic "refreshment")
role_3: OBJECT "contain entertainment control"      cnt=1 SHARED    → REMOTE ✓
role_4: REGION "support entertainment control"      cnt=1 SHARED    → SHARED_REMOTE_REGION (via structural: cnt=1, SHARED)
role_5: REGION "support seating"                    cnt=2 SHARED    → SEATING_POSITION? (depends on mapper)
```

**Raw FM relations**:
```
role_1 --[placed on]--> role_4    → ? (direction seems wrong: refreshment region placed on remote region?)
role_2 --[placed on]--> role_1    → "placed on" → needs FITS_SET_ON mapping
role_3 --[placed on]--> role_4    → "placed on" → needs FITS_ON mapping
role_5 --[near]-->      role_1    → "near" → needs NEAR_SEAT mapping
role_5 --[near]-->      role_4    → "near" → needs ACCESSIBLE_FROM_BOTH_SEATS mapping
```

**Raw FM groups**:
```
group_1: "refreshment setting"              tool=role_2 target=role_1 cnt=2 rels=["placed on"] ctx=role_5 ctx_rels=["near"]
group_2: "entertainment control placement"  tool=role_3 target=role_4 cnt=1 rels=["placed on"] ctx=role_5 ctx_rels=["near"]
```

**Result**: Role mapping partially succeeds via structural inference (cnt+policy). The blockers are:
1. "contain refreshment item" (role_2) → `map_living_room_object_payload_role()` uses multi-signal check for "refreshment" and needs candidate_categories to contain cup/saucer keywords → likely returns `None`.
2. "support seating" (role_5) → `map_living_room_fixed_target_role()` should catch "seating" → `SEATING_POSITION`, but entity_kind is REGION not FIXED_TARGET, so the fixed_target mapper may not engage.
3. All relation phrases ("placed on", "near") are unmapped by the current `canonicalize_living_room_relation()`.

**Fix needed**: (a) Map "contain refreshment item" → `CUP_SAUCER_SET` when candidate_categories mention cups/drinkware, (b) handle REGION "support seating" → `SEATING_POSITION`, (c) map "placed on" → `FITS_SET_ON`/`FITS_ON` and "near" → `NEAR_SEAT`.

### W3 (Workshop, feasible)

**Raw FM output** (4 roles):
```
role_1: REGION "fastening target"              cnt=1 DISTINCT  cats=[workbench surface, marked workbench location, assembly fixture]
role_2: OBJECT "fastening component"           cnt=1 DISTINCT  cats=[screw, bolt, nut, washer, fastener]
role_3: OBJECT "fastening implement"           cnt=1 REUSABLE  cats=[screwdriver, drill, wrench, tool]
role_4: REGION "equipment placement surface"   cnt=1 SHARED    cats=[workbench surface, table top]
```

**Raw FM groups**:
```
group_1: "fastening operation" tool=role_3 target=role_2 cnt=1
         rels=["role_3 manipulates role_2", "role_2 fits into role_1"]
         ← ARITY VIOLATION: 2 phrases instead of atomic relation
```

**Result**: 
- role_1 → `workshop_frame_joint` via `map_workshop_fixed_target_role()` ✓
- role_2 → `fastener` via `map_workshop_role_function()` ✓ (cats contain "screw", "fastener")
- role_3 → `CAN_FASTEN` (WRONG!) because `map_workshop_role_function()` sees "fastening" and matches `has_fastener_action` at line 500. This misclassifies it as a fastener component instead of a driver tool. The causal-position fallback in `semantic_compiler.py` only fires when `mapped is None`, so it never corrects this misclassification. Then `semantic_compiler.py` line 109 maps `CAN_FASTEN` → `fastener`, creating a COLLISION with role_2 (already mapped to `fastener`). Two roles mapping to the same canonical name → `CANONICALIZATION_AMBIGUITY`.
- role_4 → `MAIN_WORKBENCH_ZONE` via context region mapper ✓

**Group failure**: required_relations contains compound phrases ("role_3 manipulates role_2") that include role IDs — these are not atomic relation predicates. The sanitizer passes them through but the relation mapper can't parse them.

**Fix needed**: (a) In `map_workshop_role_function()`, add "fastening implement"/"fastening tool" → `CAN_DRIVE_SCREW`. (b) In structural_sanitizer or semantic_compiler, strip role ID references from relation phrases before mapping (generic normalization).

---

## §9 File Modification Summary

| File | Changes |
|---|---|
| `mujoco_scenes/kitchen_vlm_functional_graph.py` | Extend `map_binary_relation()` with generic relation phrases |
| `mujoco_scenes/environment_vlm_requirements.py` | Extend `map_living_room_role_function()` structural inference; extend `canonicalize_living_room_relation()` phrase matching; extend `LIVING_INTERACTION_GROUP_ALIASES` |
| `mujoco_scenes/workshop_phase1/requirements.py` | Extend `map_workshop_role_function()` with "fastening implement/tool" aliases |
| `mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py` | Extend `causal_position()` source detection; strengthen workshop instrument fallback |
| `scripts/evaluate_vlm_functional_tamp.py` | Clarify `raw_vlm_role_recall` metric naming |
| `mujoco_scenes/functional_tamp_pipeline/evaluation_metrics.py` | Verify candidate_goal_coverage semantics |

---

## §10 Testing Plan

### §10.1 Unit Tests (Before Replay)

```bash
# Existing test suite must still pass
python -m pytest mujoco_scenes/functional_tamp_pipeline/tests/ -x -q

# New tests to add:
# 1. Test generic relation phrase mapping for each domain
# 2. Test causal_position() returns correct positions for source/dest/instrument
# 3. Test that no GT leakage exists (grep for reference fixtures in runtime path)
```

### §10.2 Anti-Leakage Audit

```bash
# Verify no runtime code imports GT fixtures
grep -rn 'GTSpecProvider\|gt_spec\|reference_spec\|ground_truth' \
    mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py \
    mujoco_scenes/functional_tamp_pipeline/structural_sanitizer.py \
    mujoco_scenes/functional_tamp_pipeline/executability.py \
    mujoco_scenes/functional_tamp_pipeline/grounding.py \
    mujoco_scenes/functional_tamp_pipeline/domains/ \
    mujoco_scenes/kitchen_vlm_functional_graph.py \
    mujoco_scenes/environment_vlm_requirements.py \
    mujoco_scenes/workshop_phase1/requirements.py
# Only raw_semantic_evaluation.py and evaluation_metrics.py should reference GT
```

### §10.3 Replay Regression (After Fixes)

1. Replay v3_rerun5 saved outputs through fixed code
2. Compare with v3_rerun5 baseline: expect STRICTLY better or equal metrics
3. Verify 0% false completion maintained
4. Verify VLM requests = 1.0, replans = 0.0

---

## §11 Acceptance Gates

The final 32-variant evaluation must satisfy ALL of:

| Gate | Criterion |
|---|---|
| Invariant Integrity | 32 variants, 20 feasible, 12 infeasible, 13 recovery; 1 VLM call each; 0 replans; single commit; clean tree |
| False Completion | 0% (no infeasible variant claims full success) |
| Kitchen Full-Task Success | > 0% (at least one feasible Kitchen variant achieves all 4 goals) |
| Overall Goal Coverage | > 0% (non-trivial progress on feasible variants) |
| No Regression | Every metric ≥ v3_rerun5 baseline |
| Anti-Leakage | Zero GT imports in runtime pipeline code |

**Stretch goals** (not required for acceptance):
- Kitchen full-task success ≥ 50% (3+ of 6 feasible variants)
- Workshop partial-plan coverage ≥ 67% (2/3 goals in most variants)
- Overall outcome_correct > 25%

---

## §12 Execution Handoff

### Priority Order

1. **Stage 0**: Validate replay infrastructure (30 min)
2. **Stage 1**: Kitchen semantic compiler fixes (60-90 min)
3. **Stage 2**: Living room semantic compiler fixes (60-90 min)
4. **Stage 3**: Workshop semantic compiler fixes (45-60 min)
5. **Stage 4**: Evaluation metric corrections (30 min)
6. **Stage 6**: Replay validation with fixed code (30 min + analysis)
7. **Stage 7**: FM prompt refinement (ONLY if needed, 30-60 min)
8. **Stage 8**: Final live 32-variant evaluation (30-60 min)
9. **Stage 5**: Verify full-task coverage (integrated with Stage 6/8)

### Estimated Total: 5-7 hours

### Key Decision Points
- After Stage 6 replay: If Kitchen achieves >50% full-task success on replay alone, skip Stage 7.
- After Stage 6 replay: If Living Room achieves >0% full-task success, skip living room prompt changes.
- After Stage 8: If acceptance gates are met, DONE. If not, analyze which specific failures remain and iterate on the weakest domain.

---

## §13 Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| FM outputs lack structural signals (binding_policy/count) needed for inference | Living Room role mapping won't improve | Fall back to FM prompt refinement (Stage 7) |
| Relation phrase space is larger than anticipated | Many relations remain unmapped | Build a generic phrase-to-predicate fuzzy matcher using embedding similarity (last resort) |
| Kitchen planning compiler has deeper issues beyond role availability | 0 actions even with all roles grounded | Debug the symbolic planning domain operators and preconditions |
| Replay harness has hidden bugs | Metrics don't match original run | Compare per-variant records field-by-field |
| VLM server unavailability during Stage 8 | Cannot run live evaluation | Schedule server time in advance; have replay results as fallback |

---

*Document created: 2026-09-06T19:50+05:30*
*Audit commit: d0282997837c89d45a4a2c32c5f0c1eba954b1e1*
*Author: Automated pipeline audit*
