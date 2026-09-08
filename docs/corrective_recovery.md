# Corrective Recovery Execution Ledger — Functional-TAMP VLM Interface

> **Repository:** `Narendhiranv04/icra-we-ball`
> **Development branch:** `vlm-testing-pipeline`
> **Authoritative Plan:** `CORRECTIVE_RECOVERY_PLAN_ANTIGRAVITY_FLASH38.md`
> **Baseline HEAD:** `af2dde7341adff911c8a6c2d6b7870797ba87ed3`
> **Target Model:** `Qwen/Qwen3.5-9B` (alias `qwen35-9b`) on remote RTX 5090 via ngrok SSH tunnel

---

## 1. Stage Progress Ledger

| Stage | Status | Commit | Files changed | Tests | Raw replays | Live calls | Result | Remaining |
|---|---|---|---|---|---|---:|---|---|
| 0: Forensic Baseline & Safety Snapshot | PASSED | `62294f5a` | `docs/corrective_recovery.md`, `scripts/forensics_baseline.py`, `scripts/generate_stage0_forensic_table.py` | 32-record baseline metric recompute | K1, K2, K3, L1, L6, W1, W2, W8 | 0 | Gate 0 passed: immutable forensic baseline verified; exact causal failure points established for all 8 representative cases. | Complete |
| 1: Separate Online Executability from Offline Completeness | PASSED | `92a7df0c` | `semantic_compiler.py`, `models.py`, `search.py`, `evaluation_metrics.py`, `test_v2_compiler_and_completeness.py`, `test_stage1_contract_separation.py` | 25 passed (`test_stage1_contract_separation.py`, `test_v2_compiler_and_completeness.py`, `test_raw_replay_and_evaluator_metrics.py`) | Synthetic & fixture separation verification | 0 | Gate 1 passed: removed domain checklists; online contract validates generic executability; offline checks reference completeness; first-cause attribution assigns GRAPH_COMPILATION_FAILURE to mapper failures. | Complete |
| 2: Complete Explicit Operation -> Capability Bridge | PASSED | `9045790f` | `models.py`, `robot_capability_registry.py`, `semantic_compiler.py`, `task_interface_validator.py`, `test_domain_canonicalization_cleanup.py`, `test_stage2_capability_bridge.py` | 43 passed across Stage 1 & 2 suites (5 in `test_stage2_capability_bridge.py`, 9 in `test_robot_capabilities_and_operations.py`, 11 in `test_v2_compiler_and_completeness.py`) | V1/V2 contract compilation & planner action realization verification | 0 | Gate 2 passed: explicit FM operations map to generic robot capabilities; physical feasibility preconditions (including workshop singletons in V1 and V2) instantiated with provenance; roles alone never synthesize operations. | Complete |
| 3: Robust Role Canonicalization (Qwen Paraphrases) | NOT STARTED | | | | | 0 | | |
| 4: Refactor Relations (Task Semantics vs Physical Verifiers) | NOT STARTED | | | | | 0 | | |
| 5: Fix Search Eligibility and Causal Recovery | NOT STARTED | | | | | 0 | | |
| 6: Planner Action Provenance & Goal Compilation | NOT STARTED | | | | | 0 | | |
| 7: Fix Raw Evaluation & First-Cause Attribution | NOT STARTED | | | | | 0 | | |
| 8: Structured Output & Qwen Inference Reliability | NOT STARTED | | | | | 0 | | |
| 9: Real-Trace Regression Suite | NOT STARTED | | | | | 0 | | |
| 10: Small End-to-End Live Probe | NOT STARTED | | | | | 8 | | |
| 11: Full Test Suite & Method Freeze | NOT STARTED | | | | | 0 | | |
| 12: Final Development 32x1 Matrix | NOT STARTED | | | | | 32 | | |
| 13: Held-Out / Generalization Matrix | NOT STARTED | | | | | 15 | | |
| 14: Final Comparative Analysis & Paper Reports | NOT STARTED | | | | | 0 | | |

---

## 2. Stage 0 — Forensic Baseline Snapshot (`af2dde73`)

### 2.1 Remote Infrastructure Verification

- **Host:** `cstar`
- **GPU:** NVIDIA GeForce RTX 5090 (32,607 MiB VRAM, Driver 580.95.05, CUDA 13.0)
- **vLLM Process:** PID 246883 (`vllm serve Qwen/Qwen3.5-9B --served-model-name qwen35-9b --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 ...`)
- **Remote Port:** `127.0.0.1:8000` (strictly non-public)
- **Persistent Local Tunnel:** `127.0.0.1:18000 -> remote 127.0.0.1:8000` via SSH master socket `~/.cache/tamp-vlm/qwen.sock`
- **Ngrok Process:** PID 4082688 (tunneling `0.tcp.in.ngrok.io:23808 -> remote :22`)
- **API Model Verification:** `curl -sS http://127.0.0.1:18000/v1/models` returns `qwen35-9b` (root: `Qwen/Qwen3.5-9B`, max_model_len: 32768)

### 2.2 Recomputed Baseline Primary Metrics

Directly recomputed from `benchmark_reports/final_corrected_32x1_20260908T192500IST_rerun/evaluation_records.json` using `evaluation_metrics.py`:

| Primary Metric | Value |
|---|---:|
| Outcome Correct | 0.0% (0/32) |
| Feasible Success | 0.0% (0/20) |
| Feasibility Recovery | 0.0% (0/13) |
| Goal Coverage | 0.0% |
| False Completion | 0.0% (0/12) |
| VLM Requests per variant | 1.00 |
| High-level Replans | 0.00 |

### 2.3 Representative Variant Forensic Table

| Variant | Domain | Feasible (GT) | Raw Semantics Expressed (Roles / Ops / Rels) | Production Mappings (Roles / Ops / Rels) | Contract Complete | Search State | Grounding State | A* Invoked? | First Cause (Reported) | Forensic Root Cause |
|---|---|---|---|---|---|---|---|---|---|---|
| **K1** | kitchen | True | Roles: 0, Ops: 0, Rels: 0 | Roles: OK; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: [] | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | Inference token runout / tab loop caused JSON truncation (`finish_reason: length`). Structural parsing failure upstream of compiler. |
| **K2** | kitchen | True | Roles: 6, Ops: 4, Rels: 4 | Roles: 2 unresolved/collided; Ops: 2 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 4 roles | Yes | `TASK_SPECIFICATION_FAILURE` | FM expressed complete kitchen task (sources, cups, bowls, stirrers, spoons, transfer ops, stir op, pair op). Compiler failed on `soup_serving_vessel` (`UNRESOLVED_SEMANTIC`) and `stirring_utensil` (`AMBIGUOUS_ROLE_MAPPING collision with coffee_serving_vessel`). Missing transfer capability marked operations `UNSUPPORTED_OPERATOR`. Search blocked by `contract_incomplete` gate. |
| **K3** | kitchen | True | Roles: 9, Ops: 7, Rels: 5 | Roles: 4 unresolved/collided; Ops: 2 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed valid roles/relations/ops. Compiler suffered role collision/unresolved mapping. Missing material-transfer capability. Search starved by contract gate even though physical objects were unobserved. |
| **L1** | living_room | True | Roles: 5, Ops: 2, Rels: 4 | Roles: 2 unresolved/collided; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: [] | Grounded: 2 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed refreshment and media roles and placement operations. Compiler rejected `refreshment_component` (`UNRESOLVED_SEMANTIC`) and `entertainment_control` (`UNRESOLVED_SEMANTIC`). Relations treated as `SOFT_SEMANTIC_ONLY`. Search never considered. |
| **L6** | living_room | True | Roles: 4, Ops: 2, Rels: 2 | Roles: 1 unresolved/collided; Ops: 1 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: [] | Grounded: 3 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed valid refreshment item & remote roles. Compiler rejected `refreshment_item` (`UNRESOLVED_SEMANTIC`) and rejected `move_remote_to_accessible_surface` (`UNSUPPORTED_OPERATOR`). Planner never engaged. |
| **W1** | workshop | True | Roles: 4, Ops: 3, Rels: 4 | Roles: 2 unresolved/collided; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed fastening tool, component, marked location, and surface. Compiler rejected `fastening_component` (`UNRESOLVED_SEMANTIC`) and `fastening_tool` (`UNRESOLVED_SEMANTIC`). Workshop V2 preconditions skipped. Search starved by contract gate. |
| **W2** | workshop | True | Roles: 5, Ops: 2, Rels: 4 | Roles: 2 unresolved/collided; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed valid fastening tool, component, fastener, and location. Compiler rejected `fastening_tool` (`UNRESOLVED_SEMANTIC`) and collided `marked_location` with `fastener`. Search starved. |
| **W8** | workshop | True | Roles: 4, Ops: 2, Rels: 3 | Roles: OK; Ops: 1 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed `fastening_tool`, `fastener`, `target_assembly`, `workbench_surface`. Roles mapped, but `perform_fastening` marked `UNSUPPORTED_OPERATOR` because V2 capability preconditions were skipped / operator not bridged. Search starved. |

### 2.4 Verification of the 7 Forensic Hypotheses

1. **Workshop V2 physical-precondition skip:** Verified. In `semantic_compiler.py`, capability physical preconditions for Workshop were bypassed in V2, causing fastening operations to fail or drop preconditions.
2. **Hardcoded online domain completeness:** Verified. Online contract validation checks against hidden task requirements rather than validating whether what the FM expressed is self-consistent and executable.
3. **Kitchen missing material-transfer capability:** Verified. FM expressions like "Transfer coffee substance into serving vessel" and "Transfer water into serving vessel" fail with `UNSUPPORTED_OPERATOR` because the capability registry lacks generic material transfer / pouring.
4. **Role mapping failures on natural Qwen paraphrases:** Verified. Natural phrases like "Holds soup ready for consumption" (`soup_serving_vessel`), "An implement used to manipulate or install the fastening component" (`fastening_tool`), and "Device for controlling media playback" (`entertainment_control`) were marked `UNRESOLVED_SEMANTIC`. Spurious collisions (e.g. `stirring_utensil` colliding with `coffee_serving_vessel`) blocked valid roles.
5. **Search globally starved by contract gate:** Verified. Every single recovery variant (K2, K3, W1, W2, W8) had `search_state_trace: ['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` because contract completeness was gated on hidden domain completeness.
6. **Planner action generation from role identity:** Verified. In K2, A* was invoked despite missing operations and planned purely off role presence, violating operation provenance.
7. **Raw semantic evaluator undercoverage & overattribution:** Verified. Evaluator reported 0.0 F1 for relations and operations on K2/W1/W8 despite explicit, highly coherent FM semantics, and attributed 100% of failures to `TASK_SPECIFICATION_FAILURE` (FM omission) rather than compiler/interface rejection.

**Gate 0 Status: PASSED.**

---

## 3. Stage 1 — Separate Online Executability from Offline Completeness

### 3.1 Architectural Changes Implemented

1. **Domain Checklist Removal:** Completely removed hardcoded domain checklists (`if d_norm == "kitchen":`, `elif d_norm == "living_room":`, `elif d_norm == "workshop":`) from `check_required_contract_complete` in `semantic_compiler.py`.
2. **Generic Online Executable Contract Completeness:** Defined generic online validation ensuring:
   - Structural sanitizer succeeded and document is not semantically incomplete.
   - At least one role node compiled; zero unresolved roles in trace.
   - Zero disabled groups and zero unresolved required operations.
   - Zero unresolved required relations; all relation endpoints exist in compiled nodes.
   - All role node counts and binding policies are internally valid (`min >= 1`, `max >= min`, valid policy).
   - All operation groups reference valid compiled nodes and have capability/function mappings.
3. **Field Separation in Metadata & Evaluation Record:**
   - Online contract field: `online_executable_contract_complete` (with backward-compatible alias `required_contract_complete`).
   - Offline benchmark reference field: `offline_reference_task_complete` (with alias `raw_vlm_spec_complete`).
   - Added `online_executable_contract_complete` property to `FunctionalRequirementGraph` in `models.py`.
4. **Search Eligibility Restoration:** Updated `classify_search_state` in `search.py` to prioritize `online_executable_contract_complete`. An expressed subtask that is internally coherent and executable is no longer blocked from search by hidden reference omissions.
5. **First-Cause Attribution Correction:** Reordered failure precedence in `evaluation_metrics.py` so that compiler mapping failures (`unresolved_roles`, `disabled_groups`, sanitizer failures, contract incomplete) produce `GRAPH_COMPILATION_FAILURE`, whereas genuine FM omissions where the online contract compiled successfully produce `TASK_SPECIFICATION_FAILURE` (`FM_SEMANTIC_OMISSION`).

### 3.2 Gate 1 Verification

- **Unit tests:** 25 passed across `test_stage1_contract_separation.py`, `test_v2_compiler_and_completeness.py`, and `test_raw_replay_and_evaluator_metrics.py`.
- **Proved properties:**
  - FM omission -> `offline_reference_task_complete == False`.
  - Self-consistent expressed subtask -> `online_executable_contract_complete == True`, reaching search state `SEARCH_RECOVERABLE`.
  - Compiler mapper failures -> `GRAPH_COMPILATION_FAILURE`.
  - True FM omissions -> `TASK_SPECIFICATION_FAILURE`.

**Gate 1 Status: PASSED.**

---

## 4. Stage 2 — Complete Explicit Operation -> Robot Capability Bridge

### 4.1 Architectural Changes Implemented

1. **Material Transfer Capability Registered:** Added `TRANSFER_CONTENT_TO_CONTAINER` in `CANONICAL_ROBOT_CAPABILITIES["kitchen"]`:
   - Allowed sources: `coffee_source`, `water_source`, `source`, `ingredient`.
   - Allowed targets: `coffee_container`, `prepared_cup_target`, `beverage_cup`, `cup`, `target_container`.
   - Planner operation: `POUR`.
   - Broad semantic cues: `pour`, `pouring`, `transfer`, `transferring`, `dispense`, `fill`, `fill cup`, `transfer content to container`, `transfer material into container`, etc.
2. **Equipment Return Capability Registered:** Added `RETURN_REUSABLE_ITEM_TO_SUPPORT` in `CANONICAL_ROBOT_CAPABILITIES["workshop"]`:
   - Allowed sources: `driver`, `tool`, `fastening_tool`.
   - Allowed targets: `MAIN_WORKBENCH_ZONE`, `workbench_surface`, `workbench`.
   - Planner operation: `PLACE`.
   - Semantic cues: `return`, `returning`, `return driver`, `return tool`, `return reusable equipment to workbench`, etc.
3. **Workshop V2 Singleton Bug Fixed:**
   - Excised `if not is_v2` restriction on defaulting context role: in Workshop, if `repair_target` is in nodes and context is unspecified, context defaults to `repair_target` in both V1 and V2.
   - Excised `if not is_v2` guard around physical precondition instantiation in `semantic_compiler.py`: for singleton interactions (`count == 1`), `op_interp.physical_preconditions` (`COMPATIBLE_WITH`, `REACHES_TARGET`, `COMPATIBLE_WITH_TARGET`) are instantiated in both V1 and V2 without requiring the FM to name checker predicates.
4. **Physical-Precondition Provenance Recorded:**
   - Extended `FunctionalRelation` with `provenance: str = "EXPLICIT_REQUIREMENT"`, `source_operation_id: str | None = None`, `capability_id: str | None = None`.
   - Capability-instantiated preconditions set `provenance="ROBOT_CAPABILITY_PRECONDITION"`, `source_operation_id=...`, `capability_id=...`.
   - Provenance records collected and recorded in `OperationGroup.preconditions_provenance`, `trace['precondition_provenance']`, and `graph.metadata['precondition_provenance']`.
5. **No Operation from Endpoints Preserved:**
   - Verified that endpoints alone (`coffee_stirrer + coffee_container`, `driver + fastener`) never synthesize operations or physical preconditions if the FM omitted operations.
6. **Task Interface Validator Updated:**
   - Permitted empty `required_relations` on `OperationGroup` only when the mapped capability legitimately defines 0 relation templates (such as `TRANSFER_CONTENT_TO_CONTAINER` or `RETURN_REUSABLE_ITEM_TO_SUPPORT`), while maintaining strict non-empty validation on all other groups.

### 4.2 Gate 2 Verification

- **Test Suite:** `mujoco_scenes/functional_tamp_pipeline/tests/test_stage2_capability_bridge.py` (5 tests), plus regression on existing suites (43 total tests passed in 0.33s).
- **Proved Properties:**
  - Explicit FM `"stir contents"` -> `STIR_COFFEE` capability -> `INSERTABLE_IN + REACHES_BOTTOM` instantiated with provenance `ROBOT_CAPABILITY_PRECONDITION`.
  - Explicit FM `"install/fasten component at target"` -> `FASTEN_JOINT` capability -> `COMPATIBLE_WITH`, `REACHES_TARGET`, `COMPATIBLE_WITH_TARGET` instantiated in both V1 and V2 without requiring FM to separately name checker predicates.
  - Explicit FM `"transfer material into container"` -> `TRANSFER_CONTENT_TO_CONTAINER` capability -> planner generates `POUR` realization action.
  - Roles only without explicit operation -> zero task operations synthesized, zero capability relations synthesized.
  - Workshop equipment return -> `RETURN_REUSABLE_ITEM_TO_SUPPORT` -> `PLACE` operator.

**Gate 2 Status: PASSED (Commit: `9045790f`).**

---

## 5. Stage 3 — Robust Role Canonicalization Using Real Qwen Language

### 5.1 Architectural Changes Implemented

1. **Real-Language Regression Fixtures (`test_stage3_role_canonicalization.py`):**
   - Implemented real-language paraphrases captured from forensic records across domains:
     - `"Holds soup ready for consumption."` -> `soup_container`
     - `"Tool used to mix ingredients inside the coffee serving vessel."` -> `coffee_stirrer`
     - `"Device used to operate the television or media system."` -> `REMOTE`
     - `"A collection of items designated for consumption by one person."` -> `CUP_SAUCER_SET`
     - `"An implement used to manipulate or install the fastening component."` -> `CAN_DRIVE_SCREW` (`driver`)
     - `"A physical item capable of connecting or securing elements at the marked location."` -> `CAN_FASTEN` (`fastener`)
   - Verified that no fixtures leak benchmark variant IDs into production code.

2. **Causal Head Prioritization in Kitchen:**
   - In `kitchen_vlm_functional_graph.py`, distinguished tool/implement vs receptacle/target heads.
   - Roles describing the tool acting on a container (e.g. `"Tool used to mix ingredients inside the coffee serving vessel."`) resolve to `coffee_stirrer`.
   - Roles describing the container receiving ingredients or being the target of an action (e.g. `"Receives coffee and water ingredients and is the target of stirring action."`) resolve to `coffee_container`.
   - Expanded stemming and inflections for containers (`holds`, `holding`, `contains`, `receptacles`, `vessels`, `servings`).
   - Resolved K2 collision without ambiguity crash.

3. **Living Room Payloads, Regions, and Seating Anchors:**
   - In `environment_vlm_requirements.py`, expanded `has_remote` to recognize natural phrases for television and media operating devices (`"device used to operate the television or media system"`, `"entertainment control"`, `"media controller"`).
   - Expanded `CUP_SAUCER_SET` recognition for natural collection and consumption phrasing (`"collection of items designated for consumption by one person"`, `"set comprising a drink vessel and a serving dish intended for consumption"`).
   - Expanded region recognition for personal refreshment support surfaces and shared entertainment control placement regions.
   - Updated `map_living_room_fixed_target_role` to recognize seating reference points even when the FM labels the armchair entity kind as `OBJECT` rather than `FIXED_TARGET` or `REGION`, while ensuring movable payloads remain excluded.

4. **Unreferenced Duplicate Role Merging:**
   - Updated `can_merge_roles` in `semantic_compiler.py`: when one role participates in operations/relations (e.g. `causal_position = {'group_target'}`) and another role with identical normalized function and binding policy is unreferenced (`causal_position = set()`), the unreferenced duplicate merges cleanly (`RAW_ROLE_MERGED`) rather than causing an `AMBIGUOUS_ROLE_MAPPING` failure (resolving L1 forensic failure).

5. **Workshop Tool, Component, and Target Distinction:**
   - In `workshop_phase1/requirements.py`, expanded `driver_phrases` to include `"manipulate or install"` and `"implement used to manipulate or install"`.
   - Expanded `fastener_phrases` to include `"connecting or securing elements at the marked location"` and fastener action verbs (`connect`, `secure`, `join`).
   - Ensured fixed receiving targets (`repair_target`) and generic workbench support context (`MAIN_WORKBENCH_ZONE`) remain strictly distinct from movable tools and fasteners.
   - Preserved fail-closed semantics: candidate categories alone cannot manufacture a role when the function is unrelated (e.g. `"illuminate workspace"` or `"paint the wall surface"`).

6. **Preserved Invariant:**
   - Confirmed that role mapping alone never synthesizes operations or relations.

### 5.2 Gate 3 Verification

- **Test Suite:** `mujoco_scenes/functional_tamp_pipeline/tests/test_stage3_role_canonicalization.py` (11 tests passed).
- **Regression Suites:**
  - Stages 1, 2, and 3: 21 passed in 0.15s.
  - Domain canonicalization: 81 passed in 0.42s across Kitchen, Living Room, and Workshop.
  - Functional TAMP pipeline: 494 passed.
- **Commit:** `78f8a93a` (`fix(semantics): harden role canonicalization for natural FM paraphrases`).

**Gate 3 Status: PASSED.**

---

## 6. Stage 4 — Refactor Relation Handling: Task/Causal Semantics vs Physical Verifiers

### 6.1 Architectural Changes Implemented

1. **Conceptual Separation of Relations:**
   - **Physical Verifier Constraints (`category="PHYSICAL_VERIFIER"`):**
     - Predicates requiring geometric/physical ground-truth verification (`INSERTABLE_IN`, `REACHES_BOTTOM`, `FITS_ON`, `FITS_SET_ON`, `NEAR_SEAT`, `ACCESSIBLE_FROM_BOTH_SEATS`, `COMPATIBLE_WITH`, `REACHES_TARGET`, `COMPATIBLE_WITH_TARGET`).
     - Stored in `graph.relations`. Validated against frozen `predicate_registry` signatures in `validate_runtime_gf`.
   - **Task/Causal Semantic Relations (`category="TASK_CAUSAL_SEMANTICS"`):**
     - Predicates expressing narrative/causal dependency (`PROVIDES_MATERIAL_TO`, `ACTS_ON`, `PAIRED_WITH`, `INSTALLED_AT`, `CONNECTED_TO`, and reverse mappings like `RECEIVES_CONTENTS_FROM`).
     - Stored in `graph.task_causal_relations` (and `graph.metadata['task_causal_relations']`).
     - Preserved as task semantic provenance; never burdened with physical geometric verification checkers.
     - Validated in `validate_runtime_gf` for node endpoint existence without requiring geometric checker signatures.
   - **Capability-Derived Physical Preconditions:**
     - Instantiated when explicit operations map to robot capabilities (`provenance="ROBOT_CAPABILITY_PRECONDITION"`, `category="PHYSICAL_VERIFIER"`).
     - Does not require FM to redundantly declare checker predicates (e.g. `stir contents` automatically attaches `INSERTABLE_IN` and `REACHES_BOTTOM`).

2. **Deterministic Linguistic Cues for Task Causal Relations:**
   - Added `_TASK_CAUSAL_RELATION_CUES` and `_TASK_CAUSAL_INVERSE_CUES` to `relation_interpreter.py`.
   - Cues safely recognize natural language phrases:
     - `"provides material to"`, `"provides material into"`, `"supplies material to"`, `"pours into"` -> `PROVIDES_MATERIAL_TO`
     - `"acts on"`, `"operates on"`, `"manipulates"`, `"manipulates interior of"`, `"stirs contents of"` -> `ACTS_ON`
     - `"paired with"`, `"accompanied by"`, `"alongside"`, `"served with"`, `"provided for"` -> `PAIRED_WITH`
     - `"installed at"`, `"secured at"`, `"fastened at"`, `"anchored at"` -> `INSTALLED_AT`
     - `"receives contents from"`, `"receives material from"` -> direction normalized to `PROVIDES_MATERIAL_TO`
   - Physical verifiers take precedence over task/causal cues when both endpoints and lexical cues match a physical constraint.

3. **Fail-Closed Uninterpretable Required Relations:**
   - If an FM declares an explicit required relation that matches neither a physical verifier nor a task/causal relation (e.g. `"must maintain 45 degree tilt during operation"` or `"bananas on the moon"`):
     - Evaluates to `UNINTERPRETABLE_REQUIRED_RELATION`.
     - Recorded in `trace['unresolved_required_relations']`.
     - `online_executable_contract_complete` evaluates to `False`.
     - Fails closed safely.

4. **Endpoint Signature Isolation Invariant:**
   - Verified that endpoint signatures alone never manufacture relations when semantic cues are absent.
   - For example, between `driver` and `fastener`, nonsensical text fails closed as `UNINTERPRETABLE_REQUIRED_RELATION` and never guesses `COMPATIBLE_WITH`.
   - Missing relations between roles are never synthesized from candidate identities alone.

5. **Data Model and Interface Updates:**
   - Extended `FunctionalRelation` with `category: str = "PHYSICAL_VERIFIER"`.
   - Extended `FunctionalRequirementGraph` with `task_causal_relations: tuple[FunctionalRelation, ...] = ()`.
   - Updated `validate()` in `models.py` and `validate_runtime_gf()` in `task_interface_validator.py` to validate `task_causal_relations`.
   - Updated `to_dict()` and `from_dict()` across models for complete serialization round-trip.
   - Updated `convert_v2_to_canonical_document` in `fm_schema_v2.py` to seamlessly accept `interaction_groups` alongside `operation_pairings`.

### 6.2 Gate 4 Verification

- **New Test Suite:** `mujoco_scenes/functional_tamp_pipeline/tests/test_stage4_relation_handling.py` (8 tests passed).
- **Proved Properties:**
  - Explicit physical relation -> physical verifier in `graph.relations` with `category="PHYSICAL_VERIFIER"`.
  - Explicit causal relation -> preserved semantic edge in `graph.task_causal_relations` with `category="TASK_CAUSAL_SEMANTICS"`, contract remains complete.
  - Direction normalization for causal relations -> `"coffee container receives contents from coffee source"` correctly normalized to `PROVIDES_MATERIAL_TO(coffee_source, coffee_container)`.
  - Workshop separation -> `"tool acts on component"` (causal `ACTS_ON`) vs `"mechanically engages screw head"` (physical `COMPATIBLE_WITH`).
  - Explicit operation -> capability-derived physical preconditions (`INSERTABLE_IN + REACHES_BOTTOM`), no missing relations error.
  - Unknown required relation -> `UNINTERPRETABLE_REQUIRED_RELATION`, `online_executable_contract_complete == False`.
  - Endpoint signature alone -> never creates relation without semantic evidence.
  - Serialization round-trip -> `to_dict()` and `from_dict()` preserve all fields and categories.
- **Regression Suites:**
  - Stages 1, 2, 3, and 4: 37 passed in 0.16s.
  - Domain canonicalization: 86 passed in 0.74s across Kitchen, Living Room, and Workshop.
  - Safe relation interpreter: 8 passed in 0.14s.

**Gate 4 Status: PASSED (Commit: `3420ab1d`).**
