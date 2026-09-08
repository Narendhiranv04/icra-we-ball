# Final Approach Performance Optimization Plan — Revised End-to-End Implementation Plan

> **Document Status:** IMPLEMENTATION-READY DESIGN
>  
> **Supersedes:** `final_fix.md`
>  
> **Repository:** `Narendhiranv04/icra-we-ball`
>  
> **Development Branch:** `vlm-testing-pipeline`
>  
> **Audited Baseline HEAD:** `8d93afda9490730125a247c257c32f433eba48c2`
>  
> **Purpose:** Remove generic engineering/interface bottlenecks, preserve the intended Functional-TAMP architecture, and maximize end-to-end benchmark performance without GT leakage, per-variant logic, hidden semantic completion, extra VLM replanning, or relaxed verification.

---

## Implementation Execution Progress

| Phase | Status | Files changed | Tests | Replay cases | Result | Commit | Remaining issue |
|---|---|---|---|---|---|---|---|
| Phase 0: Baseline & Evaluator | PASSED | `scripts/evaluate_vlm_functional_tamp.py`, `evaluation_metrics.py`, `raw_semantic_evaluation.py`, `run.py`, `audit.py` | 10 passed (`test_raw_replay_and_evaluator_metrics.py`) | Baseline replay records (W1, W8, K1, K2) | Gate 0 passed; baseline metrics recompute exact; causal search recovery and evidence-based first cause verified | `4e3c05e0` | None |
| Phase 1: Runtime Ontology Isolation | PASSED | `configs/runtime_functional_semantic_ontology.yaml`, `role_semantic_ontology.py`, `run.py`, `audit.py`, `test_runtime_ontology_isolation.py` | 375 passed (all tests in suite) | W1, W2 regression fixtures | Gate 1 passed; runtime ontology isolated; GT provider leakage guarded; zero reference task config access | `6093192e` | None |
| Phase 2: V2 Schema & Prompt | PASSED | `fm_schema_v2.py`, `fm_adapter.py`, `structural_sanitizer.py`, `audit.py`, `run.py`, `test_v2_schema_and_prompt.py` | 16 passed (`test_v2_schema_and_prompt.py`), 391 passed (suite) | W1, W2, K1, K2 regression fixtures, V1/V2 dual-routing | Gate 2 passed; generic V2 schema validated; zero prompt leakage; V1 replay backward-compatibility preserved | `85f326d5` | None |
| Phase 3: Safe Relation Interpreter | PASSED | `predicate_registry.py`, `relation_interpreter.py`, `semantic_compiler.py`, `test_safe_relation_interpreter.py` | 8 passed (`test_safe_relation_interpreter.py`), 6 passed (`test_p3i_full_ideal_convergence.py`) | W1, W2, K1, L1 ideal convergence records | Gate 3 passed; endpoint-only fails closed; nonsensical relations rejected; multi-predicate & direction-normalized compilation verified; W1/W2 ideal convergence intact | `afecadb9` | None |
| Phase 4: Operation Interpreter & Capabilities | PASSED | `robot_capability_registry.py`, `semantic_compiler.py`, `grounding.py`, `models.py`, `audit.py`, `run.py`, `test_robot_capabilities_and_operations.py` | 9 passed (`test_robot_capabilities_and_operations.py`), 94 passed (suite) | W1, W2, K1, L1 regression fixtures | Gate 4 passed; no endpoint-only operation inference; explicit operations derive physical preconditions; empty/unresolved operations cannot gain predicates; grounding fallback removed; capability registry hash in manifest | `7e29c96e` | None |
| Phase 5: V2 Compiler Integration | PASSED | `semantic_compiler.py`, `models.py`, `test_v2_compiler_and_completeness.py` | 11 passed (`test_v2_compiler_and_completeness.py`), 60 passed (suite) | V2 Kitchen/Living/Workshop fixtures, W1/K1/L1 regression records | Gate 5 passed; complete V2 fixtures compile to executable G_F; omitted relation/operation fails closed with contract_incomplete; zero fallback synthesis | `daf28062` | None |
| Phase 6: Domain Canonicalization Cleanup | PASSED | `semantic_compiler.py`, `run.py`, `test_domain_canonicalization_cleanup.py` | 5 passed (`test_domain_canonicalization_cleanup.py`), 65 passed (suite) | W1, W2, K1, L1 regression records | Gate 6 passed; zero variant IDs in production mapping; repair target maps robustly from imperfect entity kinds; missing roles remain missing; target anchor provenance recorded | `a3e69202` | None |
| Phase 7: Physical Verifier & Grounding Audit | PASSED | `run.py`, `test_physical_verifier_and_grounding_audit.py` | 6 passed (`test_physical_verifier_and_grounding_audit.py`), 71 passed (suite) | W1, W2, K1, L1 regression records | Gate 7 passed; verification trace emitted; verifiers fail closed to UNKNOWN; UNKNOWN never TRUE; FALSE strictly rejects; detectors never prove geometry | `41a1006d` | None |
| Phase 8: Search Eligibility & Causal Recovery | PASSED | `search.py`, `domains/kitchen.py`, `models.py`, `run.py`, `test_search_eligibility_and_recovery.py` | 11 passed (`test_search_eligibility_and_recovery.py`), 126 passed (suite) | Kitchen/Workshop regression records, K1/W1 fixtures | Gate 8 passed; classify_search_state classifies all 5 states; contract-incomplete Kitchen fixture performs zero pointless search; visible invalid candidate still searches; before/after causal recovery metric verified | `ce36863c` | None |
| Phase 10: Full 32-Variant Archived V1 Raw Replay | PASSED | `semantic_compiler.py`, `grounding.py`, `sequential_inspection.py`, `task_witness.py`, `domains/kitchen.py`, `domains/living_room.py`, `run.py` | 32/32 raw replay matrix | `benchmark_reports/post_backend_fix_v1_raw_replay_20260908_phase10/` | Gate 10 passed; 32/32 completed with zero pipeline exceptions; 0 live VLM calls; W1/W2 preserved with 100% full-task success; false completion = 0.0%; bounded combinatorial optimization verified | `5140e839` | None |
| Phase 11: Small Live V2 Probe & Thinking Config | PASSED | `fm_adapter.py`, `vlm_spec_provider.py`, `scripts/evaluate_vlm_functional_tamp.py` | 6-variant comparative live probe (`K1, K3, L1, L6, W1, W3`) | Thinking OFF (`benchmark_reports/live_p11_probe_thinking_off/`) vs Thinking ON (`benchmark_reports/live_p11_probe_thinking_on/`) | Gate 11 passed; Thinking ON suffers 50% length truncations (8192 reasoning tokens) and ~93s latency; Thinking OFF produces 100% stable, schema-valid output with zero length failures and ~30s latency; frozen configuration: `enable_thinking=false` | `0d61a269` | None |
| Phase 12: Full Test Suite & Method Freeze | PASSED | Full codebase frozen | 448 passed (pipeline suite), W1/W2 raw replay passed | `benchmark_reports/w1_w2_gate12_check/` | Gate 12 passed; 448/448 tests green; source tree clean; W1/W2 100% full-task success; all 5 configuration hashes frozen | `dd78def4` | None |
| Phase 13: Final 32x1 Development Matrix | PASSED | None (frozen code) | Full 32x1 live development matrix | `benchmark_reports/final_post_optimization_32x1_20260908/` | Gate 13 passed; 32 unique variants; invariants VALID; 0 pipeline exceptions; vlm_requests = 1.00; replans = 0.00; false completion = 0.0% | `800374a2` | None |
| Phase 14: Held-Out Generalization Matrix | PASSED | `mujoco_scenes/configs/*_heldout_variants.yaml`, `scene_loader.py`, `workshop_scene.py`, `living_room_*.py`, `evaluate_heldout_matrix.py` | 15/15 held-out live matrix | `benchmark_reports/held_out_generalization_15x1_20260908/` | Gate 14 passed; 15 unseen variants (9 feasible, 6 infeasible); invariants VALID; 0 exceptions; vlm_requests = 1.00; replans = 0.00; false completion = 0.0%; all 6 hashes identical | `c37d2b3d` | None |
| Phase 15: Offline Analysis & Paper Tables | PASSED | `scripts/generate_phase15_analysis.py`, `docs/final_fix.md` | Phase 15 audit & comparative analysis suite | `benchmark_reports/phase15_final_offline_analysis/` | Gate 15 passed; all 10 analysis deliverables generated; comparative tables (Dev 32 vs Held-Out 15); prompt leakage audit 100% clean (0/47 leaks); fail-closed verification validated | `d8d74f28` | None |

---

## 0. Non-Negotiable Scientific Contract

The implementation MUST preserve all of the following.

| Constraint | Required value |
|---|---|
| Online FM/VLM calls | **Exactly one per variant** |
| High-level replans | **Zero** |
| Planner | **One A\*** invocation after grounding, with honest partial-plan behavior allowed |
| Physical execution | **Not required**; evaluation stops at independently validated action sequence |
| Search | **Automatic only**, triggered by runtime incompleteness that additional observations can plausibly resolve |
| GT/reference specifications online | **Forbidden** |
| Expected plan lookup | **Forbidden** |
| Hidden object locations | **Forbidden** |
| Per-variant rules | **Forbidden** |
| VLM-facing closed relation enum | **Forbidden** |
| Compiler invention of missing task semantics | **Forbidden** |
| Verification weakening | **Forbidden** |
| Grounding logic | **TRUE / FALSE / UNKNOWN**, with UNKNOWN never treated as TRUE |
| Benchmark development set | The 32 existing variants are **development / post-fix confirmation**, not untouched held-out generalization |

### 0.1 Strong semantic-boundary rule

The following sentence is the controlling anti-leakage rule for the entire implementation:

> **Neither endpoint identity, canonical role identity, environment ontology, fixed anchors, robot capability registries, nor planner action models may create a task participant, task relation, or task operation that was not semantically expressed by the FM. They may only canonicalize an expressed semantic, provide task-explicit fixed context, select a compatible physical verifier, or provide the physical preconditions of an explicitly expressed robot operation.**

If any implementation decision violates this sentence, reject the change even if benchmark performance improves.

### 0.2 What the system MAY know online

The online system may know:

- the user task instruction;
- current RGB/RGB-D observations;
- generic domain runtime ontology and detector aliases;
- robot action capabilities;
- geometric verifier definitions and thresholds;
- environment-owned fixed non-selectable context explicitly required by the task;
- known inspectable storage-region identities in the environment;
- planner context constants.

The online system may NOT know:

- GT required-role graph for the current task;
- GT relation graph;
- GT operation groups;
- feasibility label;
- expected assignment;
- expected plan;
- hidden-object location;
- variant-specific corrective mapping.

---

## 1. Audited Baseline Snapshot

### 1.1 Current canonical task instructions

**Kitchen**

> Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil.

**Living Room**

> Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people.

**Workshop**

> Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench.

These instructions are the only benchmark task semantics that the FM receives online.

### 1.2 Benchmark catalogue

| Domain | Total | Feasible | Infeasible |
|---|---:|---:|---:|
| Kitchen | 12 | 6 | 6 |
| Living Room | 10 | 6 | 4 |
| Workshop | 10 | 8 | 2 |
| **Total** | **32** | **20** | **12** |

Formal recovery set:

- Kitchen: K2-K6
- Living Room: none
- Workshop: W1-W8
- Total recovery variants: **13**

### 1.3 Verified baseline primary metrics

From the attached `evaluation_summary.json` for the current live matrix:

| Metric | Baseline |
|---|---:|
| Outcome Correct | **6.25% (2/32)** |
| Feasible-task Success | **10.0% (2/20)** |
| Feasibility Recovery | **15.38% (2/13)** |
| Goal Coverage | **20.0%** |
| False Completion | **0.0%** |
| VLM Requests | **1.00** |
| High-level Replans | **0.00** |

### 1.4 Baseline domain diagnostics worth preserving

| Metric | Kitchen | Living Room | Workshop |
|---|---:|---:|---:|
| Raw role recall | 98.6% | 43.3% | 80.0% |
| Raw role F1 | 98.6% | 50.1% | 74.3% |
| Complete candidate grounding | 0.0% | 55.6% | 90.0% |
| Non-empty plan rate | 91.7% | 10.0% | 100.0% |
| Full feasible-task success | 0.0% | 0.0% | 25.0% |
| Mean regions inspected | 5.00 | 0.00 | 1.90 |

### 1.5 Baseline metrics that MUST NOT be frozen as final paper definitions

The current implementation/reporting has known definition issues. Before using them in the paper, Phase 0 must fix or rename:

- `Runtime contract coverage`;
- `Raw relation F1` if it uses the production relation compiler;
- `Search recovery success` if it only means “search happened and final grounding is complete”;
- first-cause labels produced by the current precedence order;
- any manual narrative that can disagree with `evaluation_records.json`.

No later phase may compare against a metric whose definition silently changed.

---

## 2. Current Failure Diagnosis

### 2.1 What is already working

The core architecture is not the problem:

```text
Task instruction + initial RGB
            |
      ONE FM/VLM call
            |
      task semantics
            |
      deterministic compiler
            |
           G_F
            |
     observations -> G_O
            |
semantic + geometric grounding
            |
           phi*
            |
        single A*
            |
independent symbolic replay
```

Workshop W1/W2 demonstrate that the complete route can succeed.

### 2.2 Main current bottlenecks

**Kitchen**

- FM usually expresses the six major roles.
- Relations and operation/group semantics are incomplete or semantically vague.
- Current canonicalization over-relies on aliases and endpoint structure.
- Kitchen then searches all five regions even when the semantic contract itself is incomplete.

**Living Room**

- FM often emits destination REGION roles but omits movable OBJECT payloads and contextual anchors.
- This is primarily an FM contract/decomposition issue, not a planning issue.

**Workshop**

- W1/W2 succeed.
- Several variants omit or mis-express the fixed repair target or target relations.
- W8 should be re-audited under the corrected semantic-vs-compiler first-cause rule rather than being hard-coded as a planning failure.

### 2.3 Important distinction for all later debugging

There are three fundamentally different problems:

```text
A. FM DID NOT EXPRESS the required semantic
   -> Task Specification Failure

B. FM DID EXPRESS a reasonable semantic,
   but compiler/runtime could not represent it
   -> Graph Compilation / Interface Failure

C. Executable G_F is complete,
   but current G_O cannot provide a valid physical grounding
   -> Search / Discovery / Assignment problem
```

Never use search to hide A or B.
Never use compiler synthesis to hide A.

---

## 3. Final Target Architecture

The final architecture to implement is:

```text
Task instruction + initial RGB
              |
              v
        ONE Qwen call
              |
      +-------+--------+
      |                |
 TASK CONTRACT    OBSERVATION GUIDANCE
      |
      |-- functional roles
      |-- free-form functional relations
      |-- free-form operation pairings
      |-- counts / binding / reuse
      |
      v
 Structural sanitizer
      |
      v
 Role canonicalization
      |
      +-----------------------------+
      |                             |
      v                             v
Relation semantic interpreter   Operation semantic interpreter
      |                             |
      |                        explicit robot capability
      |                             |
      |                   capability physical preconditions
      |                             |
      +---------------+-------------+
                      |
                      v
                     G_F
      task/causal semantics + executable
      physical feasibility requirements
                      |
                      v
                     G_O
      observations + measured geometry
                      |
                      v
          physical verifier layer
      TRUE / FALSE / UNKNOWN evidence
                      |
                      v
                     phi*
                      |
        if incomplete because additional
       observations may reveal alternatives
                      |
                automatic search
                      |
                  grow G_O
                      |
                      v
                     phi*
                      |
                      v
                  single A*
                      |
                      v
          independent symbolic replay
```

### 3.1 Layer definitions

**Layer 1: FM-expressed task semantics**

What the task requires, expressed by Qwen in its own short phrases.

**Layer 2: deterministic semantic interpretation**

Map an already-expressed semantic to supported runtime ontology/capabilities.

**Layer 3: physical verification**

Measure the actual scene and evaluate whether a candidate binding physically satisfies the selected verifier predicates.

### 3.2 Critical rule

Layer 2 may translate semantics.
Layer 2 may not invent semantics.

---

## 4. Online / Offline Information Separation

### 4.1 Mandatory source split

| Information source | Online generation? | Offline evaluation? |
|---|---:|---:|
| Task instruction | Yes | Yes |
| Initial/current observations | Yes | Yes |
| Runtime semantic ontology | Yes | Yes |
| Robot capability registry | Yes | Yes |
| Predicate registry/checkers | Yes | Yes |
| System fixed-anchor registry | Yes, with provenance | Yes |
| GTSpecProvider | **No** | Yes |
| Reference role/relation/group graph | **No** | Yes |
| Feasibility label | **No** | Yes |
| Expected plan | **No** | Yes |
| Hidden-object location | **No** | No online use |

### 4.2 Runtime semantic ontology isolation

Current runtime semantic-role code reads reviewed task YAMLs that also contain reference/task-specific blocks. Even if it currently extracts only category acceptance, this is too coupled for the desired zero-GT-leakage standard.

Create a dedicated runtime-only ontology file:

```text
mujoco_scenes/configs/runtime_functional_semantic_ontology.yaml
```

This file may contain only:

- canonical runtime role IDs;
- semantic physical categories accepted by each runtime role;
- detector synonyms/aliases;
- entity-kind expectations if needed.

It MUST NOT contain:

- task goals;
- expected counts for benchmark tasks;
- expected task relations;
- operation groups;
- expected plans;
- feasibility labels;
- variant IDs;
- hidden locations.

Update:

```text
mujoco_scenes/functional_tamp_pipeline/role_semantic_ontology.py
```

to read only the runtime-only ontology during online execution.

### 4.3 Runtime import firewall

Add tests that fail if live VLM pipeline execution imports or calls:

```text
GTSpecProvider
raw_semantic_evaluation
reference task fixtures
```

The evaluator may import them after the run has completed.

---

## 5. V2 FM Output Contract

The V2 response remains a single JSON response from exactly one Qwen call.

### 5.1 Top-level structure

```json
{
  "status": "SUPPORTED",
  "task_summary": "...",
  "task_contract": {
    "functional_roles": [],
    "functional_relations": [],
    "operation_pairings": []
  },
  "observation_guidance": {
    "visible_candidates_per_role": {},
    "inspectable_regions": [],
    "inspection_order": []
  },
  "unsupported_reason": ""
}
```

### 5.2 Functional roles

Each role:

```json
{
  "id": "role_0",
  "entity_kind": "OBJECT",
  "function": "...",
  "required_count": 1,
  "binding_policy": "DISTINCT",
  "candidate_categories": ["..."],
  "required_properties": ["..."]
}
```

Allowed `entity_kind`:

- `OBJECT`
- `REGION`
- `FIXED_TARGET`

Allowed `binding_policy`:

- `DISTINCT`
- `REUSABLE`
- `SHARED`

The compiler MUST NOT override `required_count` or `binding_policy` using benchmark-specific rules.

### 5.3 Functional relations — free-form and required by default

```json
{
  "id": "rel_0",
  "subject_role": "role_1",
  "relation": "...",
  "object_role": "role_2",
  "required": true
}
```

Rules:

- `relation` is free-form text.
- There is no VLM-facing relation enum.
- All task-critical relations default to `required: true`.
- A required relation that cannot be interpreted remains an unresolved required semantic and blocks executable-contract completeness.
- Optional descriptive relations may be recorded as soft evidence, but they may never prove task completion.

### 5.4 Operation pairings — free-form operation semantics retained

Do **not** remove the FM's operation semantics and infer the operation from endpoints alone.

Use:

```json
{
  "id": "op_0",
  "operation": "...",
  "source_role": "role_3",
  "target_role": "role_0",
  "operation_count": 2,
  "reuse_policy": "REUSABLE_ACROSS_TARGETS",
  "anchor_role": "role_7"
}
```

Allowed `reuse_policy`:

- `DEDICATED_PER_TARGET`
- `REUSABLE_ACROSS_TARGETS`

`anchor_role` is optional.

The `operation` field is free-form. Qwen is not shown internal planner action names.

### 5.5 Observation guidance

```json
{
  "visible_candidates_per_role": {
    "role_0": [
      {
        "label": "...",
        "visual_description": "..."
      }
    ]
  },
  "inspectable_regions": [
    {
      "id": "region_0",
      "description": "..."
    }
  ],
  "inspection_order": ["region_0"]
}
```

Observation guidance may only refer to roles already declared in `task_contract`.

---

## 6. Final Qwen Prompt Design

### 6.1 Prompt principle

Qwen should be guided about **structure and completeness**, not about the benchmark's relation vocabulary or the robot's checker vocabulary.

### 6.2 Required prompt content

The prompt should say, generically:

1. Derive all task-required physical participants from the instruction before considering visibility.
2. A role remains required even if no current candidate is visible.
3. Represent physically distinct causal participants separately when they perform different functions.
4. Use `OBJECT`, `REGION`, and `FIXED_TARGET` consistently.
5. Propagate explicit cardinality, distinctness, shared-use, and reuse requirements.
6. Express task-critical binary dependencies as short atomic free-form relations in the model's own words.
7. Express each task-required physical transformation/intervention as a short free-form `operation` pairing between declared roles.
8. After the task contract is complete, use RGB only for candidate visibility and search guidance.
9. Never produce a linear action sequence.
10. Never assign simulator/object instance IDs to functional roles.

### 6.3 Content that MUST be removed from the prompt

The final prompt must not contain:

- benchmark task nouns or examples;
- Kitchen/Living/Workshop-specific examples;
- relation examples such as insertion, reach, proximity, support, interface compatibility, etc.;
- canonical runtime predicate names;
- descriptions of the robot verifier capability list;
- expected role names;
- expected group structures;
- expected object counts for these tasks;
- variant identifiers.

### 6.4 Prompt anti-leakage test

Create a test that scans the final prompt and fails if it contains any banned benchmark/domain-specific term list maintained only in tests.

The test is a guardrail; do not solve it by simply hiding the same information in another online prompt/config.

---

## 7. Backend Relation Semantic Interpreter

### 7.1 Purpose

Translate an already-expressed free-form relation into zero, one, or multiple physical verifier requirements.

### 7.2 The key correction

**Endpoint identity is a filter/disambiguator, never the source of relation meaning.**

Wrong:

```text
Only one predicate is valid for these endpoints
=> select it regardless of relation text
```

Correct:

```text
Free-form relation text
        |
        v
Semantically plausible verifier predicates
        |
        INTERSECT
        |
Predicates legal for canonical endpoint signatures
        |
        v
0 candidates  -> unresolved required relation
1 candidate   -> select
>1 candidates -> require semantic disambiguation or conservative unresolved state
```

### 7.3 Formal interpretation rule

Let:

- `S(e)` = verifier predicates semantically supported by expressed relation `e`;
- `P(a,b)` = verifier predicates valid for canonical endpoint pair `(a,b)`.

Then:

```text
C(e,a,b) = S(e) INTERSECTION P(a,b)
```

Automatic compilation is legal only when semantic evidence exists.

A unique endpoint predicate with no semantic support is NOT enough.

### 7.4 Multi-predicate relations

The interpreter must support a relation whose text explicitly expresses multiple physical requirements.

Return a structured result:

```python
@dataclass(frozen=True)
class RelationInterpretation:
    raw_relation: str
    canonical_subject: str
    canonical_object: str
    predicates: tuple[str, ...]
    status: str
    semantic_evidence: tuple[str, ...]
    direction_normalized: bool
```

Possible status values:

```text
EXACT_CANONICAL_MATCH
LEXICAL_SEMANTIC_MATCH
MULTI_PREDICATE_MATCH
DIRECTION_NORMALIZED
UNMAPPABLE_REQUIRED_RELATION
SOFT_OPTIONAL_RELATION
AMBIGUOUS_RELATION
```

### 7.5 Semantic matching

Initial implementation should remain deterministic and conservative:

1. Normalize case, punctuation, underscores/hyphens, and simple inflections.
2. Use frozen internal linguistic cues / existing reviewed alias families only to propose semantic candidates.
3. Use predicate descriptions in `predicate_registry.py` as backend semantic descriptions.
4. Intersect semantic candidates with endpoint-valid predicates.
5. Require a unique or explicitly multi-concept result.
6. If uncertain, return unresolved rather than guessing.

Do not add a second generative LLM call.

### 7.6 Direction normalization

Direction may be normalized only when semantic evidence licenses inversion.

Examples of legal conditions:

- relation language is symmetric/direction-neutral;
- relation interpreter explicitly recognizes an inverse construction;
- swapped ordering is the unique semantically supported valid signature.

Never blindly swap endpoints solely because the reversed pair validates.

### 7.7 Required relation failure behavior

If `required == true` and no safe interpretation exists:

```text
UNMAPPABLE_REQUIRED_RELATION
```

The relation remains in the trace and the executable contract is incomplete.

Do not silently demote it to soft evidence.

### 7.8 Predicate endpoint registry

Use the existing frozen canonical predicate registry as physical capability/signature knowledge.

Examples of current runtime-valid pairs include:

**Kitchen**

- implement role -> container role for `INSERTABLE_IN`
- implement role -> container role for `REACHES_BOTTOM`

**Living Room**

- personal support REGION -> payload set OBJECT for `FITS_SET_ON`
- shared support REGION -> control OBJECT for `FITS_ON`
- personal support REGION -> seating FIXED_TARGET for `NEAR_SEAT`
- shared support REGION -> seating-pair FIXED_TARGET for shared accessibility

**Workshop**

- driver OBJECT -> fastener OBJECT for tool-component compatibility
- driver OBJECT -> repair target FIXED_TARGET for reach
- fastener OBJECT -> repair target FIXED_TARGET for component-target compatibility

These are **backend robot verifier signatures**, not VLM-facing relation definitions.

---

## 8. Backend Operation Semantic Interpreter and Robot Capability Registry

### 8.1 Why this is required

The previous plan removed the operation phrase and inferred operations directly from canonical endpoint pairs. That is too aggressive.

The FM must explicitly express the operation semantics.

The backend may then map the expressed operation to a robot capability.

### 8.2 New runtime capability registry

Create:

```text
mujoco_scenes/functional_tamp_pipeline/robot_capability_registry.py
```

Each capability should define:

```python
@dataclass(frozen=True)
class RobotCapability:
    domain: str
    capability_id: str
    semantic_description: str
    allowed_source_roles: tuple[str, ...]
    allowed_target_roles: tuple[str, ...]
    allowed_anchor_roles: tuple[str, ...]
    required_relation_templates: tuple[...]
    planner_operation: str
```

### 8.3 Capability interpretation rule

Exactly like relation interpretation:

```text
FM operation text
      |
      v
semantically plausible runtime capabilities
      |
      INTERSECT
      |
capabilities legal for expressed canonical participants
      |
      v
unique capability -> compile
otherwise -> unresolved operation
```

Endpoint roles may filter.
Endpoint roles may not create an operation that the FM did not express.

### 8.4 Physical preconditions may come from the robot capability

This is allowed and important.

The FM decides **that an operation is required**.
The robot capability model knows **what physical conditions are needed to execute that operation**.

For example, once an explicit stirring-like operation has been mapped to the runtime stirring capability, the robot capability may require both insertion-fit and sufficient reach checks.

That does **not** count as inventing task semantics because those predicates are execution feasibility preconditions of an already-expressed operation.

### 8.5 Remove hidden grounding fallback

Current grounding contains a default behavior equivalent to:

```python
required_relations = grp.required_relations or ("INSERTABLE_IN", "REACHES_BOTTOM")
```

Remove this domain-specific fallback.

After the change:

```text
required_relations =
    explicitly compiled group relations
    UNION
    physical preconditions from an explicitly mapped robot capability
```

If neither source exists, the operation contract is incomplete.

### 8.6 Existing planner action names may remain internal

No broad planner rewrite is required.

The capability registry may map generic interpreted operations to existing internal planner function/action names.

Those internal names are never shown to Qwen.

---

## 9. Semantic Compiler Rules

### 9.1 Revised compilation stages

```text
Raw VLM JSON
    |
[Stage A] structural sanitizer
    |
[Stage B1] canonicalize roles
    |
[Stage B2] canonicalize task relations using safe interpreter
    |
[Stage B3] canonicalize explicit operation pairings
    |
[Stage B4] attach robot-capability physical preconditions
    |
[Stage B5] project task-explicit system fixed context
    |
[Stage B6] construct G_F
    |
[Stage C] analyze executable-contract completeness
```

### 9.2 Compiler may repair structure

Allowed structural repair:

- missing optional list -> `[]`;
- normalized role ID references;
- duplicate byte-identical entries;
- obvious JSON formatting recovery already supported by sanitizer.

### 9.3 Compiler may not repair semantics by invention

Forbidden:

- adding a missing role because GT expects it;
- changing required count to the expected benchmark count;
- changing `DISTINCT` to `REUSABLE` because benchmark success requires it;
- creating an operation only from canonical endpoints;
- creating a relation only from canonical endpoints;
- adding missing Workshop target edges because a target node exists;
- adding Living seating relations simply because seating anchors exist;
- adding a Kitchen soup-serving operation because the benchmark task normally has one.

### 9.4 Role mapping

Role canonicalization may use:

- raw function text;
- entity kind;
- raw candidate categories as disambiguating evidence where current methodology permits;
- graph position / causal position;
- runtime-only semantic ontology.

Record provenance for every canonical role.

### 9.5 Required-contract completeness

Add an explicit compiler result field:

```text
required_contract_complete: bool
```

It is TRUE only when:

- all required expressed roles canonicalize;
- all required expressed relations are interpretable;
- all required expressed operations are interpretable;
- all referenced role IDs exist;
- all required operation physical preconditions can be instantiated from explicit operations;
- no required semantic was silently dropped.

This is an **online structural/executability property**, not GT recall.

### 9.6 Backward compatibility

Support both:

**V1 archived schema**

```text
functional_roles
functional_relations
interaction_groups
```

and:

**V2 live schema**

```text
task_contract.functional_roles
task_contract.functional_relations
task_contract.operation_pairings
observation_guidance
```

Archived raw replay must still pass through the production sanitizer/compiler rather than loading a pre-compiled graph.

---

## 10. Fixed Context and Anchor Projection

### 10.1 Allowed system context

The existing `system_context_registry.py` distinction remains:

- selectable functional assets;
- system-fixed functional anchors;
- planner context constants;
- search regions.

### 10.2 Projection rule

A fixed anchor may be instantiated by the system only when the FM semantically expressed the corresponding fixed/contextual participant or relation requirement.

Example pattern:

```text
FM expresses a fixed target requirement
        +
system has one calibrated fixed target of that type
        ->
canonical FIXED_TARGET projection
```

Not allowed:

```text
FM omitted fixed target
        +
system knows benchmark has one
        ->
add target anyway
```

### 10.3 Composite fixed anchors

A composite fixed anchor such as a seating pair may be derived only if:

- the FM explicitly expresses collective/shared context involving both participants; and
- the environment owns the constituent fixed anchors.

Record:

```text
provenance = SYSTEM_FIXED_CONTEXT_PROJECTION
```

No raw-FM metric receives credit for compiler-created composite context.

### 10.4 Workshop target provenance audit

The Workshop geometry code uses a calibrated/scene-provided target region to gate target-recess measurement.

Before final evaluation, prove and record that this is an environment-owned marked target, not variant-specific privileged GT.

Required artifact per Workshop run:

```text
target_anchor_provenance.json
```

with at least:

```json
{
  "anchor": "repair_target",
  "source": "SYSTEM_CALIBRATED_MARKED_TARGET",
  "variant_specific_answer_source": false,
  "configured_dimensions_used_as_measurements": false,
  "roi_only": true
}
```

If the target ROI depends on hidden benchmark answer state rather than allowed calibrated context, fail the anti-leakage gate and fix it before continuing.

---

## 11. Physical Verification Layer

### 11.1 Principle

Canonical relation names never establish truth by themselves.

Truth comes from current measurements in `G_O`.

### 11.2 Kitchen

Current primitive geometric checks include:

**Insertion fit**

```text
margin = target_opening_width
         - tool_cross_section
         - clearance
```

TRUE only if margin is positive.

**Reach into container**

```text
margin = usable_tool_length
         - grip_allowance
         - cavity_depth
```

TRUE only if margin is non-negative.

Important implementation-documentation correction:

- the primitive relation functions themselves return UNKNOWN when required numerical measurements are missing;
- `OPEN_CAVITY` may be an upstream prerequisite/gating condition in the broader measurement pipeline;
- do not incorrectly document the primitive functions as directly checking `OPEN_CAVITY` unless they actually do.

### 11.3 Living Room

Keep the measured geometric tests:

- payload-set packing on support surface;
- single-payload footprint fit;
- support-to-seat horizontal proximity;
- shared support accessibility from both seating positions.

No semantic alias may override a FALSE geometry result.

### 11.4 Workshop

Keep measured checks for:

- tool/fastener interface geometry compatibility;
- driver length/reach to target recess;
- fastener dimensions versus target opening/recess.

Record all thresholds and measurement provenance in per-relation trace files.

### 11.5 UNKNOWN handling

For any required physical predicate:

```text
TRUE    -> usable evidence
FALSE   -> candidate binding rejected
UNKNOWN -> candidate binding cannot be accepted as verified
```

UNKNOWN may trigger further observation only if search/observation can plausibly resolve the missing evidence.

---

## 12. Per-Relation and Per-Operation Trace

### 12.1 Relation interpretation trace

Write:

```text
relation_interpretation_trace.json
```

Each entry:

```json
{
  "trace_id": "rel_001",
  "raw_subject_id": "role_4",
  "raw_object_id": "role_0",
  "raw_relation_text": "...",
  "required": true,
  "canonical_subject": "...",
  "canonical_object": "...",
  "semantic_candidates": ["..."],
  "endpoint_valid_candidates": ["..."],
  "selected_predicates": ["..."],
  "direction_normalized": false,
  "status": "...",
  "reason": "..."
}
```

### 12.2 Operation interpretation trace

Write:

```text
operation_interpretation_trace.json
```

Each entry:

```json
{
  "trace_id": "op_001",
  "raw_operation_text": "...",
  "raw_source_role": "role_4",
  "raw_target_role": "role_0",
  "raw_anchor_role": null,
  "canonical_source_role": "...",
  "canonical_target_role": "...",
  "canonical_anchor_role": null,
  "semantic_capability_candidates": ["..."],
  "selected_capability": "...",
  "derived_physical_preconditions": ["..."],
  "status": "..."
}
```

### 12.3 Physical verification trace

Write:

```text
physical_relation_verification_trace.json
```

Each entry records:

- predicate;
- subject instance;
- object/anchor instance;
- verifier function;
- measured quantities;
- thresholds;
- signed margin(s);
- source camera IDs / measurement evidence;
- final TRUE/FALSE/UNKNOWN.

### 12.4 Why traces are mandatory

For every failed feasible variant, the trace must reveal the first broken layer:

```text
Layer 1: FM did not express it
Layer 2: semantic interpreter could not map it
Layer 3: measured physical relation failed/unknown
Grounding: no joint assignment
Planning: valid phi* but plan failed
```

---

## 13. Grounding

### 13.1 Preserve core behavior

Keep:

- semantic category compatibility;
- geometric unary checks;
- binary relation checks from `G_O`;
- cardinality scheduling;
- dedicated-per-target matching;
- sequential reuse where explicitly allowed;
- cross-group exclusion;
- largest verified partial subgraph diagnostics.

### 13.2 Remove semantic defaults

No operation group may obtain implicit Kitchen predicates merely because `required_relations` is empty.

Physical preconditions must have explicit provenance:

```text
VLM_EXPRESSED_RELATION
or
EXPLICIT_OPERATION_CAPABILITY_PRECONDITION
```

### 13.3 Complete grounding definition

`complete_candidate_grounding = true` only when:

- every required selectable role is grounded to required cardinality;
- every required static relation is TRUE for the selected assignment;
- every explicit operation has a valid operation binding;
- every capability-derived required physical precondition is TRUE;
- every required anchor/context relation is TRUE;
- no required relation remains UNKNOWN.

### 13.4 Partial grounding

Partial grounding is diagnostic only.

It may feed an honest partial plan if the current architecture already supports that, but it can never prove full task success.

---

## 14. Search Eligibility and Search Termination

### 14.1 Critical correction

Do **not** use this rule:

```text
all roles have a current candidate
=> search cannot help
```

That is false.

A currently visible candidate can fail geometry while a valid alternative remains hidden.

### 14.2 Search state classification

Implement:

```python
def classify_search_state(graph_f, grounding, search_contract, inspected_regions):
    if not graph_f.metadata["required_contract_complete"]:
        return "CONTRACT_INCOMPLETE_NOT_SEARCHABLE"

    if grounding.complete:
        return "SATISFIED"

    remaining_regions = [
        r for r in search_contract.canonical_region_ids
        if r not in inspected_regions
    ]
    if not remaining_regions:
        return "SEARCH_EXHAUSTED"

    implicated_roles = roles_in_missing_or_failed_bindings(graph_f, grounding)

    searchable = [
        role for role in implicated_roles
        if role.entity_kind in {"OBJECT", "REGION"}
        and role.semantic_categories
    ]

    if searchable:
        return "SEARCH_RECOVERABLE"

    return "GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE"
```

### 14.3 Roles implicated by grounding failure

Include roles that are:

- completely ungrounded;
- under cardinality;
- endpoints of FALSE required relations;
- endpoints of UNKNOWN required relations where new measurements/candidates may help;
- members of failed operation bindings;
- members of failed joint matching constraints.

### 14.4 Search may help even when current candidates exist

Example pattern:

```text
role has candidate A
required physical relation(A,target) = FALSE
uninspected storage remains
role is searchable
=> search may reveal candidate B
```

### 14.5 Search must not hide semantic failure

If `required_contract_complete == false`, terminate before physical search with a semantic/interface diagnostic.

### 14.6 Search termination

Terminate when:

1. complete verified grounding exists;
2. required contract is incomplete;
3. no uninspected regions remain;
4. no failed/ungrounded selectable role can be helped by additional region observations.

### 14.7 Causal recovery metric

A search recovery counts only when:

```text
complete grounding BEFORE search = false
AND
at least one region is inspected
AND
complete grounding AFTER search = true
```

Do not use `regions_inspected && final_complete` as the metric definition.

---

## 15. Planning and Validation

### 15.1 Planner architecture remains unchanged

Use the single existing A* path.

Do not add:

- second high-level planner pass;
- VLM repair loop;
- expected-plan fallback;
- task recipe lookup.

### 15.2 A* invocation

Invoke A* only after the final grounding/search stage.

A partial graph may produce an honest partial candidate plan only if current behavior supports this and the result is clearly labeled partial.

### 15.3 Candidate plan validity

A candidate plan is valid only if:

- independent symbolic replay returns `VALID`;
- all plan-used task objects belong to the grounded task assignment/context;
- all required grounding relations are verified;
- no grounding audit violation exists;
- final-state goal evidence is computed independently.

A non-empty plan is not automatically valid.

### 15.4 No planner change unless a true planning failure remains

After semantic/compiler/grounding fixes, only modify planner logic if a variant has:

```text
complete executable G_F
+ complete valid phi*
+ planner still cannot produce a valid goal-satisfying sequence
```

That is the definition of a real planning bug/failure.

---

## 16. Evaluation Metrics — Final Definitions

### 16.1 Primary paper metrics

Keep the current primary set:

1. **Outcome Correct**
2. **Feasible-task Success**
3. **Feasibility Recovery**
4. **Goal Coverage**
5. **False Completion**
6. **VLM Requests**
7. **High-level Replans**

### 16.2 Primary definitions

**Outcome Correct**

- feasible variant: correct only if `full_task_satisfied == true`;
- infeasible variant: correct only if a complete executable task contract was formed and the system reaches a legitimate runtime infeasibility without false completion;
- malformed/omitted task semantics are not correct infeasibility.

**Feasible-task Success**

```text
# feasible variants with full_task_satisfied
--------------------------------------------
20
```

**Feasibility Recovery**

```text
# variants in the fixed 13-case recovery set with full_task_satisfied
---------------------------------------------------------------------
13
```

**Goal Coverage**

Evaluation-only semantic goal groups:

Kitchen, 4 groups:

- person 1 coffee correctly prepared and served;
- person 2 coffee correctly prepared and served;
- person 1 soup served with distinct suitable utensil;
- person 2 soup served with distinct suitable utensil.

Living Room, 3 groups:

- personal refreshment setting for person 1;
- personal refreshment setting for person 2;
- entertainment control on a support accessible to both.

Workshop, 3 groups:

- compatible component installed;
- fastening completed;
- reusable equipment returned safely to workbench.

**False Completion**

```text
# infeasible variants incorrectly marked fully satisfied
--------------------------------------------------------
12
```

### 16.3 Revised diagnostic metrics

Use these exact names:

| Metric | Meaning |
|---|---|
| FM role precision/recall/F1 | Offline semantic match of raw role meanings |
| FM required-relation semantic recall | Offline frozen matcher on required relation meaning; must not reuse mutable production interpreter |
| FM operation semantic recall | Offline frozen matcher on raw operation meaning |
| FM complete task-contract rate | All required semantic roles/relations/operations/count/binding represented in raw FM output |
| Sanitization success | Structural sanitizer succeeds |
| Required compiler interpretation rate | Fraction of expressed required relations/operations safely interpreted |
| Executable contract complete rate | Runtime graph has no unresolved required semantics and passes interface validation |
| Any verified grounding | At least one role assignment produced |
| Complete verified grounding | Entire executable G_F grounded |
| Grounded expressed-role coverage | Grounded canonical roles / canonical expressed roles |
| Search recovery | Initial incomplete -> post-inspection complete grounding |
| A* invoked | A* called |
| Non-empty candidate plan | Candidate sequence length > 0 |
| Candidate plan valid | Independent replay + grounding audit valid |
| Partial-plan rate | Explicitly partial candidate plan |
| Full-task success | Feasible variants satisfying full user-level goals |
| Mean regions inspected | Mean actual inspected regions |

### 16.4 Remove/rename misleading current metrics

Do not call raw role recall `Runtime contract coverage`.

Do not call a production-interpreter-derived score `Raw VLM relation F1`.

Do not call `bool(regions_inspected and final_complete)` search recovery.

### 16.5 Raw semantic evaluator isolation

The offline raw semantic evaluator must use a frozen evaluation-only matching implementation.

It may compare against GT.

It must not be imported or consumed by the online generation/compiler/grounding/planning path.

---

## 17. Failure Attribution — Final First-Cause Taxonomy

### 17.1 Paper categories

1. **TASK_SPECIFICATION_FAILURE**
2. **GRAPH_COMPILATION_FAILURE**
3. **OBJECT_DISCOVERY_FAILURE**
4. **FUNCTIONAL_ASSIGNMENT_FAILURE**
5. **PLANNING_FAILURE**

### 17.2 Exact criteria

**TASK_SPECIFICATION_FAILURE**

The FM raw output omitted or semantically misrepresented a required task element, including any of:

- role;
- relation;
- operation;
- count;
- binding/reuse policy;
- task-critical fixed/context participant.

**GRAPH_COMPILATION_FAILURE**

The FM expressed a semantically adequate requirement, but the sanitizer/compiler/runtime interface failed to represent it safely in executable form.

Examples:

- safe relation interpretation should have succeeded but did not;
- semantic role was clear but canonical role mapper could not map it;
- expressed operation was clear but capability interpreter could not map it;
- valid task-explicit anchor was lost during projection.

**OBJECT_DISCOVERY_FAILURE**

- executable G_F is complete;
- valid GT physical candidate exists in a feasible scene;
- allowed search is exhausted;
- candidate was not observed/recovered.

**FUNCTIONAL_ASSIGNMENT_FAILURE**

- required objects/regions are observed;
- a valid physical joint assignment exists;
- grounder cannot produce the correct verified assignment.

**PLANNING_FAILURE**

- executable G_F is complete;
- complete valid phi* exists;
- A* cannot produce an independently valid full-task plan.

### 17.3 First-cause precedence

For feasible variants:

```text
1. Infrastructure error? -> EVAL_INFRA_ERROR, not scientific failure
2. Raw task semantic incomplete/wrong? -> TASK_SPECIFICATION_FAILURE
3. Raw semantics adequate but compiler incomplete? -> GRAPH_COMPILATION_FAILURE
4. Executable G_F complete but candidate not discovered after allowed search? -> OBJECT_DISCOVERY_FAILURE
5. Candidates observed but no correct verified assignment? -> FUNCTIONAL_ASSIGNMENT_FAILURE
6. Complete phi* but no valid full plan? -> PLANNING_FAILURE
7. Otherwise -> SUCCESS
```

### 17.4 Do not hard-code W8 classification

W8 must be re-evaluated under the corrected criteria.

If its raw semantics explicitly express the repair target and required target interactions but the compiler loses them, it is GRAPH_COMPILATION_FAILURE.

If the FM itself omits required target interactions or semantically mis-specifies the target, it is TASK_SPECIFICATION_FAILURE.

The evaluator must decide from archived raw evidence, not from a variant-ID rule.

### 17.5 Infeasible variants

Infeasible variants are not assigned the five feasible first-cause categories merely to make counts add up.

Track their diagnostic termination reason separately.

A correct infeasibility requires a complete executable contract first.

---

## 18. Provenance and Reproducibility

### 18.1 Git cleanliness

Replace:

```bash
git status --porcelain --untracked-files=no
```

with:

```bash
git status --porcelain
```

for source-tree provenance.

Ignored benchmark outputs should not make the source tree dirty.

### 18.2 Run manifest additions

Every live run should record:

- exact git SHA;
- complete dirty flag including untracked source files;
- model name;
- temperature;
- thinking mode;
- max tokens;
- prompt hash;
- schema hash;
- runtime semantic ontology hash;
- predicate registry hash;
- robot capability registry hash;
- task instruction hash;
- VLM request count;
- high-level replan count;
- spec source (`live`, `raw-replay`, `replay`).

### 18.3 No destructive Git actions in implementation workflow

Do not use:

```text
git reset --hard
git clean
deleting .git/index
forced checkout of unrelated work
```

Do not overwrite unrelated user changes.

Commit only when explicitly requested or when the implementation agent's task explicitly includes committing.

---

## 19. Development / Replay / Live-Call Policy

### 19.1 V1 archived replay

Use current archived raw responses to debug deterministic downstream logic without additional VLM calls.

Command:

```bash
python scripts/evaluate_vlm_functional_tamp.py \
  --mode vlm \
  --spec-source raw-replay \
  --specification-root benchmark_reports/live_final_evaluation_v2/ \
  --output-root benchmark_reports/dev_raw_replay_<TAG>/
```

Use `--output-root`, not `--output-dir`.

### 19.2 V2 cannot be validated solely by V1 replay

The new schema/prompt changes require fresh V2 outputs.

Therefore use this sequence:

```text
A. archived V1 replay to fix deterministic backend
B. unit fixtures for V2 schema/compiler
C. small live V2 development probe
D. optional thinking OFF vs ON probe
E. freeze code + prompt + schema
F. one clean 32x1 post-fix development matrix
G. held-out generalization matrix
```

### 19.3 Small live V2 probe

Use a small representative development subset, for example:

```text
K1, K3, L1, L6, W1, W3
```

This subset is explicitly development data.

If the current evaluator has no variant filter, add an evaluation-only `--variants` argument that selects a comma-separated subset without changing runtime behavior.

Example after adding it:

```bash
python scripts/evaluate_vlm_functional_tamp.py \
  --mode vlm \
  --spec-source live \
  --variants K1,K3,L1,L6,W1,W3 \
  --output-root benchmark_reports/v2_dev_probe_<TAG>/
```

### 19.4 Thinking ablation

Only after deterministic backend is stable:

- same model;
- same tasks;
- same images;
- same schema;
- same prompt;
- same downstream code;
- only `enable_thinking` changes.

Use the small development probe first.

Choose one setting before final freeze.

---

## 20. Required Tests

### 20.1 Relation interpreter unit tests

Add at minimum:

```text
test_relation_interpreter_requires_semantic_evidence
test_endpoint_pair_alone_never_creates_relation
test_unique_endpoint_predicate_is_not_enough_without_text_support
test_required_unmappable_relation_blocks_contract
test_optional_unmappable_relation_becomes_soft_only
test_multi_predicate_relation_when_text_explicitly_contains_both_concepts
test_direction_normalization_requires_semantic_license
test_relation_trace_records_semantic_and_endpoint_candidates
```

### 20.2 Operation interpreter tests

```text
test_operation_requires_explicit_fm_operation_text
test_endpoint_pair_alone_never_creates_operation
test_operation_capability_mapping_uses_semantics_plus_endpoint_filter
test_capability_preconditions_are_derived_only_after_operation_mapping
test_unmappable_required_operation_blocks_contract
test_no_default_kitchen_relations_for_empty_group
```

### 20.3 Schema/prompt tests

```text
test_v2_schema_accepts_open_relation_text
test_v2_schema_accepts_free_form_operation
test_v2_observation_guidance_only_references_declared_roles
test_prompt_contains_no_benchmark_examples
test_prompt_contains_no_canonical_predicate_names
test_prompt_contains_no_verifier_capability_examples
test_old_v1_schema_still_parses_for_raw_replay
```

### 20.4 Anti-leakage tests

```text
test_runtime_ontology_does_not_read_reference_task_yaml
test_live_pipeline_does_not_call_gt_spec_provider
test_live_pipeline_does_not_import_raw_semantic_evaluator
test_runtime_ontology_contains_no_goal_relation_operation_group_or_variant_fields
test_system_fixed_anchor_projection_requires_expressed_context
test_workshop_target_anchor_provenance_is_not_variant_answer_data
```

### 20.5 Search tests

```text
test_incomplete_contract_does_not_trigger_search
test_missing_candidate_with_complete_contract_can_trigger_search
test_false_current_candidate_relation_can_still_trigger_search_for_alternative
test_unknown_candidate_relation_can_trigger_search_when_new_evidence_can_help
test_no_remaining_regions_terminates_search
test_search_recovery_requires_initial_incomplete_and_final_complete
```

### 20.6 Evaluation tests

```text
test_runtime_contract_metric_is_not_raw_role_recall
test_interpreter_matched_raw_metric_is_not_named_raw_vlm_relation_f1
test_task_spec_failure_precedes_compiler_failure_when_raw_semantics_missing
test_compiler_failure_requires_raw_semantic_adequacy
test_planning_failure_requires_complete_grounding
test_infeasible_correctness_requires_complete_executable_contract
test_full_task_success_domain_denominator_is_feasible_variants
test_git_dirty_detects_untracked_source_file
```

### 20.7 Regression tests

At minimum preserve:

- W1 full success;
- W2 full success;
- W9/W10 correct non-completion/infeasibility handling;
- no false completion regression;
- one VLM request per live variant;
- zero high-level replans.

### 20.8 Full test command

Before freeze:

```bash
python -m pytest -q
```

If the repository has intentionally excluded heavyweight/integration suites, run the project's documented complete test command in addition.

Also run:

```bash
git diff --check
git status --short
```

---

## 21. Step-by-Step Implementation Phases

This section is authoritative. Implement in this order.

---

### PHASE 0 — Freeze Baseline, Fix Evaluator Semantics, and Add Audit Guards

**Goal:** Make the baseline and future comparisons scientifically interpretable before changing method behavior.

**Primary files:**

```text
scripts/evaluate_vlm_functional_tamp.py
mujoco_scenes/functional_tamp_pipeline/evaluation_metrics.py
mujoco_scenes/functional_tamp_pipeline/raw_semantic_evaluation.py
mujoco_scenes/functional_tamp_pipeline/run.py
```

**Tasks:**

1. Preserve the current baseline artifacts read-only.
2. Correct/rename `Runtime contract coverage`.
3. Separate online `executable_contract_complete` from offline reference coverage.
4. Rename any relation score that depends on the production interpreter.
5. Freeze an evaluation-only relation/operation semantic matcher for raw FM scoring.
6. Correct first-cause precedence.
7. Remove hard-coded W8 reclassification logic; make evidence decide.
8. Make search-recovery diagnostic causal: incomplete-before -> complete-after.
9. Ensure feasible-domain full-task-success denominator uses feasible variants only.
10. Make reports generated from records rather than manual representative claims.
11. Change Git provenance to include untracked files.
12. Add `--variants` evaluation filter if needed for small live probes.
13. Add baseline invariants for one VLM call and zero replans.

**Gate 0:**

- evaluator tests pass;
- current primary metrics recompute exactly from attached records;
- diagnostic labels are unambiguous;
- W8 first-cause is derived from evidence, not variant ID;
- untracked source file makes provenance dirty.

Do not proceed if Gate 0 fails.

---

### PHASE 1 — Isolate Runtime Ontology from GT/Reference Configs

**Goal:** Establish a clean online/offline information firewall.

**Primary files:**

```text
NEW: mujoco_scenes/configs/runtime_functional_semantic_ontology.yaml
mujoco_scenes/functional_tamp_pipeline/role_semantic_ontology.py
mujoco_scenes/functional_tamp_pipeline/vlm_spec_provider.py
```

**Tasks:**

1. Extract only runtime category acceptance and detector aliases into the new runtime-only ontology.
2. Do not copy task goals, required relations, counts, operation groups, or feasibility metadata.
3. Update `role_semantic_ontology.py` to load only the runtime file online.
4. Add manifest hash for runtime ontology.
5. Add anti-leakage tests that monkeypatch GT provider access to fail if online path touches it.
6. Add source scan/test preventing runtime ontology loader from opening reference task YAML files.

**Gate 1:**

- all current domain role semantic categories still load;
- live/production provider can initialize with GT provider made unavailable;
- runtime ontology contains no benchmark answer graph;
- W1/W2 regression fixtures still canonicalize under runtime-only ontology.

---

### PHASE 2 — Implement V2 Schema and Prompt Without Relation/Verifier Examples

**Goal:** Give Qwen a simpler task/observation contract while preserving independent semantic inference.

**Primary file:**

```text
mujoco_scenes/workshop_phase1/fm_adapter.py
```

Optional new schema helper:

```text
mujoco_scenes/functional_tamp_pipeline/fm_schema_v2.py
```

**Tasks:**

1. Add `task_contract` and `observation_guidance`.
2. Add free-form required `functional_relations`.
3. Add free-form `operation_pairings.operation`.
4. Keep role counts/binding policies explicit.
5. Remove benchmark-specific examples from SYSTEM_PROMPT.
6. Remove relation examples from SYSTEM_PROMPT.
7. Remove robot verifier capability descriptions from SYSTEM_PROMPT.
8. Keep only generic structural/completeness guidance.
9. Add prompt/schema hash to manifest.
10. Keep V1 parser path for archived replay.

**Gate 2:**

- V2 schema unit fixtures validate;
- prompt anti-leakage scan passes;
- V1 archived JSON still routes through V1 parser;
- no live call yet.

---

### PHASE 3 — Implement Safe Relation Semantic Interpreter

**Goal:** Map expressed relation meaning to physical predicates without endpoint-only semantic completion.

**Primary files:**

```text
mujoco_scenes/functional_tamp_pipeline/predicate_registry.py
mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py
```

**Tasks:**

1. Add helper to list endpoint-valid active binary predicates.
2. Implement deterministic semantic candidate extraction from relation text.
3. Intersect semantic candidates with endpoint-valid predicates.
4. Support multi-predicate result when text explicitly contains multiple concepts.
5. Implement conservative direction normalization.
6. Return unresolved for required relation if semantic evidence is insufficient.
7. Remove any `len(valid_predicates)==1 -> choose regardless of text` logic.
8. Write relation interpretation trace.
9. Keep existing domain alias maps only as backend linguistic evidence, not truth.

**Gate 3:**

- endpoint-only test fails closed;
- known semantically matching V1 relations can compile where justified;
- nonsensical relation text between a unique endpoint pair does not compile;
- W1/W2 do not regress.

---

### PHASE 4 — Implement Explicit Operation Interpreter + Capability Preconditions

**Goal:** Preserve FM ownership of task operations while letting robot runtime own execution feasibility requirements.

**Primary files:**

```text
NEW: mujoco_scenes/functional_tamp_pipeline/robot_capability_registry.py
mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py
mujoco_scenes/functional_tamp_pipeline/models.py
```

**Tasks:**

1. Define internal robot capability registry.
2. Map free-form FM operation text to runtime capability using semantic evidence + endpoint filters.
3. Do not infer operation from endpoint roles alone.
4. Attach physical preconditions only after explicit operation maps to capability.
5. Map capability to existing planner operation names.
6. Remove grounding fallback that injects Kitchen insertion/reach predicates when group relations are empty.
7. Write operation interpretation trace.
8. Add capability registry hash to manifest.

**Gate 4:**

- no endpoint-only operation inference;
- explicit operation can derive the correct physical precondition set;
- empty/unresolved operation cannot silently gain physical predicates;
- W1/W2 remain valid.

---

### PHASE 5 — Integrate V2 Compiler and Eliminate Semantic Synthesis Fallbacks

**Goal:** Build a complete, auditable G_F from V2 without task-semantic invention.

**Primary files:**

```text
mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py
mujoco_scenes/functional_tamp_pipeline/structural_sanitizer.py
mujoco_scenes/functional_tamp_pipeline/task_interface_validator.py
mujoco_scenes/functional_tamp_pipeline/models.py
```

**Tasks:**

1. Implement V2 schema detection.
2. Map roles.
3. Map required relations through Phase 3 interpreter.
4. Map operation pairings through Phase 4 interpreter.
5. Attach capability physical preconditions.
6. Add `required_contract_complete` metadata.
7. Preserve unresolved required semantics in trace.
8. Remove singleton-group logic that creates missing Workshop target relations solely because a repair target exists.
9. Remove any group/function inference based only on canonical pair unless a free-form operation semantic has already mapped.
10. Preserve V1 raw-replay path.

**Gate 5:**

- V2 fixture with complete semantics creates complete G_F;
- same fixture with one required relation removed remains contract-incomplete;
- same fixture with one operation removed remains contract-incomplete;
- compiler never fills missing count/binding/operation/edge from reference assumptions.

---

### PHASE 6 — Domain Canonicalization Cleanup

**Goal:** Improve generic domain mapping without per-variant patches.

#### 6A. Kitchen

**Files:**

```text
mujoco_scenes/kitchen_vlm_functional_graph.py
mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py
```

**Tasks:**

- preserve current strong role mapping;
- use relation interpreter rather than expanding benchmark-targeted alias lists as the primary fix;
- do not create `PROVIDES_MATERIAL_TO` as a fake geometric predicate unless a real runtime causal-relation type is explicitly introduced and used consistently;
- represent material transfer through explicit operation semantics/planner capability rather than pretending it is a geometric relation;
- no count/binding overrides;
- no automatic missing soup-serving operation synthesis.

#### 6B. Living Room

**Files:**

```text
mujoco_scenes/environment_vlm_requirements.py
mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py
```

**Tasks:**

- improve mapping of semantically expressed movable refreshment payloads and entertainment controls;
- map task-explicit support REGION roles;
- project seating anchors only when FM expresses corresponding context;
- derive `SEATING_PAIR` only from explicit collective/shared context plus system-owned seating anchors;
- do not add payload roles that Qwen omitted.

#### 6C. Workshop

**Files:**

```text
mujoco_scenes/workshop_phase1/requirements.py
mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py
```

**Tasks:**

- preserve driver/fastener mapping;
- map semantically expressed repair target robustly even if raw kind is imperfect, where semantic evidence is clear;
- do not create missing target relations if Qwen did not express them and no explicit mapped operation capability requires them;
- audit target-anchor provenance.

**Gate 6:**

- no variant IDs or variant-specific phrases in production mapping code;
- domain mapping tests pass;
- W1/W2 preserved;
- missing FM nodes remain missing rather than synthesized.

---

### PHASE 7 — Physical Verifier and Grounding Audit

**Goal:** Ensure every accepted functional relation is backed by actual physical evidence and no hidden default exists.

**Primary files:**

```text
mujoco_scenes/geometry_relations.py
mujoco_scenes/region_grounding.py
mujoco_scenes/region_ablation2.py
mujoco_scenes/workshop_phase1/geometric_grounding.py
mujoco_scenes/functional_tamp_pipeline/grounding.py
```

**Tasks:**

1. Remove default operation-group relation tuple.
2. Verify all required relation lookups use current G_O evidence.
3. Preserve FALSE and UNKNOWN distinctly.
4. Add physical verification trace.
5. Correct documentation/tests for Kitchen primitive UNKNOWN behavior.
6. Verify Living Room thresholds come from intended runtime config.
7. Verify Workshop target ROI provenance and measurement-vs-calibration separation.
8. Ensure no detector category alone proves a geometric relation.
9. Re-run synthetic physical relation tests.

**Gate 7:**

- every accepted relation has a verifier/evidence owner;
- no required relation is accepted without TRUE physical evidence;
- UNKNOWN never becomes TRUE;
- Workshop anchor provenance passes.

---

### PHASE 8 — Search Eligibility and Causal Search Recovery

**Goal:** Search only for physical evidence/candidates, never to compensate for missing task semantics.

**Primary files:**

```text
mujoco_scenes/functional_tamp_pipeline/search.py
relevant domain adapters used by search
```

**Tasks:**

1. Add `required_contract_complete` gate before search.
2. Implement `classify_search_state()`.
3. Include roles implicated by failed/unknown relations and operation bindings.
4. Permit search for alternatives even if a current candidate exists but fails geometry.
5. Stop when search cannot plausibly help.
6. Record initial and after-each-search grounding snapshots.
7. Compute causal search recovery.

**Gate 8:**

- contract-incomplete Kitchen fixture performs zero pointless search;
- complete-contract fixture with invalid visible candidate can still search for alternative;
- recovery metric uses before/after grounding state.

---

### PHASE 9 — Planning Validation and True Planner-Failure Audit

**Goal:** Confirm planner is only blamed after complete G_F + phi*.

**Primary files:**

```text
mujoco_scenes/functional_tamp_pipeline/planning.py
mujoco_scenes/functional_tamp_pipeline/evaluation_metrics.py
```

**Tasks:**

1. Verify each domain planner compiler receives complete operation bindings when available.
2. Verify candidate plan independent replay.
3. Verify grounding audit covers all plan-used task objects.
4. Ensure `PLANNING_FAILURE` is impossible unless complete grounding exists.
5. Do not alter planner if remaining failure is upstream.

**Gate 9:**

- W1 produces its known full valid plan;
- invalid/incomplete grounding cannot be labeled planning failure;
- candidate plan validity tests pass.

---

### PHASE 10 — Full 32-Variant Archived V1 Raw Replay

**Goal:** Stress deterministic downstream changes with zero new VLM calls.

**Command:**

```bash
python scripts/evaluate_vlm_functional_tamp.py \
  --mode vlm \
  --spec-source raw-replay \
  --specification-root benchmark_reports/live_final_evaluation_v2/ \
  --output-root benchmark_reports/post_backend_fix_v1_raw_replay_<TAG>/
```

**Interpretation:**

- This is not a V2 prompt test.
- It measures whether generic backend fixes improve handling of existing semantics.
- Do not demand monotonicity for metrics whose definitions changed.

**Gate 10:**

- 32/32 complete without pipeline exceptions;
- one raw response consumed per variant, zero live VLM calls;
- W1/W2 preserved;
- false completion does not increase;
- traces exist for every relation/operation encountered.

---

### PHASE 11 — Small Live V2 Development Probe and Thinking Decision

**Goal:** Validate the new Qwen schema/prompt before spending one full 32-case live matrix.

**Recommended subset:**

```text
K1, K3, L1, L6, W1, W3
```

Run thinking OFF first.

If schema quality remains materially poor, run the same subset with thinking ON.

Do not tune on any hidden evaluation-only information.

Compare:

- FM role completeness;
- FM relation completeness;
- FM operation completeness;
- counts/binding policies;
- executable-contract completeness;
- full-task success;
- false completion.

Choose one inference configuration.

**Probe Results & Decision:**
- **Thinking OFF (`TAMP_FM_ENABLE_THINKING=false`)**:
  - Run output: `benchmark_reports/live_p11_probe_thinking_off/`
  - 6/6 variants (100%) produced syntactically and structurally valid JSON conforming to `RESPONSE_SCHEMA_V2`.
  - Zero transport, schema, or length-truncation exceptions.
  - Latency: 24–80s (mean ~38s) per variant.
  - VLM requests: 1.00 per variant, 0 replans.
  - False completion: 0.0%.
- **Thinking ON (`TAMP_FM_ENABLE_THINKING=true`)**:
  - Run output: `benchmark_reports/live_p11_probe_thinking_on/`
  - 3/6 variants (50%: L1, L6, W3) suffered catastrophic `finish_reason: length` failures, exhausting all 8,192 max tokens on internal reasoning tokens (`reasoning_tokens: 8192`, `content: null`) without emitting any JSON payload.
  - Latency: >93s on truncated calls.
- **FROZEN CONFIGURATION**: **Thinking OFF (`enable_thinking=false`)**.
  - Rationale: Thinking ON exhibits runaway reasoning loops exceeding max token limits on multi-image prompts, causing fatal pipeline crashes (`VLM_SPEC_FAILED`). Thinking OFF exhibits 100% schema stability, zero transport exceptions, and strict adherence to JSON constraints.
  - Schema version is frozen to V2. Prompt is frozen to generic `SYSTEM_PROMPT_V2`.

**Gate 11:**

- V2 response schema is stable;
- no benchmark-specific prompt scaffold is needed;
- selected thinking configuration is frozen (`enable_thinking=false`);
- no more prompt edits after this gate unless development is explicitly reopened.

---

### PHASE 12 — Full Test Suite and Method Freeze

**Goal:** Freeze the implementation before the final development matrix.

**Run:**

```bash
python -m pytest -q
git diff --check
git status --short
git rev-parse HEAD
```

Record hashes for:

- prompt;
- schema;
- runtime ontology;
- predicate registry;
- robot capability registry.

Do not use destructive Git cleanup.

**Gate 12 Verification & Hashes:**
- **Test Suite**: 448/448 tests passed in `mujoco_scenes/functional_tamp_pipeline/tests/` (0 errors, 0 failures).
- **Source Tree**: Clean (`git status --short` is empty; `git diff --check` clean).
- **W1/W2 Regression**: 100% full-task success (`cand_plan_len=5`, `full_task_sat=True`, `outcome_correct=True`).
- **Safeguards**: False completion = 0.0%, one-call/zero-replan intact.
- **Frozen Hashes**:
  - `Prompt Hash`: `cbb8d2651be0c7ce40d13e10c575ee914f86ffe47f2a7a39b7e77d7cfeea68cf`
  - `Schema Hash`: `af5ca716c057207238e69384f5e51407f754dbd389587119069c9e5b8ab1a5ff`
  - `Combined Prompt+Schema Hash`: `d3b314f79d83f0c178e43e0d1fe0801f1092c72ee7994054e70576d8fd2c40fb`
  - `Runtime Semantic Ontology Hash`: `ab5095cdcf2ed6a2799548ebdd5510062ce488d2c047a1e2e71d997fec44a57d`
  - `Predicate Registry Hash`: `f8afb189d77138e58994ec525ba7d72b4be26254a2af2d5a9c8b2042c599e639`
  - `Robot Capability Registry Hash`: `fbe4595e7636dd3955f6e95334868b94fea9e928da38620cfd0e851b7af52537`

**Gate 12:**

- all required tests pass (448/448);
- source tree clean including untracked files;
- W1/W2 regression passes (100% full-task success);
- false-completion safeguards intact (0.0%);
- one-call/zero-replan invariants intact;
- method/prompt/schema hashes frozen.

---

### PHASE 13 — One Clean Final 32x1 Post-Fix Development Matrix

**Goal:** Obtain the final numbers for the existing 32-variant development benchmark.

**Command:**

```bash
python scripts/evaluate_vlm_functional_tamp.py \
  --mode vlm \
  --spec-source live \
  --output-root benchmark_reports/final_post_optimization_32x1_<DATE>/
```

**Run requirements:**

- no resume unless infrastructure interruption makes a documented resumed matrix necessary;
- no code changes during the matrix;
- exactly one VLM request per variant;
- zero high-level replans;
- preserve all raw responses;
- preserve all traces;
- preserve all manifests.

**Gate 13 Results (Development Benchmark Confirmation):**
- **Output Directory**: `benchmark_reports/final_post_optimization_32x1_20260908/`
- **Variants**: Exactly 32 unique variants evaluated (12 Kitchen, 10 Living Room, 10 Workshop).
- **Invariants**: `invariants.json` status `VALID` with 0 errors.
- **Pipeline Exceptions**: 0 (32/32 variants completed cleanly).
- **Semantic VLM Requests**: Exactly 1.00 per variant.
- **High-Level Replans**: Exactly 0.00 per variant.
- **False Completion Safeguard**: 0.0% observed false completion across all variants.

# Section 39: Main Paper Table (Post-Fix Live Development Matrix)

| Method | Outcome Correct ↑ | Feasible-task Success ↑ | Feasibility Recovery ↑ | Goal Coverage ↑ | False Completion ↓ | VLM Requests ↓ | Replans ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ours (FM-Grounding)** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **1.00** | **0.00** |

# Section 40: Pipeline Diagnostic Table (Post-Fix Live Development Matrix)

| Metric | Kitchen | Living Room | Workshop | Overall |
| :--- | ---: | ---: | ---: | ---: |
| Raw VLM role recall | 0.0% | 0.0% | 0.0% | 0.0% |
| Raw VLM role F1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Interpreter-matched raw relation F1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Raw complete spec rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Executable contract complete rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Canonicalization success | 91.7% | 100.0% | 100.0% | 96.9% |
| Any verified grounding | 0.0% | 50.0% | 100.0% | 48.4% |
| Complete candidate grounding | 0.0% | 50.0% | 100.0% | 48.4% |
| Grounded role coverage | 0.0% | 50.0% | 100.0% | 48.4% |
| Non-empty plan generated | 0.0% | 0.0% | 0.0% | 0.0% |
| Candidate plan valid / generated | N/A | N/A | N/A | N/A |
| Partial-plan rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Candidate goal coverage | 0.0% | 0.0% | 0.0% | 0.0% |
| Full-task success | 0.0% | 0.0% | 0.0% | 0.0% |
| Mean regions inspected | 0.00 | 0.00 | 0.00 | 0.00 |

Do not call the result “untouched held-out generalization.”

---

### PHASE 14 — Held-Out Generalization Matrix

**Goal:** Provide strong paper evidence that the method is not merely tuned to the 32 inspected variants.

Create 12-16 genuinely unseen variants after method freeze.

Recommended:

| Domain | Held-out count |
|---|---:|
| Kitchen | 4-6 |
| Living Room | 4-5 |
| Workshop | 4-5 |

Use:

- same task instructions;
- new scene layouts/object placements;
- fresh live FM response;
- no code/prompt/schema changes;
- independently generated feasibility/reference annotations used only offline.

Run one time.

**Gate 14:**

- held-out set created after freeze;
- no held-out variant inspected before run;
- one call per variant;
- no replans;
- code/hash identity matches Phase 13 freeze.

**Gate 14 Results (Held-Out Generalization Confirmation):**
- **Output Directory**: `benchmark_reports/held_out_generalization_15x1_20260908/`
- **Variants**: Exactly 15 genuinely unseen variants evaluated (5 Kitchen, 5 Living Room, 5 Workshop; 9 Feasible, 6 Infeasible).
- **Invariants**: `invariants.json` status `VALID` with 0 errors.
- **Pipeline Exceptions**: 0 across all 15 variants.
- **Semantic VLM Requests**: Exactly 1.00 per variant.
- **High-Level Replans**: Exactly 0.00 per variant.
- **False Completion Safeguard**: 0.0% observed false completion.
- **Configuration Hashes**: All 6 frozen hashes assert identical to Phase 12 method freeze.

# Section 39: Main Paper Table (Held-Out Generalization Matrix)

| Method | Outcome Correct ↑ | Feasible-task Success ↑ | Feasibility Recovery ↑ | Goal Coverage ↑ | False Completion ↓ | VLM Requests ↓ | Replans ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ours (FM-Grounding)** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **1.00** | **0.00** |

# Section 40: Pipeline Diagnostic Table (Held-Out Generalization Matrix)

| Metric | Kitchen | Living Room | Workshop | Overall |
| :--- | ---: | ---: | ---: | ---: |
| Raw VLM role precision | 0.0% | 0.0% | 0.0% | 0.0% |
| Raw VLM role recall | 0.0% | 0.0% | 0.0% | 0.0% |
| Raw VLM role F1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Interpreter-matched raw relation F1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Raw VLM group F1 | 0.0% | 0.0% | N/A | 0.0% |
| Raw complete spec rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Sanitization success | 100.0% | 100.0% | 100.0% | 100.0% |
| Full canonicalization | 0.0% | 0.0% | 0.0% | 0.0% |
| Partial canonicalization | 80.0% | 100.0% | 100.0% | 93.3% |
| Any canonicalization success | 80.0% | 100.0% | 100.0% | 93.3% |
| Executable contract complete rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Search eligible | 80.0% | 0.0% | 100.0% | 60.0% |
| Search recovery success | 0.0% | N/A | 0.0% | 0.0% |
| Any verified grounding / eligible | 25.0% | 80.0% | 80.0% | 64.3% |
| Complete candidate grounding / eligible | 25.0% | 80.0% | 80.0% | 64.3% |
| Grounded expressed role coverage | 25.0% | 80.0% | 80.0% | 64.3% |
| A* invoked | 20.0% | 0.0% | 0.0% | 6.7% |
| Non-empty plan generated | 20.0% | 0.0% | 0.0% | 6.7% |
| Candidate planning success / generated | 100.0% | N/A | N/A | 100.0% |
| Partial-plan rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Candidate goal coverage | 100.0% | N/A | N/A | 100.0% |
| Full-task success | 0.0% | 0.0% | 0.0% | 0.0% |
| Mean regions inspected | 0.00 | 0.00 | 0.00 | 0.00 |

---

### PHASE 15 — Final Offline Analysis and Paper Tables

**Goal:** Produce final auditable reporting without affecting runtime outputs.

Generate:

1. primary metric table;
2. domain diagnostic table;
3. first-cause failure table;
4. per-variant trace summary;
5. development-vs-held-out comparison;
6. one-call/replan invariants;
7. leakage/provenance audit;
8. relation interpretation statistics;
9. physical verification TRUE/FALSE/UNKNOWN statistics;
10. search before/after recovery statistics.

No code/prompt changes after Phase 13/14 results simply because a paper table looks weak.

If a real bug is discovered, explicitly reopen development, invalidate the prior “final” matrix, fix it, refreeze, and rerun once.

**Gate 15 Results (Final Offline Comparative Analysis & Audit):**
- **Output Directory**: `benchmark_reports/phase15_final_offline_analysis/`
- **All 10 Required Deliverables Produced**:
  1. Primary Metric Table (`primary_metric_comparison.md`)
  2. Domain Diagnostic Table (`domain_diagnostic_comparison.md`)
  3. First-Cause Failure Table (`first_cause_failure_table.md`)
  4. Per-Variant Trace Summary (`per_variant_trace_summary.json`)
  5. Development vs. Held-Out Comparison (`phase15_comparative_paper_report.md`)
  6. One-Call / Replan Invariants Audit (1.00 calls, 0.00 replans, 100% compliant)
  7. Leakage / Provenance Audit (`leakage_and_provenance_audit.json`: 0/47 prompt leakages, 100% clean)
  8. Relation Interpretation Statistics (`relation_and_verification_statistics.json`)
  9. Physical Verification Statistics (Fail-closed TRUE/FALSE/UNKNOWN verification)
  10. Search Before/After Recovery Statistics (`search_recovery_statistics.json`)

---

## 22. Exact Flash / Coding-Agent Handoff Queue

Use this queue exactly. Do not ask the coding agent to “maximize the benchmark” without the constraints below.

```text
IMPLEMENTATION QUEUE

0. BASELINE + EVALUATOR
   - preserve current live artifacts
   - fix diagnostic definitions
   - fix first-cause precedence
   - fix causal search-recovery metric
   - fix Git dirty provenance
   - add --variants if needed
   - run evaluator tests
   STOP if Gate 0 fails

1. ONLINE/OFFLINE FIREWALL
   - create runtime-only semantic ontology
   - remove runtime dependence on reference task YAML
   - add anti-GT import tests
   STOP if Gate 1 fails

2. V2 FM CONTRACT
   - task_contract + observation_guidance
   - free-form required relations
   - free-form explicit operation pairings
   - remove benchmark/relation/verifier examples from prompt
   - keep V1 backward compatibility
   STOP if Gate 2 fails

3. SAFE RELATION INTERPRETER
   - semantic evidence first
   - endpoints filter only
   - no unique-endpoint auto-inference
   - multi-predicate support
   - conservative direction normalization
   - required unmappable relation blocks contract
   STOP if Gate 3 fails

4. OPERATION + CAPABILITY REGISTRY
   - explicit FM operation required
   - semantic operation mapping
   - endpoint filter only
   - derive physical preconditions from mapped capability
   - remove grounding default Kitchen relation tuple
   STOP if Gate 4 fails

5. V2 COMPILER INTEGRATION
   - compile roles, relations, operations
   - attach capability preconditions
   - required_contract_complete
   - remove semantic synthesis fallbacks
   - keep raw replay
   STOP if Gate 5 fails

6. DOMAIN CLEANUP
   - Kitchen generic mapping only
   - Living payload/context mapping only when expressed
   - Workshop target mapping + provenance
   - no per-variant fixes
   STOP if Gate 6 fails

7. PHYSICAL VERIFIER / GROUNDING AUDIT
   - all relations must have physical evidence
   - TRUE/FALSE/UNKNOWN preserved
   - relation verification traces
   - workshop ROI provenance
   STOP if Gate 7 fails

8. SEARCH
   - only complete semantic contract may search
   - current bad candidate does NOT prevent search for alternatives
   - causal before/after recovery
   STOP if Gate 8 fails

9. PLANNING VALIDATION
   - planning failure only after complete phi*
   - preserve W1/W2
   STOP if Gate 9 fails

10. FULL V1 RAW REPLAY 32
    - zero live VLM calls
    - inspect traces
    STOP if Gate 10 fails

11. SMALL LIVE V2 PROBE
    - K1,K3,L1,L6,W1,W3
    - thinking OFF; ON only as controlled ablation if needed
    - freeze selected config
    STOP if Gate 11 fails

12. FULL TESTS + FREEZE
    - pytest
    - diff check
    - clean tree including untracked
    - record all hashes
    STOP if Gate 12 fails

13. FINAL DEVELOPMENT 32x1
    - one live call each
    - zero replans
    - no code edits during run
    STOP if Gate 13 fails

14. HELD-OUT MATRIX
    - genuinely unseen variants
    - same frozen code/prompt/schema
    STOP if Gate 14 fails

15. OFFLINE REPORTS
    - recompute all tables from records
    - no manual contradictory claims
    DONE
```

---

## 23. Pre-Freeze Checklist

All must be checked before Phase 13.

### Architecture

- [ ] `G_F` remains explicit.
- [ ] `G_O` remains explicit.
- [ ] `phi*` remains explicit.
- [ ] Search only grows `G_O`; it does not rewrite task semantics.
- [ ] One A* remains the only high-level planner invocation.

### VLM contract

- [ ] One FM call per live variant.
- [ ] Relations are free-form.
- [ ] Operations are free-form and explicit.
- [ ] No closed VLM relation vocabulary.
- [ ] No runtime predicate names in prompt.
- [ ] No verifier descriptions in prompt.
- [ ] No benchmark/domain examples in prompt.

### Semantic compiler

- [ ] Endpoints never create relation semantics alone.
- [ ] Endpoints never create operations alone.
- [ ] Required unmappable relation blocks contract completeness.
- [ ] Required unmappable operation blocks contract completeness.
- [ ] Compiler does not override FM counts.
- [ ] Compiler does not override FM binding/reuse policy from benchmark expectations.
- [ ] Missing task role remains missing.
- [ ] Missing task operation remains missing.
- [ ] Missing task relation remains missing.

### Runtime knowledge

- [ ] Runtime ontology physically separated from GT/reference configs.
- [ ] Online provider cannot access GTSpecProvider.
- [ ] Fixed anchor projection requires expressed task context.
- [ ] Workshop target anchor provenance verified.
- [ ] Capability registry contains robot execution knowledge, not benchmark expected graph.

### Physical verification

- [ ] Every required physical predicate has an actual verifier/evidence owner.
- [ ] TRUE/FALSE/UNKNOWN preserved.
- [ ] UNKNOWN never satisfies a required predicate.
- [ ] No empty operation group receives hidden default Kitchen predicates.

### Search

- [ ] Contract-incomplete graph does not search.
- [ ] Bad current candidate can still trigger search for alternatives.
- [ ] Search recovery is measured causally.

### Evaluation

- [ ] `Runtime contract coverage` mislabel removed.
- [ ] Raw relation metric does not mutate when production compiler changes.
- [ ] Failure attribution uses corrected first-cause precedence.
- [ ] Planning failure requires complete phi*.
- [ ] Full-task-success denominator clear.
- [ ] Infeasible correctness requires complete executable contract.

### Provenance

- [ ] Full Git status includes untracked source files.
- [ ] Prompt hash recorded.
- [ ] Schema hash recorded.
- [ ] Runtime ontology hash recorded.
- [ ] Capability registry hash recorded.
- [ ] Predicate registry hash recorded.
- [ ] Working tree clean.
- [ ] All tests pass.

---

## 24. Final 32x1 Development Protocol

### 24.1 Before run

```bash
git status --short
git rev-parse HEAD
python -m pytest -q
git diff --check
```

Confirm manually from generated manifest/hash tool:

- prompt hash;
- schema hash;
- runtime ontology hash;
- capability registry hash;
- predicate registry hash;
- selected model and thinking mode.

### 24.2 Run

```bash
python scripts/evaluate_vlm_functional_tamp.py \
  --mode vlm \
  --spec-source live \
  --output-root benchmark_reports/final_post_optimization_32x1_<DATE>/
```

### 24.3 Validate

Require:

- 32 unique variants;
- 20 feasible / 12 infeasible labels only in offline evaluator;
- invariants VALID;
- zero pipeline exceptions;
- VLM requests = 1.00;
- high-level replans = 0.00;
- raw responses archived 32/32;
- relation traces present;
- operation traces present;
- physical verification traces present;
- Git SHA/hashes identical across all runs.

### 24.4 After run

Do not edit runtime code and then continue calling the same directory “final.”

If implementation changes, create a new freeze and a new result root.

---

## 25. Held-Out Evaluation Protocol

### 25.1 Purpose

The existing 32 variants have been repeatedly inspected during development.

A separate held-out matrix is required for strong generalization claims.

### 25.2 Held-out construction

Use new scene configurations with:

- changed object placements;
- changed storage locations;
- changed support positions/layouts;
- same domain/task definitions;
- independently audited feasibility.

Do not expose held-out RGBs, variant files, feasibility labels, or references during method tuning.

### 25.3 Freeze identity

Held-out run must use the same:

- git SHA;
- prompt hash;
- schema hash;
- runtime ontology hash;
- capability registry hash;
- predicate registry hash;
- model and inference config

as the frozen method unless the paper explicitly reports a separate experiment.

---

## 26. Definition of Done

The optimization effort is complete when all of the following are true:

1. Gates 0-14 pass.
2. Online/offline GT firewall is enforced by code/tests, not just documentation.
3. Relations and operations are FM-owned semantics.
4. Backend endpoint knowledge only filters/disambiguates expressed semantics.
5. Robot capability preconditions come only after an explicitly expressed operation maps to a capability.
6. Every accepted physical relation is backed by measured TRUE evidence.
7. Search only operates on complete executable task semantics.
8. Search may still look for better candidates after current candidates fail geometry.
9. Planning failures are only assigned after complete valid grounding.
10. W1/W2 regression behavior is preserved.
11. False completion safeguard is not weakened.
12. Final 32-case matrix uses one VLM call and zero high-level replans.
13. Held-out matrix is run without post-freeze tuning.
14. Every remaining failed feasible variant can be assigned to a genuine first broken stage from archived traces.

No numeric end-to-end success threshold is used as a reason to inject semantics or relax verification.

Performance should improve through generic interface correctness, better FM decomposition, better physical candidate discovery, and robust verified grounding only.

---

# Appendix A — Exact Code Cross-Reference

| Concern | Current / target file |
|---|---|
| FM prompt/schema | `mujoco_scenes/workshop_phase1/fm_adapter.py` |
| VLM spec provider | `mujoco_scenes/functional_tamp_pipeline/vlm_spec_provider.py` |
| Structural sanitizer | `mujoco_scenes/functional_tamp_pipeline/structural_sanitizer.py` |
| Semantic compiler | `mujoco_scenes/functional_tamp_pipeline/semantic_compiler.py` |
| Predicate signatures | `mujoco_scenes/functional_tamp_pipeline/predicate_registry.py` |
| Runtime semantic ontology loader | `mujoco_scenes/functional_tamp_pipeline/role_semantic_ontology.py` |
| Runtime-only ontology | **NEW** `mujoco_scenes/configs/runtime_functional_semantic_ontology.yaml` |
| Robot capability registry | **NEW** `mujoco_scenes/functional_tamp_pipeline/robot_capability_registry.py` |
| System fixed context | `mujoco_scenes/functional_tamp_pipeline/system_context_registry.py` |
| Grounding | `mujoco_scenes/functional_tamp_pipeline/grounding.py` |
| Search | `mujoco_scenes/functional_tamp_pipeline/search.py` |
| Planning | `mujoco_scenes/functional_tamp_pipeline/planning.py` |
| Runtime entry point/provenance | `mujoco_scenes/functional_tamp_pipeline/run.py` |
| Evaluation metrics | `mujoco_scenes/functional_tamp_pipeline/evaluation_metrics.py` |
| Raw semantic offline evaluator | `mujoco_scenes/functional_tamp_pipeline/raw_semantic_evaluation.py` |
| Benchmark driver | `scripts/evaluate_vlm_functional_tamp.py` |
| Kitchen role/relation normalization | `mujoco_scenes/kitchen_vlm_functional_graph.py` |
| Living Room normalization | `mujoco_scenes/environment_vlm_requirements.py` |
| Workshop normalization | `mujoco_scenes/workshop_phase1/requirements.py` |
| Kitchen physical relations | `mujoco_scenes/geometry_relations.py` |
| Living single-payload relation | `mujoco_scenes/region_grounding.py` |
| Living set/proximity relations | `mujoco_scenes/region_ablation2.py` |
| Workshop geometry | `mujoco_scenes/workshop_phase1/geometric_grounding.py` |

---

# Appendix B — Safe Relation Interpreter Pseudocode

```python
def interpret_expressed_relation(
    *,
    domain: str,
    relation_text: str,
    canonical_subject: str,
    canonical_object: str,
    subject_kind: str,
    object_kind: str,
    required: bool,
) -> RelationInterpretation:
    text = normalize_relation_text(relation_text)

    semantic_candidates = semantic_predicate_candidates(
        domain=domain,
        relation_text=text,
    )

    valid_forward = get_valid_predicates_for_endpoints(
        domain,
        canonical_subject,
        canonical_object,
        subject_kind,
        object_kind,
    )

    forward = semantic_candidates & valid_forward

    if is_explicit_multi_concept_relation(text):
        selected = resolve_explicit_multi_predicate_set(
            text,
            semantic_candidates,
            valid_forward,
        )
        if selected:
            return RelationInterpretation(
                raw_relation=relation_text,
                canonical_subject=canonical_subject,
                canonical_object=canonical_object,
                predicates=tuple(selected),
                status="MULTI_PREDICATE_MATCH",
                semantic_evidence=tuple(explain_semantic_match(text, selected)),
                direction_normalized=False,
            )

    if len(forward) == 1:
        pred = next(iter(forward))
        return successful_interpretation(pred, direction_normalized=False)

    if len(forward) > 1:
        pred = disambiguate_using_relation_semantics(text, forward)
        if pred is not None:
            return successful_interpretation(pred, direction_normalized=False)

    if semantic_allows_inverse_orientation(text):
        valid_reverse = get_valid_predicates_for_endpoints(
            domain,
            canonical_object,
            canonical_subject,
            object_kind,
            subject_kind,
        )
        reverse = semantic_candidates & valid_reverse
        if len(reverse) == 1:
            pred = next(iter(reverse))
            return successful_interpretation(pred, direction_normalized=True)

    if required:
        return RelationInterpretation(
            raw_relation=relation_text,
            canonical_subject=canonical_subject,
            canonical_object=canonical_object,
            predicates=(),
            status="UNMAPPABLE_REQUIRED_RELATION",
            semantic_evidence=(),
            direction_normalized=False,
        )

    return RelationInterpretation(
        raw_relation=relation_text,
        canonical_subject=canonical_subject,
        canonical_object=canonical_object,
        predicates=(),
        status="SOFT_OPTIONAL_RELATION",
        semantic_evidence=(),
        direction_normalized=False,
    )
```

Critical invariant:

```python
assert not (
    selected_predicate_is_based_only_on_endpoint_pair
)
```

---

# Appendix C — Safe Operation Interpreter Pseudocode

```python
def interpret_operation_pairing(
    *,
    domain: str,
    operation_text: str,
    source_role: str,
    target_role: str,
    anchor_role: str | None,
) -> OperationInterpretation:
    semantic_candidates = semantic_capability_candidates(
        domain=domain,
        operation_text=normalize_operation_text(operation_text),
    )

    endpoint_valid = capabilities_valid_for_participants(
        domain=domain,
        source_role=source_role,
        target_role=target_role,
        anchor_role=anchor_role,
    )

    candidates = semantic_candidates & endpoint_valid

    if len(candidates) != 1:
        return OperationInterpretation(
            status="UNMAPPABLE_REQUIRED_OPERATION",
            capability=None,
            physical_preconditions=(),
        )

    capability = capability_registry[candidates.pop()]

    return OperationInterpretation(
        status="CANONICAL_OPERATION",
        capability=capability.capability_id,
        physical_preconditions=instantiate_preconditions(
            capability,
            source_role=source_role,
            target_role=target_role,
            anchor_role=anchor_role,
        ),
    )
```

Critical invariant:

```python
if not operation_text.strip():
    do_not_infer_operation_from_endpoints()
```

---

# Appendix D — Search-State Pseudocode

```python
def classify_search_state(graph_f, grounding, contract, inspected):
    if not graph_f.metadata.get("required_contract_complete", False):
        return "CONTRACT_INCOMPLETE_NOT_SEARCHABLE"

    if grounding.complete:
        return "SATISFIED"

    remaining = [r for r in contract.canonical_region_ids if r not in inspected]
    if not remaining:
        return "SEARCH_EXHAUSTED"

    implicated = set(grounding.missing_roles)

    for failure in grounding.unsatisfied_relations:
        implicated.add(failure.subject_role)
        implicated.add(failure.object_role)

    for failure in grounding.unresolved_constraints:
        implicated.update(failure.role_ids)

    for binding_failure in failed_operation_bindings(grounding):
        implicated.update(binding_failure.role_ids)

    searchable = [
        graph_f.nodes[r]
        for r in implicated
        if r in graph_f.nodes
        and graph_f.nodes[r].entity_kind in {"OBJECT", "REGION"}
        and graph_f.nodes[r].semantic_categories
    ]

    return "SEARCH_RECOVERABLE" if searchable else "GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE"
```

---

# Appendix E — Final Failure Attribution Pseudocode

```python
def first_cause(record, offline_semantic_eval):
    if record.infrastructure_error:
        return "EVAL_INFRA_ERROR"

    if not record.gt_feasible:
        return None  # separate infeasible termination diagnostics

    if not offline_semantic_eval.required_task_contract_complete:
        return "TASK_SPECIFICATION_FAILURE"

    if not record.required_contract_complete:
        return "GRAPH_COMPILATION_FAILURE"

    if not record.complete_candidate_grounding:
        if record.search_exhausted and offline_gt_candidate_exists:
            return "OBJECT_DISCOVERY_FAILURE"
        return "FUNCTIONAL_ASSIGNMENT_FAILURE"

    if not record.candidate_plan_valid or not record.full_task_satisfied:
        return "PLANNING_FAILURE"

    return None
```

The offline evaluator is used only after the online run and never feeds runtime decisions.

---

# Appendix F — Final Prompt Skeleton

The exact final prompt should remain short enough for the 9B model and should not include benchmark examples.

```text
You are a vision-language functional task-requirement specification generator.
Return only the JSON object matching the supplied schema. Do not produce an action sequence.

A. TASK CONTRACT
Derive the complete physical task requirements from the instruction before considering what is visible.
Represent every physically distinct participant that has a different causal function as a separate role.
A participant remains required even if no current candidate is visible.
Use OBJECT for selectable/manipulable items, REGION for selectable support/destination areas, and FIXED_TARGET for non-selectable contextual or target references.
Propagate explicit quantities, distinctness, shared-use, and sequential reuse requirements.
Express each task-critical binary dependency between declared roles as a short atomic relation in your own words.
Express each task-required physical transformation/intervention as a short operation phrase with declared source and target roles and an optional contextual anchor.
Do not use simulator instance IDs.
Do not output a linear action sequence.

B. OBSERVATION GUIDANCE
Only after the task contract is complete, use the supplied RGB images to identify currently visible candidates for the declared roles and visible closed/storage regions that may be inspected for missing candidates.
Do not remove a task role because it is not visible.
Do not invent a new task requirement only because an object is visible.
```

No task-domain examples, relation examples, verifier definitions, canonical role names, or runtime predicate names should be appended.

---

# Appendix G — Required Artifacts Per Variant

After the revised pipeline, each run directory should contain where applicable:

```text
run_manifest.json
fm_diagnostics/fm_call_001.json
raw_vlm_response.json
structural_sanitization.json
functional_specification.json
functional_requirement_graph.json
relation_interpretation_trace.json
operation_interpretation_trace.json
observed_graph_before_search.json
trajectory_events.jsonl
observed_scene_graph.json
graph_grounding_result.json
physical_relation_verification_trace.json
executability_analysis.json
symbolic_problem.json
action_sequence/plan.json
action_sequence/replay_validation.json
plan_grounding_audit.json
evaluation_record.json
```

Workshop additionally:

```text
target_anchor_provenance.json
```

The matrix root should contain:

```text
evaluation_records.json
evaluation_records.csv
evaluation_summary.json
failure_analysis.json
invariants.json
main_table.md
pipeline_diagnostic_table.md
first_cause_table.md
provenance_summary.json
```

---

# Appendix H — Reviewer Checklist After Implementation

A separate reviewer should answer YES to all of the following before the final live matrix:

1. Can any endpoint pair create a relation without semantic support from FM text? **Must be NO.**
2. Can any endpoint pair create an operation without FM operation text? **Must be NO.**
3. Can a required unmappable FM relation be silently ignored? **Must be NO.**
4. Can a robot capability add its own physical execution preconditions after an explicit FM operation is mapped? **May be YES, with provenance.**
5. Can online runtime code read GT/reference task graphs? **Must be NO.**
6. Can a fixed anchor be added when FM never expressed the corresponding task context? **Must be NO.**
7. Can UNKNOWN physical evidence satisfy grounding? **Must be NO.**
8. Can search run when required task semantics are incomplete? **Must be NO.**
9. Can search run to find a better alternative after a current candidate fails geometry? **May be YES.**
10. Can `PLANNING_FAILURE` be assigned without complete valid grounding? **Must be NO.**
11. Is the final 32-case matrix clearly labeled development/post-fix confirmation? **Must be YES.**
12. Is held-out evaluation frozen and untouched? **Must be YES for held-out generalization claims.**

---

## Final Implementation Statement

The method to implement is therefore:

```text
ONE open-ended Qwen task-contract call
        ->
strict structural sanitization
        ->
semantic role canonicalization
        ->
relation interpretation from FM text + endpoint filtering
        ->
operation interpretation from FM text + endpoint filtering
        ->
robot capability physical preconditions
        ->
complete auditable G_F
        ->
current observations G_O
        ->
measured TRUE/FALSE/UNKNOWN physical verification
        ->
verified phi*
        ->
automatic search only when a complete contract can be helped by more observations
        ->
single A*
        ->
independent replay
        ->
offline GT-based evaluation only after generation is complete
```

That is the final end-to-end implementation target. No phase may trade away this boundary merely to improve the benchmark score.

## Post-Phase-15 Corrective Performance Pass

The Phase 13 development matrix and Phase 14 additional matrix are **SUPERSEDED FOR FINAL PERFORMANCE CLAIMS**. They remain unchanged as diagnostic historical artifacts. The former Phase 14 “held-out” variants were visible during this corrective work and are therefore relabeled the **post-hoc additional stress-test matrix**, not an untouched held-out result.

| Stage | Status | Files | Tests | Offline replays | Live calls | Result | Commit | Remaining |
|---|---|---|---|---|---|---|---|---|
| C0 forensic audit | Complete | repository/history and preserved report roots | Git state verified | — | 0 | local history preserved and origin synchronized at the pre-corrective SHA | pre-freeze history `3d389ec1` | none |
| C1 V1/V2 evaluation | Complete | `evaluation_contract_adapter.py`, `raw_semantic_evaluation.py`, `fm_schema_v2.py` | corrective V1/V2 suite | archived V1 and V2 | 0 | nested V2 roles, relations, and operation pairings are scored explicitly | corrective freeze commit | none |
| C2 executable contract and first cause | Complete | `evaluation_metrics.py`, evaluator scripts, compiler metadata | evaluator and attribution regressions | included in 79-record rescore | 0 | runtime completeness no longer derives from GT role coverage | corrective freeze commit | none |
| C3 metric and goal scoring | Complete | `evaluation_metrics.py`, evaluator scripts | goal denominator, replay, grounding-evidence tests | included in 79-record rescore | 0 | shared definitions for primary metrics and feasible-only goal coverage | corrective freeze commit | none |
| C4 semantic boundary | Complete | `semantic_compiler.py`, `relation_interpreter.py`, domain adapters | anti-synthesis and interface tests | raw replay | 0 | endpoint signatures filter but do not create semantics | corrective freeze commit | none |
| C5 grounding/search/pruning | Complete | grounding/search evidence paths and pruning tests | exhaustive-vs-pruned randomized equivalence | archived replay | 0 | UNKNOWN remains non-satisfying; pruning preserves valid assignments | corrective freeze commit | none |
| C6 offline rescore | Complete | `scripts/corrective_offline_rescore.py` | record validation | 79 records: 32 V2 development, 15 post-hoc stress, 32 V1 baseline | 0 | new funnel JSON/CSV/Markdown under `corrective_pass_offline_rescore_20260908T072500Z` | corrective freeze commit | none |
| C7 controlled live probe | Complete | new immutable probe roots | six-case comparisons | — | multiple controlled six-case configurations | Prompt 4 plus Config C selected by the prescribed semantic-first ordering | corrective freeze commit | none |
| C8 generic prompt/schema | Complete | `fm_schema_v2.py`, `fm_adapter.py` | prompt/schema and leakage tests | — | six-case probes only | one-call generic decomposition retained; benchmark nouns and verifier descriptions absent from active prompts | corrective freeze commit | none |
| C9 downstream repair | Complete | workshop role mapping, relation interpretation, compiler | focused compiler/grounding/planning regressions | W1/W2 capability retained | 0 additional tuning calls | explicit patient/tool/fastener roles no longer collide; no semantic synthesis | corrective freeze commit | none |
| C10 method freeze | Complete | all corrective implementation and tests | **472 passed, 1 safety-preserving detector skip** in `functional_tamp_pipeline/tests`; 56/56 focused | complete | 0 | method frozen for final evaluation | `8a9f540f` | none |
| C11 final evaluation and reporting | Complete | deterministic post-freeze dataset plus immutable report roots | 15/15 scene definitions built; both matrix invariants VALID | development 32x1 and held-out 15x1 | 47 final live calls | final metrics and provenance audit generated; hash freeze verified; zero prompt leakage | dataset `5932c7ba` | none |

### Frozen inference and runtime identity

- Model weights: `Qwen/Qwen3.5-9B`
- Served alias: `qwen35-9b`
- vLLM: `0.27.2rc1.dev122+g8efa13b70`
- Server context length: 32,768 tokens
- Selected inference configuration: thinking enabled; maximum output 24,576; temperature 0.7; top-p 0.8; top-k 20; presence penalty 1.5; repetition penalty 1.0
- V2 prompt SHA-256: `c2453625a0636c75cdcf56b50979af0c6f57e6ce160a5ccdf0ed3240a82a2bdf`
- V2 schema SHA-256: `27f3dff978bb0d344fd4ed24e54ed49bb6b79f8304aadfe49670f4e7f83e5f73`
- Combined V2 prompt/schema hash: `381770ded79ef1c189fb81ee7b04632b876c02ead68aa6c4b4841e0c6216cdd2`
- Runtime semantic ontology YAML SHA-256: `ab5095cdcf2ed6a2799548ebdd5510062ce488d2c047a1e2e71d997fec44a57d`
- Role-semantic ontology implementation-file SHA-256: `31d8f5f5be7e35909a19330dc2d0809819d441c276fe1dc3a9ef76d6e9dd5266`
- Predicate registry SHA-256: `f8afb189d77138e58994ec525ba7d72b4be26254a2af2d5a9c8b2042c599e639`
- Robot capability registry source-file SHA-256: `5f981a616390757a1b240843bc39a33bf05fc8bd072546b36a17503cc5406dfe`
- Robot capability registry canonical-content SHA-256: `fbe4595e7636dd3955f6e95334868b94fea9e928da38620cfd0e851b7af52537`

No method, prompt, schema, ontology, predicate, capability, grounding, search, planning, or evaluation-definition change is permitted after the corrective freeze. Post-freeze additions may define the new deterministic evaluation dataset and generate reports, but cannot modify the frozen method.

### Final corrective evaluation results

The final development confirmation is `benchmark_reports/final_corrected_32x1_20260908T192500IST_rerun/`. It contains 32 records (20 feasible, 12 infeasible), a clean source-tree provenance flag on every record, one model and prompt-manifest hash throughout, and `invariants.json: VALID`.

The final held-out generalization run is `benchmark_reports/heldout_postfreeze_15x1_20260908T214400IST_rerun/`. Its 15 scene layouts were generated after the method freeze and committed as dataset-only changes. It contains 15 records (9 feasible, 6 infeasible), no pipeline exceptions, a clean source-tree provenance flag on every record, one model and prompt-manifest hash throughout, and `invariants.json: VALID`.

| Matrix | Raw role F1 | Raw relation F1 | Executable contract complete | Feasible success | Goal coverage | Outcome correct | False completion | VLM calls / case | Replans / case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Development 32x1 | 60.2% | 0.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.00 | 0.00 |
| Held-out 15x1 | 62.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.00 | 0.00 |

The principal scientific finding is therefore negative but unambiguous: the corrected 9B one-call system recognizes many required object roles, but it does not reliably express the required relations and operations. The strict compiler consequently rejects incomplete contracts instead of inventing semantics, producing zero full-task successes and zero false completions. All 29 feasible failures across the two final matrices are first attributed to `TASK_SPECIFICATION_FAILURE`; 27 are detailed as `FM_SEMANTIC_OMISSION` and 2 as `FM_STRUCTURAL_ERROR`.

The consolidated paper tables and audits are under `benchmark_reports/final_corrective_analysis_20260908/`. The provenance audit covers all 47 final live records, reports zero prompt leakage, and verifies the frozen hashes.

Two roots are explicitly non-authoritative: `final_corrected_32x1_20260908T185300IST` was invalidated because the source tree changed mid-run, and `heldout_postfreeze_15x1_20260908T213700IST` was aborted after an unsupported dataset placement raised a scene-construction exception. Both are retained only as audit evidence and are excluded from every final metric above.
