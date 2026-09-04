# ViLaIn-TAMP Baseline Implementation Plan for icra-we-ball

Planning basis:

- Source branch inspected: `phase4/execution-integration-replay-contract`
- Inspected HEAD: `d0764889cfb239b716d728d7121c878a30e603d2`
- Final status observed during inspection: `mujoco_scenes/kitchen_ground_truth_execution.py` is modified.
- No files were changed, committed, pushed, or executed as part of this planning turn.
- The public [ViLaIn-TAMP paper](https://arxiv.org/abs/2506.03270), its [HTML version](https://arxiv.org/html/2506.03270), the [earlier official ViLaIn repository](https://github.com/omron-sinicx/ViLaIn), Fast Downward documentation, and relevant repository code were inspected.

The dirty source tree is an immediate branch-creation blocker. The implementation prompt must stop until that local change is resolved by you; it must not stash, reset, discard, or commit it automatically.

---

## 1. Repository Architecture Assessment

### Current Phase-3 path

The current method is centered under [functional\_tamp\_pipeline]\(/home/naren/RA\_iiith/mujoco\_scenes/functional\_tamp\_pipeline):

```
task instruction + initial scene observations
                  |
                  v
         functional specification
                  |
                  v
        G_F construction / validation
                  |
                  v
 initial G_O + task-aware region inspection
                  |
                  v
           ground_graph(G_F, G_O)
                  |
                  v
     phi* and domain-specific role assignment
                  |
                  v
     symbolic problem compilation
                  |
                  v
       deterministic common A*
                  |
                  v
 action plan + replay validation + grounding audit
```

Relevant modules:

- [models.py]\(/home/naren/RA\_iiith/mujoco\_scenes/functional\_tamp\_pipeline/models.py) defines `FunctionalRequirementGraph`, `GraphGroundingResult`, `PipelineResult`, and associated contracts.
- [run.py]\(/home/naren/RA\_iiith/mujoco\_scenes/functional\_tamp\_pipeline/run.py) orchestrates functional specification, observation/search, grounding, planning, audit, and artifact persistence.
- [search.py]\(/home/naren/RA\_iiith/mujoco\_scenes/functional\_tamp\_pipeline/search.py) performs task-aware inspection using requirements derived from `G_F`.
- [grounding.py]\(/home/naren/RA\_iiith/mujoco\_scenes/functional\_tamp\_pipeline/grounding.py) implements graph grounding and phi\* production.
- [planning.py]\(/home/naren/RA\_iiith/mujoco\_scenes/functional\_tamp\_pipeline/planning.py) compiles and invokes the common deterministic A\* planner.
- [symbolic\_planning\_core.py]\(/home/naren/RA\_iiith/mujoco\_scenes/symbolic\_planning\_core.py) supplies the current symbolic state/action planner.
- Domain adapters live in [domains]\(/home/naren/RA\_iiith/mujoco\_scenes/functional\_tamp\_pipeline/domains).

The current normalized action record is:

```
action_index: positive, one-based integer
action_instance_id: stable unique string
operator: uppercase operator name
arguments: ordered list of concrete identifiers
```

This structural representation is reusable. The provenance and meaning of the arguments are not reusable: ViLaIn-TAMP must obtain them from its own PDDL problem and Fast Downward plan.

### Current Phase-4 handoff

[phase4\_execution.py]\(/home/naren/RA\_iiith/mujoco\_scenes/phase4\_execution.py) is explicitly downstream of Phase 3. Its `Phase3Handoff` requires, among other items:

- graph-grounding result;
- phi\* assignment;
- operation bindings;
- observed `G_O`;
- inspection history;
- action plan;
- replay validation;
- plan-grounding audit;
- artifact hashes.

Consequently, ViLaIn-TAMP must not call `load_phase3_handoff()`, fabricate a `Phase3Handoff`, or generate dummy grounding artifacts.

The current domain adapters are likewise method-specific:

- [phase4\_kitchen.py]\(/home/naren/RA\_iiith/mujoco\_scenes/phase4\_kitchen.py) derives kitchen controller inputs from phi\* assignment and operation bindings.
- [phase4\_living\_room.py]\(/home/naren/RA\_iiith/mujoco\_scenes/phase4\_living\_room.py) converts Phase-3 artifacts into the older mobile-execution layout.
- [phase4\_workshop.py]\(/home/naren/RA\_iiith/mujoco\_scenes/phase4\_workshop.py) requires Phase-3 driver/fastener assignments and observed graph data.
- [run\_phase4\_execution.py]\(/home/naren/RA\_iiith/mujoco\_scenes/run\_phase4\_execution.py) is the proposed-method Phase-4 runner.

These cannot serve as the baseline interface.

### Low-level execution infrastructure

The lower-level physical facilities can be reused read-only:

- Kitchen scene and controller infrastructure:
  - [scene\_loader.py]\(/home/naren/RA\_iiith/mujoco\_scenes/scene\_loader.py)
  - [kitchen\_ground\_truth\_execution.py]\(/home/naren/RA\_iiith/mujoco\_scenes/kitchen\_ground\_truth\_execution.py)
  - [kitchen\_object\_manipulation.py]\(/home/naren/RA\_iiith/mujoco\_scenes/kitchen\_object\_manipulation.py)
- Living-room infrastructure:
  - [living\_room\_region\_scene.py]\(/home/naren/RA\_iiith/mujoco\_scenes/living\_room\_region\_scene.py)
  - [living\_room\_mobile\_execution.py]\(/home/naren/RA\_iiith/mujoco\_scenes/living\_room\_mobile\_execution.py), particularly `LivingRoomMobileExecutor`
  - [pick\_motion.py]\(/home/naren/RA\_iiith/mujoco\_scenes/pick\_motion.py), `PickExecutor`
  - [place\_motion.py]\(/home/naren/RA\_iiith/mujoco\_scenes/place\_motion.py), `PlaceExecutor`
  - [mobile\_motion.py]\(/home/naren/RA\_iiith/mujoco\_scenes/mobile\_motion.py), `MobileMoveExecutor`
- Workshop infrastructure:
  - [workshop\_scene.py]\(/home/naren/RA\_iiith/mujoco\_scenes/workshop\_scene.py)
  - [workshop\_ground\_truth\_execution.py]\(/home/naren/RA\_iiith/mujoco\_scenes/workshop\_ground\_truth\_execution.py)
- Generic opening and geometric utilities:
  - [drawer\_motion.py]\(/home/naren/RA\_iiith/mujoco\_scenes/drawer\_motion.py)
  - [open\_motion.py]\(/home/naren/RA\_iiith/mujoco\_scenes/open\_motion.py)
  - generic IK, collision, and trajectory utilities already used by these controllers.

The baseline will use additive wrappers around these components, passing object and destination IDs taken directly from its symbolic actions.

### Scene and variant organization

Benchmark configuration currently resides in:

- [kitchen\_feasibility\_variants.yaml]\(/home/naren/RA\_iiith/mujoco\_scenes/configs/kitchen\_feasibility\_variants.yaml)
- [living\_room\_variants.yaml]\(/home/naren/RA\_iiith/mujoco\_scenes/configs/living\_room\_variants.yaml)
- [workshop\_variants.yaml]\(/home/naren/RA\_iiith/mujoco\_scenes/configs/workshop\_variants.yaml)
- [final\_paper\_variant\_labels.py]\(/home/naren/RA\_iiith/mujoco\_scenes/final\_paper\_variant\_labels.py)

The runner may use a requested variant to instantiate the physical scene. The variant name, feasibility label, hidden contents, and canonical answer must not appear in any ViLaIn prompt.

### Protected/read-only areas

The default implementation plan permits no edits to existing files. In particular, these remain protected:

```
mujoco_scenes/functional_tamp_pipeline/**
mujoco_scenes/symbolic_planning_core.py
mujoco_scenes/phase4_execution.py
mujoco_scenes/phase4_kitchen.py
mujoco_scenes/phase4_living_room.py
mujoco_scenes/phase4_workshop.py
mujoco_scenes/phase4_workshop_entities.py
mujoco_scenes/run_phase4_execution.py
mujoco_scenes/*grounding*
mujoco_scenes/*functional*
mujoco_scenes/*phase1*
mujoco_scenes/*phase2*
mujoco_scenes/*phase3*
existing scene, controller, motion, configuration, and test files
```

Existing controller modules may be imported as read-only infrastructure. They must not be altered for baseline convenience.

---

## 2. ViLaIn-TAMP Method Contract

### Paper-faithful pipeline

The implementation contract is:

```
task instruction L
+ scene observation S
+ fixed PDDL domain D and natural-language domain descriptions
                 |
                 v
object estimation: labels + bounding boxes + object characteristics
                 |
                 v
initial-state estimation as PDDL predicates
                 |
                 v
goal-state estimation as PDDL predicates
                 |
                 v
complete generated PDDL problem
                 |
                 v
Fast Downward symbolic planning
                 |
                 v
sequence-level geometric/motion refinement
                 |
          failure? ---- no ----> final executable plan
             |
            yes
             v
Corrective Planning revises the PDDL problem
             |
             +---- repeat, maximum three corrections
```

The paper treats the PDDL domain as fixed knowledge. ViLaIn generates or revises the problem, not the domain.

### Paper-faithful components

- Object estimator: `Qwen2.5-VL-7B-Instruct`.
- Initial-state estimator: GPT-4o.
- Goal-state estimator: GPT-4o.
- Corrective Planning: GPT-4o.
- Symbolic planner: Fast Downward.
- Geometric planner: MoveIt Task Constructor in the original system.
- Maximum CP corrections in the main condition: three.
- Symbolic and motion/refinement failures can trigger CP.
- Motion feedback contains the failed action/stage and a natural-language explanation such as collision or reachability failure.
- The original problem and correction/error history remain available to later corrections.
- The domain is immutable throughout CP.
- Execution occurs after a symbolically and geometrically feasible sequence is obtained.

The earlier official ViLaIn implementation confirms the broad architecture—PDDL generation, Fast Downward, validation/error capture, and whole-problem correction—but its older GroundingDINO/BLIP2/GPT-4 configuration must not replace the newer ViLaIn-TAMP model configuration.

### Required MuJoCo adaptations

| CategoryDecision      |                                                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ROS/MoveIt            | Do not add ROS or MoveIt solely for this baseline.                                                                                                                                          |
| MTC refinement        | Implement a baseline-owned, cloned-MuJoCo sequence preflight using generic IK, trajectory, and collision checks. Clearly label it an MTC approximation.                                     |
| Learned skills        | Treat POUR, STIR, and SCREW as existing black-box controller skills with explicit start/end feasibility envelopes, matching the paper’s treatment of learned skills.                        |
| Fixed objects         | Known invariant infrastructure such as robot bases, serving surface, workbench, and target hole may be domain knowledge. Variant-dependent presence or contents must come from observation. |
| Object poses          | Estimate rough 3-D object locations from RGB-D detections. Do not give simulator segmentation or body names to the VLM.                                                                     |
| Execution             | Use the existing MuJoCo controllers through baseline-only adapters.                                                                                                                         |
| Physical task success | Use a post-execution hidden evaluator rather than the generated PDDL goal.                                                                                                                  |

### Optional ablations, not part of the primary paper-faithful condition

- Model-matched: use the same model family as the proposed method but make independent calls with ViLaIn prompts.
- Common-solver control: replace Fast Downward with the repository’s A\* only as a clearly labeled solver ablation.
- CP limits `0`, `1`, and `3`.
- No geometric refinement.
- Online execution-feedback CP. This is not primary because the paper applies CP during planning/refinement, before scored execution.

---

## 3. Fair Baseline Boundary

| SAFE SHARED INFRASTRUCTUREMETHOD-SPECIFIC INFORMATION — MUST NOT SHARE                                             |                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Same MuJoCo XML, scene constructors, physics engine, timestep, and randomization seed                              | `G_F`, its natural-language specification, nodes, edges, constraints, or hashes                          |
| Same raw RGB/depth and camera calibration                                                                          | `G_O`, observed graph, semantic/geometric relation evaluations                                           |
| Same camera names and task-independent camera trajectories                                                         | `ground_graph()` results or internal diagnostics                                                         |
| Legal inventory of possible storage regions and fixed scene infrastructure                                         | phi\*, its assignment, witnesses, alternatives, or role identities                                       |
| Generic enumeration of movable bodies for execution association                                                    | Task-aware detector vocabulary generated from `G_F`                                                      |
| Generic body/AABB/contact queries after the baseline has selected a concrete PDDL object                           | Functional satisfaction results or unsatisfied functional requirements                                   |
| Existing generic IK, collision checking, RRT, grasp, pick, place, mobile-base, drawer-opening, and controller code | Proposed-method inspection ranking, early stopping, or search history                                    |
| Fixed, generic PDDL action physics                                                                                 | Proposed-method operation bindings                                                                       |
| Existing execution logging and telemetry machinery where it does not require Phase-3 artifacts                     | `graph_grounding_result.json`, plan-grounding audit, functional witnesses, canonical grounding witnesses |
| Task-independent region order used by the full-inspection condition                                                | GT role assignments, GT selected tools, compatible-object answer sets during planning                    |
| Existing controller semantics for POUR/STIR/SCREW                                                                  | Proposed method’s generated action sequence                                                              |
| Hidden benchmark evaluator after baseline planning and execution terminate                                         | Variant feasibility label or expected answer in any model prompt                                         |
| Variant ID inside runner/scene factory and final evaluator                                                         | Hidden evaluator results fed back to CP                                                                  |
| Fixed benchmark task instruction                                                                                   | Simulator body names, segmentation masks, hidden poses, or hidden storage contents supplied to the VLM   |
| Method-independent execution-effect ledger populated only after successful controller actions                      | Proposed-method symbolic state, generated predicates, plan replay, or execution handoff                  |

Additional enforcement:

- Baseline package production code must not import `mujoco_scenes.functional_tamp_pipeline`.
- It must not import any `phase4_*` adapter or call `load_phase3_handoff`.
- A boundary test should scan baseline imports and source text for forbidden identifiers.
- Hidden evaluator modules must never be passed to the interpreter, planner, refiner, or CP loop.

---

## 4. Target Architecture

```
                              SAME BENCHMARK VARIANT
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
        PROPOSED METHOD — READ ONLY             ViLaIn-TAMP BASELINE
        ---------------------------             --------------------
        raw observations                         raw observations
                    |                                     |
                    v                                     v
              G_F construction                    object estimation
                    |                                     |
              task-aware search                  initial/goal estimation
                    |                                     |
                    v                                     v
                   G_O                            generated PDDL problem
                    |                                     |
                    v                                     v
              ground_graph()                       Fast Downward
                    |                                     |
                    v                                     v
             phi* / assignments               cloned-scene refinement
                    |                                     |
                    v                               failure -> CP
             existing A*                                  |
                    |                                     v
                    v                         baseline symbolic actions
           Phase-3 artifacts                              |
                    |                                     v
                    v                         execution projection:
           existing Phase-4                    PDDL IDs -> body IDs
                    |                            from baseline evidence
                    |                                     |
                    +------------------+------------------+
                                       |
                                       v
                     SHARED LOW-LEVEL MUJOCO CONTROLLERS
                                       |
                                       v
                            PHYSICAL TERMINAL STATE
                                       |
                    +------------------+------------------+
                    |                  |                  |
                    v                  v                  v
            generated-goal     execution integrity   hidden actual-task
              evaluation          evaluation             evaluation
```

The paths meet only at:

1. raw benchmark observations;
2. generic physical/controller infrastructure;
3. final physical state and independent evaluation.

---

## 5. Proposed New File Tree

All implementation files are additive.

```
mujoco_scenes/
├── baselines/
│   ├── __init__.py
│   └── vilain_tamp/
│       ├── __init__.py
│       ├── README.md
│       ├── config.py
│       ├── contracts.py
│       ├── artifacts.py
│       ├── observations.py
│       ├── fm.py
│       ├── prompts.py
│       ├── interpreter.py
│       ├── pddl.py
│       ├── planner.py
│       ├── identity.py
│       ├── refinement.py
│       ├── corrective_planning.py
│       ├── runner.py
│       ├── requirements-vilain-tamp.txt
│       ├── configs/
│       │   ├── paper_faithful.yaml
│       │   └── model_matched.yaml
│       ├── domains/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── kitchen/
│       │   │   ├── domain.pddl
│       │   │   └── knowledge.yaml
│       │   ├── living_room/
│       │   │   ├── domain.pddl
│       │   │   └── knowledge.yaml
│       │   └── workshop/
│       │       ├── domain.pddl
│       │       └── knowledge.yaml
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── kitchen.py
│       │   ├── living_room.py
│       │   └── workshop.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── kitchen.py
│       │   ├── living_room.py
│       │   └── workshop.py
│       └── tests/
│           ├── __init__.py
│           ├── fixtures/
│           │   ├── object_estimates/
│           │   ├── generated_problems/
│           │   ├── fast_downward/
│           │   └── corrections/
│           ├── test_boundary.py
│           ├── test_contracts.py
│           ├── test_artifacts.py
│           ├── test_domains.py
│           ├── test_observations.py
│           ├── test_fm.py
│           ├── test_interpreter.py
│           ├── test_planner.py
│           ├── test_identity.py
│           ├── test_refinement.py
│           ├── test_corrective_planning.py
│           ├── test_kitchen_execution.py
│           ├── test_living_room_execution.py
│           ├── test_workshop_execution.py
│           ├── test_evaluation.py
│           └── test_offline_pipeline.py
└── run_vilain_tamp_baseline.py
```

Responsibilities:

- `config.py`: typed, validated configuration and model-condition selection.
- `contracts.py`: minimal frozen dataclasses/enums shared across baseline stages.
- `artifacts.py`: atomic artifact writing, hashing, manifest assembly, and event logging.
- `observations.py`: initial-only and fixed-full-inspection acquisition protocol.
- `fm.py`: independent Qwen/OpenAI model transports, call metrics, and mock interfaces.
- `prompts.py`: paper-derived object, initial-state, goal-state, and CP prompt builders.
- `interpreter.py`: object → initial → goal → complete PDDL problem orchestration.
- `pddl.py`: PDDL extraction, structural parsing, immutable-domain checks, and atom normalization.
- `planner.py`: external Fast Downward and VAL subprocess adapters plus plan parser.
- `identity.py`: detection-derived PDDL-object to simulator-entity association.
- `refinement.py`: cloned-scene, sequence-level geometric preflight.
- `corrective_planning.py`: bounded CP state machine and history construction.
- `runner.py`: baseline-only orchestration without Phase-3 or Phase-4 handoffs.
- `requirements-vilain-tamp.txt`: isolated optional dependencies; the existing requirements stay unchanged.
- `configs/*.yaml`: declared experimental conditions, never benchmark answers.
- `domains/registry.py`: loads only fixed domains and their descriptions.
- `domain.pddl`: immutable generic dynamics.
- `knowledge.yaml`: natural-language predicate/action/object descriptions used by ViLaIn.
- `execution/base.py`: controller-neutral execution protocol and effect ledger.
- Domain execution modules: direct controller adapters from the baseline plan.
- `evaluation/*`: post-terminal hidden benchmark evaluators.
- `run_vilain_tamp_baseline.py`: thin CLI entry point.
- Tests: offline/unit-first coverage, including explicit anti-leakage tests.

---

## 6. Interfaces and Data Contracts

Use frozen dataclasses, enums, tuples, and JSON `to_dict()` serialization. Do not introduce a general workflow framework.

### `ViLaInObservation`

Fields:

```
domain
observation_mode
stage_id
camera_frames[]
opened_region_id | null
capture_timestamp
inspection_ordinal | null
content_hash
```

Each camera frame contains paths to RGB, depth, and calibration. Internal run metadata may know the variant; the object given to model prompt builders must not.

### `ObjectEstimate`

```
object_id                 # ViLaIn-created stable ID
label
pddl_type
description
detections[]              # camera, xyxy box, confidence
estimated_centroid_m | null
centroid_covariance | null
observation_stage_ids[]
status                    # OBSERVED / AMBIGUOUS / LOST
```

No `functional_role`, `phi`, or expected-action fields.

### `GeneratedPDDLProblem`

```
attempt_index
source                    # INITIAL or CP
domain_name
domain_sha256
problem_text
declared_objects
initial_atoms
goal_atoms
raw_response_artifact
problem_sha256
```

### `PDDLValidationResult`

```
valid
stage                     # INTERNAL, TRANSLATOR, PLAN_VAL
diagnostics[]
exit_code | null
stdout_artifact | null
stderr_artifact | null
elapsed_seconds
```

### `SymbolicAction`

The existing four-field shape is sufficient:

```
action_index
action_instance_id
operator
arguments[]
```

Use IDs such as `vilain_00_001_pick_from`, making provenance visibly independent.

### `SymbolicPlan`

```
attempt_index
planner_name
planner_version
search_configuration
actions[]
plan_cost | null
planner_time_seconds
raw_plan_artifacts[]
plan_sha256
```

### `RefinementFailure`

```
attempt_index
action_index | null
action_instance_id | null
operator | null
arguments[]
stage
reason_code
summary
robot_or_arm | null
involved_entities[]
collision_pair | null
numeric_evidence{}
backend_trace_artifact | null
recoverable_by_problem_revision
```

Stages include `ENTITY_RESOLUTION`, `GRASP_GENERATION`, `IK`, `TRAJECTORY`, `COLLISION`, `SKILL_ENVELOPE`, and `STATE_TRANSITION`.

### `CorrectivePlanningAttempt`

```
correction_index
initial_problem_sha256
prior_problem_sha256
failure
history_problem_hashes[]
history_error_hashes[]
model
request_artifact
raw_response_artifact
revised_problem_sha256 | null
status
latency_and_usage
```

### `BaselineExecutionPlan`

```
selected_attempt_index
domain
symbolic_plan
refinement_certificate
normalized_actions[]
```

### `ExecutionProjection`

For each action:

```
action_instance_id
pddl_operator
pddl_arguments[]
controller_operator
controller_arguments[]
resolved_entities[]
binding_method
binding_confidence
binding_evidence_artifacts[]
skill_parameters{}
```

This is an execution bridge, not phi\*. It must never contain roles such as `coffee_stirrer`, `selected_driver`, or references to functional requirements. A tool becomes relevant because it appears in `STIR(tool,mug)` or `SCREW(driver,fastener,target)`.

### `BaselineRunResult`

```
run_status
selected_attempt_index | null
planning_status
refinement_status
execution_status
generated_goal_status
benchmark_status
artifact_paths
metrics
```

### `BenchmarkGoalEvaluation`

```
domain
variant
ground_truth_feasibility
requirement_checks[]
actual_task_success
predicted_infeasible
correct_infeasibility_recognition
benchmark_outcome_correct
evidence_artifacts[]
```

This structure is created only after planning/execution terminate and is never supplied to CP.

---

## 7. PDDL Design

All three domains use fixed generic schemas. Object existence, locations, compatibility, contents, and task goals belong in generated problem files.

### Kitchen

Types:

```
entity
movable - entity
vessel source utensil - movable
content
location
surface storage - location
```

Core predicates:

```
(at object location)
(holding object)
(handempty)
(accessible location)
(open storage)
(contains vessel content)
(can-dispense source content)
(can-stir utensil vessel)
(can-serve-with utensil vessel)
(inside utensil vessel)
(stirred vessel)
```

Operators:

- `open-storage(storage)`
- `pick-from(movable, location)`
- `place-on(movable, surface)`
- `pour(source, vessel, content)`
- `stir(utensil, vessel)`
- `place-in(utensil, vessel)`

Allowed fixed physics:

- picking requires accessibility and an empty hand;
- pouring transfers/establishes content;
- stirring requires a suitable tool and target;
- placing an utensil inside a vessel requires compatibility;
- an object cannot be in two locations simultaneously.

Must be inferred/generated:

- which vessels, sources, and utensils exist;
- where each is;
- source contents;
- utensil-vessel compatibility;
- which concrete vessels are coffee/soup targets;
- the complete task goal;
- unavailable or inaccessible objects.

The domain must not encode “object X is the coffee stirrer” or per-variant selections.

### Living room

Types:

```
movable
cup saucer remote - movable
location
support - location
seat
```

Predicates:

```
(at movable location)
(holding movable)
(handempty)
(present location)
(accessible location)
(supports support movable)
(personal-to support seat)
(shared support)
```

Operators:

- `pick-from(movable, location)`
- `place-on(movable, support)`

Allowed fixed physics:

- placement requires a present, accessible support;
- the movable must be held;
- a released object becomes located on its destination.

Must be generated:

- detected payload objects;
- support presence;
- left/right/shared relations;
- concrete goal mapping of cups/saucers/remotes to supports;
- any missing support or object.

No per-variant table availability may be hard-coded.

### Workshop

Types:

```
movable
driver fastener - movable
location
storage surface target - location
```

Predicates:

```
(at movable location)
(holding movable)
(handempty)
(accessible location)
(open storage)
(driver-compatible driver fastener)
(fits fastener target)
(can-reach driver target)
(inserted fastener target)
(fastened fastener target)
```

Operators:

- `open-storage(storage)`
- `pick-from(movable, location)`
- `insert(fastener, target)`
- `drive(driver, fastener, target)`
- `place-on(movable, surface)`

Allowed fixed physics:

- insertion requires fastener/target fit;
- driving requires the compatible driver, inserted fastener, and reachability;
- tool placement releases the tool;
- target hole and main workbench may be fixed known infrastructure.

Must be generated:

- observed drivers and fasteners;
- storage locations;
- compatibility and reachability beliefs;
- target goal;
- concrete driver and fastener choice.

For full inspection, observation stages are ordered. The goal estimator may use that order to interpret “first compatible encountered.” The domain must not contain the canonical selected driver as a fixed fact.

### Corrective geometric predicates

The fixed schemas may include generic predicates such as:

```
(blocked movable location)
(cannot-reach movable target)
(collision-risk movable movable)
```

These may only be inserted into a revised problem when supported by a recorded refinement failure. CP cannot modify the predicate or action definitions.

---

## 8. Observation / Partial-Observability Protocol

Two conditions should be implemented and reported separately.

### Primary: `initial_observation_only`

This is the paper-fidelity condition.

- Capture the fixed canonical RGB-D camera set from the initial scene.
- Storage remains closed.
- Supply raw RGB images to the object estimator.
- Depth and calibration are used only by the generic pose/identity bridge.
- Do not perform active inspection.
- Do not provide a list of hidden contents or simulator segmentation.
- The VLM sees the task instruction, fixed domain descriptions, and initial images.

This condition will predictably struggle when benchmark-critical objects are hidden. That is a meaningful limitation of applying the published single-observation method to this benchmark.

### Stronger/fairness condition: `fixed_full_inspection`

This controls for observability without donating the proposed method’s search policy.

Protocol:

1. Capture initial views.
2. Open every legal inspectable region in a fixed domain-specific order.
3. Capture the same camera set after each opening.
4. Never stop early because the task appears solvable.
5. Never reorder regions using the task, a model, `G_F`, or detected objects.
6. Give the object estimator raw observations from all stages.
7. Log every opening and its cost.

Fixed orders:

```
Kitchen:
D1 -> D2 -> C2 -> B1 -> C1

Workshop:
LEFT_DRAWER -> RIGHT_DRAWER -> TOOL_CABINET

Living room:
no storage-opening sequence; use the canonical initial multiview capture
```

The inspection scene should be a planning copy. Scored execution starts from a fresh scene. If open storage is required for executing the selected plan, replay the same fixed openings as observation-preparation actions and log them separately from task actions.

### Information presented to models

Models may receive:

- task instruction;
- RGB images;
- image/stage ordering;
- natural-language camera/view descriptions;
- fixed domain predicate/action descriptions;
- object estimates generated by their own previous stage.

Models must not receive:

- variant labels such as `F3` or `I1`;
- object body names;
- XML joint/body metadata;
- masks from simulator segmentation;
- ground-truth hidden contents;
- proposed-method inspection or grounding artifacts.

Recommendation: use initial-observation-only as the principal ViLaIn-TAMP baseline and fixed-full-inspection as the stronger observation-matched/fairness condition. Report both, not a pooled score.

---

## 9. Perception and Object-Identity Bridge

### Perception path

```
RGB camera frames
        |
        v
Qwen2.5-VL labels + boxes + descriptions
        |
        v
stable ViLaIn IDs: mug_1, mug_2, spoon_1, ...
        |
        v
depth-backed 3-D centroid estimates
        |
        v
multi-view association of the same ViLaIn object
```

IDs must be stable and deterministically formed from normalized label, observation order, and spatial ordering.

### Simulator bridge

The physical execution layer then associates each already-created ViLaIn object with a simulator entity using:

- bbox/depth-derived 3-D centroid;
- visible candidate body AABBs;
- coarse generic class compatibility;
- one-to-one assignment;
- distance and ambiguity thresholds;
- observation-stage visibility;
- explicit confidence/evidence logging.

Allowed fixed entity metadata:

- body is movable vs fixed;
- broad executable class such as vessel, tool, fastener, support;
- geometry/AABB;
- public fixed-location identities such as main workbench or serving surface.

Forbidden metadata:

- “best stirrer”;
- selected coffee/soup role;
- compatible driver answer;
- expected goal destination;
- proposed-method assignments or relation scores.

Ambiguous or unresolved association is a baseline failure. It must not silently select the object that makes the plan succeed.

The mapping is used for execution only. Prompt builders receive ViLaIn IDs and visual descriptions, not simulator body names.

---

## 10. Symbolic Planning

### Validation pipeline

For every generated/revised problem:

1. Extract exactly one PDDL problem from the raw response.
2. Perform internal structural parsing:
   - balanced forms;
   - correct problem/domain declaration;
   - declared objects;
   - recognized types and predicates;
   - valid arity;
   - non-empty goal;
   - no domain modification.
3. Run Fast Downward translation.
4. Run Fast Downward search if translation succeeds.
5. Normalize all plan files.
6. Validate the selected plan with VAL.
7. Persist all stdout, stderr, commands, return codes, and timings.

### Fast Downward configuration

Primary:

- Fast Downward, pinned and external to the repository.
- Satisficing LAMA-family search.
- Recommended locked invocation: Fast Downward 24.06 with `--alias lama-first`.
- Overall timeout: 200 seconds, inherited as a disclosed implementation choice from the earlier ViLaIn setup rather than claimed as paper-specified.
- Deterministic process environment and logged seed where supported.

The paper identifies Fast Downward but not a fully reproducible release/search configuration. Therefore exact version, alias, command, and binary hash must be reported.

### Plan parsing

Accept standard forms such as:

```
0: (PICK-FROM MUG_1 COUNTERTOP) [1]
```

Normalize case and emit the four-field `SymbolicAction` structure. Ignore comments and cost lines. Reject:

- undeclared objects;
- unknown actions;
- incorrect arity;
- duplicate action indices;
- empty plans unless the initial state satisfies the goal.

### Error handling

CP-eligible:

- malformed problem PDDL;
- translator/type/predicate errors;
- valid but unplannable problem;
- no plan;
- VAL-invalid plan;
- geometric/refinement failure.

Infrastructure-terminal, not CP:

- missing Fast Downward/VAL executable;
- unavailable model service;
- artifact I/O failure;
- subprocess launch failure;
- global resource cancellation.

A planner timeout should be terminal `SYMBOLIC_TIMEOUT` by default; treating it as a semantic correction trigger would confound planner resource limits with PDDL correctness.

### Optional A\* control

The existing A\* may be wrapped only in a separate `common_solver_control` condition. It must not be the primary ViLaIn result and must not consume proposed-method compiled states or actions.

---

## 11. Geometric Refinement

### Required separation

```
symbolic planning
      |
      v
planning-copy/cloned-scene refinement
      |
      v
refinement certificate or structured failure
      |
      v
fresh live scene for scored execution
```

Actual scored execution must not serve as the primary refinement loop.

### MuJoCo analogue of MTC

Implement a `MuJoCoSequenceRefiner` over a fresh cloned `MjModel`/`MjData`:

For each action, in sequence:

1. Resolve the ViLaIn entity.
2. Generate generic grasp or interaction candidates.
3. Test IK/reachability.
4. Plan collision-checked approach/transfer/retreat paths.
5. Test object-centric start/end skill envelopes.
6. Apply the predicted terminal transition to the planning copy.
7. Continue from that predicted/refined state.

Action stage mappings:

| Symbolic actionRefinement stages |                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------ |
| OPEN-STORAGE                     | handle pose, reachability, opening trajectory, collision                       |
| PICK-FROM                        | grasp candidates, pregrasp IK, approach, attach, retreat                       |
| PLACE-ON / INSERT                | transfer, target pose candidates, IK, collision, release/insert envelope       |
| POUR                             | source grasp, target-relative pour pose, reachability, collision, return pose  |
| STIR                             | tool grasp, target-relative insertion/stir envelope, reachability, collision   |
| DRIVE                            | driver grasp, fastener-axis alignment, reachability, collision, drive envelope |

POUR, STIR, and DRIVE remain black-box skills. Refinement validates their object-centric preconditions and planned approach/end poses rather than attempting to learn their internal dynamics.

### Approximation label

This must be reported as:

> “MuJoCo cloned-scene sequence preflight, an adaptation of ViLaIn-TAMP’s MoveIt Task Constructor refinement.”

It is not an exact MTC reproduction.

### Refinement output

On success:

- per-stage candidate/trajectory references;
- resolved entity IDs;
- chosen grasp/target pose;
- collision/reachability certificate;
- predicted terminal state;
- elapsed time.

On failure:

- `RefinementFailure` with the failed action, stage, collision pair or reachability evidence, and an actionable summary.

Do not use proposed-method semantic/geometric requirement checks.

---

## 12. Corrective Planning

### Loop

```
P_initial = first generated PDDL problem
history_problems = []
history_errors = []
correction_count = 0

attempt TAMP(P_initial)

while failed and correction_count < 3:
    CP receives:
        immutable fixed domain + descriptions
        task instruction
        original object estimates
        P_initial
        current failed problem
        structured current failure
        prior problem/error history
    CP returns one complete revised PDDL problem
    validate and attempt TAMP(revised_problem)
    append problem/error history
    correction_count += 1
```

The primary condition allows three CP calls, hence at most four total TAMP attempts.

### Trigger conditions

- PDDL extraction or structural validation failure.
- Fast Downward translator failure.
- Symbolic no-plan result.
- VAL-invalid plan.
- Entity-resolution failure attributable to hallucinated/ambiguous problem objects.
- IK, trajectory, collision, grasp, or skill-envelope refinement failure.

### CP inputs

- Fixed domain PDDL, read-only.
- Natural-language descriptions of domain types, predicates, and actions.
- Task instruction.
- The original baseline object estimates.
- Original initial problem.
- Current problem.
- Current structured error.
- All prior revised-problem hashes/text and error summaries.
- Failed plan trace, including the failed action and stage.
- No hidden benchmark evaluation.

### CP output

Exactly one complete replacement PDDL problem. Raw output is retained. CP does not output an action plan.

### Allowed modifications

- object declarations;
- initial-state predicates;
- goal predicates;
- geometric constraint facts supported by feedback;
- removal of hallucinated objects or facts;
- alternate concrete goal objects, if justified by observation/error history.

### Forbidden modifications

- domain name or domain text;
- action or predicate schemas;
- addition of unknown predicates/types;
- Fast Downward settings;
- controller code;
- observation history;
- task instruction;
- hidden GT facts;
- proposed-method artifacts;
- explicit action sequence;
- benchmark evaluator conditions.

### Termination

- Success: symbolically valid, VAL-valid, fully refined plan.
- Failure: three correction calls exhausted.
- Failure: repeated identical problem hash without new supported information.
- Failure: unrecoverable infrastructure error.
- Failure: CP output cannot be extracted or validated; that invalid revision consumes one CP iteration.
- Execution failure after refinement is scored but does not trigger CP in the primary condition.

---

## 13. Baseline → Execution Integration

The existing Phase-4 handoff cannot be used. The baseline introduces its own direct projection.

### Projection path

```
Fast Downward action
        |
        v
normalized SymbolicAction
        |
        v
domain-specific syntactic projection
        |
        v
ViLaIn object ID -> simulator body association
        |
        v
controller operation + concrete controller arguments
```

### Kitchen projection

| PDDL actionNeutral/controller action |                          |
| ------------------------------------ | ------------------------ |
| `OPEN-STORAGE(region)`               | `OPEN(region)`           |
| `PICK-FROM(object, location)`        | `PICK(object)`           |
| `PLACE-ON(object, surface)`          | `PLACE(object, surface)` |
| `POUR(source, target, content)`      | `POUR(source, target)`   |
| `STIR(tool, target)`                 | `STIR(tool, target)`     |
| `PLACE-IN(tool, vessel)`             | `PLACE(tool, vessel)`    |

Controller input is derived from action arguments:

- coffee and water sources are the objects appearing as POUR sources;
- targets are the vessels appearing in POUR/STIR;
- stirrer-target pairing is exactly `STIR(tool,target)`;
- soup utensil-target pairing is exactly `PLACE-IN(tool,vessel)`;
- serving destinations are exactly those in `PLACE-ON`.

A baseline-only `KitchenControllerContract` should duck-type the controller fields where necessary. It must not import a GT planner assignment or call `assignment_from_handoff()`.

### Living-room projection

| PDDL actionController action   |                           |
| ------------------------------ | ------------------------- |
| `PICK-FROM(payload, location)` | `PICK(payload)`           |
| `PLACE-ON(payload, support)`   | `PLACE(payload, support)` |

Do not call `run_mobile_execution()` with fabricated Phase-1/Phase-2 directories. The additive adapter should invoke the existing mobile, pick, and place executor classes directly.

### Workshop projection

| PDDL actionController action      |                                   |
| --------------------------------- | --------------------------------- |
| `OPEN-STORAGE(region)`            | `OPEN(region)`                    |
| `PICK-FROM(object, location)`     | `PICK(object, location)`          |
| `INSERT(fastener, target)`        | `PLACE(fastener, target)`         |
| `DRIVE(driver, fastener, target)` | `SCREW(driver, fastener, target)` |
| `PLACE-ON(driver, surface)`       | `PLACE(driver, surface)`          |

The driver and fastener needed by the controller are taken directly from `DRIVE(driver,fastener,target)`. A baseline-only controller contract may expose these fields to the existing dispatcher, but it must not represent them as a functional assignment.

### Execution-effect ledger

Some benchmark effects—liquid transfer and stirring—are not physically simulated as particles/fluids. The common backend should emit a method-independent, controller-certified effect only after a physical action succeeds:

```
POUR_COMPLETED(source, target, content)
STIR_COMPLETED(tool, target)
DRIVE_COMPLETED(driver, fastener, target)
```

The ledger is created during execution. It is not initialized from the generated goal and is not fed back to planning.

### Failure behavior

- Unresolved entity: fail the action; do not guess.
- Unsupported action: fail with `UNSUPPORTED_CONTROLLER_ACTION`.
- Controller exception: record and terminate execution at the first failure.
- Postcondition failure: record separately from controller return status.
- No CP after scored execution in the primary condition.

---

## 14. Independent Benchmark Evaluator

The evaluator must run only after planning and execution terminate. It receives:

- final MuJoCo state;
- contact/pose data;
- method-independent execution-effect ledger;
- hidden benchmark specification/variant metadata.

It does not receive the generated PDDL goal as its authority.

### Kitchen

Hidden true requirements:

- two distinct coffee vessels exist and are served in the correct serving area;
- each received required coffee ingredients through successful pours;
- each was successfully stirred with a physically suitable utensil;
- two distinct soup vessels are served;
- each soup vessel contains a suitable eating utensil;
- distinctness/reuse constraints follow the benchmark task, not the generated PDDL;
- no required object remains held;
- required placements are physically stable and supported.

Because fluids are abstracted, liquid/stirring checks combine physical execution success with the controller-certified ledger.

### Living room

Hidden true requirements:

- left-side cup and saucer are physically supported by the left personal side table;
- right-side cup and saucer are physically supported by the right personal side table;
- remote is physically supported by the shared coffee table;
- support presence and payload availability reflect the actual variant;
- objects are released, stable, inside the support footprint, free of floor contact, and free of invalid penetration.

Reuse the same method-independent physical ON criteria as the benchmark, not the planner’s `at` atom.

### Workshop

Hidden true requirements:

- a physically compatible fastener is inserted in the target joint/hole;
- insertion depth and tip/head orientation satisfy benchmark tolerances;
- the joint is repaired/fastened;
- the used driver is physically compatible;
- the selected driver is left safely on the main workbench;
- for the fixed-inspection condition, “first compatible encountered” is checked against the fixed observation order;
- no object remains held.

### Three distinct outcomes

Every run records:

1. **Planner-generated goal satisfaction**
   Whether the final symbolic/physical interpretation satisfies the goal ViLaIn wrote.
2. **Execution integrity**
   Whether each projected action executed, its controller and physical postconditions held, and no unresolved bindings occurred.
3. **Actual benchmark task success**
   Whether the hidden domain evaluator confirms the true physical task.

For infeasible variants, additionally report:

- baseline predicted infeasible;
- GT task infeasible;
- correct infeasibility recognition.

A no-plan outcome on a feasible variant is not a success.

---

## 15. Artifact / Provenance Layout

Default run root:

```
runs/vilain_tamp/<domain>/<variant>/<observation_mode>/<model_condition>/<run_id>/
├── baseline_manifest.json
├── run_config.json
├── events.jsonl
├── observation_manifest.json
├── observations/
│   ├── inspection_trace.json
│   └── stages/
│       ├── 000_initial/
│       │   └── cameras/<camera>/
│       │       ├── rgb.png
│       │       ├── depth.npy
│       │       └── camera.json
│       └── 001_<region>/...
├── perception/
│   ├── request.json
│   ├── raw_response.txt
│   ├── model_metadata.json
│   └── object_estimates.json
├── interpreter/
│   ├── initial_state_request.json
│   ├── initial_state_raw.txt
│   ├── initial_state.pddlfrag
│   ├── goal_request.json
│   ├── goal_raw.txt
│   ├── goal.pddlfrag
│   ├── domain.pddl
│   └── problem_initial.pddl
├── attempts/
│   ├── 00_initial/
│   │   ├── problem.pddl
│   │   ├── pddl_validation.json
│   │   ├── planner/
│   │   │   ├── command.json
│   │   │   ├── stdout.txt
│   │   │   ├── stderr.txt
│   │   │   ├── sas_plan*
│   │   │   ├── symbolic_plan.json
│   │   │   └── plan_validation.json
│   │   ├── refinement/
│   │   │   ├── refinement.json
│   │   │   ├── failures.json
│   │   │   └── traces/
│   │   └── execution_projection.json
│   └── 01_cp/...
├── corrective_planning/
│   └── attempt_01/
│       ├── request.json
│       ├── history_manifest.json
│       ├── raw_response.txt
│       └── revised_problem.pddl
├── final_action_plan.json
├── execution_projection.json
├── execution/
│   ├── entity_resolution.json
│   ├── execution_trace.json
│   ├── effect_ledger.json
│   └── execution_result.json
├── benchmark/
│   ├── generated_goal_evaluation.json
│   └── benchmark_goal_evaluation.json
├── metrics.json
└── baseline_run_result.json
```

`baseline_manifest.json` records:

- repository commit and dirty status;
- config hashes;
- domain/problem/plan hashes;
- model names and immutable snapshots;
- request IDs, token usage, call latency;
- Qwen checkpoint/revision;
- Fast Downward and VAL versions, binary paths, hashes, and commands;
- Python/package versions;
- MuJoCo version;
- seeds;
- observation mode;
- CP limit;
- all material artifact hashes;
- implementation adaptation label for MuJoCo refinement.

Raw images and prompts are retained. Secrets and API keys are never persisted.

---

## 16. Metrics and Experimental Conditions

### Required metrics

- Actual benchmark task success.
- Feasible-task physical success rate.
- Correct infeasibility recognition.
- Generated-goal satisfaction.
- PDDL extraction and syntax validity.
- Fast Downward translation validity.
- Plannability.
- VAL plan validity.
- Refinement success.
- Failure stage distribution.
- CP calls and iterations.
- Success by CP iteration.
- Object-estimator, initial-state, goal-state, and CP FM calls.
- Input/output tokens and API cost.
- FM latency.
- Symbolic planning time.
- Geometric planning/refinement time.
- Execution time.
- End-to-end time.
- Execution action failures and physical postcondition failures.
- Entity-resolution ambiguity/failure.
- Number of inspected regions.
- Inspection travel/opening/time cost.
- Symbolic plan length.
- Controller action count.

### Primary conditions

1. `paper_faithful_initial_only_cp3`
   - Qwen2.5-VL-7B-Instruct object estimator.
   - GPT-4o initial/goal/CP.
   - Fast Downward.
   - MuJoCo cloned-scene refinement.
   - maximum three CP corrections.
   - initial observations only.
2. `paper_faithful_fixed_full_inspection_cp3`
   - identical method/model configuration;
   - fixed, task-independent complete inspection;
   - reported as stronger/fairness condition.

The paper names GPT-4o but does not provide a fully reproducible snapshot. Use `gpt-4o-2024-08-06` as the recommended frozen pre-paper snapshot, explicitly report it as an implementation choice, and never use the moving `gpt-4o` alias for final experiments. OpenAI documents the available GPT-4o snapshots and multimodal input support on the [official GPT-4o model page](https://developers.openai.com/api/docs/models/gpt-4o).

Run five independent model generations per benchmark instance, matching the paper’s repeated-sampling evaluation style. Treat each as a separate run and preserve its seed/request metadata.

### Optional ablations

- CP limits 0 and 1.
- Model-matched ViLaIn prompts using the proposed method’s model family, with new independent calls.
- Original older ViLaIn perception configuration.
- No geometric refinement.
- Common A\* solver control.
- Oracle/full visibility as an upper bound, clearly labeled and not a primary baseline.
- Online execution-feedback CP, clearly marked non-paper-faithful.
- Strict versus assisted low-level execution, if the proposed method is also evaluated under the same controller policy.

---

## 17. IMPLEMENTATION ROADMAP

Recommended total: **15 stages**. None includes a full baseline simulation or expensive experiment.

### Stage 1 — Baseline Boundary, Contracts, and Configuration

**Goal:**
Create the isolated package skeleton, core contracts, validated configuration, artifact writer, and anti-coupling tests.

**Read first:**

- `mujoco_scenes/functional_tamp_pipeline/models.py`
- `mujoco_scenes/functional_tamp_pipeline/planning.py`
- `mujoco_scenes/phase4_execution.py`
- `mujoco_scenes/requirements.txt`

**Add:**

- `mujoco_scenes/baselines/__init__.py`
- `mujoco_scenes/baselines/vilain_tamp/__init__.py`
- `mujoco_scenes/baselines/vilain_tamp/README.md`
- `mujoco_scenes/baselines/vilain_tamp/contracts.py`
- `mujoco_scenes/baselines/vilain_tamp/config.py`
- `mujoco_scenes/baselines/vilain_tamp/artifacts.py`
- `mujoco_scenes/baselines/vilain_tamp/configs/paper_faithful.yaml`
- `mujoco_scenes/baselines/vilain_tamp/configs/model_matched.yaml`
- `mujoco_scenes/baselines/vilain_tamp/tests/__init__.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_contracts.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_artifacts.py`

**Modify:**
None.

**DO NOT MODIFY:**

- Every file outside the additions above.
- Especially `functional_tamp_pipeline/**`, `phase4_*`, scene/controller code, and existing requirements.

**Implementation:**

- Define the Section 6 dataclasses/enums without behavior-heavy orchestration.
- Validate domain, observation mode, model condition, CP limit, timeout, and external tool paths.
- Implement atomic JSON/text writes and SHA-256 manifests.
- Add source/import scanning that rejects `functional_tamp_pipeline`, `phase4_`, `ground_graph`, `phi`, `G_F`, `G_O`, `GraphGroundingResult`, and Phase-3 artifact names in production baseline code.
- README must state the fair-boundary rules and MuJoCo adaptation.
- No model calls or simulator imports.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py mujoco_scenes/baselines/vilain_tamp/tests/test_contracts.py mujoco_scenes/baselines/vilain_tamp/tests/test_artifacts.py -q
python -m compileall -q mujoco_scenes/baselines/vilain_tamp
git diff --check
git status --short
```

**Commit:**

```
Add isolated ViLaIn-TAMP baseline contracts
```

**Push:**

```
git push origin naren/ViLaIn-TAMP
```

**STOP CONDITION:**
Tests pass, diff contains only Stage-1 additions, commit is pushed, and the commit hash is reported.

**Dependencies:**
Clean branch initialized from the approved Phase-4 source HEAD.

---

### Stage 2 — Fixed PDDL Domains

**Goal:**
Add immutable generic PDDL domains and natural-language domain knowledge for all three environments.

**Read first:**

- `mujoco_scenes/baselines/vilain_tamp/contracts.py`
- `mujoco_scenes/functional_tamp_pipeline/domains/kitchen.py`
- `mujoco_scenes/functional_tamp_pipeline/domains/living_room.py`
- `mujoco_scenes/functional_tamp_pipeline/domains/workshop.py`
- benchmark task strings and variant YAML files, read-only

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/domains/__init__.py`
- `mujoco_scenes/baselines/vilain_tamp/domains/registry.py`
- all six `domain.pddl` and `knowledge.yaml` files from Section 5
- `mujoco_scenes/baselines/vilain_tamp/pddl.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_domains.py`

**Modify:**
None.

**DO NOT MODIFY:**
All existing repository files and Stage-1 contracts unless a Stage-1 test defect blocks loading; report instead of broadening scope.

**Implementation:**

- Encode Section 7 schemas only.
- No per-variant constants, answers, object availability, compatibility assignments, or goals.
- Registry returns domain text, hash, type/predicate/action signatures, and descriptions.
- `pddl.py` performs internal structural/signature checks and domain immutability comparison.
- Add synthetic valid/invalid problems inside the test module.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_domains.py -q
python -m compileall -q mujoco_scenes/baselines/vilain_tamp
git diff --check
git status --short
```

**Commit:**

```
Add fixed ViLaIn-TAMP PDDL domains
```

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
All domain tests pass and no variant-specific truth appears in domain or knowledge files.

**Dependencies:**
Stage 1.

---

### Stage 3 — Observation and Fixed-Inspection Protocol

**Goal:**
Implement observation manifests, RGB-D capture interfaces, and deterministic full-inspection sequencing.

**Read first:**

- `mujoco_scenes/baselines/vilain_tamp/contracts.py`
- scene capture APIs in `scene_loader.py`, `living_room_region_scene.py`, `workshop_scene.py`
- `sequential_inspection.py`, read-only for generic mechanics only

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/observations.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_observations.py`

**Modify:**

- `mujoco_scenes/baselines/vilain_tamp/contracts.py` only if additional camera-frame fields are required.

**DO NOT MODIFY:**
Scene code, sequential inspection, functional search, configs, or controllers.

**Implementation:**

- Implement `initial_observation_only` and `fixed_full_inspection`.
- Hard-code only the task-independent region orders in Section 8.
- Store RGB, metric depth, intrinsics, extrinsics, hashes, and opening trace.
- Provide abstract scene/capture/open protocols so tests use fakes.
- Prompt-facing serialization must omit variant and simulator entity names.
- Do not run MuJoCo in tests.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_observations.py mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py -q
git diff --check
git status --short
```

**Commit:** `Add fair ViLaIn observation protocols`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Mock tests prove fixed complete ordering, no early stop, and prompt-facing metadata redaction.

**Dependencies:**
Stages 1–2.

---

### Stage 4 — Model Transports and Paper Prompts

**Goal:**
Add independent model adapters and faithful prompt construction without making live calls.

**Read first:**

- `mujoco_scenes/baselines/vilain_tamp/domains/*/knowledge.yaml`
- `mujoco_scenes/workshop_phase1/fm_adapter.py` for transport/error conventions only
- paper supplementary prompt descriptions
- earlier ViLaIn prompting code

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/fm.py`
- `mujoco_scenes/baselines/vilain_tamp/prompts.py`
- `mujoco_scenes/baselines/vilain_tamp/requirements-vilain-tamp.txt`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_fm.py`

**Modify:**

- baseline YAML configs only.

**DO NOT MODIFY:**
Existing FM adapters or root requirements.

**Implementation:**

- Define mockable protocols for Qwen object estimation and GPT-4o text/multimodal calls.
- Paper configuration: Qwen2.5-VL-7B-Instruct and `gpt-4o-2024-08-06`.
- Preserve raw response, sanitized request, call ID, usage, latency, model/revision.
- Prompts cover object estimates, initial predicates, goal predicates, and CP.
- PDDL stages request PDDL text, not proposed-method JSON schemas.
- Separate dependency set for a dedicated `.venv-vilain-tamp`.
- No auto-download, model startup, network call, or Fast Downward installation.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_fm.py mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py -q
python -m compileall -q mujoco_scenes/baselines/vilain_tamp
git diff --check
```

**Commit:** `Add ViLaIn model adapters and prompts`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Mock transports reproduce and log all four call types without accessing proposed-method adapters.

**Dependencies:**
Stages 1–3.

---

### Stage 5 — Vision-Language Interpreter

**Goal:**
Implement object → initial-state → goal → complete problem generation using mocked outputs.

**Read first:**

- `contracts.py`
- `pddl.py`
- `fm.py`
- `prompts.py`
- fixed domains

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/interpreter.py`
- object-estimate and generated-problem fixtures
- `mujoco_scenes/baselines/vilain_tamp/tests/test_interpreter.py`

**Modify:**
Baseline contracts only if fixture serialization exposes a missing field.

**DO NOT MODIFY:**
Everything outside the baseline package.

**Implementation:**

- Deterministically normalize Qwen labels/boxes into `ObjectEstimate`.
- Estimate rough 3-D centroids from depth without segmentation.
- Call initial and goal estimators independently.
- Assemble a complete problem against the fixed domain.
- Preserve raw fragments and reject unknown predicates/types.
- Never add planner actions, roles, or hidden object identities.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_interpreter.py mujoco_scenes/baselines/vilain_tamp/tests/test_domains.py -q
git diff --check
```

**Commit:** `Implement ViLaIn PDDL problem generation`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Kitchen, living-room, and workshop fixtures yield deterministic valid problem artifacts entirely from mocked baseline outputs.

**Dependencies:**
Stages 1–4.

---

### Stage 6 — Fast Downward and VAL Integration

**Goal:**
Add external symbolic planner invocation, plan normalization, and plan validation.

**Read first:**

- `pddl.py`
- `contracts.py`
- earlier official ViLaIn Fast Downward wrapper
- Fast Downward and VAL command formats

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/planner.py`
- Fast Downward output fixtures
- `mujoco_scenes/baselines/vilain_tamp/tests/test_planner.py`

**Modify:**
Baseline config files for tool paths/version/search/timeout fields.

**DO NOT MODIFY:**
`symbolic_planning_core.py`, functional planning, or root dependencies.

**Implementation:**

- Use argument lists with `subprocess`, never shell interpolation.
- Validate executable presence/version before running.
- Primary alias `lama-first`, 200-second configurable timeout.
- Parse multiple `sas_plan*` files deterministically.
- Emit existing four-field action shape with ViLaIn-specific IDs.
- Run VAL through a separate adapter.
- Tests use fake executables or stored outputs; do not install/run real planners.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_planner.py -q
git diff --check
```

**Commit:** `Add Fast Downward planning for ViLaIn baseline`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Mock translator/search/timeout/no-plan/VAL cases pass and no common A\* code is invoked.

**Dependencies:**
Stages 1–5.

---

### Stage 7 — Object Identity and Execution Projection

**Goal:**
Bind ViLaIn IDs to simulator entities using only baseline visual/geometric evidence and project PDDL actions.

**Read first:**

- `contracts.py`
- `interpreter.py`
- scene body naming and AABB APIs
- current controller action signatures

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/identity.py`
- `mujoco_scenes/baselines/vilain_tamp/execution/__init__.py`
- `mujoco_scenes/baselines/vilain_tamp/execution/base.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_identity.py`

**Modify:**
None outside baseline.

**DO NOT MODIFY:**
Existing entity resolvers, grounders, Phase-4 adapters, controllers.

**Implementation:**

- Implement one-to-one geometric/class association with thresholds and ambiguity failure.
- Implement the Section 13 syntactic action projections.
- Create `ExecutionProjection` records with evidence.
- Explicitly reject functional-role fields and proposed artifact inputs.
- No actual controller calls yet.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_identity.py mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py -q
git diff --check
```

**Commit:** `Add baseline object binding and action projection`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Synthetic scenes demonstrate deterministic correct, ambiguous, and unresolved binding outcomes.

**Dependencies:**
Stages 1–6.

---

### Stage 8 — Cloned-Scene Geometric Refinement

**Goal:**
Implement the MuJoCo sequence-preflight contract and structured refinement failures.

**Read first:**

- `identity.py`
- `execution/base.py`
- existing generic IK/collision/motion modules
- low-level controller planning entry points

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/refinement.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_refinement.py`

**Modify:**
Baseline contracts if a certificate field is missing.

**DO NOT MODIFY:**
Motion planners, controllers, scene code, grounding geometry checks.

**Implementation:**

- Define pluggable stage refiners and a cloned-scene sequence coordinator.
- Support all stage categories in Section 11.
- Keep contact-rich skill internals as black-box envelopes.
- Emit detailed `RefinementFailure`.
- Never touch the live scored scene.
- Tests use fake stage backends; no full MuJoCo run.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_refinement.py -q
git diff --check
```

**Commit:** `Add MuJoCo refinement contract for ViLaIn`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Sequence propagation, first-failure stopping, certificates, and structured error serialization pass with mock backends.

**Dependencies:**
Stages 1–7.

---

### Stage 9 — Corrective Planning Loop

**Goal:**
Implement the bounded, history-preserving CP state machine.

**Read first:**

- `corrective_planning.py` does not yet exist
- `fm.py`, `prompts.py`, `pddl.py`, `planner.py`, `refinement.py`
- paper CP algorithm and supplementary prompts

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/corrective_planning.py`
- correction fixtures
- `mujoco_scenes/baselines/vilain_tamp/tests/test_corrective_planning.py`

**Modify:**
`prompts.py` only to finalize CP formatting.

**DO NOT MODIFY:**
Fixed domains or any proposed-method files.

**Implementation:**

- Maximum three corrections.
- Pass original problem, current problem, structured error, and complete history.
- Validate revised problem against unchanged domain hash.
- Detect identical revisions.
- Separate CP-eligible failures from infrastructure termination.
- Never pass benchmark evaluator output.
- Use mocked planner/refiner/model in tests.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_corrective_planning.py -q
git diff --check
```

**Commit:** `Implement bounded ViLaIn corrective planning`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Success at attempts 0–3, exhaustion, repeated revision, invalid correction, and infrastructure-error cases all pass.

**Dependencies:**
Stages 1–8.

---

### Stage 10 — Kitchen Execution Adapter

**Goal:**
Execute projected kitchen actions without Phase-3 assignments.

**Read first:**

- `execution/base.py`
- `identity.py`
- `kitchen_ground_truth_execution.py`
- `kitchen_object_manipulation.py`
- `phase4_kitchen.py` only to understand controller calling conventions

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/execution/kitchen.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_kitchen_execution.py`

**Modify:**
None outside baseline.

**DO NOT MODIFY:**
Kitchen controller, Phase-4 adapter, GT planner/state, or functional pipeline.

**Implementation:**

- Construct controller data only from projected action arguments and identity mappings.
- Implement baseline-native inventory and controller contract.
- Support OPEN, PICK, PLACE, POUR, STIR.
- Emit effect ledger entries only after controller and postcondition success.
- Do not call `assignment_from_handoff`, GT inventory construction, or any role assignment.
- Tests use fake dispatcher/controller objects.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_kitchen_execution.py -q
git diff --check
```

**Commit:** `Add direct kitchen execution for ViLaIn plans`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
A synthetic POUR/STIR/PLACE plan produces correct direct controller calls and no assignment artifacts.

**Dependencies:**
Stages 1–9.

---

### Stage 11 — Living-Room Execution Adapter

**Goal:**
Execute baseline PICK/PLACE actions through mobile manipulation without fabricated Phase directories.

**Read first:**

- `execution/base.py`
- `living_room_mobile_execution.py`
- `pick_motion.py`
- `place_motion.py`
- `mobile_motion.py`

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/execution/living_room.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_living_room_execution.py`

**Modify:**
None outside baseline.

**DO NOT MODIFY:**
`run_mobile_execution`, Phase adapters, region grounding, or controller code.

**Implementation:**

- Invoke lower-level mobile/pick/place executors directly.
- Compute destinations from the concrete support named in the symbolic action.
- Preserve existing physical ON postcondition checks.
- Do not create Phase-1/Phase-2 artifacts.
- Tests use fake motion executors.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_living_room_execution.py -q
git diff --check
```

**Commit:** `Add direct living-room execution for ViLaIn`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Synthetic plans call mobile/pick/place components with only projected payload/support IDs.

**Dependencies:**
Stages 1–10.

---

### Stage 12 — Workshop Execution Adapter

**Goal:**
Execute baseline workshop PICK/INSERT/DRIVE/PLACE actions without a Phase-3 `WorkshopAssignment`.

**Read first:**

- `execution/base.py`
- `workshop_ground_truth_execution.py`
- `workshop_scene.py`
- `phase4_workshop.py` for calling conventions only

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/execution/workshop.py`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_workshop_execution.py`

**Modify:**
None outside baseline.

**DO NOT MODIFY:**
Workshop controllers, GT planner, Phase-4 adapter, grounding/search code.

**Implementation:**

- Extract driver, fastener, and target solely from `DRIVE`.
- Use `INSERT` and `PLACE-ON` arguments directly.
- Supply a baseline-specific controller contract where duck typing is required.
- Emit drive effect only after physical controller success.
- Tests use a fake dispatcher.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_workshop_execution.py -q
git diff --check
```

**Commit:** `Add direct workshop execution for ViLaIn`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
The mock plan drives exactly the plan-specified driver/fastener and produces no role-assignment artifact.

**Dependencies:**
Stages 1–11.

---

### Stage 13 — Hidden Benchmark Evaluator

**Goal:**
Implement independent post-terminal task evaluation for all domains.

**Read first:**

- benchmark variant YAML files
- final paper variant mapping
- current physical verification functions
- baseline execution effect ledger

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/evaluation/__init__.py`
- `evaluation/base.py`
- `evaluation/kitchen.py`
- `evaluation/living_room.py`
- `evaluation/workshop.py`
- `tests/test_evaluation.py`

**Modify:**
None outside baseline.

**DO NOT MODIFY:**
Existing evaluators, domain adapters, variant configs, planners.

**Implementation:**

- Implement Section 14 requirements.
- Keep generated-goal evaluation separate.
- Support feasible and infeasible outcome accounting.
- Evaluator entry point accepts only terminal state snapshot, effect ledger, and hidden benchmark context.
- Add a test proving evaluator output is never an input type accepted by CP/planner/interpreter.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests/test_evaluation.py mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py -q
git diff --check
```

**Commit:** `Add independent benchmark evaluation for ViLaIn`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
Synthetic success/failure/infeasible fixtures for all three domains pass independently of generated goals.

**Dependencies:**
Stages 1–12.

---

### Stage 14 — Baseline Orchestrator and CLI

**Goal:**
Connect all implemented components through a baseline-native runner.

**Read first:**

- all baseline modules
- existing runner CLI conventions
- scene factory APIs

**Add:**

- `mujoco_scenes/baselines/vilain_tamp/runner.py`
- `mujoco_scenes/run_vilain_tamp_baseline.py`

**Modify:**

- `artifacts.py`
- baseline configs
- baseline README

**DO NOT MODIFY:**
Existing runners, scene/controllers, Phase-3/4 files, requirements.

**Implementation:**

- CLI supports domain, variant, observation mode, model condition, output directory, dry/offline model fixtures, CP limit, planner paths, planning-only, and execute flags.
- Default must be planning-only; scored execution requires explicit `--execute`.
- Pipeline order: observation → interpretation → planning/refinement/CP → projection → optional execution → hidden evaluation → metrics.
- Do not load or accept Phase-3 run directories.
- Refuse execution when repository provenance or required artifacts are inconsistent.
- No live models or simulation in tests at this stage.

**Checks:**

```
python -m compileall -q mujoco_scenes/baselines/vilain_tamp mujoco_scenes/run_vilain_tamp_baseline.py
python -m mujoco_scenes.run_vilain_tamp_baseline --help
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests -q
git diff --check
```

**Commit:** `Add ViLaIn baseline runner and CLI`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
CLI imports, help works, full baseline unit suite passes, and no external calls occur.

**Dependencies:**
Stages 1–13.

---

### Stage 15 — Offline End-to-End Contract and Reproducibility Audit

**Goal:**
Prove the complete artifact/provenance pipeline with mocks and finalize documentation.

**Read first:**

- `runner.py`
- `artifacts.py`
- all tests and configs
- this implementation plan

**Add:**

- remaining complete mock fixtures under `tests/fixtures`
- `mujoco_scenes/baselines/vilain_tamp/tests/test_offline_pipeline.py`

**Modify:**

- baseline README
- baseline requirement/config files
- baseline artifact and runner modules only for defects exposed by the offline test

**DO NOT MODIFY:**
All non-baseline files, including the top-level CLI unless an import-only defect is found and reported in the diff.

**Implementation:**

- Execute a fully mocked run for each domain.
- Exercise a no-CP success, a refinement-failure-plus-CP success, and an exhausted/infeasible outcome.
- Assert exact artifact layout, hashes, metrics, and isolation.
- Document external Fast Downward/VAL installation paths, separate venv, model setup, and experimental command examples.
- Do not run MuJoCo, Fast Downward, VAL, Qwen, GPT-4o, or a full experiment.

**Checks:**

```
python -m pytest mujoco_scenes/baselines/vilain_tamp/tests -q
python -m compileall -q mujoco_scenes/baselines/vilain_tamp mujoco_scenes/run_vilain_tamp_baseline.py
git diff --check
git status --short
```

**Commit:** `Complete offline ViLaIn baseline integration`

**Push:** `git push origin naren/ViLaIn-TAMP`

**STOP CONDITION:**
All offline tests pass, the documented artifact tree is reproduced, the commit is pushed, and no live planning/simulation/model call has occurred.

**Dependencies:**
Stages 1–14.

---

## 18. Git / Branch Initialization

The source working tree is currently dirty. Therefore the following process must stop immediately on its first run until you resolve the modification yourself.

```
git status --short --branch
git branch --show-current
git rev-parse HEAD
```

Required results before proceeding:

- branch is `phase4/execution-integration-replay-contract`;
- working tree and index are clean;
- HEAD is the source commit you intend to use.

Do not stash, reset, restore, discard, or auto-commit dirty changes.

Once clean:

```
git fetch origin phase4/execution-integration-replay-contract
git rev-parse HEAD
git rev-parse origin/phase4/execution-integration-replay-contract
```

If the hashes differ, stop and report both hashes. Do not pull, merge, or rebase automatically.

If they match:

```
git switch -c naren/ViLaIn-TAMP
git push -u origin naren/ViLaIn-TAMP
git status --short --branch
git branch --show-current
git rev-parse HEAD
```

Expected:

- current branch: `naren/ViLaIn-TAMP`;
- clean working tree;
- HEAD still equals the approved source commit;
- upstream set to `origin/naren/ViLaIn-TAMP`.

---

## 19. Guardrail Checklist for Every Medium Turn

```
[ ] Work only on naren/ViLaIn-TAMP.
[ ] Run git status, branch --show-current, and rev-parse HEAD first.
[ ] Stop if the tree is dirty before starting the requested stage.
[ ] Implement only the named stage.
[ ] Add/modify only files explicitly allowed by that stage.
[ ] Treat every existing proposed-method file as read-only.
[ ] Do not import or reuse G_F, G_O, ground_graph(), phi*, assignments,
    operation bindings, functional witnesses, audits, search rankings, or plans.
[ ] Do not fabricate Phase-3 or Phase-4 handoff artifacts.
[ ] Do not expose variant truth, simulator body names, or evaluator conditions
    to ViLaIn planning.
[ ] Do not perform broad refactors or dependency upgrades.
[ ] Do not make live FM calls or run simulations unless the stage explicitly says so.
[ ] Run only the specified checks.
[ ] Inspect git diff --check, git diff, and git status.
[ ] Commit only the completed stage with the specified message.
[ ] Push origin naren/ViLaIn-TAMP.
[ ] Report checks, changed files, commit hash, push result, and stop.
```

---

## 20. Risks / Decisions Requiring My Approval

Only four genuine decisions remain.

1. **Dirty source-branch modification**
   `mujoco_scenes/kitchen_ground_truth_execution.py` is currently modified. You must decide whether to commit it to the source branch, preserve it elsewhere, or discard it. The implementation agent must not decide.
2. **MTC adaptation approval**
   The public system depends on MoveIt Task Constructor, and a complete public ViLaIn-TAMP implementation was not located. The recommended primary adaptation is cloned-MuJoCo sequence preflight using the existing IK/collision/controller infrastructure. If exact ROS/MTC reproduction is required instead, the architecture, dependencies, and schedule change substantially.
3. **Frozen GPT-4o snapshot**
   The paper reports GPT-4o but not a reproducible snapshot. The recommended primary snapshot is `gpt-4o-2024-08-06`. Approve this before any paid/live experimental runs.
4. **Inspection-opening accounting**
   The recommended full-inspection condition opens storage in a planning copy, counts every opening as inspection cost, and replays necessary openings on the fresh execution scene. Confirm whether those replay openings should also count against the physical action budget. They should always remain separate in telemetry.

No existing proposed-method modification is required by the default architecture.

---

## A. Recommended total number of implementation stages

**15 stages**, followed by separately authorized smoke/integration experiments. Full experiments are deliberately outside the implementation roadmap.

---

## B. Exact first Medium-model prompt: branch initialization + Stage 1 only

```
You are implementing Stage 1 only of the approved “ViLaIn-TAMP Baseline
Implementation Plan for icra-we-ball”.

Do not work ahead. Do not run simulations, make model calls, install
dependencies, or modify existing proposed-method files.

BRANCH INITIALIZATION

From /home/naren/RA_iiith, first run:

git status --short --branch
git branch --show-current
git rev-parse HEAD

The required source branch is:
phase4/execution-integration-replay-contract

If the working tree or index is dirty, STOP immediately and report the exact
status. Do not stash, reset, restore, discard, or commit the existing changes.

If clean, run:

git fetch origin phase4/execution-integration-replay-contract
git rev-parse HEAD
git rev-parse origin/phase4/execution-integration-replay-contract

If the two hashes differ, STOP and report them. Do not pull, merge, or rebase.

If they match, create and publish:

git switch -c naren/ViLaIn-TAMP
git push -u origin naren/ViLaIn-TAMP

Verify the new branch and clean status.

STAGE 1 — BASELINE BOUNDARY, CONTRACTS, AND CONFIGURATION

Goal:
Create the isolated baseline package skeleton, core contracts, validated
configuration, atomic artifact writer, README boundary specification, and
anti-coupling tests.

Read first:
- mujoco_scenes/functional_tamp_pipeline/models.py
- mujoco_scenes/functional_tamp_pipeline/planning.py
- mujoco_scenes/phase4_execution.py
- mujoco_scenes/requirements.txt

Add exactly:
- mujoco_scenes/baselines/__init__.py
- mujoco_scenes/baselines/vilain_tamp/__init__.py
- mujoco_scenes/baselines/vilain_tamp/README.md
- mujoco_scenes/baselines/vilain_tamp/contracts.py
- mujoco_scenes/baselines/vilain_tamp/config.py
- mujoco_scenes/baselines/vilain_tamp/artifacts.py
- mujoco_scenes/baselines/vilain_tamp/configs/paper_faithful.yaml
- mujoco_scenes/baselines/vilain_tamp/configs/model_matched.yaml
- mujoco_scenes/baselines/vilain_tamp/tests/__init__.py
- mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py
- mujoco_scenes/baselines/vilain_tamp/tests/test_contracts.py
- mujoco_scenes/baselines/vilain_tamp/tests/test_artifacts.py

Modify:
- none

DO NOT MODIFY:
- any file outside the exact additions above
- especially functional_tamp_pipeline/**
- symbolic_planning_core.py
- phase4_execution.py
- phase4_kitchen.py
- phase4_living_room.py
- phase4_workshop.py
- run_phase4_execution.py
- existing scenes, controllers, configs, requirements, or tests

Implementation requirements:
1. Add minimal frozen dataclasses/enums for:
   ViLaInObservation, ObjectEstimate, GeneratedPDDLProblem,
   PDDLValidationResult, SymbolicAction, SymbolicPlan, RefinementFailure,
   CorrectivePlanningAttempt, BaselineExecutionPlan, ExecutionProjection,
   BaselineRunResult, and BenchmarkGoalEvaluation.
2. Preserve the existing neutral action fields:
   action_index, action_instance_id, operator, arguments.
3. Do not include G_F, G_O, phi, role assignment, operation binding,
   functional witness, or plan-grounding fields.
4. Add validated configuration for domain, observation mode, model condition,
   CP maximum, timeouts, output root, and external tool paths.
5. paper_faithful.yaml must declare:
   Qwen2.5-VL-7B-Instruct, gpt-4o-2024-08-06, Fast Downward,
   lama-first, a 200-second symbolic timeout, maximum CP corrections 3,
   and initial_observation_only as the primary mode.
6. model_matched.yaml is an optional condition and must not reuse model outputs
   or prompts from the proposed method.
7. artifacts.py must atomically write JSON/text, compute SHA-256, and construct
   manifests without writing secrets.
8. test_boundary.py must scan production baseline source/imports and reject:
   functional_tamp_pipeline, phase4_, ground_graph, GraphGroundingResult,
   G_F, G_O, phi*, Phase3Handoff, graph_grounding_result,
   plan_grounding_audit, and functional grounding witness references.
   Allow explanatory mentions inside README/tests, but not production code.
9. README must state that the baseline is a parallel method and document the
   no-information-leakage boundary.
10. Do not add runner, PDDL, perception, planner, refinement, CP, execution,
    evaluator, dependency, or simulator implementation yet.

Run only:
python -m pytest \
  mujoco_scenes/baselines/vilain_tamp/tests/test_boundary.py \
  mujoco_scenes/baselines/vilain_tamp/tests/test_contracts.py \
  mujoco_scenes/baselines/vilain_tamp/tests/test_artifacts.py -q
python -m compileall -q mujoco_scenes/baselines/vilain_tamp
git diff --check
git status --short
git diff --stat
git diff

If checks pass and the diff contains only Stage-1 files, commit:

git add \
  mujoco_scenes/baselines/__init__.py \
  mujoco_scenes/baselines/vilain_tamp

git commit -m "Add isolated ViLaIn-TAMP baseline contracts"
git push origin naren/ViLaIn-TAMP

Then report:
- source and final branch
- starting HEAD
- tests/checks
- exact changed files
- commit hash
- push result

STOP. Do not begin Stage 2.
```

---

## C. Reusable compact Medium-model prompt template for Stages 2 onward

```
Implement Stage <N> only from the approved ViLaIn-TAMP Baseline
Implementation Plan for icra-we-ball.

GUARDRAILS

1. cd /home/naren/RA_iiith
2. Run:
   git status --short --branch
   git branch --show-current
   git rev-parse HEAD
3. Required branch: naren/ViLaIn-TAMP.
4. If the branch is wrong or the tree/index is dirty, STOP and report.
   Do not stash, reset, restore, discard, rebase, or merge.
5. Implement only this stage. No work from later stages.
6. Existing proposed-method files are read-only.
7. Do not reuse or import G_F, G_O, ground_graph(), phi*, functional roles,
   assignments, operation bindings, task-aware search, grounding witnesses,
   plan-grounding audits, Phase-3/4 handoffs, or proposed-method plans.
8. Do not expose hidden variant truth, body names, segmentation, or evaluator
   conditions to the baseline planner.
9. Do not run simulations, live model calls, or unlisted commands.
10. Add/modify only the files listed below.

STAGE <N> — <NAME>

Goal:
<copy verbatim from the plan>

Read first:
<copy verbatim>

Add:
<copy verbatim>

Modify:
<copy verbatim>

DO NOT MODIFY:
<copy verbatim>

Implementation:
<copy verbatim>

Checks:
<copy verbatim>

After checks:
- run git diff --check
- inspect git status --short
- inspect git diff --stat and git diff
- confirm only the allowed stage files changed

Commit exactly:
git add <explicit stage files>
git commit -m "<approved commit message>"
git push origin naren/ViLaIn-TAMP

Report:
- starting and ending HEAD
- checks and results
- changed files
- commit hash
- push result

STOP CONDITION:
<copy verbatim>

Dependencies:
<copy verbatim>

STOP after pushing this one stage. Do not begin the next stage.
```

# POST-STAGE-15 LIVE INTEGRATION ROADMAP

Stages 1–15 are complete. The codebase is currently **OFFLINE / CONTRACT
COMPLETE**. The remaining work is live runtime integration and validation; it
is not a redesign of ViLaIn-TAMP.

Existing proposed-method files remain read-only except when exact upstream
source-branch commits are merged into this baseline branch. Full benchmark
experiments must not begin until all live-integration stages below are
complete.

## Stage 16 — Synchronize Shared Execution Backend

**Goal:**
Bring `naren/ViLaIn-TAMP` up to the current committed state of
`origin/phase4/execution-integration-replay-contract` without manually changing
the proposed method and without rewriting ViLaIn history.

**Method:**
Merge the source branch into `naren/ViLaIn-TAMP`.

This is backend synchronization only. Do not manually edit proposed-method
files during this stage. Do not cherry-pick selected behavior, duplicate fixes
in baseline code, or rebase the published baseline history. If the merge
conflicts, stop and report instead of inventing a resolution.

**Checks:**

- baseline offline tests;
- compileall;
- relevant source-side non-simulation regression tests;
- Git diff/check/status;
- confirmation that the baseline isolation boundary remains intact.

## Stage 17 — Concrete MuJoCo Observation Backends

**Goal:**
Connect `ObservationProtocol` to the actual three benchmark scenes.

Implement baseline-owned concrete adapters for canonical RGB-D camera capture,
intrinsics/extrinsics extraction, `initial_observation_only`,
`fixed_full_inspection`, and generic region-opening mechanics.

Do not use functional search, `G_F`-derived vocabulary, proposed-method region
ranking, or early stopping. Make no FM calls in this stage.

## Stage 18 — Live ViLaIn Model Transports

**Goal:**
Replace injected mock transports with real baseline-owned model transports.

The primary condition uses `Qwen2.5-VL-7B-Instruct` for object estimation and
`gpt-4o-2024-08-06` for initial-state, goal-state, and CP calls.

Requirements:

- use the dedicated `.venv-vilain-tamp`;
- do not change proposed-method dependencies;
- log exact models and revisions;
- retain raw request/response artifacts;
- never write API secrets;
- never reuse proposed-method FM outputs;
- make no model calls during automated unit tests.

Validate one standalone object-estimation call before integrating further.

## Stage 19 — Concrete TAMP Attempt Runner

**Goal:**
Provide the real `TAMPAttemptRunner` used by `CorrectivePlanningLoop`.

Pipeline:

```text
GeneratedPDDLProblem
    -> internal PDDL validation
    -> Fast Downward
    -> VAL
    -> symbolic action normalization
    -> independent identity binding
    -> execution projection
    -> geometric refinement
    -> TAMPAttemptOutcome
```

Do not use the repository common A* in the primary ViLaIn condition.

## Stage 20 — Concrete MuJoCo Sequence Refinement

**Goal:**
Connect `MuJoCoSequenceRefiner` to real planning-copy geometric
infrastructure.

Implement baseline-owned stage backends for entity resolution, grasp
candidates, IK, trajectory, collision, skill envelope, and predicted state
transition. Use a cloned/planning scene only. Do not use scored execution as
the normal refinement loop. POUR, STIR, and DRIVE remain black-box controller
skills whose start/end envelopes are checked.

## Stage 21 — Live Controller and Terminal-State Bridges

**Goal:**
Connect the implemented Kitchen, Living Room, and Workshop execution adapters
to the existing generic controllers.

Also implement actual terminal-state extraction, effect-ledger population,
`HiddenBenchmarkContext` loading, generated-goal evaluation, and actual hidden
benchmark evaluation. Concrete controller arguments must originate from the
ViLaIn symbolic plan and its own object-identity bridge. No functional
assignments may be imported.

## Stage 22 — Live Planning-Only Smoke Validation

**Goal:**
Prove that real observation -> FM -> PDDL -> Fast Downward -> refinement works
before physical scored execution.

Run only one simple feasible/all-visible case per domain initially, in this
order:

1. Kitchen.
2. Living Room.
3. Workshop.

Use `initial_observation_only` first. Inspect all intermediate artifacts
manually after each domain. Do not run the full variant matrix.

## Stage 23 — Live Execution / CP / Inspection Smoke Validation

After Stage 22 succeeds:

- execute one easy feasible case per domain;
- verify projected action arguments;
- verify terminal physical state;
- verify the hidden evaluator;
- test one `fixed_full_inspection` case;
- test one genuine refinement failure -> CP -> successful revised plan case;
- test one infeasible case.

Do not automatically tune failures by repeatedly modifying code. Diagnose each
failure from saved artifacts and handle it in a separate authorized turn.

## Stage 24 — Experimental Harness and Full Benchmark

Begin only after Stages 16–23 are frozen.

Implement experiment scheduling and results aggregation for
`initial_observation_only`, `fixed_full_inspection`, five independent
generations per benchmark instance, the CP=3 primary condition, and required
ablations.

Collect:

- actual task success;
- correct infeasibility recognition;
- generated-goal satisfaction;
- PDDL validity, plannability, and VAL validity;
- refinement success and CP iterations;
- FM calls, tokens, cost, and latency;
- symbolic, refinement, execution, and end-to-end time;
- inspected-region count;
- execution and entity-resolution failures;
- plan length.

No experimental result may alter hidden task truth or feed the terminal
evaluator back into planning.

## Live-Integration Global Rule

Complete one stage per Medium-model turn. Every stage must:

1. pass its specified checks;
2. be inspected with Git diff/status;
3. be committed;
4. be pushed to `origin/naren/ViLaIn-TAMP`;
5. report the commit hash and push status;
6. stop before the next stage.
