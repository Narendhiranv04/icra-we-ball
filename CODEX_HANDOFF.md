# Codex Project Handoff

Updated: 2026-08-26 (Asia/Kolkata)

This file is the working context for continuing the project from another Codex
session after SSHing into this machine. Start the next session by asking Codex
to read this file, inspect the current worktree, and continue without resetting
or discarding uncommitted changes.

## Latest change: 1/3/5-camera baseline benchmark

- VLM-TAMP and OWL-TAMP Kitchen/Living Room runners accept
  `--camera-count {1,3,5}`. The nested fixed subsets are top; left/right/top;
  and all five (adding front/close).
- `baseline_common.run_plan_gt_batch` accepts `--camera-counts 1,3,5` and
  `--resume`. The intended paper grid is 10 seeds per variant for K1-K12 and
  L1-L10, for both methods and all three image counts.
- `baseline_common.summarize_plan_gt_batch` creates compact `table4.csv` and
  `table4.json` containing outcome accuracy and mean ordered GT LCS-F1.
- OWL-TAMP Kitchen now uses the same auditable shared task vocabulary as
  VLM-TAMP for evaluation only: OPEN->INSPECT, CLOSE excluded, and
  PLACE_SERVING_UTENSIL->PLACE. OWL-TAMP also records outcome agreement.
- Full copy-paste procedure: `docs/BASELINE_BENCHMARK_RUNBOOK.md`.
- Camera render smoke passed for both scenes at 1/3/5 views. Focused tests:
  `53 passed`.
- Fixed a Kitchen VLM-TAMP startup regression in `method_manifest.json`: it
  referenced `result.model_calls` before the planner had run. The manifest now
  records the configured `max_model_calls`; focused validation remains
  `53 passed`.
- After the first live smoke output, removed the explicit "Search the closed
  kitchen storage" instruction from the shared Kitchen goal. Inspection is now
  a model decision based on the visible state, not a goal directive. OWL-TAMP
  malformed sketch/constraint output is recorded as `INVALID_MODEL_OUTPUT`
  (`UNRESOLVED`, zero sequence score) instead of crashing and losing the trial.
  A subsequent smoke trace exposed a no-op verification mismatch: an initial
  `source_region` now satisfies an already-true `PLACED` subgoal, matching the
  empty plan returned by PDDLStream instead of producing
  `effect_not_observed`. Focused validation after these fixes: `61 passed`.

## Latest change: OWL-TAMP Kitchen/Living baseline

- Added isolated `owl_tamp_baseline/`. It does not modify or use the VLM-TAMP
  planner and does not use the proposed YOLO/functional-search path.
- Implements observable-only relaxed grounding, a neutral five-image sketch
  prompt, `Executed(i)` compilation, per-action constraint prompting with a
  restricted DSL, symbolic search, and the paper's 500-sample/action and
  five-skeleton limits.
- Added planning-only Kitchen and Living Room launchers and private
  `EXPECTED_GT_ACTIONS` exact/LCS/F1 comparison.
- Closed Kitchen storage contents remain hidden. K2-K12 intentionally expose
  the single-shot paper protocol's observability limitation; do not silently
  add automatic inspection or a privileged inventory.
- This is a paper-derived reimplementation of arXiv:2411.08253v4 because no
  official author repository was public. Report it that way.
- `BASELINE_FIDELITY.md` contains the permitted OWL-TAMP claims and explicit
  domain/observability adaptations.
- Focused validation: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q owl_tamp_baseline/tests` -> 3 passed.

## 2026-08-26: planning-only Kitchen baseline completion

- `vlm_tamp_baseline.run_kitchen --planning-only --variant K1 ...` now runs one
  two-stage VLM-TAMP planning round, PDDLStream refinement, symbolic rollout,
  and private GT comparison without moving MuJoCo.
- K1--K12 planning-only VLM-TAMP and OWL-TAMP now construct variants directly;
  they no longer depend on an older proposed-framework Phase-1 run. Object IDs
  are anonymous and derived from instance segmentation; backend names remain
  private until GT comparison.
- Symbolic `INSPECT` reveals a region without physics. It does not trigger a
  second VLM round in the primary initial-plan condition.
- Kitchen results save both raw execution-vocabulary metrics and shared-task
  metrics (`OPEN -> INSPECT`, GT-only `CLOSE` excluded,
  `PLACE_SERVING_UTENSIL -> PLACE`).
- Added deterministic inference `--seed` support and
  `baseline_common.run_plan_gt_batch` for Kitchen/Living variant grids.

## Workspace and Git state

- Repository root: `/home/boreddog/Documents/RRC/LH_Extension/V1`
- Current branch: `kitchen_livingroom_integration`
- The branch starts at `8a2f8ae` (`llm3_baseline`). The dirty worktree combines
  the pending cleanup with the Phase-C kitchen/living perception and physical
  execution port. It is intentionally not committed or pushed yet.
- The newest integrated upstream revision is `b4dcbd6`. Its complete L2
  Ablation 1/2/3 implementation and realistic movie-night assets are retained.
- Generated `reports/` media is deliberately excluded; report generators,
  source, configuration, assets, and tests are retained.
- Do **not** run `git reset`, `git checkout --`, `git clean`, or otherwise remove
  changes unless the user explicitly requests it.
- Remote: `https://github.com/Narendhiranv04/icra-we-ball.git`
- Latest fetched point-cloud comparison ref:
  `origin/naren/googlePointCloudIntegration-phaseC` at `6065a55` ("Generalize
  living-room physical execution across all 13 variants"). It is divergent
  this branch at the current divergent tips. Its production symbolic action
  contracts are `PICK`, `PLACE`, `POUR`, `STIR` for kitchen and `PICK`, `PLACE`
  for living room. The newer `OPEN`, `CLOSE`, `MOVE`, serving, and refinement
  operations are oracle/execution helpers, not part of the planning contract.
  The current branch ports the reusable perception, registry, entity-resolution,
  motion, and verification pieces. The non-oracle kitchen symbolic compiler
  and deterministic A* sequencer are now included; generated reports and
  oracle planners remain excluded.
- The similarly named local branch `naren/pointCloudExtraction` is stale and
  should not be used for comparisons. Compare against the remote ref directly
  after fetching.

Before editing, run:

```bash
cd /home/boreddog/Documents/RRC/LH_Extension/V1
git status --short
git branch --show-current
git log -3 --oneline --decorate
```

## Paper-baseline fidelity update (2026-08-23)

The LLM3 and VLM-TAMP folders are now algorithm-faithful ports to the shared
MuJoCo/Google-robot kitchen. Their exact reproduction boundary and reporting
rules are authoritative in `BASELINE_FIDELITY.md`.

- VLM-TAMP now uses two model stages (English intermediate goals, then grounded
  predicates), sequential subgoal refinement through the real pinned
  PDDLStream/FF stack, three object reducers, up to 12 skeletons, and
  failure-triggered reprompting. The old fixed action-template refiner is only
  the explicit `--refiner catalog-ablation` condition.
- Install its local dependency with
  `bash vlm_tamp_baseline/setup_pddlstream.sh`. The required revision is
  `b38137e47fd4a4116a3e36bc4be691cbe5da6cb0`; its pinned Fast Downward binary
  is built below the ignored `.paper_deps/` directory.
- LLM3 now asks for a complete plan with discrete arguments and bounded
  continuous parameters, returns motion feedback and the last three plan
  traces to the model, and permits parameter resampling or symbolic
  backtracking. PLACE, POUR, and STIR consume the sampled values physically.
- Both baselines use the same frozen observations, full supported manipulable
  registry, MuJoCo skills, IK/collision logic, and effect verifier. The frozen
  sample resolves all 15 supported manipulable objects; it no longer restricts
  baselines to the proposed planner's selected 11 objects.
- `--decoding paper` is the default: VLM-TAMP uses temperature 0.2 and LLM3
  uses temperature 0; thinking is disabled in both. `model-native` must be
  reported as a separate ablation.
- Validation on this worktree: `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q` completed
  with 558 passed and 4 skipped. Pinned-PDDLStream regression tests cover
  visible object-in-receptacle placement and inspect-before-use of a target in
  closed storage.
- These are defensible algorithm ports in a common new domain, not exact
  reproductions of either paper's original robot, simulator, task set, or
  foundation model. Preserve that wording in paper tables and captions.
- The first live VLM-TAMP run exposed a contradictory constrained response:
  `GOAL_COMPLETE` with non-empty English steps. Prompt/schema version 3 makes
  status/cardinality combinations mutually exclusive. The executive also
  records malformed completions and reprompts with `invalid_vlm_output`
  instead of crashing the episode.
- The second live run showed Qwen repeatedly returning a schema-valid
  `GOAL_COMPLETE` while the observed state explicitly had
  `goal_satisfied=false`. Prompt/schema version 4 therefore removes terminal
  completion from both model stages. Only the independent simulator verifier
  may return `GOAL_COMPLETE`; the VLM can return steps/subgoals or their
  `NO_VALID_*` status. A non-structured model completion claim is rejected.
- Prompt version 5 removes the blocked-receptacle example, free-hand hint, and
  other task-specific sequencing guidance from the English stage. The
  benchmark goal no longer instructs the model to search storage. Inspection
  remains available, but whether and where to inspect must now follow from the
  neutral task goal and supplied scene evidence.
- The first prompt-v5 run reached PDDLStream but exposed a workspace-adapter
  bug: symbolic `right_side` was sent directly to the legacy navigator, whose
  public destination is `cupboard2`. The read-only RRT* stream now uses the
  canonical `WORKSPACE_DESTINATIONS` mapping and resolves that alias back to
  the `right_side` physical pose. Invalid workspace names fail the stream
  instead of crashing PDDLStream.
- The repository audit repaired the PDDL domain/stream mismatch that made
  `PLACED(spoon,mug)` impossible. Regions and receptacle objects now have
  distinct pick/place operators, POUR/STIR require the target's region to be
  accessible, target receptacles enter the goal-related object reducer, and
  current observed locations override frozen source locations after motion.
- Stream sampling errors now become infeasible certificates with recorded
  diagnostics rather than process crashes. The pinned fork's global
  visualization/statistics side effects are disabled at the adapter boundary;
  method-specific run artifacts remain authoritative.
- LLM3 prompt/schema version 3 makes `PLAN` non-empty and `NO_VALID_PLAN`
  empty, while only the independent verifier may declare goal completion.
  Invalid model output is saved and reprompted without executing motion. The
  custom same-label override of `NO_VALID_PLAN` was removed because it was not
  part of the LLM3 paper algorithm.
- Both live baseline runners now reject non-empty output directories so failed
  and successful episodes cannot silently share model calls or traces. Use a
  fresh explicit `--output-dir` for every experiment.
- Root `conftest.py` selects EGL for tests unless the caller explicitly sets a
  renderer, preventing MuJoCo/GLFW from aborting plain headless pytest runs.

## Second repository sweep (2026-08-23)

- Baseline execution now separates VLM observations from verifier-only state
  observations. LLM3 and VLM-TAMP render the five RGB views only for model
  calls; post-action and post-subgoal checks use `observe_state()`. Camera IDs
  are resolved once per episode and each PNG is encoded once before being
  written and embedded. This removes repeated five-camera rendering and
  duplicate PNG compression from the physical execution loop.
- Plan authorization is a typed failure boundary. Invalid arguments,
  collisions, and backend failures during `prepare()` no longer escape and
  crash the episode; they enter bounded replanning in the primary grounded
  executive, LLM3, VLM-TAMP, and the functional task executive. Direct skill
  start errors are likewise returned as action failures.
- PDDLStream activation and PDDL file loading occur once per refiner instance,
  rather than running dependency Git checks and reading both PDDL files for
  every subgoal/trial. Sampling and motion certificates remain fresh.
- Duplicate VLM-TAMP succeeded subgoals are suppressed. LLM3 rejects
  self-placement, self-pouring, and self-stirring before motion.
- Optional dependency injection now distinguishes `None` from valid falsy
  values across planners, inference transports, semantic/geometry configs,
  event logs, and model profiles. Empty test/config inputs can no longer be
  silently replaced by production defaults.
- Baseline JSON traces use one shared atomic writer, so an interrupted update
  cannot leave a partially written episode or model-call artifact.
- The remaining structural debt is concentrated in older monolithic scene and
  report modules (`observed_state.py`, `kitchen_object_manipulation.py`,
  `generic_manipulation.py`, `task_witness.py`, and `scene_loader.py`). They are
  large but exercised; splitting them should be a separately reviewed refactor,
  not mixed into experiment behavior changes.

## Cohesion and execution audit (2026-08-23)

- The user explicitly deferred the workshop scene. This audit did not change
  workshop source, configuration, behavior, or tests.
- Kitchen Phase C now validates its registry and plan boundaries, rejects
  malformed or duplicate steps, validates finite motion parameters, and
  replays live PICK/PLACE/SERVE locations. Objects moved out of storage are no
  longer incorrectly filtered by their immutable discovery location before a
  later POUR or STIR.
- The live kitchen observer now reports current object locations maintained by
  the physical dispatcher while retaining discovery provenance separately.
  Planning and verification therefore consume the post-action state rather
  than reconstructing stale locations from the Phase-1 snapshot.
- Phase B/C dispatch, planner-action parsing, symbolic search, task-witness
  loading, inference transports, semantic grounding, SAM output, IK inputs,
  camera rendering, and scene configuration now fail at typed boundaries with
  useful errors instead of leaking `KeyError`, `IndexError`, `-1` MuJoCo IDs,
  non-finite values, or partially written artifacts.
- Region-grounding and living-room capture paths validate mapping-shaped YAML,
  positive image dimensions, known inspection labels, camera interfaces, and
  physical support/body names before indexing MuJoCo arrays. TV dusting also
  validates the complete robot/tool/screen interface during construction.
- VLM-TAMP intentionally retains its documented paper-condition complete
  object-name universe. Hidden locations and contents remain unavailable; the
  production planner and LLM3 continue to receive observation-only state.
- Repository validation completed with `631 passed, 4 skipped` using
  `bash mujoco_scenes/scripts/validate_repository.sh`. The skips are optional
  dependency conditions; the only warnings are third-party Matplotlib
  deprecations. `git diff --check`, module entry-point checks, and the full
  pytest suite are included in that validator.

## User preferences

- Use `uv` for the Python environment.
- Run natively for now; Docker support can be used later.
- Keep code small, clean, readable, and free of unnecessary abstractions or
  excessive generated prose.
- Preserve existing user changes in the dirty worktree.
- Ground-truth action testing should be driven from a text file rather than a
  console prompt.
- Foundation models must receive only visible image-derived state. They must
  not be given hidden simulator objects or privileged MuJoCo state.
- Simulator ground truth is permitted only as an explicit oracle/evaluation
  backend and must be clearly labelled.
- Functional predicates should remain simple, such as `can_stir` or
  `can_clean`, instead of subjective notions such as efficiency.
- Each functional ranking should contain 10–15 distinct concrete searchable
  categories, never umbrella labels such as `utensil`, `tool`, or `object`.

## Overall research direction

This is a Robust TAMP research codebase using MuJoCo. The intended pipeline is:

1. Provide the foundation model with visible scene images.
2. Call the FM once to decompose the goal into simple functional requirements,
   propose/rank 10–15 concrete object or method types, and optionally
   provide abstract subgoal precedence. The FM must not claim that an
   unobserved type is present in the scene.
3. Follow the experiment's configured region search order. The finalized
   workshop production policy uses one fixed search order; fixed/random/FM
   orders may remain diagnostic ablations elsewhere.
4. Identify visible instances from RGB with a lightweight semantic model and
   assign generic persistent IDs. YOLO-World has struggled with the small,
   unusually viewed simulator objects, so SAM 3.1 and lightweight proposal +
   crop-classification alternatives should be evaluated.
5. Accumulate visible instances across every inspected region and build RGB-D
   point clouds for deterministic target-specific geometry such as
   insertability, usable length, cross-section, cavity depth, support,
   tool-fastener mating, hole fit, and approach clearance.
6. Construct complete functional witnesses from the full accumulated observed
   registry. Components of one method may come from different regions.
7. Stop immediately at the first complete method that passes every semantic
   and geometric check. Do not keep searching for a higher-ranked method that
   might be hidden elsewhere.
8. Give the verified object/region handoff to a downstream sequencer. Kitchen
   currently uses deterministic A* over generic `PICK`, `PLACE`, `POUR`, and
   `STIR`; continuous grasp, placement, IK, and collision checks remain in the
   physical refinement layer. This is the proposed-method path; the separate
   paper-faithful VLM-TAMP baseline now uses pinned PDDLStream.
9. Execute one action or short safe prefix, verify its observed effect, and
   deterministically re-run the symbolic sequencer after execution failures.
   Do not use the FM as the replanner.

Examples include choosing a spoon or chopstick instead of a coffee stirrer,
while rejecting undesirable objects such as a pen or knife after semantic/FM
ranking and geometric validation.

The FM ranking is a bounded commonsense proposal, not global optimality. Under
first-feasible early termination, a valid lower-ranked method found earlier in
the fixed search is intentionally selected. Ranking only affects which
currently constructible candidates are checked first when more than one is
available at the same observation stage. Scientific correctness is membership
in the deterministically verified feasible set, not agreement with one
preferred method.

The persistent observation, tracking, semantic grounding, task-witness,
same-evidence ablation, and report-generation source from Naren's branch is now
integrated. The kitchen object-alternative and living-room region-alternative
benchmarks are present. The branch also contains the redesigned compact
workshop physical scene described below. See
`THREE_SCENE_BENCHMARKS.md`.

The integrated test command is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  .venv/bin/python -m pytest mujoco_scenes/tests -q
```

The focused workshop suite passes 12 tests, including the new workshop
point-cloud runner. The complete repository result is recorded in the
Validation section after each full run.
Pytest is declared in `mujoco_scenes/requirements-dev.txt`.

## Finalized FM, search, and sequencing boundary

Naren's controlled S1/L2 search runners do **not** call an FM. Their task
YAML files are explicitly manual future-FM contracts
(`generated_from_foundation_model: false`). They run real RGB/RGB-D perception,
semantic and geometric grounding, same-evidence policy ablations, and emit a
verified handoff. Kitchen now has a separate deterministic Phase-2 compiler and
Phase-C execution entry point; the search runner itself still does not move the
robot.

The existing client in `mujoco_scenes/foundation_model.py` can call an
OpenAI-compatible vLLM/SGLang endpoint for the older required-function ranking
path. The separate `inference_server` workspace provides multimodal functional
decomposition only. The new `llm3_baseline/` workspace is the deliberately
separate direct-action comparison: it receives actual camera images, a visible
image-derived registry, and a natural-language goal, then returns a validated
scene action sequence. It rejects invented object and region IDs. Its bounded
executive re-prompts after shared recoverable failures. The live MuJoCo
adapter in `llm3_baseline/execution.py` shares the same
`MuJoCoSkillDispatcher` as the functional planner. The standalone client
remains planning-only; `llm3_baseline.run_kitchen` owns live execution.

The agreed execution architecture is:

```text
visible images + goal
        -> one FM decomposition/type-ranking call
        -> fixed sequential region inspection
        -> persistent cross-region observed-object registry
        -> first complete semantic + geometric witness
        -> deterministic symbolic plan construction
        -> guarded execution and deterministic replanning
```

The proposed Kitchen Phase 2 uses the non-oracle deterministic A*
implementation in `symbolic_planning.py`, beginning only after a verified
witness exists. Separately, `vlm_tamp_baseline/` invokes pinned PDDLStream to
preserve that paper's subgoal-refinement algorithm. The production discovery
executive remains separate: it owns inspection, unknown regions, evidence
updates, and early termination.

```text
mujoco_scenes/planning/
  domain.pddl
  stream.pddl
  problem_builder.py
  streams.py
  executor.py
  replanner.py
```

## Robot and environments

### Google Robot

The active robot is MuJoCo Menagerie's Google Robot, not Fetch. Menagerie is
expected at:

```text
/home/boreddog/Documents/RRC/LH_Extension/third_party/mujoco_menagerie/google_robot
```

The Google robot has been calibrated for kitchen navigation/manipulation,
compact carry poses, safer collision checking, reduced wobble, and faster arm
motion. Calibration guidance is maintained in
`mujoco_scenes/ROBOT_CALIBRATION.md`.

### Kitchen

Kitchen behavior exists and was ported from main to Google Robot. It includes
navigation, picking/placing, spoon/jar interactions, cupboards, box/drawer
actions, IK work, and calibration support. `run_kitchen_goal_execution.py` now
validates the configured natural-language goal with the FM, consumes a complete
observed witness, grounds the coffee/water sources, sequences 31 generic
actions, resolves all IDs one-to-one, and opens a live execution viewer.

### Living room

The living room is a compact rigid-only environment with:

- a continuous fixed L-shaped sofa along the west and south sides;
- a fixed coffee table and enlarged centered rug;
- remote, mug, hardback book, coasters, game controller, and rigid duster;
- wall-mounted TV with per-cell dust removal and power action;
- media console with two drawers;
- full-height north and west walls;
- improved lighting and warmer visual design;
- navigation, drawer, pick/place, storage, TV, dusting, and action-file support.

Important living-room files:

- `mujoco_scenes/assets/living_room_base.xml`
- `mujoco_scenes/living_room_scene.py`
- `mujoco_scenes/living_room_navigation.py`
- `mujoco_scenes/living_room_manipulation.py`
- `mujoco_scenes/living_room_actions.py`
- `mujoco_scenes/living_room_tamp.py`
- `mujoco_scenes/LIVING_ROOM_ENVIRONMENT.md`

The coffee table and sofa are intentionally static. Earlier push/pull table
experiments were removed.

### Workshop / makers-lab benchmark

The branch implements a deliberately simplified, single-arm workshop repair
cell. It is one front-facing workbench intended to keep Google Robot at
one central stance. Two wooden frame members are already rigidly held in a
fixture, and a captive guide can retain a staged screw. The artificial hinged
joint guard was removed. An ordinary closed tabletop tool cabinet supplies a
second tool/fastener region, and the orange tray is for staging the selected
screw. The earlier lock and key were removed because they distracted from the
joint-repair alternatives.

A small transparent polycarbonate cover with a rubber gasket and yellow pull
tab now sits directly over the joint guide. It is a stable free-jointed object,
not floating decorative geometry. It must be removed into the staging tray
before screw insertion. The observable task state reports whether joint access
is covered or clear. `--remove-seal` and `move_joint_seal_to_tray()` provide an
explicit ground-truth debug transition; they are not a substitute for the
future calibrated grasp-and-place action.

The left drawer contains a manual Phillips driver and an unusably short screw.
The tool cabinet contains a powered Phillips driver and the feasible long
screw. Both start closed and may be inspected independently. The intended
future execution has this partial order:

```text
remove_joint_seal -> insert_screw -> drive_screw
observe_long_screw -> insert_screw
```

Seal removal and region inspection are independent until screw insertion.
After opening the cabinet, the robot rejects the short screw geometrically,
stages and releases the long screw in the captive guide, grasps either
compatible driver, drives the screw, and verifies the repaired joint. The
fixture and guide replace the second hand, so the robot carries only one object
at a time.

This still supports the desired research variables: manual versus powered
driver object alternatives, short versus long screw geometry, fixed-order
drawer/cabinet region search, cross-region driver/fastener composition, the
geometric seal-removal prerequisite, and first-feasible termination. Cutting,
loose-frame assembly, lock-and-key manipulation, and vertical
mounting were removed because they introduced several unrelated high-risk
manipulation skills into one benchmark.

`mujoco_scenes/workshop_scene.py` compiles with `--robot google` and
`--robot none`, exposes five fixed cameras, preserves region-gated visibility,
and reports non-privileged open/closed container state. The inspection rig is
`LEFT_DRAWER` then `TOOL_CABINET` and produces fresh RGB-D evidence for their
contents. Launch it with:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot google --viewer
```

`mujoco_scenes/workshop_pointcloud.py` is the workshop-only five-view capture
entry point. It captures `INITIAL -> LEFT_DRAWER -> TOOL_CABINET`, temporarily
opens only the active region, closes it immediately after capture, and exports
RGB, depth, mask debug views, per-camera PLY files, fused object PLY files,
stage summaries, and a root manifest. The viewer therefore starts with the
drawer and cabinet closed. Its default oracle mode launches without a model
server:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_pointcloud \
  --robot google --segmentation oracle --viewer
```

For image-only masks, start `perception_server`, set `SAM3_BASE_URL`, and use
`--segmentation sam3`. Both modes use the same RGB-D backprojection, region
gating, and fusion. The cameras are calibrated virtual region-facing cameras;
the Google Robot is loaded in the same MuJoCo model but these five views are
not robot-mounted.

Current boundary: this change implements only the scene, observable state,
camera/inspection configuration, tests, and documentation. It does **not**
implement grasping, physical seal removal, screw driving, functional
grounding, PDDLStream, or execution. The existing
hammer/nail single-joint prototype has been removed rather than retained as a
misleading contract. The next workshop-pipeline phase must build directly on
the current driver/screw/seal scene and its observed point-cloud evidence.

### Repository cleanup

The unreachable Fetch composer and `gymnasium-robotics` dependency have been
removed; production scene composition supports Google Robot and no-robot mode.
The superseded `pick_motion.py`, `place_motion.py`, `open_motion.py`, and
`drawer_motion.py` stack was also removed because the live kitchen panel uses
`generic_manipulation.py` and the scene container API. Stale PDDL sketches,
the old workshop alternatives validator/config, and unreferenced debug
artifacts were retired. The two TAMP literature surveys, active standalone
ablation/report CLIs, perception/inference workspaces, and LLM3 baseline were
preserved.

The final repository-wide audit also removed the unused Phase-B evidence
freezer, living-room recorder, duplicate geometry helper modules, one unused
visual-revision config, developer screenshots, and the local paper copy.
Reusable within-region execution calibration moved into
`kitchen_execution_entities.py`. Artifact-coupled kitchen/living tests now use
self-contained measured-registry fixtures, and `pytest.ini` includes the LLM3
suite. Mink moved from runtime to development requirements because it is used
only by the standalone IK backend comparison; calibrated live motion still
uses its constrained DLS implementation. Ignored caches and `MUJOCO_LOG.TXT`
were cleared; `runs/`, `.venv/`,
model caches, all active configs, report generators, and research notes were
preserved. `EXECUTION_AND_TESTING.md` is now the canonical runbook, and
`mujoco_scenes/scripts/validate_repository.sh` is the one-command validation.

## Final camera topology

Do not revert this to five foot cameras. The user explicitly clarified the
required topology:

- exactly **two low cameras** for seeing beneath the sofa;
- exactly **five upper cameras** above the Google robot for observing the rest
  of the room and all directions;
- the wrist camera remains available for close manipulation but is not part of
  the region-observation rig.

Definitions are in `mujoco_scenes/living_room_cameras.py`.

Low cameras:

- `left_foot_camera`
- `right_foot_camera`

Upper 360-degree rig:

- `top_front_camera`
- `top_front_left_camera`
- `top_rear_left_camera`
- `top_rear_right_camera`
- `top_front_right_camera`

The five upper views are spaced at 72-degree yaw intervals with overlapping
90-degree fields of view. They are attached to `google:base_link` on a virtual
observation mast above the fixed head/compact arm silhouette. They rotate and
translate with the mobile base but do not move with the arm.

The original `head_camera_rgb` is repurposed as `top_front_camera` in the living
room, preventing an accidental sixth upper observation camera. The kitchen's
Google robot behavior remains separate.

## Lost-remote scenario

`living_room_scene.py` supports:

- `--scenario standard`
- `--scenario lost_remote`

The lost-remote scenario places the remote in a real 24 cm clearance beneath
the raised rigid sofa. The couch has rigid legs rather than a solid floor-level
base.

The robot navigates to a collision-checked sofa observation pose through RRT*.
The final calibrated `couch` pose is approximately `(0.10, -1.15, pi/2)`.

`mujoco_scenes/living_room_sofa.py` implements active under-sofa perception:

- captures RGB and depth from only the two low cameras;
- supports explicit `oracle` and image-only `sam3` mask backends;
- back-projects masked depth into world coordinates;
- accepts evidence only inside the bounded under-sofa volume;
- requires both low cameras to provide sufficient evidence;
- updates observed state only after inspection;
- saves RGB, masks, camera calibration, region points, and metadata under
  `runs/living_room_sofa/<mode>/`.

Last validated end-to-end result from the actual RRT* sofa pose:

```text
navigation couch None 2198
Remote observed beneath sofa by 2/2 foot cameras
left_foot_camera: 984 valid region points
right_foot_camera: 1038 valid region points
```

The ordinary top-down pick remains disabled for a remote still beneath the
sofa. Physical extraction with a hook/reacher tool has not yet been implemented.

## Live camera debugging

`mujoco_scenes/living_room_camera_debug.py` provides a Tk live-debug window for
all seven observation feeds:

- two ground views;
- five upper views;
- throttled rendering to avoid unnecessarily slowing simulation;
- an option to show the exact latest ground-camera segmentation overlays while
  keeping the top-camera feeds live.

Launch it with:

```bash
MUJOCO_GL=glfw uv run python -m mujoco_scenes.living_room_scene \
  --scenario lost_remote \
  --sofa-perception oracle \
  --robot-debug-view \
  --viewer
```

`--sofa-debug-view` remains a compatibility alias. The Actions panel also has
an **Open live robot-camera view** button.

For under-sofa testing:

1. Move to `Couch`.
2. Press **Inspect beneath sofa with foot cameras**.
3. Enable the latest-mask option in the debug view.
4. Inspect `runs/living_room_sofa/oracle/inspection.json` and camera images.

For SAM 3.1, run the separate `perception_server` workspace, set
`SAM3_BASE_URL`, and replace `--sofa-perception oracle` with
`--sofa-perception sam3`.

## Action text file

The grounded action file is:

```text
mujoco_scenes/configs/living_room_actions.txt
```

The Actions window loads it with **Reload and run**. Useful commands include:

```text
move couch
inspect sofa
state observed
gt
```

Other verbs include move, open, close, pick, place, and functional task calls.
`gt` is deliberately explicit ground-truth output for evaluation.

## Point-cloud branch audit

Naren's point-cloud branch was initially misunderstood. Its kitchen XML declares
six task cameras, but its geometry fusion rig uses exactly five logical views:

- inspection left;
- inspection right;
- inspection top;
- inspection front;
- inspection close.

Those are virtual region-facing cameras dynamically repositioned from
`configs/inspection_rigs.yaml`; they are not five robot wrist cameras.

The current branch has integrated versions of:

- `geometry_checker.py`
- `geometry_properties.py`
- `configs/geometry_inference.yaml`
- `configs/inspection_rigs.yaml`
- `observed_geometry.py`
- `inspection_policy.py`
- `perception.py`
- `sam3_client.py`
- point-cloud tests and documentation
- `observed_state.py`
- `sequential_inspection.py`
- `semantic_grounding.py`
- `task_witness.py`
- `generate_grounding_report.py`

The local `geometry_checker.py` additionally supports learned image masks and
cross-camera centroid association rather than relying solely on MuJoCo object
segmentation. Do not blindly replace these integrated files with another
branch; preserve the Google Robot, interactive living room, S1/L2 ablations,
and local stability fixes.

The last integrated Naren baseline is `b4dcbd6`; the newest fetched comparison
tip is `origin/naren/googlePointCloudIntegration-phaseC` at `a15d7ed`. Fetch
and compare the remote Phase-C ref before any future integration:

```bash
git fetch --prune origin
git log --oneline --decorate -10 origin/naren/googlePointCloudIntegration-phaseC
git diff --name-status b4dcbd6..origin/naren/googlePointCloudIntegration-phaseC -- mujoco_scenes
```

## Current perception boundary

The current state is intentionally honest:

- The controlled S1/L2/workshop observation pipeline, persistent registry,
  YOLO-World semantics, point-cloud geometry, witnesses, ablations, and reports
  are integrated.
- The interactive living room has two lower cameras for the sofa and five upper
  robot-mounted cameras for general coverage.
- SAM 3.1 is integrated through `sam3_client.py` for learned mask inspection
  and the under-sofa path, but the S1/L2 semantic CLI runners still expose only
  `yolo_world` and `none` as built-in semantic detector choices.
- YOLO-World identification quality is insufficient on several small or
  unusually viewed workshop/kitchen objects. The next perception experiment
  should connect SAM 3.1 to the common semantic backend and compare identical
  saved RGB evidence against YOLO-World. A lighter alternative is
  MobileSAM-style proposals plus a lightweight image-text crop classifier.
- Require multi-view support, persistent generic IDs, and an explicit
  `UNKNOWN` outcome rather than forcing every mask into a known category.
- Simulator segmentation and names remain oracle/offline-evaluation evidence,
  never learned-path semantic inputs.

## Foundation-model inference

Remote inference is intended to use an OpenAI-compatible server. The repository
contains separate workspaces for server deployment:

- `inference_server/` for Docker-first vLLM/SGLang multimodal model serving;
- `perception_server/` for SAM 3.1.

`inference_server/` is self-contained and can be rsynced directly to the RTX
5090 machine. `models.json` profiles Qwen3.5-9B, GLM-4.6V-Flash,
Qwen3-VL-8B-Thinking, InternVL3.5-14B-HF, and the updated
Kimi-VL-A3B-Thinking-2506 checkpoint. It starts one model at a time, retains
weights/cache on the host, supports up to eight images, and uses load-time FP8
for the two 15B/16B-total checkpoints. `./serve` provides `doctor`, `list`,
`up`, `logs`, `down`, and redacted `command` operations; `smoke_test.py` checks
the OpenAI-compatible endpoint with text or multiple local images. vLLM is the
default backend and SGLang is a per-launch fallback. Muse Glimmer remains a
disabled profile until an official local checkpoint/serving recipe exists;
Muse Spark is a hosted Meta Model API model, not a local 5090 checkpoint.

`inference_server/NEW_PC_SETUP.md` is the standalone end-to-end runbook for a
fresh client: native `uv`/vLLM setup on the Fish-based GPU server, both server
processes, keyless loopback binding, the two-port SSH tunnel from Bash, camera
image transfer, prompt submission, shutdown, and common failure modes.

The non-Qwen profiles are now wired through the same functional API rather
than merely listed as launch placeholders. Every registry entry contains its
own planner mode, sampling values, and output budget. Qwen3.5 and
GLM-4.6V-Flash support a thinking/non-thinking toggle; GLM launches with the
vLLM `glm45` reasoning parser. Qwen3-VL-8B-Thinking and
Kimi-VL-A3B-Thinking-2506 are fixed-thinking checkpoints. InternVL3.5-14B-HF
now also defaults to its prompt-driven thinking mode, using temperature 0.6,
top-p 0.95, and a 12,288-token output budget. Qwen3-VL uses presence
penalty 0 rather than the Qwen3.5 setting, and the other profiles likewise no
longer inherit Qwen3.5 sampling accidentally.

All thinking samplers were checked against the current checkpoint guidance and
are protected by a regression test: Qwen3.5 uses temperature 1.0/top-p
0.95/top-k 20/min-p 0/presence 1.5/repetition 1.0; GLM-4.6V-Flash uses
0.8/0.6/2/repetition 1.1; Qwen3-VL visual inference uses 1.0/0.95/20/presence
0/repetition 1.0; InternVL thinking uses temperature 0.6/top-p 0.95; and Kimi
2506 uses temperature 0.6 from that checkpoint's bundled generation config.
Qwen3.5 now has a 24,576-token planner cap after it reached the former 12K
limit before emitting final JSON. The remaining 12,288-token caps and Qwen's
larger cap are local deployment bounds, not creator-recommended maxima.

`server.py` now permits a missing inference API key only when the raw model
server is bound to loopback. This makes the registry-backed native command
usable with the existing SSH tunnel: set `INFERENCE_MODEL`, set
`INFERENCE_HOST=127.0.0.1`, erase `INFERENCE_API_KEY`, and run
`python3 inference_server/server.py`. Network-facing and Docker launches still
require a key. `NEW_PC_SETUP.md` contains the exact Fish commands for switching
both the model process and the planner process between all five profiles.

InternVL and Kimi are intentionally experimental for functional JSON because
their reasoning modes are not connected to dedicated vLLM parsers here. Their
requests embed the JSON schema in the prompt, strip `<think>` or `◁think▷`
markers, extract the final object, and apply the same deterministic validator.
Raw multimodal serving is still available if a sample fails that contract.
`blobfile` was added to the native requirements for Kimi's remote model code.

`./serve up MODEL --detach` starts the raw OpenAI-compatible model backend on
port 8000 and an authenticated functional-decomposition API on port 8080. The
direct VLM action planner was removed because action sequencing belongs after
search and deterministic feasibility checks. The relevant files are:

- `functional_catalog.json`: the standalone copy of the simple repository
  function registry, forbidden generic labels, and the required range of
  10–15 ranked candidates;
- `functional_planner.py`: image/goal prompt, strict decomposition schema,
  request validation, ranked-type validation, and dependency-cycle checks;
- `planner_api.py`: `GET /health`, `GET /v1/functions`, and
  `POST /v1/decompose`, with authentication for non-loopback deployments;
- `functional_client.py`: local-image CLI for testing the API.

The VLM selects configured functions such as `can_hold_liquid` or `can_stir`
and ranks 10–15 concrete object or region types for each replaceable role.
These are commonsense priors, not detections: the prompt forbids claims that a
type is present, visible, graspable, or geometrically feasible. It also forbids
object IDs, simulator names, action sequences, search order, and geometry
results. The production system prompt contains no concrete candidate examples,
and a regression test protects this candidate-unseeded condition. The
downstream search must ground observed instances and run the
task-specific semantic and point-cloud checks. The response envelope states
that search, semantic grounding, geometry verification, and execution have not
started. Kitchen, living-room, and workshop labels are accepted.

The decomposition validator normalizes harmless `unsupported_reason`
placeholders (`none`, `N/A`, or `not applicable`) to the required empty string
for `DECOMPOSED`. It still rejects a genuinely non-empty reason, a decomposed
response with zero requirements, or an unsupported response that contains
requirements.

Native SSH-tunnel testing also supports no-key operation. `planner_api.py`
defaults to `127.0.0.1`, omits upstream authorization when no inference key is
configured, and accepts unauthenticated clients only on a loopback bind. It
refuses keyless startup on `0.0.0.0` or another non-loopback host. The smoke
test and functional client likewise omit rather than fabricate Authorization headers.
This is intended for the user's `ssh -L` workflow; network-facing Docker
deployment remains authenticated.

Qwen3.5 thinking is enabled for functional decomposition through
`chat_template_kwargs.enable_thinking=true`. It can be disabled without a code
change using `PLANNER_ENABLE_THINKING=false` for the low-latency ablation. Its
profile output budget is 24,576 tokens. A reasoning-only or otherwise null
final response reports the finish reason and recommends increasing the budget
or, for a toggleable checkpoint, disabling thinking. Empty
`PLANNER_MAX_TOKENS` and `PLANNER_ENABLE_THINKING` values now select the model
profile defaults.

Do not use greedy decoding for Qwen3.5 thinking. It repeatedly consumed the
entire 4K, 8K, and 12K budgets without reaching final JSON. Functional requests
now use the official Qwen3.5 thinking sampler: temperature 1.0, top-p 0.95,
top-k 20, min-p 0, presence penalty 1.5, and repetition penalty 1.0. The
non-thinking ablation uses temperature 0.7 and top-p 0.8.

The client code remains backend-neutral. No Hugging Face training pipeline is
required; this project uses existing models for inference only. Actual model
loading still needs validation on the RTX 5090 host because the local machine
does not provide that GPU or download the checkpoints.

## LLM3-style zero-training baseline

`llm3_baseline/` is a separate comparison workspace. It does not vendor the
original PyBullet project and contains no training or fine-tuning code. It
calls the same raw OpenAI-compatible VLM endpoint as the other clients and
loads the selected model's thinking mode, sampling recipe, token limit,
reasoning markers, and structured-output capability from
`inference_server/models.json`.

The baseline includes strict kitchen, living-room, and workshop action
catalogues; a visible-state observation schema; a multimodal single-plan CLI;
and a bounded `LLM3Executive` that returns recoverable execution failures to
the VLM. Output validation rejects invented actions, missing arguments,
unobserved object IDs, and unknown region IDs. `INSPECT` may refer to a known
uninspected region, but no hidden contents may be included in the observation.
An independent goal flag/verifier owns completion rather than the VLM.

Run one planning-only request with:

```bash
python3 -m llm3_baseline.client \
  --goal "Stir the contents of the visible mug" \
  --observation llm3_baseline/example_observation.json \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png
```

Set `LLM3_MODEL_BASE_URL` to the raw tunneled model endpoint and
`LLM3_PROFILE` to a configured profile. See `llm3_baseline/README.md` for the
full command and contract.

`LLM3MuJoCoExecutor` now connects `LLM3Executive` to the same
`MuJoCoSkillDispatcher` used by the functional planner. It maps shared IK,
collision, path, grasp, placement, and precondition failures into the bounded
replanning loop. The VLM supplies bounded continuous parameters; PLACE, POUR,
and STIR consume them in the shared physical layer. The standalone client is
planning-only, while `llm3_baseline.run_kitchen` constructs the observer,
private goal verifier, and physical dispatcher.

## VLM-TAMP zero-training baseline

`vlm_tamp_baseline/` is an isolated algorithm-faithful VLM-TAMP port based on
Yang et al., ICRA 2025. It uses the existing frozen OpenAI-compatible VLM
profiles, installs the official pinned PDDLStream dependency under ignored
`.paper_deps/`, and does not train a model. The exact local prompts are in
`vlm_tamp_baseline/prompt.py`.

Unlike LLM3, the VLM never receives the primitive action catalogue. The first
query produces ordered English intermediate goals; the second grounds them as
formal predicates using the goal, images/visible relations, complete planning
object names, history, and failure feedback. Hidden object locations and
region contents are not exposed. `PDDLStreamSubgoalRefiner` solves each formal
subgoal through the paper's three-attempt reducer protocol before the shared
physical dispatcher executes it. `VLMTAMPExecutive` independently checks
subgoal and final-goal effects and reprompts after failed grounding,
refinement, execution, or effect validation.

The live paper protocol supplies the complete manipulable object-name universe
while keeping private locations in the TAMP world model. The standalone CLI is
planning-only:

```bash
VLM_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1 \
VLM_TAMP_PROFILE=qwen35-9b \
.venv/bin/python -m vlm_tamp_baseline.client \
  --goal "Stir the contents of the visible mug" \
  --observation llm3_baseline/example_observation.json \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png
```

This should be reported as a **VLM-TAMP algorithm port to the shared kitchen
domain**, not an exact reproduction of the official PR2 kitchen benchmark.
The original release's PDDLStream/FastDownward planning protocol is preserved;
PyBullet/PR2 perception, IK, and control are replaced by the same MuJoCo Google
Robot substrate used by the other methods for a controlled comparison.

The two baseline implementations are now structurally independent:

- `llm3_baseline/` owns the direct-action prompt, planner, executive,
  execution adapter, CLI, tests, and live runner;
- `vlm_tamp_baseline/` owns the subgoal prompt/catalogue, planner, refiner,
  executive, execution adapter, CLI, tests, and live runner; and
- `baseline_common/` owns only neutral observation/action records, the shared
  physical-skill catalogue, generic image/model transport helpers, and the
  final MuJoCo dispatcher bridge.

Neither baseline imports the other. Folder-boundary tests enforce this rule so
their execution loops and effect ledgers can evolve independently while both
retain the same low-level controller for a fair comparison.

## Live LLM3 and VLM-TAMP kitchen baselines

Added end-to-end entry points on 2026-08-23:

- `python -m llm3_baseline.run_kitchen` runs direct VLM action planning,
  physical execution, observed-effect verification, and bounded replanning;
- `python -m vlm_tamp_baseline.run_kitchen` runs VLM subgoal proposal,
  pinned PDDLStream refinement, physical execution, subgoal verification, and
  bounded reprompting;
- `mujoco_scenes/baseline_kitchen_runtime.py` is policy-neutral simulator
  plumbing shared by both. It owns the fresh Google Robot scene, Naren's five
  inspection-rig views, physical `INSPECT`, sanitized observed state, viewer, effect ledger,
  and private benchmark evaluator;
- successful `SkillResult.effects` now survive the common baseline adapter and
  are stored in both execution histories;
- placement destinations may be a known region or visible receptacle object,
  enabling physical spoon-in-bowl goals; and
- the private evaluator accepts functionally interchangeable spoon instances
  after the shared physical checks, instead of requiring the reference
  planner's exact stirrer ID. Re-placing an object removes its stale placement
  effect.

The reference deterministic plan is used only inside `_private_evaluation/` to
construct the fixed benchmark and resolve observed generic IDs. It is never
included in either VLM prompt. Hidden region contents become visible only
after a successful physical inspection. The legacy defaults remain
`runs/.../llm3_e2e` and `runs/.../vlm_tamp_e2e`, but a non-empty directory is
refused; paper runs should always provide a fresh explicit `--output-dir`.

Both live baseline runners call the raw OpenAI-compatible endpoint and must use
the `18000 -> server:8000` SSH tunnel. The `18080 -> server:8080` tunnel is only
for `planner_api.py` functional decomposition and has no
`/v1/chat/completions` route. The baseline runbooks were corrected accordingly
on 2026-08-23.

The first live LLM3 attempt returned `NO_VALID_PLAN` at revision 1 and executed
zero actions. An earlier safeguard forced inspection while unopened regions
remained; this was intentionally removed on 2026-08-23. Both baselines now plan
from the current visible observation without automatic search. `INSPECT`
remains available through the action/formal-subgoal catalogues and runs only
when proposed by the VLM. Both runners save parsed proposals under
`model_calls/` and print concise numbered plans in the terminal; VLM-TAMP also
prints each deterministic subgoal refinement.

One live attempt inspected all five regions, successfully picked and poured
from the kettle, returned it to the countertop, then failed
`PICK(object_0004)` with `GRASP_FAILED`. Qwen subsequently returned
`NO_VALID_PLAN`. That decision is preserved as baseline behavior rather than
overridden by a custom alternative-selection heuristic. Model-call artifacts
include observation revision and incoming failure feedback. The external
effect ledger clears stale `holding(object)` after a successful PLACE.

The third live LLM3 run (`runs/llm3_e2e_test_03`) exposed execution-boundary
bugs that were fixed on 2026-08-23:

- `KitchenExecutionObserver` reverse-maps the low-level held MuJoCo body to its
  observed generic ID and fails closed if no binding exists;
- `LLM3Executive` validates the complete proposed hand-state sequence before
  authorizing motion. Invalid PICK/PLACE/POUR/STIR/INSPECT ordering becomes a
  recoverable `precondition_failed`; VLM-TAMP's refiner also reads the runtime
  `robot.held_object` key and refuses a second PICK or held inspection;
- occupied-gripper exceptions are recoverable precondition failures instead
  of terminal internal errors. Full JSON-safe physical failure telemetry is
  kept in episode history but omitted from history sent back to the VLM; and
- dynamic PLACE preserves the requested grasp rotation during both approach
  and descent and rotates the held body/gripper offset correctly. This fixes
  source return after a side-wall jar grasp.

The exact headless regression `PICK(object_0010) ->
POUR(object_0010,object_0001) -> PLACE(object_0010,countertop)` was replayed
against `runs/kitchen_live_02`; all three physical skills and strict placement
postconditions passed. The Google kitchen arm command rate was raised from
`0.60` to `0.85 rad/s`; IK, collision, contact, tracking, and settle thresholds
were not relaxed.

## Kitchen/living-room Phase-C integration

The dirty worktree ports the reusable Kitchen and Living-Room parts through
Naren's `5559650` execution-complete tip:

- assignment-driven kitchen role cardinality and multi-view semantic conflict
  handling;
- inspection-derived source provenance rather than oracle-derived location;
- kitchen Phase A/B/C access, PICK/PLACE, POUR/STIR, entity resolution,
  manipulation stance selection, guarded IK/collision motion, and ledgers;
- all 10 controlled living-room variants (L1-L10), realistic rigid assets, Phase-1
  region-function grounding, dynamic mobile stance refinement, and physical
  placement verification;
- a shared observed-registry adapter and symbolic-to-physical dispatcher; and
- the production `GroundedPlanExecutive`, which consumes a verified witness,
  calls only a deterministic sequencer on failure, and owns observed execution;
- `KitchenGroundedExecution`, `run_kitchen_planner_execution.py`, and the
  living Phase-1/Phase-3 CLIs.

The missing kitchen goal-to-execution bridge was completed on 2026-08-19:

- `symbolic_planning.py`, `symbolic_planning_core.py`, and
  `run_kitchen_symbolic_pipeline.py` are the observed-state deterministic
  Phase-2 planner ported from Naren's Phase-C branch;
- `kitchen_execution_bundle.py` writes the validated plan, execution inventory,
  one-to-one entity resolution, frozen registry, and witness bundle;
- `run_kitchen_goal_execution.py` sends the goal plus stage-000 images to the
  functional API, validates the exact configured task scope, runs source
  grounding and Phase 2, and launches live Google-robot Phase-C execution;
- the exact model prompt is
  `inference_server/prompts/functional_decomposition.txt`;
- YOLO-World is installed with CPU-only PyTorch and both S1 semantic configs
  specify `device: cpu`; no local GPU or YOLO server is required;
- `minimum_rim_enclosure_ratio` is now 0.66 (8/12 angular sectors), while
  open-centre and multiview-interior checks remain mandatory. This fixed two
  handled mugs that were conservatively `UNKNOWN`;
- `coffee_source` was added to the primary semantic vocabulary so the coffee
  tin is not consumed as a third mug.

`runs/kitchen_live_02` was produced locally as a real CPU perception run. It
reached `COMPLETE` at C1, the planner generated 31 actions, and all 11 observed
IDs resolved one-to-one. Isolated `PICK(object_0010)` succeeded physically in
13 seconds. A sequential smoke replay completed kettle PICK and its first POUR,
then was manually stopped during the second POUR because the full 31-action
physical run is lengthy; this was not a controller failure.

Rendered benchmark reports, frozen `runs/` evidence, videos, and oracle-only
diagnostics are excluded. The compact Living-Room Phase-1 production inputs
and Phase-2 symbolic outputs required by the execution CLI are included under
`mujoco_scenes/benchmark_reports/`; this keeps all L1-L10 dry runs reproducible
without importing generated images or recordings. See `EXECUTION_AND_TESTING.md`
for the commands, implemented boundaries, and artifact contracts.

## VLM-TAMP live-run corrections (2026-08-24)

The API verification run exposed and fixed two baseline integration errors:

- the VLM no longer receives the frozen complete Phase-1 object inventory.
  Its object universe is accumulated only from episode observations, so hidden
  objects become planner-visible only after an executed inspection reveals
  them. The complete inventory remains private to PDDLStream geometry checks;
- Phase-1 provenance values `INITIAL`, `TABLE`, and `TABLETOP` now map to the
  execution domain's open `countertop` region. Previously, PDDLStream treated
  initially visible tabletop objects as if they occupied a closed region and
  could not refine even `HOLDING(object_0010)`.

The regression fixture uses the real `source_container=null`,
`observed_source_region=INITIAL`, `source_kind=TABLE` shape and verifies that
the coffee source refines to `PICK`. The isolated LLM3/VLM-TAMP suites pass:
`63 passed` with external pytest plugin autoload disabled.

A subsequent live run exposed that installed Mink was being selected by the
generic `auto` default for calibrated physical manipulation. All jar candidates
failed in DAQP before any arm trajectory existed, so repeated PICK attempts
looked frozen. The default is restored to the calibrated constrained DLS
backend; Mink remains opt-in through `MUJOCO_IK_BACKEND=mink`. Physical mode
transitions are now printed once as `[physical] ...`, and Phase-B preserves the
detailed PICK failure message instead of reducing it to `GRASP_FAILED`.

`NO_VALID_SUBGOALS` from the formal grounding stage now consumes one VLM call
and reprompts with neutral structured failure feedback while calls remain. It
no longer terminates a three-call episode after the first ungroundable English
proposal.

Phase C also now prefers the registry's canonical `source_region` over raw
`last_evidence_source_region` and normalizes stage-000 table tags. Without
this, valid POUR/STIR actions targeting initially visible vessels were removed
from the authorized plan because raw `INITIAL` was mistaken for storage,
causing `POUR_TARGET_RESOLUTION_FAILED` after a successful PICK.
The synchronous held-payload stance searches now publish one physical status
before POUR/STIR planning so their stationary search interval is visible.
VLM-TAMP terminal subgoals, refined skills, and PDDLStream plans retain exact
generic IDs while appending aliases learned from episode observations, e.g.
`object_0010 (coffee_source)`. Hidden-object labels are not preloaded merely
for display.

Liquid-bearing transport was made an explicit physical invariant on
2026-08-24. `JAR_SOURCE` and `KETTLE` picks preserve the payload's local +Z
axis against world +Z, record the maximum observed carry tilt, and fail closed
above 10 degrees. A generic compact-arm fold is prechecked against the same
constraint. This exposed the previous Phase-C POUR recovery as physically
invalid: the coffee jar's generic navigation fold predicted 109.5 degrees of
tilt even though the final recorded carry was upright. POUR now searches from
the live upright carry, returns directly through a Cartesian upright recovery
to the recorded post-PICK carry, and only then restores the base. A successful
POUR marks its target as liquid-bearing so later PICK/MOVE operations inherit
the upright constraint. The exact headless `PICK(object_0010) ->
POUR(object_0010, object_0001)` replay completed with `PICK_SUCCESS` and
`POUR_MOTION_VERIFIED`; the recovery trace contains
`RECOVER_UPRIGHT_POST_PICK_CARRY_ARM` and no generic compact fold. The focused
manipulation/Phase-B/Phase-C suite passes (`40 passed`).

On 2026-08-25 the coffee-jar POUR default was increased from 35 to 55 degrees
so the kinematic transfer proxy is visually unambiguous. The exact headless
`PICK(object_0010) -> POUR(object_0010, object_0001)` replay still completed
with `POUR_MOTION_VERIFIED`, a 55-degree measured maximum tilt, and a positive
1.51 cm outlet-interior alignment margin. A proposed kettle increase to 25
degrees was rejected during validation: the first geometry contacted the
countertop, while raising the outlet made the strict pre-pour pose unreachable.
The kettle therefore retains its previously verified 10-degree calibration;
no invalid family-wide change was kept. The focused suite now passes (`41
passed`).

On 2026-08-25 a repeated VLM-TAMP `MODEL_CALL_BUDGET_EXHAUSTED` run was traced
to two deterministic failures rather than insufficient generation tokens.
For a STIRRED subgoal while the coffee jar was held, equal-cost PDDL actions
allowed `place-object` to park the jar in the cup; that unsafe placement then
tripped the liquid-upright guard and left no useful third model call. The
PDDLStream adapter now exposes object-relative placement only for an exact
`PLACED(object, receptacle)` subgoal and returns incidental held objects to
their observed source region. A regression confirms STIR refinement produces
`PLACE jar countertop -> PICK spoon -> STIR`, while explicit spoon-in-mug
placement remains supported. Physical replay also found that the soft grasp
weld let the jar lag the wrist after POUR (about 15 degrees despite an upright
recovery target). Held-task trajectories now stiffen the already contact-
verified grasp weld and use strict final joint tracking. The exact headless
`PICK -> POUR cup -> POUR mug` replay passed both pours and ended at 6.87 and
6.89 degrees, below the unchanged 10-degree transport limit. The focused
VLM-TAMP/Phase-C suite passes (`43 passed`).

## Validation

The latest complete repository test result (rerun after the VLM-TAMP
model-budget/PDDL placement and rigid held-payload recovery fixes) is:

```text
644 passed, 4 skipped, 14 warnings
```

This was rerun after the liquid-safe transport/recovery correction, the second
repository audit, Phase-C integration, production
grounded executor, deterministic kitchen Phase-2 planner, live goal runner,
self-contained execution fixtures, LLM3 test discovery, the isolated VLM-TAMP
subgoal/refinement baseline, both five-camera live baseline runners, and
enforced cross-baseline folder boundaries, held-state safeguards, failure
telemetry, and grasp-preserving source placement on 2026-08-23.
Kitchen, living room, and workshop were also launched directly with Google
Robot; kitchen and living room produced headless render frames.

The warnings were matplotlib/pyparsing deprecations, not test failures. The
four skips are HTTP socket integration tests because this Codex sandbox denies
local socket creation; planner validation and transport tests pass, and those
HTTP tests run normally on an unrestricted machine.

## Shared visual-baseline input contract (2026-08-25)

LLM3 and VLM-TAMP now receive the same evaluation evidence in their live
kitchen runners: the goal, five RGB views, persistent instance/region IDs, and
a semantic-neutral textualized state. The RGB views are annotated with generic
IDs such as `object_0001` and `C1`; they never contain semantic names such as
`mug`, detector output, or functional tags. The textual state contains IDs,
observable locations/relations, region open/inspected state, and robot state
only. Hidden objects are absent until an executed inspection reveals them.

MuJoCo instance segmentation supplies the bounding boxes and cross-view ID
correspondence. It is therefore an oracle **instance tracker**, explicitly
recorded in `shared_observation_contract.json`, but it supplies no semantic
class. LLM3 retains its method-specific textualized-state/full continuous-plan
prompt, while VLM-TAMP retains its two-stage English-subgoal/formal-grounding
prompts and PDDLStream refinement. Both VLM-TAMP stages receive the same five
images because the grounding stage must resolve generic IDs visually.

Every call trace records `model_visible_input` without embedding base64 data.
Raw frames, ID-annotated frames, and per-camera bounding boxes are stored below
`observations/`. `latest_observation.json` is model-safe; semantic labels and
the private goal contract were moved below `_private_evaluation/`. Terminal
aliases remain human-only and are not included in prompts. The exact protocol
and paper claim boundary are documented in `BASELINE_FIDELITY.md`.

The original VLM-TAMP used semantically annotated/named observations, while
the original LLM3 release was text-only. Consequently these must be reported
as shared-domain visual adaptations, not exact input-level replications.

Useful validation commands:

```bash
git diff --check
./mujoco_scenes/scripts/validate_repository.sh
```

The five upper camera renders were visually checked at the standard home pose.
They have distinct overlapping directions and are mounted high enough that the
head and compact arm do not substantially block the environmental views.

## Native setup

From the repository root:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r mujoco_scenes/requirements.txt
```

Use `MUJOCO_GL=glfw` for interactive native windows and `MUJOCO_GL=egl` for
headless rendering/tests.

## Suggested prompt for the next Codex session

```text
Read CODEX_HANDOFF.md completely, then inspect the current dirty worktree.
Preserve all existing changes on `kitchen_livingroom_integration`. The dirty
worktree combines cleanup with Naren Phase-C perception/execution and shared
planner adapters. Read `EXECUTION_AND_TESTING.md`, rerun the complete suite,
and inspect the diff before committing. Keep the production functional
search separate: one-shot FM decomposition, fixed search, first-feasible
stopping, and no FM replanning. The LLM3 and VLM-TAMP comparison baselines may
reprompt on shared refinement/motion failure. Do not add oracle inventories or generated evidence to
planner-visible state.
```

## Latest execution/variant integration (2026-08-25)

The reusable Kitchen and Living-Room changes through upstream commit
`5559650` were ported onto `kitchen_livingroom_integration` while retaining
the Google Robot and the local VLM-TAMP execution adapter. Generated videos,
benchmark-report archives, expected-action exports, and Workshop scene changes
were deliberately not copied.

The active Kitchen benchmark is now the two-person 12-variant suite
`K1`–`K12`: six feasible placement/search variants and six single-cause
infeasible variants. The physical GT path has stable serving slots, direct
consecutive source pours, a larger visible pour angle, strict vertical
stirring geometry, five-camera recording, strict/assisted modes, and short
label support. Phase C retains `authorize_plan()` so VLM-TAMP replans are
frozen before dispatch.

The Living-Room benchmark has ten variants `L1`–`L10`, comprising six
feasible initial payload arrangements and four missing-region infeasible
cases. Its mobile execution, symbolic planning, region oracle, recorders, and
five-camera scene support are present.

The new one-call Qwen Kitchen functional-graph pipeline is available through
`mujoco_scenes.run_kitchen_vlm_pipeline`. It receives the goal, five raw
initial RGB views, the fixed response schema, generic checker API, and
observable region handles. It does not receive hidden contents, simulator
object identities/poses, oracle outcomes, GT assignments, or GT actions.
Downstream code validates the graph, grounds observed evidence, checks
geometry, stops search at the first complete witness, and produces an A*
symbolic action sequence. This runner intentionally stops before physical
execution.

See `docs/SCENE_VARIANT_CATALOGUE.md`,
`docs/KITCHEN_LIVING_ROOM_VLM_REQUIREMENTS_INTEGRATION.md`, and section 7 of
`EXECUTION_AND_TESTING.md`.

Final validation after completing the compact Living-Room runtime-artifact
integration: `792 passed, 4 skipped`; all public
Kitchen, Living-Room, and VLM-TAMP CLI `--help` imports passed, the `K1`
ground-truth dry run passed symbolic preflight, and `git diff --check` is
clean.

## Living Room planning-only VLM-TAMP baseline (2026-08-25)

`vlm_tamp_baseline.run_living_room` now runs L1-L10 from fresh five-camera,
ID-only annotated Living Room observations. It preserves the existing
two-query VLM-TAMP prompt and sequential PDDLStream refinement, but replaces
physical dispatch with a validated symbolic PICK/PLACE rollout because this
experiment compares plans only. The runner writes exact and ordered-LCS GT
sequence metrics against `EXPECTED_GT_ACTIONS/living_room/<Lx>` and reports
outcome agreement separately. GT actions, semantic roles, and simulator
backend resolution remain in `_private_evaluation/` and are not sent to the
model. An explicit model `NO_VALID_SUBGOALS` is an infeasibility prediction;
retry exhaustion remains unresolved.
The default is one two-stage model call (`INITIAL_PLAN_ONLY`) to avoid treating
unchanged RGB as a new physical observation; higher model-call limits are
recorded as `SYMBOLIC_REPROMPT_ABLATION`.

Validation after this addition: the Living Room focused baseline/runtime suite
passed `65 passed`; the complete headless repository suite passed
`799 passed, 4 skipped`. All L1-L10 adapter variants resolved successfully,
and a measured-geometry Living Room PLACED subgoal refined to PICK/PLACE with
the pinned PDDLStream backend.

## Expected GT action catalogue refresh (2026-08-25)

`EXPECTED_GT_ACTIONS/kitchen/K3`, `K4`, and `K6` were regenerated from the
current `5559650` Kitchen GT planner. The stale catalogues had retained an old
countertop-staging step for hidden soup bowls and the old soup-utensil pairing.
Their authoritative action counts are now 25, 28, and 31 respectively. A clean
all-variant dry run passed 12/12 symbolic preflights, and canonical operator,
argument, and reason comparison matched every K1-K12 catalogue with zero
mismatches. No physical execution was used during this refresh.

## Baseline smoke-test audit (2026-08-26)

The planning-only Table 4 benchmark now scores the predicted symbolic action
sequence against `EXPECTED_GT_ACTIONS`. Its compact columns are GT exact-match
rate and ordered LCS-F1; feasible/infeasible outcome labels remain diagnostic
metadata and are not paper-table scores.

The one-seed K1/L1 smoke grid for VLM-TAMP and OWL-TAMP at 1, 3, and 5 images
is under `runs/codex_smoke_20260826_01/{kitchen,living_room}`. All 12 processes
completed and wrote artifacts, but none returned a GT-overlapping sequence.
Observed terminal causes include invalid model JSON, `NO_VALID_SUBGOALS`,
`NO_PLAN`, symbolic-plan failure, and one Kitchen PDDLStream refinement
failure. These are baseline/model outcomes, not physical-execution results.

Two runtime defects found by the smoke run were fixed: OWL-TAMP now records
invalid JSON as `INVALID_MODEL_OUTPUT` instead of crashing, and Living-Room
region annotations use visible support geometry rather than transparent
logical marker geoms. The pre-fix Living-Room artifacts are archived under
`runs/codex_smoke_20260826_01/living_room_before_region_annotation_fix` and
must not be included in summaries. The focused baseline suite passes
`65 passed`, and `git diff --check` is clean.

## Baseline prompt hardening and retest (2026-08-27)

The completed 720-episode Kitchen batch was audited before reuse. All episode
artifacts and camera counts were complete, but OWL-TAMP returned no comparable
sequence in any trial and VLM-TAMP returned no exact match. Prompt/runtime
hardening was therefore applied before any new full batch: VLM-TAMP prompt v7
uses camera-count-neutral wording, bounds English step strings, restricts
formal arguments to observed IDs, discourages blanket object enumeration, and
records exact request text/schema plus raw responses without duplicating image
bytes. OWL-TAMP prompt v2 removes contradictory fake IDs, requests a concise
partial skeleton, compacts the unchanged grounded-action list, uses strict JSON
schemas, constrains helper-expression syntax, records raw responses, and writes
its prompt version into episode artifacts.

Targeted live tests showed that these changes remove VLM token-limit truncation
and OWL malformed sketch JSON, but do not solve Qwen3.5-9B's planning quality:
VLM still tends to enumerate visible IDs, and OWL still fills the 64-action
schema and produces an unsatisfiable Executed-subsequence constraint. Do not
rerun the full 720-episode batch until image annotation legibility and these
semantic planning failures are addressed. Focused validation after the schema
changes: `66 passed`; `git diff --check` clean.

## Complete Kitchen region annotations (2026-08-27)

The shared Kitchen baseline image generator now annotates every region exposed
in its textual observation. In addition to articulated storage regions `B1`,
`C1`, `C2`, `D1`, and `D2`, the static `countertop` and `serving_area` IDs are
grounded to the visible `counter_surface` and `serving_surface` MuJoCo geoms.
Initialization rejects a public region without annotation geometry, and a
regression test covers both static region mappings. A fresh K1 five-view EGL
capture confirmed all seven region IDs in all five annotation manifests.
## Kitchen benchmark annotation legibility

- Kitchen benchmark images now default to 640x480 for both VLM-TAMP and OWL-TAMP.
- `countertop` and `serving_area` annotations follow their segmentation masks instead of using oversized rectangular boxes.
- Object labels use collision-aware placement so they do not cover the objects they identify.
- Focused kitchen annotation tests pass (9 tests); the broader baseline suite passed before the final label-placement refinement (62 tests).
- Existing benchmark images are unchanged; rerun trials to generate images with the improved annotations.
- Visible objects and living-room regions use semantic aliases only in RGB annotations. Stable IDs remain internal to manifests and planner state. Both comparison baselines receive the same annotated RGB input.
- Living-room duplicate classes receive unique aliases (`cup_1`, `cup_2`, etc.). Both prompts receive an observable alias-to-ID map, removing the prior ID-only prompt contradiction. Living-room labels avoid object and label overlap.
- OWL-TAMP Living Room symbolic search previously counted every grounded-action attempt against its node-expansion budget, causing false `NO_SYMBOLIC_PLAN` results even for an exact valid L1 sketch. It now counts expanded states. A five-object/ten-action regression passes; old OWL results from the prompt-v8 batch must be discarded.
