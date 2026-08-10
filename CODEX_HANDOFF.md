# Codex Project Handoff

Updated: 2026-08-10 (Asia/Kolkata)

This file is the working context for continuing the project from another Codex
session after SSHing into this machine. Start the next session by asking Codex
to read this file, inspect the current worktree, and continue without resetting
or discarding uncommitted changes.

## Workspace and Git state

- Repository root: `/home/boreddog/Documents/RRC/LH_Extension/V1`
- Current branch: `workshop_joint_prerequisites`
- The branch currently points to `a7760e7`, the committed workshop-only
  five-view point-cloud runner. The dirty worktree removes the lock/key task
  and ensures both storage regions are closed outside their active capture.
- The newest integrated upstream revision is `b4dcbd6`. Its complete L2
  Ablation 1/2/3 implementation and realistic movie-night assets are retained.
- Generated `reports/` media is deliberately excluded; report generators,
  source, configuration, assets, and tests are retained.
- Do **not** run `git reset`, `git checkout --`, `git clean`, or otherwise remove
  changes unless the user explicitly requests it.
- Remote: `https://github.com/Narendhiranv04/icra-we-ball.git`
- Latest fetched point-cloud comparison ref:
  `origin/naren/googlePointCloudIntegration` at `d898687`. The current branch
  retains the integrated shared geometry layer from the earlier merge and does
  not blindly copy the newer kitchen/living-room benchmark phases.
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
- Object alternatives should normally be capped at three.

## Overall research direction

This is a Robust TAMP research codebase using MuJoCo. The intended pipeline is:

1. Provide the foundation model with visible scene images.
2. Call the FM once to decompose the goal into simple functional requirements,
   propose/rank at most three common object or method types, and optionally
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
8. Give the verified object/region handoff to a downstream sequencer. The
   agreed direction is PDDLStream for grounded symbolic action ordering plus
   grasp, placement, IK, navigation, and collision-free trajectory streams.
9. Execute one action or short safe prefix, verify its observed effect, and
   deterministically re-run PDDLStream after execution failures. Do not use the
   FM as the replanner.

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
benchmarks are present. The dirty worktree now also contains the redesigned
compact workshop physical scene described below. See
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

Naren's controlled S1/L2 benchmark runners do **not** call an FM. Their task
YAML files are explicitly manual future-FM contracts
(`generated_from_foundation_model: false`). They run real RGB/RGB-D perception,
semantic and geometric grounding, same-evidence policy ablations, and emit a
verified handoff. They do not perform navigation, manipulation, task ordering,
PDDLStream, or TAMP execution.

The separate existing client in `mujoco_scenes/foundation_model.py` can call an
OpenAI-compatible vLLM/SGLang endpoint. Today it receives a required function
plus structured visible candidate IDs/categories/facts and returns a functional
subset and ranking. It does not yet receive the actual image payload or derive
the functional requirements from a natural-language goal. Replacing the manual
future-FM contract with validated structured multimodal output remains future
integration work.

The agreed execution architecture is:

```text
visible images + goal
        -> one FM decomposition/type-ranking call
        -> fixed sequential region inspection
        -> persistent cross-region observed-object registry
        -> first complete semantic + geometric witness
        -> PDDLStream problem construction/refinement
        -> guarded execution and deterministic replanning
```

There is no PDDLStream invocation in the repository yet. When added, keep the
outer discovery executive separate initially: it owns inspection, unknown
regions, evidence updates, and early termination. PDDLStream begins only after
a verified witness exists and owns the grounded execution sequence and
continuous samplers. Suggested minimal layout:

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
actions, IK work, and calibration support. The current design discussion is
focused on the third compact workshop benchmark; do not start unrelated kitchen
changes without a new request.

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

The dirty worktree implements a deliberately simplified, single-arm workshop
repair cell. It is one front-facing workbench intended to keep Google Robot at
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
`configs/workshop_joint_alternatives.yaml` and `workshop_alternatives.py` still
describe the older hammer/nail single-joint prototype and must be redesigned
in the next workshop-pipeline phase rather than treated as the new scene's
contract.

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

The integrated Naren reference is now
`origin/naren/googlePointCloudIntegration` at `b4dcbd6`. Fetch that exact
remote ref before any future comparison:

```bash
git fetch origin naren/googlePointCloudIntegration
git log --oneline --decorate -10 origin/naren/googlePointCloudIntegration
git diff --name-status b4dcbd6..origin/naren/googlePointCloudIntegration -- mujoco_scenes
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

- `inference_server/` for vLLM-oriented model serving;
- `perception_server/` for SAM 3.1.

The client code should remain backend-neutral enough to support vLLM or SGLang.
No Hugging Face training pipeline is required; this project uses existing
models for inference only.

## Validation

The latest complete repository test result is:

```text
366 passed, 14 warnings
```

The warnings were matplotlib/pyparsing deprecations, not test failures.

Useful validation commands:

```bash
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  .venv/bin/python -m pytest mujoco_scenes/tests -q
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
Preserve all existing changes. The compact workshop physical scene has been
implemented; do not redesign it without a new request. The next workshop work
is the separately authorized functional-grounding/search/sequencing phase.
Before changing that pipeline, inspect the scene inventory and explicitly
separate visible evidence from hidden drawer truth. Keep the one-shot FM,
fixed search, first-feasible stopping, and no-FM-replanning decisions.
```
