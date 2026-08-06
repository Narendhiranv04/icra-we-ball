# Codex Project Handoff

Updated: 2026-08-06 (Asia/Kolkata)

This file is the working context for continuing the project from another Codex
session after SSHing into this machine. Start the next session by asking Codex
to read this file, inspect the current worktree, and continue without resetting
or discarding uncommitted changes.

## Workspace and Git state

- Repository root: `/home/boreddog/Documents/RRC/LH_Extension/V1`
- Current branch: `three_scene_benchmarks`
- The branch contains the three-scene integration commit `412f092` and the
  newest controlled merge from `origin/naren/googlePointCloudIntegration`.
- The newest integrated upstream revision is `b4dcbd6`. Its complete L2
  Ablation 1/2/3 implementation and realistic movie-night assets are retained.
- Generated `reports/` media is deliberately excluded; report generators,
  source, configuration, assets, and tests are retained.
- Do **not** run `git reset`, `git checkout --`, `git clean`, or otherwise remove
  changes unless the user explicitly requests it.
- Remote: `https://github.com/Narendhiranv04/icra-we-ball.git`
- Integrated point-cloud reference:
  `origin/naren/googlePointCloudIntegration` at `b4dcbd6`.
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
2. Ask it to decompose a task into functional/geometric requirements.
3. Inspect regions in a fixed, random, or FM-ranked order for ablations.
4. Semantically ground visible object candidates from RGB.
5. Build RGB-D point clouds and test geometric requirements such as
   insertability, usable length, cross-section, cavity depth, or support.
6. Ask the FM to rank only visible feasible alternatives.
7. Select the first ranked candidate that passes deterministic geometry.
8. Terminate inspection/planning early when the functional witness is complete.
9. Execute and verify the chosen physical action.

Examples include choosing a spoon or chopstick instead of a coffee stirrer,
while rejecting undesirable objects such as a pen or knife after semantic/FM
ranking and geometric validation.

The persistent observation, tracking, semantic grounding, task-witness,
same-evidence ablation, and report-generation source from Naren's branch is now
integrated. The kitchen object-alternative and living-room region-alternative
benchmarks are present, and a compact workshop scaffold combines region and
object-system alternatives. See `THREE_SCENE_BENCHMARKS.md`.

The integrated test command is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  .venv/bin/python -m pytest mujoco_scenes/tests -q
```

It currently passes 361 tests, including both server workspaces, all three L2
region ablations, and fresh workshop RGB-D evidence.
Pytest is declared in `mujoco_scenes/requirements-dev.txt`.

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
actions, IK work, and calibration support. The user currently wants living-room
work prioritized; do not start unrelated kitchen changes.

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

The current branch has local versions of:

- `geometry_checker.py`
- `geometry_properties.py`
- `configs/geometry_inference.yaml`
- `configs/inspection_rigs.yaml`
- `observed_geometry.py`
- `inspection_policy.py`
- `perception.py`
- `sam3_client.py`
- point-cloud tests and documentation

The local `geometry_checker.py` additionally supports learned image masks and
cross-camera centroid association rather than relying solely on MuJoCo object
segmentation.

However, the following complete modules from Naren's branch are not currently
present in this worktree:

- `observed_state.py`
- `sequential_inspection.py`
- `semantic_grounding.py`
- `task_witness.py`
- `generate_grounding_report.py`

Do not blindly copy `scene_loader.py` or the whole remote branch because that
would overwrite Google Robot and living-room work. Port relevant modules
selectively and adapt their kitchen-region assumptions.

The remote branch has advanced beyond the previously audited `64719f1` commit.
Fetch and review the changes through the current `7ca4633` commit before doing
the persistent tracking integration:

```bash
git fetch origin naren/pointCloudExtraction
git log --oneline --decorate -10 origin/naren/pointCloudExtraction
git diff --name-status 3f2e377..origin/naren/pointCloudExtraction -- mujoco_scenes
```

## Current perception boundary

The current state is intentionally honest:

- The two lower cameras are fully integrated with the lost-remote inspection.
- The five upper cameras render valid distinct live images and provide
  full-surround visual coverage.
- The five upper feeds are **not yet connected** to persistent semantic/
  point-cloud region tracking or the foundation-model visible-state payload.
- That integration is the next major task.

A reasonable next implementation sequence is:

1. Define compact living-room observation regions and navigation poses.
2. Capture the five upper RGB-D feeds at each inspection pose.
3. Run SAM 3.1/semantic grounding only on visible images.
4. Associate detections across the five upper views without MuJoCo IDs.
5. Persist generic object tracks across region inspections.
6. Pass only visible labels/images/properties to the foundation model.
7. Add fixed/random/FM region-order ablations and early termination.
8. Keep oracle segmentation available only for evaluation.

## Foundation-model inference

Remote inference is intended to use an OpenAI-compatible server. The repository
contains separate workspaces for server deployment:

- `inference_server/` for vLLM-oriented model serving;
- `perception_server/` for SAM 3.1.

The client code should remain backend-neutral enough to support vLLM or SGLang.
No Hugging Face training pipeline is required; this project uses existing
models for inference only.

## Validation

The latest complete repository test result during this conversation was:

```text
Ran 118 tests in 18.988s
OK
```

The latest focused living-room result after final camera placement was:

```text
Ran 38 tests in 0.728s
OK
```

Useful validation commands:

```bash
git diff --check
.venv/bin/python -m py_compile \
  mujoco_scenes/living_room_cameras.py \
  mujoco_scenes/living_room_scene.py \
  mujoco_scenes/living_room_sofa.py \
  mujoco_scenes/living_room_camera_debug.py \
  mujoco_scenes/living_room_actions.py
.venv/bin/python -m unittest discover -s mujoco_scenes/tests
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

Use `MUJOCO_GL=glfw` for interactive native windows and `MUJOCO_GL=osmesa` for
headless rendering/tests.

## Suggested prompt for the next Codex session

```text
Read CODEX_HANDOFF.md completely, then inspect the current dirty worktree.
Preserve all existing changes. Continue by wiring the five upper Google Robot
cameras into living-room visible-state semantic/point-cloud tracking, using
origin/naren/pointCloudExtraction only as a selective reference. Do not expose
hidden simulator objects to the learned path, and keep oracle behavior explicit.
Before editing, tell me what has changed on the remote branch since 64719f1 and
which pieces you intend to port.
```
