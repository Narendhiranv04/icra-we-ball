# Robust TAMP MuJoCo benchmarks

The current benchmark layout is documented in
[THREE_SCENE_BENCHMARKS.md](THREE_SCENE_BENCHMARKS.md): kitchen object
alternatives, living-room region alternatives, and a compact workshop that
combines both. Google Robot remains the default interactive robot.

## Intended Workflow

The **S1 environment** demonstrates container-level missing-object search with
Google Robot, plus a deterministic `--no-robot` virtual-inspection mode.

The intended execution sequence is:

1. Google Robot observes the closed workstation.

2. It detects the following visible objects on the countertop:

   * Kettle
   * Coffee jar
   * Sugar jar
   * Spoon

3. It determines that the required **mug is missing**.

4. Using its mug-location prior, the controller selects container **C1** as the first search location.

5. Google Robot navigates to a suitable inspection pose in front of C1.

6. It opens C1 and observes the objects inside:

   ```text
   {mug, glass}
   ```

7. It selects the mug and transfers it to the countertop.

8. A later Task and Motion Planning (TAMP) system will:

   * Prepare the coffee.
   * Move the completed result to the serving area.

## Container-Opening Behaviour

Currently, `open_container()` directly commands the container actuator.

This implementation is intended as a **search and debugging action**, allowing the missing-object search pipeline to be developed independently of contact-based door manipulation.

Physical Google Robot door/drawer opening remains a later calibration milestone;
the current search path deliberately uses the deterministic container actuator.

## Environment Documentation

The complete region definitions, object layout, and usage guide are available in:

[S1_ENVIRONMENT.md](mujoco_scenes/S1_ENVIRONMENT.md)

The separate rigid-only Google Robot living room—with an L-shaped couch,
fixed coffee table, rigid tabletop objects, a wall-mounted TV, book-ledge and
two-drawer storage, remote-controlled TV power, and guarded TV dusting—is documented in
[LIVING_ROOM_ENVIRONMENT.md](mujoco_scenes/LIVING_ROOM_ENVIRONMENT.md). Both
environment families now use the same entry point and independently select
Google Robot or the robot-free virtual-observation mode:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --environment living-room --robot google --viewer

MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --environment kitchen --scene S1_coffee_missing_mug --no-robot --viewer
```

The compact `lost_remote` variant uses two RGB-D cameras mounted on the lower
front of the Google mobile base to inspect beneath the raised rigid sofa. A
separate five-camera rig above the robot head provides full-surround room views:

```bash
MUJOCO_GL=glfw uv run python -m mujoco_scenes.living_room_scene \
  --scenario lost_remote --sofa-perception oracle \
  --robot-debug-view --viewer
```

Remote functional-alternative ranking through either vLLM or SGLang is
documented in
[FOUNDATION_MODEL.md](mujoco_scenes/FOUNDATION_MODEL.md). Both servers use the
same small OpenAI-compatible client; no inference package is installed in the
MuJoCo environment.

The complete observe-assess-plan-execute-verify loop, living-room storage
alternatives, offline smoke-test mode, and extension points are documented in
[TAMP_PIPELINE.md](mujoco_scenes/TAMP_PIPELINE.md).

The separate [inference_server](inference_server/README.md) workspace contains
Docker-first vLLM and SGLang profiles for the project's multimodal models. It
can be rsynced directly to a 5090 server and has no MuJoCo dependency. Its
functional API accepts scene images plus a goal and returns simple functional
requirements with 10–15 ranked concrete object or region types. It
does not claim those types are present, search the scene, verify geometry,
sequence actions, or execute anything.

The separate [llm3_baseline](llm3_baseline/README.md) folder is a zero-training
comparison method. A frozen VLM directly proposes observation-bounded scene
actions, and its bounded executive can return shared motion failures for a new
proposal. The current command-line path produces validated plans only; live
MuJoCo skill dispatch remains an explicit integration step.

The complete native server and new-client-PC procedure is in
[NEW_PC_SETUP.md](inference_server/NEW_PC_SETUP.md).

The separate [perception_server](perception_server/README.md) workspace runs
SAM 3.1 on a GPU server. The simulator sends it RGB images and text prompts,
then performs depth back-projection and cross-camera association locally.

## Cameras

### Scene Cameras

The original five cameras remain available, with an additional lateral view:

* `left_shoulder_camera`
* `right_shoulder_camera`
* `overhead_camera`
* `side_camera`
* `front_camera`
* `wrist_camera`

The `wrist_camera` is mounted on the Google Robot gripper when the robot is enabled.

### Robot Camera

Google Robot additionally provides:

* `head_camera_rgb`

## Robot Backend

Google Robot is the default and is loaded from the workspace-level
`third_party/mujoco_menagerie` checkout:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug \
  --robot google \
  --viewer
```

Use `--robot google` or `--robot none`. Google Robot supports
scene loading, free/fixed/head/wrist cameras, joint targets, collision-checked
Actions-panel navigation, and an S1-calibrated vertical sugar-jar pick/place at
the serving area. It also supports the main-branch far-tip spoon pick and
gravity-settled vertical carry; spoon placement remains gated. Other scenes
and Google grasps remain gated until they pass the process in
[ROBOT_CALIBRATION.md](mujoco_scenes/ROBOT_CALIBRATION.md).

## Running the Scene Locally

Navigate to the repository's `V1` directory, then launch the scene:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug \
  --viewer \
  --camera front_camera
```

The launch also opens an `Actions` panel. Under `Move`, select `Home`,
`Cupboard 1`, `Cupboard 2`, or `Box`; each button executes the corresponding
collision-checked mobile-base trajectory. `Cupboard 2` and `Box` share one
physical right-side pose. Google Robot exposes the physically checked
sugar-jar pick/place and spoon pick/carry. Its navigation home is farther from
the serving table; manipulation actions automatically approach the work stance
and return to a collision-checked compact navigation state before Move is
enabled. Container inspection uses the deterministic scene-controller opening
path and immediately captures fresh region-facing evidence. Pass
`--no-actions-panel` to suppress this panel.

## Fixed-order observed-resource witness

The five-view persistent object graph is evaluated using point-cloud geometry
only. Categories and semantic function mappings do not participate in
property extraction, graph candidates, witness selection, or stopping.
This command observes the fully closed scene first, follows only the supplied
fixed inspection order, and stops only when a globally distinct all-`TRUE`
geometric witness is found:

```bash
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD/runs:/output" \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug \
  --no-robot \
  --task-requirements configs/s1_find_open_receptacle.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root /output \
  --run-id open_receptacle_region_evidence_demo \
  --point-cloud-width 320 \
  --point-cloud-height 240
```

This mode contains no robot or mobile-navigation action. Every stage saves
fresh, region-gated per-object measurement evidence and `witness.json`;
`latest_witness.json`, the global registry, and graph are atomically replaced
at run level. Historical cumulative clouds remain visualization-only and are
guarded from property extraction. See
[mujoco_scenes/README.md](mujoco_scenes/README.md) for the evidence layout,
camera/volume configuration, universal geometry schema and provenance,
geometry-only task documents, and the joint RGB semantic + point-cloud
geometry counterexample experiments. The joint experiment uses actual
YOLO-World RGB detections, generic mask-associated object IDs, measured
relations, and emits a verified role-assignment handoff without executing
robot motion or TAMP.

The repository also includes Ablation 3, which builds a complete measured
tool–target compatibility matrix and applies reuse/distinctness on task-level
function groups. Run its actual-detector demonstration and presentation with:

```bash
./mujoco_scenes/scripts/run_ablation3_multi_target_demo.sh \
  ablation3_multi_target_demo
xdg-open reports/ablation3_multi_target_demo/presentation_report.html
```

See the Ablation 3 section in
[mujoco_scenes/README.md](mujoco_scenes/README.md) for the four same-evidence
diagnostics, exact Docker command, measured matrix, scene variants, and output
layout.

The final integrated Scene 1 stress test combines persistent discovery,
semantic and unary grounding, target-specific binary geometry, exact
multi-target assignment, function-scoped reuse/distinctness, and
semantic-first relation pruning across six visible containers and the full
`INITIAL → D1 → D2 → C2 → B1 → C1` inspection horizon:

```bash
./mujoco_scenes/scripts/run_s1_integrated_kitchen_demo.sh \
  s1_integrated_kitchen_demo
xdg-open reports/s1_integrated_kitchen_demo/presentation_report.html
```

The broad goal and functional requirements are still manually connected; no
FM parsing, robot manipulation, action planning, or TAMP execution occurs.
See the integrated benchmark section in
[mujoco_scenes/README.md](mujoco_scenes/README.md) for exact local/Docker
commands, scene variants, measured progression, outputs, and limitations.

## Living-room region-functional grounding

The separate L2 benchmark grounds a destination region for the exact goal
“Place the refreshment tray on a suitable living-room surface within easy
reach of the sofa.” It demonstrates, from one shared RGB-D + YOLO-World
observation stream, that geometry-only incorrectly selects a rug,
semantic-only incorrectly selects an undersized side table, and joint
semantic–geometric verification selects the coffee table.

Run the complete benchmark and presentation package with:

```bash
./mujoco_scenes/scripts/run_l2_region_ablation1_demo.sh \
  l2_living_room_region_ablation1_demo
xdg-open reports/l2_living_room_region_ablation1_demo/presentation_report.html
```

See [LIVING_ROOM_ENVIRONMENT.md](mujoco_scenes/LIVING_ROOM_ENVIRONMENT.md)
for scene variants, evidence guarantees, exact local and Docker commands,
region registry, predicates, relations, outputs, and limitations.
