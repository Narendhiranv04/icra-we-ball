# S1: Coffee Preparation with a Missing Mug

## Intended Workflow

The **S1 environment** is designed to demonstrate container-level missing-object search using the Fetch mobile manipulator.

The intended execution sequence is:

1. Fetch observes the closed workstation.

2. It detects the following visible objects on the countertop:

   * Kettle
   * Coffee jar
   * Sugar jar
   * Spoon

3. It determines that the required **mug is missing**.

4. Using its mug-location prior, Fetch selects container **C1** as the first search location.

5. Fetch navigates to a suitable manipulation pose in front of C1.

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

In a later version, the Fetch gripper will physically open the same door joint through contact-based manipulation.

## Environment Documentation

The complete region definitions, object layout, and usage guide are available in:

[S1_ENVIRONMENT.md](S1_ENVIRONMENT.md)

## Cameras

### Scene Cameras

The original five cameras remain available, with an additional lateral view:

* `left_shoulder_camera`
* `right_shoulder_camera`
* `overhead_camera`
* `side_camera`
* `front_camera`
* `wrist_camera`

The `wrist_camera` is now mounted on the Fetch gripper.

### Fetch Camera

Fetch additionally provides:

* `head_camera_rgb`

## Running the Scene Locally

Navigate to the project directory:

```bash
cd /home/naren/RA_iiith
```

Launch the S1 environment using the front camera:

```bash
MUJOCO_GL=glfw /home/naren/miniconda3/bin/python \
  -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug \
  --viewer \
  --camera front_camera
```

The launch also opens an `Actions` panel. Under `Move`, select `Home`,
`Cupboard 1`, `Cupboard 2`, or `Box`; each button executes the corresponding
collision-checked mobile-base trajectory. `Cupboard 2` and `Box` share one
physical right-side pose. Under `Pick`, select a reachable table object or an
object exposed in a fully open drawer to run its vertical pre-grasp,
contact-aware close, lift, and return to the object-in-gripper carry pose.
Once an object is held, `Place` offers `Serving table`, `Table`, `Drawer 1`,
and `Drawer 2` when their physical preconditions are satisfied. `Table`
automatically selects the safe counter strip nearest the robot's current
home, left, or right base pose.
Placed objects remain selectable for another pick whenever the robot is at
the base pose corresponding to that strip. At the shared right-side pose,
`Open` → `Box` approaches B1's handle along +Y with vertical fingers, confirms
bilateral contact, and carries the real lid joint around its hinge to the
intentional 100-degree open position. It then opens the gripper, retreats
vertically above the lid, and returns to the empty carry-hover pose.
At Home, `Open` → `Drawer 1` and `Drawer 2` use mirrored front grasps on
identical, level U-handles. The gripper reaches each handle through a safe
high corridor, approaches along +Y, confirms bilateral contact, and pulls the
physical slide straight back along -Y to its full `0.25 m` limit. It then
releases, retreats horizontally from the opened handle, and returns to the
same empty carry pose.
Once open, D1/D2 contents become selectable under `Pick`. Utensils use
vertical handle-end grasps and a smooth, compliant working-end-down transition;
the D2 tissue uses a vertical centre pinch. `Place` also exposes `Drawer 1`
and `Drawer 2` while fully open. Both execute a vertical hover/descent/release
and vertical retreat, preferring the object's vacated stable tray slot before
sampling another buffered point.
Pass `--no-actions-panel` to suppress this panel.

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
