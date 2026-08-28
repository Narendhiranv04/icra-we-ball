# Three-scene grounding benchmarks

The repository now uses one observation contract across three controlled
experiments. The foundation model receives only the visible image state and
returns functional requirements or a ranking over visible IDs. Geometry is
then validated from RGB-D evidence. Hidden simulator inventory is never sent
to the model.

| Scene | Experimental variable | Production success condition |
|---|---|---|
| Kitchen S1 | Alternative object types | A ranked observed utensil satisfies semantic and target-specific geometric requirements. |
| Living room L2 | Alternative destination regions | An observed region is semantically appropriate, supports the payload, fits it, and is near the sofa. |
| Workshop W1 | Regions and object types together | A visible tool/fastener pair has the required simple functions and passes hole, reach, and mating checks. |

The functional FM returns 10--15 concrete type priors per requirement. The
grounding experiments may expose a smaller controlled set of observed
instances. The deterministic stage rejects unobserved IDs and checks geometry
in rank order; search stops as soon as a complete valid witness is available.

## Native uv setup

From `V1`:

```bash
uv venv --python 3.11
uv pip install --torch-backend cpu --python .venv/bin/python \
  -r mujoco_scenes/requirements-dev.txt
```

Download the semantic checkpoints once:

```bash
.venv/bin/python -m mujoco_scenes.scripts.prepare_semantic_models
```

Use `MUJOCO_GL=glfw` for an interactive desktop viewer and `MUJOCO_GL=egl`
for headless RGB-D capture.

## Kitchen: object alternatives

The integrated kitchen scene is
`S1_integrated_kitchen_object_function_primary`. It uses the incoming
five-view region-facing RGB-D pipeline, persistent object registry, semantic
role gating, pairwise geometry, and early completion after a valid global
witness is observed.

Run natively:

```bash
./mujoco_scenes/scripts/run_s1_integrated_kitchen_native.sh kitchen_trial_01
```

This search command intentionally runs without a robot. After it reports a
`COMPLETE` witness, send the configured goal and execute the generated plan in
the live Google-robot viewer:

```bash
GOAL='Prepare and serve coffee and soup for three people using the available kitchenware. Stir all three coffees and provide each soup bowl with a suitable utensil.'
MUJOCO_GL=glfw .venv/bin/python -m \
  mujoco_scenes.run_kitchen_goal_execution \
  --phase1-run-dir runs/kitchen_trial_01 \
  --base-url http://127.0.0.1:18080/v1 \
  --goal "$GOAL" \
  --camera free
```

The functional gateway and SSH tunnel must already be running. See
`EXECUTION_AND_TESTING.md` for the complete contract and offline-response
option. The manual interactive Google Robot kitchen also remains available:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot google --viewer
```

## Living room: region alternatives

The controlled L2 movie-night environment now contains three complementary
region ablations. Every policy within an ablation consumes the same saved
five-view RGB-D and semantic evidence:

- Ablation 1 ranks `SOFA_SEAT_PATCH`, `SMALL_SIDE_TABLE`, and `COFFEE_TABLE`.
  Geometry-only accepts the sofa cushion, semantic-only accepts the undersized
  C-table, and joint grounding selects the coffee table.
- Ablation 2 compares always-shared, always-distinct, and function-aware
  allocation. The correct solution uses two distinct personal refreshment
  regions and one shared controls region.
- Ablation 3 compares target-agnostic counting, greedy target assignment, and
  deterministic global one-to-one matching between tables and seats.

The controlled L2 benchmarks use five virtual region-facing cameras. The
separate interactive living room retains the physical Google Robot rig: five
upper cameras for room coverage and two low cameras for the under-sofa task.

Run the controlled native benchmark after preparing semantic models:

```bash
MUJOCO_GL=egl .venv/bin/python -m mujoco_scenes.run_l2_region_ablation \
  --scene L2_living_room_region_ablation1_primary \
  --robot none \
  --run-id living_region_trial_01
```

The additional runners are `mujoco_scenes.run_l2_region_ablation2` and
`mujoco_scenes.run_l2_region_ablation3`.

Run the interactive living room:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.living_room_scene \
  --robot google --viewer
```

## Workshop: joint region and object alternatives

W1 is a single-arm frame-joint repair cell. Two wooden members are already
immobilized in a bench fixture, and a captive guide can retain a staged screw.
The robot never needs to hold the workpiece, screw, and driver simultaneously.
A transparent polycarbonate transport cover sits flush over the guide. It has
a rubber gasket and a yellow pull tab, and must be removed into the adjacent
staging tray before a screw can enter the joint. This adds a visible geometric
access prerequisite without requiring another tool.
An ordinary closed tabletop tool cabinet stores a second tool/fastener option.
It has no lock or key. The adjacent orange tray is reserved for staging the
selected screw.

The left drawer hides a manual Phillips driver and a geometrically inadequate
short screw. The tool cabinet hides a powered Phillips driver and the feasible
long screw. Once both regions have been observed, either driver can form a
complete cross-region system with the long screw. This retains object choice,
region search, and geometric rejection while removing cutting, loose-frame
assembly, vertical mounting, and the unrelated lock-and-key task.

The intended partial order is:

```text
remove_joint_seal -> insert_screw -> drive_screw
observe_long_screw -> insert_screw
```

Seal removal and either storage inspection may occur in any order. Only the
dependent actions above are ordered. Both containers begin closed.

Inspect it with Google Robot:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot google --viewer
```

Run the complete five-view workshop inspection and then open the final scene:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_pointcloud \
  --robot google --segmentation oracle --viewer
```

The runner captures `INITIAL`, `LEFT_DRAWER`, and `TOOL_CABINET` in that order.
It temporarily opens only the region being photographed and closes it again
immediately after capture. Consequently, the interactive viewer starts with
both containers fully closed. Each stage writes the five RGB images, depth
maps, mask overlays, per-view clouds, fused object clouds, and a summary
beneath a timestamped `runs/workshop_pointcloud/` directory.

`oracle` means explicit MuJoCo instance masks for simulator debugging. To use
the same RGB-D reconstruction with image-only masks from the separate SAM 3.1
server, set its endpoint and select the learned backend:

```bash
export SAM3_BASE_URL=http://127.0.0.1:8010
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_pointcloud \
  --robot google --segmentation sam3 --viewer
```

These are the workshop's five calibrated virtual region-facing cameras, not
five cameras physically mounted on the Google Robot. The robot is still loaded
in the scene for visual and future execution integration.

Open both storage regions directly for perception debugging:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot none --open LEFT_DRAWER --open TOOL_CABINET \
  --remove-seal --viewer
```

The point-cloud runner establishes workshop observation and reconstruction,
but not task execution. Updating the functional-requirement, search, and
sequencing pipeline is intentionally deferred. The `--remove-seal` option applies a
labelled ground-truth debug transition; calibrated grasp-and-place execution
must eventually replace it.

## Current boundary

The observation, grounding, ablation, early-termination, and evidence-report
paths are implemented for the existing benchmarks. Kitchen and living-room
robot manipulation remains in their calibrated controllers. Workshop grasping,
screw driving, and task sequencing are not implemented yet. The current W1
boundary is scene construction, region-gated observations, and observable
task-state transitions.
