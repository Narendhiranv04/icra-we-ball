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

Each ranker may return at most three alternatives. Ranking belongs to the
foundation model; the deterministic stage only rejects unobserved IDs and
checks geometry in rank order. Planning stops as soon as a valid witness is
available.

## Native uv setup

From `V1`:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python \
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

This benchmark intentionally runs without a robot because it measures
grounding and search, not manipulation execution. The interactive Google
Robot kitchen remains available separately:

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

W1 is deliberately compact: one workbench, one frame joint, a visible
hammer/nail near-miss, and two drawers. The left drawer contains a screwdriver
and a screw that is too short. The right drawer contains the compatible
driver/screw pair. This produces both region search and object-system choice
without making the room large.

Inspect it with Google Robot:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot google --viewer
```

Open drawers directly for perception debugging:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot none --open LEFT_DRAWER --open RIGHT_DRAWER --viewer
```

The FM-facing task contract is
`mujoco_scenes/configs/workshop_joint_alternatives.yaml`. The deterministic
validator is `mujoco_scenes.workshop_alternatives.evaluate_ranked_alternatives`.
It accepts only observed IDs and simple functions (`can_hammer`, `can_screw`,
`can_fasten`); it does not use efficiency scores or hidden scene contents.

## Current boundary

The observation, grounding, ablation, early-termination, and evidence-report
paths are implemented. Kitchen and living-room robot manipulation remains in
their existing calibrated controllers. Workshop arm grasps and physical
fastening are not calibrated yet; W1 currently ends at a verified tool-system
witness. That boundary keeps grounding results separate from future execution
failures.
