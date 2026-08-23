# Workshop fixed-pair environment

The Workshop environment is a fixed-furniture inspection-and-fastening benchmark. Its canonical instruction is:

> Find the compatible screw and the first compatible driver encountered while inspecting the storage regions. Insert the screw tip-down into the fixed workbench repair hole and drive it fully.

## Scene contents

Inspectable articulated storage is fixed across every variant: `LEFT_DRAWER`, `RIGHT_DRAWER`, and `TOOL_CABINET`. The search order is left drawer, right drawer, then tool cabinet. The workbench and repair hole are fixed. The tool cart is static background furniture.

There are exactly four movable benchmark objects: `workshop_long_phillips_driver`, `workshop_power_driver`, `workshop_medium_phillips_screw`, and `workshop_wooden_hammer`. Both drivers match the screw's Phillips recess. The screw fits the target hole. The hammer is a distractor. No active bolts, pliers, wrenches, alternate screws, trays, bins, shelf alternatives, or surface-obstruction variants remain.

## Variants

`configs/workshop_variants.yaml` defines eight feasible position/presence permutations and two infeasible missing-role cases. No variant changes region geometry or furniture layout. See `docs/WORKSHOP_VARIANT_VISUAL_CATALOGUE.md` for the matrix and five-view snapshots.

## Observation boundary

Opening storage actuates its physical joint but does not return inventory. Production-facing code observes synchronized RGB-D views, instance proposals, persistent tracks, and measured geometry. Variant IDs, backend names, declared storage contents, semantic functions, and exact dimensions remain privileged construction/evaluation data.

## Grounding and execution

Grounding searches compatible `(driver, screw)` pairs. `MAIN_WORKBENCH_ZONE` is the fixed target, not a grounded alternative. Search may stop after one, two, or three regions. If both compatible drivers have been observed, the first encountered in inspection order is selected.

The GT executor uses collision-audited motion, actuated storage joints, constrained robot-carried payloads, fixed-pose staging after support contact, vertical screw alignment, and measured terminal validation. Manual driving visibly ratchets the driver. Power driving keeps the casing stationary while the screw rotates and advances.

## Commands

```bash
.venv/bin/python -m mujoco_scenes.workshop_scene --list-variants
.venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot google --variant F0_MANUAL_FIRST_ONE_REGION --viewer
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.run_workshop_ground_truth_execution --variant all \
  --output-root runs/workshop_fixed_pair_gt_execution
.venv/bin/python -m mujoco_scenes.render_workshop_variant_catalogue
```
