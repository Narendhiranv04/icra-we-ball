# Workshop fixed-pair pipeline readiness

## Current scope

The Workshop benchmark has been restructured around one task: find one Phillips screw and the first compatible driver encountered, then install that screw in the fixed main-workbench repair hole.

The only movable scene objects are one manual Phillips screwdriver, one power driver, one compatible Phillips screw, and one wooden hammer distractor. Variants change only object position or presence. Furniture, storage geometry, the workbench target, lighting, and camera definitions do not vary. Parts trays, hardware bins, alternate shelves, bolts, pliers, wrenches, and alternate screw types are not active benchmark elements.

## Variant and feasibility contract

There are 10 variants: eight feasible and two deliberately infeasible. The fixed inspection order is `LEFT_DRAWER`, `RIGHT_DRAWER`, then `TOOL_CABINET`. Search stops when both required roles have been observed.

| Validated gate | Result |
|---|---:|
| Complete variant suite | 10/10 |
| Feasible physical executions | 8/8 |
| Infeasible inspect-and-terminate executions | 2/2 |
| Typed actions completed | 290/290 |
| Grounding-to-execution contracts | 10/10 exact |

| Role | Accepted objects | Required checks |
|---|---|---|
| `CAN_DRIVE_SCREW` | Manual Phillips screwdriver or power driver | Tool reaches target and its tip profile matches the screw recess. |
| `CAN_FASTEN` | Phillips screw | Diameter/length fit the fixed repair hole and the recess matches the driver. |

`MAIN_WORKBENCH_ZONE` is a fixed insertion target, not a predicted alternative role. The hammer is part of detector vocabulary only as a negative distractor. A variant is infeasible only after exhaustive inspection establishes `NO_COMPATIBLE_DRIVER` or `NO_COMPATIBLE_SCREW`.

The authoritative matrix and five-camera images are in [WORKSHOP_VARIANT_VISUAL_CATALOGUE.md](WORKSHOP_VARIANT_VISUAL_CATALOGUE.md).

## GT action implementation

Each feasible GT plan:

1. Opens and inspects each required storage region in order.
2. Picks the selected driver using a vertical top-down drawer approach or a
   horizontal cabinet-front approach.
3. Stages the driver on the workbench and fixes it at its established contact pose.
4. Closes its source region.
5. Retrieves the screw, aligns it tip-down/head-up, and inserts it vertically.
6. Closes the screw source region.
7. Retrieves and engages the staged driver.
8. Drives the screw gradually, returns the driver safely, and verifies the terminal state.

The manual branch visibly ratchets the complete tool while coupling rotation to screw advance. The power branch keeps the casing and wrist still while the screw rotates and advances, representing the internal powered spindle. Objects are carried by explicit grasp constraints rather than animated independently.

Storage mechanisms use front-facing handle approaches. Drawer opening/closing
follows the horizontal slide axis; cabinet opening/closing follows the hinge
arc. Objects in drawers are approached from overhead, grasped with the gripper
vertical, and lifted vertically clear of the drawer. Objects in the cabinet are
approached with the gripper horizontal, grasped inside the cabinet, and first
withdrawn straight through the opening before any lift. Motion uses
collision-audited base corridors, Cartesian approach/retreat waypoints, bounded
joint rates at half the prior Workshop speed, and settled holds.

A payload constraint may activate only after 25 consecutive simulation steps
with contacts on both finger sides. Activation is rejected unless bilateral
contact still exists and the required attachment translation is at most 4 mm.
The validated executions therefore do not use proximity-triggered or
pose-snapping pickup. The physical trace records the finger/object contact
pairs, contact duration, attachment snap, Cartesian waypoints, and cabinet
withdrawal error for every pick.

## Validation gates

Every run emits an assignment, typed plan, symbolic preflight, compiled-scene audit, physical trace, terminal report, and summary. Feasible success requires the expected inspection prefix, all opened storage closed, the intended first-observed driver, empty gripper, vertical screw orientation, measured installed depth, active installed constraint, and the correct driving mode.

Infeasible runs must inspect and close all three regions, execute no insertion, and return the exact missing-role reason.

The execution profile is `PHYSICS_ASSISTED_GT_EXECUTION`. It demonstrates coherent robot-carried GT motion and measured terminal physics; it does not claim learned grasping, force/torque control, or autonomous motion planning.

## Grounding and foundation-model boundary

`ManualWorkshopFMContract` currently supplies the two role descriptions and four-label detector vocabulary once at episode initialization. Inspection, persistent tracking, compatibility checks, first-observed selection, normalized handoff, planning, execution, and validation are independent of that provider.

The GT decision-to-execution schema is 10/10 exact, but this is not a frozen
production-grounding result. The old 14-variant YOLO decisions are intentionally
not reused because their role schema and object universe are invalid for this
redesign. Before claiming that only VLM integration remains, run and freeze the
redesigned 10-variant YOLO-World-L five-view grounding suite, then validate its
generic track-to-execution entity handoff. After that, the remaining integration
step is replacing the manual requirement provider with a live VLM/FM response.

## Reproduction

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.run_workshop_ground_truth_execution \
  --variant all --output-root runs/workshop_fixed_pair_gt_execution

.venv/bin/python -m mujoco_scenes.render_workshop_variant_catalogue
```

Add `--record --resolution 640x360 --fps 20` to the execution command to create synchronized five-view videos.

The redesigned production-grounding command is:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /home/naren/miniconda3/bin/python -m \
  mujoco_scenes.run_workshop_phase1 \
  --variant all --robot none \
  --config mujoco_scenes/configs/workshop_phase1_yoloworld_l_five_view_close.yaml \
  --output outputs/workshop_fixed_pair_yoloworld_l_five_view
```

## Interactive Actions panel

The Workshop scene now has its own companion Actions panel. It exposes physical
move, storage open/inspect/close, object pick, workbench placement, screw
insertion, manual/power driving, repair verification, reset, and step/run GT
controls. Buttons are enabled from the same symbolic preconditions used by the
validated executor. During interactive exploration, either present compatible
driver may be selected; this is labelled as an interactive override and does
not modify the variant's frozen GT first-observed assignment.

Use the Conda Python on this workstation because it includes Tk support:

```bash
/home/naren/miniconda3/bin/python -m mujoco_scenes.workshop_scene \
  --robot google --variant F0_MANUAL_FIRST_ONE_REGION --viewer
```

Add `--no-actions-panel` for the original passive MuJoCo viewer.
