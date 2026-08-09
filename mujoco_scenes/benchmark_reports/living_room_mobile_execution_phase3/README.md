# Living-Room Mobile Execution Phase 3A

This milestone refines the frozen Phase-2 `PICK`/`PLACE` plan for
`L2_integrated_living_room_region_function_F0_BASE` into physical Google Robot
execution. The task planner's order and allocation are immutable. The refiner
tests the current base first with the execution IK and collision checker, then
generates deterministic target-facing ring stances and inserts `MOVE` only
when the current pose is infeasible.

## Boundaries

- Inputs are the frozen Phase-1 observed payload/region registries and the
  frozen Phase-2 plan.
- Generic IDs are resolved to simulator bodies only inside the simulation
  adapter, by semantic consistency plus nearest observed centroid.
- Placement position and yaw consume the exact selected Phase-1
  `FITS_SET_ON.fit_evidence.selected_packing`: arrangement, 0/90-degree
  orientation, measured oriented footprints, edge clearance, and 2.5 cm
  payload clearance. Both pre- and post-execution checks use oriented
  rectangles rather than enclosing circles.
- Navigation uses the existing deterministic RRT* and base actuators.
- Manipulation uses the existing Google-arm IK, segment collision checking,
  bilateral finger contact, inactive-to-active grasp weld, lift/carry,
  release, retreat, and strong post-place physical verification. A resumed
  PLACE controller is admitted only after simulator equality state proves the
  intended weld is active, exclusive, attached to the intended body, close to
  the gripper, off the floor, and accompanied by a physically closed gripper.
- Held state is revalidated before and after every carried MOVE.
- Final `ON(object, region)` requires the intended support contact, no floor
  contact, the actual oriented footprint inside the observed Phase-1 support,
  rectangular pair clearance, no invalid furniture penetration, correct
  supported height and orientation, and settled linear and angular velocity.
- After the complete sequence, all six frozen Phase-2 goals are independently
  recomputed from the final MuJoCo state.
- The runner does not reorder task actions, reallocate witness bindings,
  invoke an FM, or use PDDLStream/TAMP.

The invisible grasp sites and initially inactive welds are execution-only
annotations. They are absent from the no-robot perception model and therefore
cannot affect frozen Phase-1 RGB-D evidence.

## Run

```bash
export MUJOCO_GL=egl
PYTHON=/home/naren/miniconda3/bin/python
PHASE1=runs/living_room_region_phase1/living_room_region_phase1_final_closure_v3_20260809/F0_BASE
PHASE2=mujoco_scenes/benchmark_reports/living_room_symbolic_phase2/variants/F0_BASE

$PYTHON -m mujoco_scenes.run_living_room_mobile_execution \
  --phase1-dir "$PHASE1" \
  --phase2-dir "$PHASE2" \
  --output-dir runs/living_room_mobile_execution_phase3/f0_plan_only

$PYTHON -m mujoco_scenes.run_living_room_mobile_execution \
  --phase1-dir "$PHASE1" \
  --phase2-dir "$PHASE2" \
  --output-dir runs/living_room_mobile_execution_phase3/f0_full_execution \
  --execute
```

Use `--start-task-action N --max-task-actions 1` for isolated category trials.
Plan-only mode may propagate the selected base pose directly in its private
planning state. `--execute` never teleports the base or payload during normal
actions; it uses MuJoCo actuators and physics.

## Artifacts

- `execution_entity_resolution.json`: generic-ID/backend resolution boundary.
- `dynamic_placement_targets.json`: measured, bounded, non-overlapping targets.
- `phase1_selected_packing_realization.json`: direct Phase-1 packing-to-pose
  provenance, including world XYZ/yaw, corners and predicted margins.
- `stance_reachability_audit.json`: current-pose-first candidate tests, IK,
  collision segments, and rejection reasons.
- `refined_mobile_plan.json`: frozen task actions plus conditionally inserted
  MOVE actions.
- `held_object_validation.json`: physical grasp evidence before/after carry.
- `physical_goal_validation.json`: PLACE-time strong `ON` checks.
- `final_physical_goal_validation.json`: independent simultaneous final goals.
- `physical_metrics.json`: contact, boundary, overlap, retention and stability
  rates and margins.
- `scientific_guard_report.json`: artifact-derived closure guards.
- `physical_execution.json`: action outcomes and post-place verification.
- `provenance_manifest.json`: hashes of frozen Phase-1/2 inputs.
- `run_summary.json`: concise terminal result.

## Current scope

Authoritative closure validation achieved 5/5 repeated first-object trials,
6/6 fresh-reset individual payload executions, and 3/3 complete F0 trials.
The final authoritative run retained all six objects, inserted 12 conditional
MOVEs without changing the frozen task order, and physically revalidated all
six `ON` goals simultaneously. See `validation_summary.json` for numeric
metrics and `authoritative/execution_timeline.{gif,mp4}` for the visual audit.

This phase is deterministic simulation execution only. It does not address
real-robot calibration, perception updates after each physical move, grasp
learning, continuous task replanning, or recovery after a structured failure.
It is not a general TAMP system.
