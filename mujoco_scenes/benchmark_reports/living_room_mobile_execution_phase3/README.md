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
- Placement centres, support boundaries, and non-overlap checks use measured
  Phase-1 point-cloud geometry.
- Navigation uses the existing deterministic RRT* and base actuators.
- Manipulation uses the existing Google-arm IK, segment collision checking,
  bilateral finger contact, inactive-to-active grasp weld, lift/carry,
  release, retreat, and post-place physical verification.
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
- `stance_reachability_audit.json`: current-pose-first candidate tests, IK,
  collision segments, and rejection reasons.
- `refined_mobile_plan.json`: frozen task actions plus conditionally inserted
  MOVE actions.
- `physical_execution.json`: action outcomes and post-place verification.
- `provenance_manifest.json`: hashes of frozen Phase-1/2 inputs.
- `run_summary.json`: concise terminal result.

## Current scope

This phase is deterministic simulation execution only. It does not address
real-robot calibration, perception updates after each physical move, grasp
learning, continuous task replanning, or recovery after a structured failure.
