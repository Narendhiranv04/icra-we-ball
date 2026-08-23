# Kitchen and Living-Room GT Pipeline Readiness

Status: **Kitchen GT execution closed; Living-Room grounding/planning
revalidated for the fixed-table variant family; final physical videos pending
regeneration; VLM planner integration pending**.

This statement applies to the ground-truth demonstration pipeline. It does not
claim that every Kitchen manipulation is an unassisted contact-rich motion or
that liquid dynamics are simulated. The distinction is intentional and is
recorded in every Kitchen execution artifact.

## Final validation matrix

| Environment | Variants | Grounding / role assignment | Symbolic planning | GT execution | Recorded evidence |
| --- | ---: | --- | --- | --- | --- |
| Kitchen | 16 (10 feasible, 6 infeasible) | 16/16 YOLO-World-L task classification; privileged GT binding for execution | 16/16 GT preflight | **16/16 passed** | Five-camera MP4 for every variant |
| Living Room | 10 (6 feasible, 4 infeasible) | Production generic object/region bindings | **10/10 revalidated** | Executor/tests ready; new videos not yet regenerated | Pending for six feasible variants |

For feasible Kitchen variants, all 410 planned actions completed. Each one has
six POUR and three STIR actions, giving 60 POUR and 30 STIR proxy executions
across the suite. The six infeasible variants each executed their six-action
storage-search plan and terminated as `INFEASIBLE_CONFIRMED`. The complete
Kitchen suite therefore contains 446 successful action records.

For Living Room, all six feasible variants compile deterministically from the
new Phase-1 witness. F0, F2, F3, and F5 contain five PICK/PLACE pairs; F1 and
F4 skip the already-correct saucer and contain four pairs. All five goals pass
independent replay. The four missing-table variants are rejected before the
planner. The physical executor and placement allocator consume these new
artifacts, but any recordings from the retired 13-variant family are not
evidence for this replacement family and must be regenerated.

## Kitchen Phase C and assisted execution

Kitchen Phase C was already complete for its frozen POUR/STIR operator scope:

- POUR pair coverage: 4/4.
- Kettle repeatability: 3/3.
- Jar repeatability: 3/3.
- Sequential source tests: 2/2.
- STIR pair coverage: 2/2.
- STIR repeatability: 3/3.
- Integrated Phase-C ledger: 6/6.

The old `complete_plan_execution.json` failure occurred later in the Phase-B
serving tail, after all Phase-C events. It was not a Phase-C POUR/STIR failure.

The all-variant closure uses execution profile
`ASSISTED_DETERMINISTIC_DEMONSTRATION`. It has these explicit semantics:

- Container search is physically reflected in MuJoCo through deterministic
  articulation actuation.
- A missed PICK is recovered by presenting the selected object at a bounded
  gripper-frame carry pose and enabling its exact object-specific weld. Success
  is accepted only when the live held-state check reports one exclusive active
  payload weld.
- PLACE uses the assigned target pose. Conservative serving-footprint failure
  may use the frozen semantic serving row. This is a direct payload pose write.
- POUR and STIR use visible bounded wrist gestures while the selected payload
  remains attached. They are kinematic action proxies; no fluid transfer or
  contact-based mixing is claimed.
- The symbolic effect is applied only after the proxy and live held-state check
  succeed.

This profile is appropriate for GT sequence demonstration and VLM-planner
evaluation. Strict unassisted manipulation remains a separate robotics metric
and must not be inferred from the assisted-suite score.

## Evidence

Kitchen suite summary:
[`runs/kitchen_ground_truth_execution_assisted_suite/suite_summary.json`](../runs/kitchen_ground_truth_execution_assisted_suite/suite_summary.json)

Kitchen videos and per-variant assignments, plans, traces, final states, and
camera manifests:
[`runs/kitchen_ground_truth_execution_assisted_suite/`](../runs/kitchen_ground_truth_execution_assisted_suite/)

Living-Room Phase-1 production report:
[`mujoco_scenes/benchmark_reports/living_room_region_feasibility_phase1/`](../mujoco_scenes/benchmark_reports/living_room_region_feasibility_phase1/)

Living-Room Phase-2 symbolic report:
[`mujoco_scenes/benchmark_reports/living_room_symbolic_phase2/`](../mujoco_scenes/benchmark_reports/living_room_symbolic_phase2/)

Focused Living-Room grounding, planning, resolver, and execution-adapter tests:
**114/114 passed**.

## Reproduction

Kitchen full assisted GT suite with five-camera recordings:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python \
  -m mujoco_scenes.run_kitchen_ground_truth_execution \
  --variant all --assisted-suite --record \
  --camera-resolution 426x240 \
  --output-root runs/kitchen_ground_truth_execution_assisted_suite
```

Kitchen strict physical-primitives path with assisted recovery retained:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python \
  -m mujoco_scenes.run_kitchen_ground_truth_execution \
  --variant F1_INITIAL_COMPLETE --record
```

Living-Room full execution suite:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python \
  -m mujoco_scenes.run_living_room_execution --variant all --record
```

Robust GT-demonstration profile for Living Room:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python \
  -m mujoco_scenes.run_living_room_execution \
  --variant all --assisted-suite --record
```

## Remaining VLM integration boundary

The remaining integration replaces the privileged/deterministic plan producer,
not the action executors:

1. The VLM consumes the observation registry, region state, functional witness,
   and task instruction.
2. It returns the existing typed action schema and selected generic bindings.
3. Existing schema/precondition validation rejects unknown objects, regions,
   operators, invalid ordering, incomplete assignments, and infeasible tasks.
4. The validated plan is dispatched to the already closed scene-specific
   executor.

Kitchen's accepted execution operators are `OPEN`, `CLOSE`, `PICK`, `PLACE`,
`POUR`, `STIR`, `SERVE_COFFEE`, `SERVE_SOUP`, and
`PLACE_SERVING_UTENSIL`. Living Room uses generic `PICK` and `PLACE`, with MOVE
inserted by execution refinement. The VLM must not output simulator body names;
those remain confined to the execution resolver.

Thus the GT task definitions, functional-role satisfaction, action sequences,
preconditions/effects, motion dispatch, failure handling, recording, and suite
validation are ready. The pending work is wiring and evaluating the VLM as the
plan/binding producer at this frozen boundary.
