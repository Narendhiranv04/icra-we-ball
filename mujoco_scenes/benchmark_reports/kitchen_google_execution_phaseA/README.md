# Kitchen Google Robot Execution — Phase A

This report freezes the first physical kitchen-execution boundary for the
authoritative `S1_integrated_kitchen_object_function_primary` scene.  It does
not claim complete kitchen-task execution.  Phase A covers collision-checked
mobile workspace selection and physical Google Robot `OPEN`/`CLOSE` of D1,
D2, C1, C2 and B1.

## Result

The authoritative combined run completed all ten articulation requests:

| Container | Mechanism | Required workspace | OPEN | CLOSE |
|---|---|---|---|---|
| D1 | prismatic drawer | HOME | PASS | PASS |
| D2 | prismatic drawer | HOME | PASS | PASS |
| C1 | hinged cupboard | LEFT_SIDE | PASS | PASS |
| C2 | hinged cupboard | RIGHT_SIDE | PASS | PASS |
| B1 | hinged box lid | RIGHT_SIDE | PASS | PASS |

The dispatcher inserted two necessary mobile moves (HOME to LEFT_SIDE and
LEFT_SIDE to RIGHT_SIDE) and omitted moves when the physical workspace was
already correct. C2 and B1 intentionally share RIGHT_SIDE.

Every successful articulation record reports bilateral handle contact, a
temporary relative weld, collision-checked IK, an active live collision guard,
the expected final joint postcondition, no direct container actuator use, no
live articulated-qpos write, and no unexpected motion of another mechanism.

## Architecture

`kitchen_execution_policy.py` maps each physical request to a required
workspace. `KitchenGoogleExecutionDispatcher` refines a request with an
actuator-driven mobile `MOVE` only when needed.  It then calls the generic
`GoogleKitchenArticulationExecutor`.

The articulation executor samples the target articulated joint, calculates
the corresponding world-frame handle trajectory on a planning copy, solves a
continuous Google-arm IK path, and collision-checks the path.  At runtime it
approaches the real handle, requires contact from both gripper sides, activates
an initially inactive relative weld, makes the container actuator passive,
and drives the articulated joint through arm motion.  It then releases the
handle, retreats, and verifies the physical postcondition before scene-state
bookkeeping is updated.

`KitchenScene.open_container()` and `close_container()` remain unchanged as
direct deterministic adapters for perception and inspection. They are not
called by the physical execution dispatcher.

## Workspace poses

| Workspace | x (m) | y (m) | yaw (rad) |
|---|---:|---:|---:|
| HOME | 0.000 | -1.100 | 0.000 |
| LEFT_SIDE | -1.025 | -0.100 | -1.570796 |
| RIGHT_SIDE | 1.025 | -0.100 | 1.570796 |

The cupboard actions use bounded local manipulation offsets after reaching
their workspace: C1 `(0, -0.15, 0)` and C2 `(0, +0.15, 0)`. The base retracts
after the primitive.

## Evidence map

- `validation_summary.json`: authoritative outcome, moves and diagnostic run.
- `workspace_policy.json`: exact poses, aliases and container requirements.
- `articulation_specs.json`: joints, handles, targets, tolerances and sampling.
- `scientific_guard_report.json`: artifact-derived execution guards.
- `authoritative/combined_workspace_sequence.json`: complete raw run record.
- `authoritative/<container>/{open,close,cycle_summary}.json`: compact evidence.
- `authoritative/direct_actuation_guard.json`: physical-motion source audit.
- `environment.json`: execution environment.
- `test_summary.txt`: validation commands and results.
- `reproduction_commands.sh`: reproducible headless and viewer commands.

## Reliability note

The authoritative combined run is a complete 1/1 pass. The retained file
`runs/kitchen_phaseA_reliability_5cycles/execution_results.json` is an earlier
non-authoritative attempt that stopped during cycle 3; it is intentionally not
presented as a five-cycle pass. The later focused and full isolated test runs
were reported passing. This distinction prevents stale diagnostic evidence
from being mistaken for authoritative execution evidence.

## Scope boundary and next phase

Phase A does not implement integrated-object PICK/PLACE calibration, POUR,
STIR, motion-level execution of the full symbolic plan, or replanning. The
recommended Phase B starting point is to retain this dispatcher and workspace
policy, then calibrate generic Google Robot PICK/CARRY/PLACE for the observed
integrated-kitchen objects before adding liquid-transfer or stirring actions.
