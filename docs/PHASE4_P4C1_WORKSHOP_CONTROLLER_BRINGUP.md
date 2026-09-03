# Phase 4 P4-C1 — Workshop strict controller bring-up

Source branch state was `e4d9a17a431003e258e3c9659c289386116c5b7e`, clean. This pass changed only Phase-4 execution/controller files.

## Current GT handoffs

All were regenerated in GT mode with `--search-order auto --dry-run` under `/tmp/p4-c1-phase3`.

- W1: `INCOMPLETE`; inspected `LEFT_DRAWER`, `RIGHT_DRAWER`, `TOOL_CABINET`. Current upstream Phase 3 blocked; no end-to-end claim.
- W6: `INCOMPLETE`; inspected `TOOL_CABINET`, `LEFT_DRAWER`, `RIGHT_DRAWER`. Current upstream Phase 3 blocked; no end-to-end claim.
- L1: `ACTION_SEQUENCE_READY`; strict execution passed 10/10. `strict_execution=true`, no strict violation, no assisted fixture, no direct-state fallback.

## Workshop controller evidence

Runs are explicitly `CONTROLLER_DEVELOPMENT_ONLY` and do not contribute to TAMP metrics.

- TOOL_CABINET OPEN passed using `tool_cabinet_door_handle_col`: 25 bilateral contact steps, contact-gated zero-snap handle grasp, 0.0 m translation snap, 0.00000119023 rad angle snap, measured hinge 1.038878484 rad, and no direct container actuator.
- LEFT_DRAWER OPEN stopped as `COLLISION_BLOCKED`: right fingertip versus `left_drawer_shell_top_col`, -0.003 m signed distance. The collision was not allowed or repaired.
- Cabinet wooden-hammer PICK improved the historical gross 0.4292 m calibration issue by re-centering the base from the live resolved grasp geometry, but stopped at an actuator settle residual of 0.0445 rad; no bilateral object contact or grasp attachment was claimed.
- Cabinet medium-screw PICK stopped as `PREGRASP_POSITION_ERROR`, measured 0.2219 m versus a 0.040 m gate. No tolerance widening or attachment occurred.
- Drawer PICK, workbench PLACE, and insertion could not be physically reached because the prerequisite strict OPEN/PICK failed. Consequently there are no honest release/settling or insertion measurements to report.

The strict surface path now skips the legacy staging weld, releases only the live contact-gated grasp, settles under free-body physics, and requires workbench contact, bounded position, inactive grasp, no held object, and velocity limits. The strict insertion path preserves the live grasp transform, moves the robot through hover/entry/push waypoints, releases, and measures tip lateral error (<=3 mm), axis error (<=0.05 rad), depth (8–18 mm), target contact, held state, and settling velocity. It does not invoke alignment/installed welds or payload/task-state writes. These paths are implemented and unit-gated but are not claimed physically validated in this pass.

Strict SCREW remains blocked as `STRICT_PHYSICAL_SCREW_UNAVAILABLE`.

## Commands

```text
MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie MUJOCO_GL=egl /home/naren/miniconda3/bin/python -m mujoco_scenes.functional_tamp_pipeline.run --domain workshop --variant W1 --mode gt --search-order auto --dry-run --output-root /tmp/p4-c1-phase3
MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie MUJOCO_GL=egl /home/naren/miniconda3/bin/python -m mujoco_scenes.functional_tamp_pipeline.run --domain workshop --variant W6 --mode gt --search-order auto --dry-run --output-root /tmp/p4-c1-phase3
MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie MUJOCO_GL=egl /home/naren/miniconda3/bin/python -m mujoco_scenes.functional_tamp_pipeline.run --domain living_room --variant L1 --mode gt --search-order auto --dry-run --output-root /tmp/p4-c1-phase3
MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie MUJOCO_GL=egl /home/naren/miniconda3/bin/python -m mujoco_scenes.run_phase4_execution --domain living_room --variant L1 --mode gt --phase3-run /tmp/p4-c1-phase3/living_room/L1/gt --output-root /tmp/p4-c1-execution
MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie MUJOCO_GL=egl /home/naren/miniconda3/bin/python -m mujoco_scenes.run_workshop_phase4_controller_development --variant <scene> --actions-json <actions> --output /tmp/p4-c1-controller/<case>.json
MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie /home/naren/miniconda3/bin/python -m pytest -q mujoco_scenes/tests/test_workshop_ground_truth_execution.py mujoco_scenes/tests/test_phase4_execution.py
```

Test result: 44 passed. Artifacts are in `/tmp/p4-c1-phase3`, `/tmp/p4-c1-execution`, and `/tmp/p4-c1-controller` and are intentionally not committed.

## Status

`P4-C1 BLOCKED_AT_PHYSICAL_INSERTION`: the fixture-free insertion controller is implemented but cannot be exercised until strict storage access and PICK succeed. Next pass: **P4-C1.1 — insertion mechanics only**; it must first use a physically acquired fastener and must not weaken the current gates.
