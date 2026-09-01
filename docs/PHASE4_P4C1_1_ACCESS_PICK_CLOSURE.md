# Phase 4 P4-C1.1 — strict Workshop access and PICK closure

Tested controller commit: `6dec5d60e3a66c2afe04b8c57c76c29603f5156a` (GitHub-resolvable, clean tree). The code was rebased above Phase-3 P3-G commit `579d2e882f09e3b16c3c6eea302b84d868afc01c`. No Phase-3 file or `ground_graph()` changed.

## Current GT status

- W1 GT: `INCOMPLETE`; inspected LEFT_DRAWER, RIGHT_DRAWER, TOOL_CABINET; `CURRENT_UPSTREAM_PHASE3_BLOCKED`.
- W6 GT: `INCOMPLETE`; inspected TOOL_CABINET, LEFT_DRAWER, RIGHT_DRAWER; `CURRENT_UPSTREAM_PHASE3_BLOCKED`.
- L1 GT: `ACTION_SEQUENCE_READY`; exact-code strict execution passed 10/10 with no strict violation, assisted fixture, state write, payload write, or post-release dynamics modification.

## Physical controller evidence

All Workshop runs are `CONTROLLER_DEVELOPMENT_ONLY` and excluded from TAMP metrics.

- LEFT_DRAWER OPEN: prior -3 mm shell-top collision eliminated by a front-facing handle frame and a mirrored contact-gated mobile-base pull. Final slide 0.249186 m; exact handle; 25 bilateral frames; attachment snap 0.000002497 m / 0 rad; no shell contact observed; no direct articulation write.
- RIGHT_DRAWER OPEN: final slide 0.249139 m; exact handle; 25 bilateral frames; attachment snap 0.000002468 m / 0 rad; no shell contact observed; no direct articulation write.
- TOOL_CABINET OPEN: 1.214745 rad; exact handle; 25 bilateral frames; attachment snap 0 m / 0.000001190 rad; no direct articulation write.
- LEFT drawer long-driver PICK: preclose 0.006752 m; 25 bilateral `*_col_handle` frames; snap 0.000004116 m / 0.000046202 rad; object displacement about 0.159 m; held identity preserved.
- RIGHT drawer power-driver PICK: preclose 0.009545 m; 25 bilateral `*_col_handle` frames; snap 0.000006627 m / 0.000015742 rad; object displacement about 0.175 m; held identity preserved.
- Cabinet screw PICK: old 0.2219 m miss was a frame/door-aperture problem. Full contact-driven door motion, the saved collision-free handle IK seed, world-coordinate base centering, and physically narrowed free-space jaw preshape reduce preclose error to 0.006870 m. It records 25 bilateral shaft contacts, 0.000002462 m / 0.000005026 rad snap, and about 0.210 m retrieval/lift displacement. This is a real fastener acquisition.
- Cabinet long-driver PICK remains `PREGRASP_POSITION_ERROR`: 0.1998 m. The driver lies near the cabinet right wall; the front-facing gripper cannot enter to its handle without pad/wall interference. No tolerance increase, attachment, substitution, or state repair was accepted.

The earlier hammer actuator failure is now precisely attributed by actuator telemetry rather than hidden by a global settle relaxation. Contact-driven cabinet base following uses a local contact-stall mode, but success still depends on measured articulation and the frozen exact-handle gate.

Strict telemetry aggregation now treats `target_alignment_constraint_used`, `installed_fastener_constraint_used`, and `staging_constraint_used` as assisted task fixtures. A nested regression proves both aggregate and strict-violation flags become true.

PLACE and insertion mechanics were not tuned. `_destination_position("MAIN_WORKBENCH_ZONE")` remains `P4-C1.2 REQUIRED CLEANUP`; strict insertion remains physically unvalidated and its broad `frame_*`/`fixture_*` contact policy is not expanded. SCREW remains blocked.

## Commands and artifacts

Commands used the prefixes `MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie`, `MUJOCO_GL=egl`, and `/home/naren/miniconda3/bin/python`:

```text
-m mujoco_scenes.functional_tamp_pipeline.run --domain workshop --variant W1 --mode gt --search-order auto --dry-run --output-root /tmp/p4-c11-final-phase3
-m mujoco_scenes.functional_tamp_pipeline.run --domain workshop --variant W6 --mode gt --search-order auto --dry-run --output-root /tmp/p4-c11-final-phase3
-m mujoco_scenes.functional_tamp_pipeline.run --domain living_room --variant L1 --mode gt --search-order auto --dry-run --output-root /tmp/p4-c11-final-phase3
-m mujoco_scenes.run_workshop_phase4_controller_development --variant F0_MANUAL_FIRST_ONE_REGION --case left_drawer_driver_pick --output /tmp/p4-c11-exact/left_driver.json
-m mujoco_scenes.run_workshop_phase4_controller_development --variant F3_POWER_FIRST_TWO_REGIONS --case right_drawer_power_pick --output /tmp/p4-c11-exact/right_power.json
-m mujoco_scenes.run_workshop_phase4_controller_development --variant F5_POWER_FIRST_THREE_REGIONS --case cabinet_screw_pick --output /tmp/p4-c11-exact/cabinet_screw.json
-m mujoco_scenes.run_workshop_phase4_controller_development --variant F5_POWER_FIRST_THREE_REGIONS --case cabinet_driver_pick --output /tmp/p4-c11-exact/cabinet_driver.json
-m mujoco_scenes.run_phase4_execution --domain living_room --variant L1 --mode gt --phase3-run /tmp/p4-c11-final-phase3/living_room/L1/gt --output-root /tmp/p4-c11-exact-execution
-m pytest -q mujoco_scenes/tests/test_workshop_ground_truth_execution.py mujoco_scenes/tests/test_phase4_execution.py
```

Tests: 45 passed. Large artifacts/videos are not committed.

## Status

`P4-C1.1 = BLOCKED_AT_CABINET_DRIVER_PICK`.

Next Phase-4 work must remain P4-C1.1 and solve the cabinet-side driver aperture/approach geometry. Do not start P4-C1.2 until that PICK family is physically closed.
