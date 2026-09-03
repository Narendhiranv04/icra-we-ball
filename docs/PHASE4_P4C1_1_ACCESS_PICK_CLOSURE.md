# Phase 4 P4-C1.1 — strict Workshop access and PICK closure

Tested code commit: `34b76b46c0de87bc143711fa01b121aad9b4c7e9` (clean, pushed, and contained by `origin/naren/pipeline_check`). It is based on current integrated Phase-3 head `56a448571dfbfbd0c38dd2044c05718433d84025`. No Phase-3 source or `ground_graph()` changed.

## Changes frozen in the tested code

- Strict Workshop reaches use exact task-contact geoms; furniture bodies are no longer broad collision exemptions.
- Object attachment requires live bilateral contact on reviewed grasp geoms and both snap gates: translation <= 4 mm and angle <= 0.02 rad.
- PICK records live pre/post AABBs and requires displacement plus geometric source clearance.
- OPEN changes observation state only after verified physical articulation; failed OPEN stays closed and does not reveal hidden contents.
- Storage collision audit covers drawer/cabinet envelopes and records the minimum signed contact pair.
- The controller-development harness now applies the common nested strict telemetry audit to successes and failures.

## Exact-code physical evidence

All Workshop artifacts are `CONTROLLER_DEVELOPMENT_ONLY` and excluded from TAMP metrics. Each case was run twice from a fresh F0/F3/F5 simulator. Both repetitions were byte-identical and failed closed with no strict-fixture violation:

- LEFT_DRAWER OPEN: exact handle bilateral contact reached 25 steps and articulation reached 0.249216 m, but the left fingertip penetrated `left_drawer_col_front` by 8.244 mm. Strict result: failure.
- RIGHT_DRAWER OPEN: exact handle bilateral contact reached 25 steps and articulation reached 0.249168 m, but the left fingertip penetrated `right_drawer_col_front` by 6.055 mm. Strict result: failure.
- TOOL_CABINET OPEN: collision checking stopped articulation follow at `google:link_finger_tip_right` versus `tool_cabinet_door_col`, signed distance -7 mm. Strict result: `COLLISION_BLOCKED`.
- Cabinet long-driver PICK: cannot begin because its required physical cabinet OPEN fails at the same door-panel collision. No object attachment or fallback occurred.

The exact geom audit therefore invalidates earlier apparent OPEN/PICK successes that depended on whole-furniture body exemptions. P4-C1.1 cannot freeze. The remaining issue is physical handle/panel/gripper compatibility and a collision-free handle-follow controller; changing collision geometry, tolerances, spawn poses, or task state was explicitly rejected.

## Cross-phase controls and regression

- Fresh W1 GT: `INCOMPLETE`, inspected LEFT_DRAWER, RIGHT_DRAWER, TOOL_CABINET; `CURRENT_UPSTREAM_PHASE3_BLOCKED`.
- Fresh W6 GT: `INCOMPLETE`, inspected TOOL_CABINET, LEFT_DRAWER, RIGHT_DRAWER; `CURRENT_UPSTREAM_PHASE3_BLOCKED`.
- Fresh L1 GT regeneration was unavailable because its semantic detector weights were absent locally; no downloaded model was introduced into execution. The previously persisted GT handoff was executed on the tested code and passed strict physical execution 10/10 with every prohibited strict flag false.
- Tests on the integrated code: 86 passed (Workshop/Phase-4 plus shared Living execution regressions).

Artifacts: `/tmp/p4-c11-final-pushed-34b76b46/`.

## Status

`P4-C1.1 = BLOCKED_AT_STRICT_STORAGE_OPEN_GEOMETRY`.

Next Phase-4 step: redesign the robot-actuated storage-handle contact/follow strategy using the unchanged scene geometry so bilateral handle contact does not penetrate the drawer fronts or cabinet door. Do not start P4-C1.2.
