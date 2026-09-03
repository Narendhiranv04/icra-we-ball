# Phase-4 strict-boundary closure

Date: 2026-09-01
Starting Phase-3 revision: `850db0228e5ca7818ea6dbd1562a42e94dd2fefb`

Final remote Phase-3 revision before the Phase-4 commit:
`618f76adf0f9b03571d257345d16a2babdecc0c3`. That intervening P3-E commit is
limited to the Kitchen VLM canonicalizer/tests; the requested clean GT
repeatability runs remain the P3-D.1 `850db022` controls below.

This pass changes execution only. It does not tune controllers or implement
insertion or fastening.

## Strict simulation contract

A zero-snap grasp equality is allowed only after robot approach, physical
finger closing, and sustained bilateral object contact, with no payload pose
recovery or substitution. It models grasp attachment. Alignment welds,
staging welds, installed-target fixtures, direct payload/velocity/task-state
writes, assisted postconditions, and runtime post-release dynamics changes
are forbidden task assistance.

Workshop strict execution intercepts fastener PLACE at the frame before the
legacy alignment/installed fixture path and reports
`STRICT_PHYSICAL_INSERTION_UNAVAILABLE`. Surface PLACE is likewise blocked
before the staging weld and reports `STRICT_PHYSICAL_SURFACE_PLACE_UNAVAILABLE`.

Workshop backend association remains explicitly
`SIMULATION_BACKEND_GROUND_TRUTH_ASSOCIATION_ONLY`: variant storage metadata
provides backend membership, but cannot choose or replace a frozen phi object.
Every mapping additionally requires unique geometric correspondence within
the Phase-3 Workshop association-evaluation radius of 0.16 m. Semantic labels
remain audit-only.

## Clean current-head GT results

Artifacts were generated from a detached clean worktree (`git_dirty=false`).

- K1 run A: `runs/phase4_sync1_phase3_850db022_a_20260901`; `INCOMPLETE`;
  inspected D1, D2, C2, B1, C1. Result SHA-256
  `abcc7b4b3e151d424bbdc5b80bc9440c9388ebb0f177d33850fe7c0cb91b773a`.
- K1 run B: `runs/phase4_sync1_phase3_850db022_b_20260901`; identical
  `INCOMPLETE` result, inspection order, result hash, and grounding hash
  `54cd5bdc4ad733b9b2d74f63cc33d696e40b213a68d53cc43d232fb4ae1268e3`.
  This is repeatable upstream incompleteness, not Phase-4 nondeterminism.
- L1: `runs/phase4_sync1_phase3_850db022_20260901`; ready plan SHA-256
  `7860472159f4b0e85d6a180e578901d7ce4a41067330bffdb33fe6c219fb84ef`.
- W1 and W6 at clean `850db022`: both `INCOMPLETE`, so current Phase 4
  correctly classifies them `CURRENT_UPSTREAM_PHASE3_BLOCKED` without action.

## Physical boundary smokes

Execution artifacts:

- Current L1: `runs/phase4_sync1_execution_850db022_20260901`. Nine of ten
  actions completed. Final PLACE failed strict angular settling at
  `0.24941 rad/s` versus `0.12 rad/s`; `post_release_dynamics_modified=false`.
- W1 boundary regression used the most recent prior certified ready handoff
  only because current `850db022` is upstream-incomplete:
  `runs/phase4_sync1_execution_boundary_20260901`. Both physical inspection
  OPENs and PICK succeeded; execution stopped at action 2 with
  `STRICT_PHYSICAL_INSERTION_UNAVAILABLE` (1/5), before legacy insertion.
- W6 boundary regression preserved the frozen identity despite its semantic
  mislabel, physically opened the cabinet, then failed the existing PICK
  preclose calibration by 0.4292 m. No role substitution occurred.

All strict telemetry categories were false in these smokes: no direct task or
payload state write, task fixture, post-release dynamics modification, or
other strict violation was detected.

Focused execution tests: `94 passed in 67.45s`.

Next Phase-4 work is controller implementation/calibration after valid current
handoffs exist: Living release settling, Workshop cabinet PICK, fixture-free
insertion/surface release, and finally physical SCREW. None is part of this
boundary pass.
