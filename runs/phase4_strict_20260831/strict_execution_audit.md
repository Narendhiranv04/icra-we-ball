# Phase-4 strict execution audit — 2026-08-31

Branch baseline: `naren/pipeline_check` at `b48ed553`, with Phase-4 baseline
`32bc113ec75854fcbcc33f5282ba24d311eda791`.

## Contract enforced

- Phase-4 runners are strict-only; `--assisted-suite` and `--strict-pick` were removed.
- Kitchen uses `assisted_suite=False` and `allow_assisted_pick_recovery=False`.
- Workshop `_assisted_result`, direct searched-state restoration, direct pose/weld
  recovery, and direct fastener seating were removed from the Phase-4 adapter.
- Persisted `inspected_regions` are replayed before the immutable task plan through
  the existing robot-actuated physical OPEN primitives.
- Living-Room Phase 4 starts from the model reset without the observed-payload qpos
  rewrite. Its normal post-release damping remains transparent and does not write
  payload pose or velocity.
- The legacy Workshop SCREW routine is blocked before invocation because it writes
  fastener qpos and `joint_repaired`; no Workshop SCREW is certified strict until a
  force/joint-based simulator primitive exists.

All execution results record `strict_execution=true` and
`direct_task_state_fallback_used=false`, separate `inspection_execution` and
`task_plan_execution`, and SHA-256 hashes for the consumed Phase-3 manifest,
grounding result, final plan, and functional specification when present.

## Smoke results

| Variant | Inspection | Task execution | Result |
|---|---:|---:|---|
| K1 | 0/0 | 5/24 | Failed action 6 PLACE: second utensil settled outside assigned bowl |
| L1 | 0/0 | 10/10 | Final verification failed: left pair clearance 0.0068 m < 0.025 m |
| W1 | 1/1 physical OPEN | 3/5 | Stopped before prohibited legacy SCREW state writes |

## Full strict status

Kitchen inspection replay succeeded for all 11 requested OPENs. No Kitchen variant
completed its task plan:

- K1: 5/24, action 6 PLACE containment failure.
- K2: 0/26, action 1 PICK grasp failure after C2 OPEN.
- K3: 1/24, action 2 PLACE containment failure after B1 OPEN.
- K4: 1/26, action 2 PLACE containment failure after C2/B1 OPENs.
- K5: 6/24, action 7 PICK grasp failure after D1/D2 OPENs.
- K6: 1/26, action 2 PLACE containment failure after all five OPENs.

Living Room has no inspection OPENs. L5 passed strictly (8/8 plus final
verification). L1 and L6 completed all actions but failed final verification; L2
and L3 failed the final PLACE postcondition because the released payload had not
settled below the strict velocity threshold; L4 failed its first PICK collision
check.

Workshop inspection replay succeeded for all 13 requested OPENs in W1-W5 and
W7-W8. W1, W2, and W4 reached SCREW and stopped before the prohibited legacy direct
state writes. W3 failed action 3 PICK bilateral contact. W5, W7, and W8 failed
action 1 PICK. W6 was rejected as `BLOCKED_UPSTREAM_PHASE3` because its persisted
driver source disagrees with the simulator scene; Phase 4 did not alter it.

Overall current strict result: **1 passed, 18 Phase-4 physical failures, and 1
upstream Phase-3 blocked**. No assisted result is counted as success.

## Artifacts and tests

- Kitchen suite: `runs/phase4_strict_20260831/kitchen_suite/`
- Living-Room suite: `runs/phase4_strict_20260831/living_suite/`
- Final Workshop suite: `runs/phase4_strict_20260831/workshop_suite_final/`
- Final W1 smoke: `runs/phase4_strict_20260831/final_smoke/workshop/W1/gt/`
- Focused regression: 87 passed in 74.04 s.

Next Phase-4 work is controller/model work: implement a non-kinematic Workshop
fastening mechanism first, then fix Kitchen utensil containment/grasp reliability
and the remaining Living-Room placement settling/clearance failures without
relaxing postconditions.
