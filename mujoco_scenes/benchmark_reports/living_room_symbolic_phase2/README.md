# Living-Room Symbolic Phase 2

This tracked report compiles the frozen production-only Region-Function Phase-1
witnesses into minimal classical placement problems. It does not rerun MuJoCo,
RGB-D perception, semantic detection, geometry, or functional allocation.

For every `COMPLETE` witness, the compiler maps all five generic payload IDs to
the exact generic region IDs selected by Phase 1. The initial abstraction is
`AVAILABLE(object)` plus `HAND_EMPTY`; no unobserved staging surface is
fabricated. Deterministic A* searches grounded `PICK` and `PLACE` operators.
An independent replay implementation then checks every precondition/effect and
all final `ON(object, region)` goals.

For every `INFEASIBLE` witness, compilation returns
`FUNCTIONAL_WITNESS_NOT_COMPLETE` and the planner is not invoked.

## Reproduce

```bash
python -m mujoco_scenes.run_living_room_phase2_symbolic_benchmark
```

Inspect `benchmark_summary.json`, `scientific_guard_report.json`, and each
directory under `variants/`. Feasible variants contain the symbolic state,
goals, generated PDDL, searched plan, and independent replay result. Rejected
variants deliberately contain no plan.

## Boundary

This phase is pure symbolic planning. It performs no robot execution, IK,
motion planning, PDDLStream/TAMP, or foundation-model inference. The minimal
one-shot `AVAILABLE` abstraction is appropriate to this fixed placement task;
later manipulation-aware replanning will require perception-grounded source
locations and motion feasibility.
