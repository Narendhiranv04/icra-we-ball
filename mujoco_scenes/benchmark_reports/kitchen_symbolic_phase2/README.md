# Phase 2: perception-grounded symbolic task planning

This report evaluates the pure symbolic boundary:

`Phase-1 COMPLETE witness + frozen observed symbolic evidence -> compiler -> deterministic state-space search -> independent symbolic replay`.

The domain contains exactly four generic operator types: `PICK`, `PLACE`, `POUR`, and `STIR`. There is no `SERVE`, task-specific pour/place action, exploration, execution-time perception, replanning, FM/VLM, robot execution, IK, motion planning, collision checking, PDDLStream, or TAMP.

All nine production Phase-1 `COMPLETE` outputs produced independently replay-valid plans. The Phase-1 F0 detector miss and all Phase-1 infeasible outputs are excluded from the Phase-2 success denominator because they never supply a COMPLETE witness.

Generic `POUR(source, target)` derives transferred content from the static observed `provides(source, content)` fact. Generic `PLACE(object, destination)` represents source return, utensil provision (`at(tool, bowl)`), and serving placement (`at(vessel, serving_area)`). Coffee compatibility and soup one-to-one assignments come only from the Phase-1 witness.

## Results

| Variant | Coffee tools | Plan length | Expanded | Runtime (s) | Valid |
|---|---:|---:|---:|---:|---|
| F1_INITIAL_COMPLETE | 1 | 31 | 1870 | 0.043091 | True |
| F2_DISTRIBUTED_COFFEE_TWO | 2 | 31 | 4254 | 0.072506 | True |
| F3_DISTRIBUTED_COFFEE_THREE | 3 | 31 | 4653 | 0.079855 | True |
| F4_EARLY_RELOCATION | 1 | 31 | 2221 | 0.038487 | True |
| F5_LATE_RELOCATION | 1 | 31 | 2221 | 0.040461 | True |
| F6_DECOY_HEAVY | 1 | 31 | 4049 | 0.065576 | True |
| F7_COUNT_SURPLUS | 1 | 31 | 4049 | 0.066559 | True |
| P0_LAYOUT_BASE | 1 | 31 | 4049 | 0.068411 | True |
| P1_LAYOUT_SWAPPED | 1 | 33 | 4591 | 0.081036 | True |

## Reproduce

```bash
.venv/bin/python -m mujoco_scenes.run_phase2_symbolic_benchmark \
  --phase1-report mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1 \
  --benchmark-id phase2_kitchen_reproduction
```

The frozen Phase-1 raw evidence referenced by its manifest must be present locally. Phase 2 itself performs no rendering or detector inference.
