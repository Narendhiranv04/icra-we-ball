# Phase 1 kitchen feasibility closure

Phase 1 evaluates: **Given a fixed task instruction and an exhaustively inspected controlled environment, does the observed semantic-geometric grounding system determine whether a complete functional assignment exists?**

All 16 variants use the identical instruction. Coffee requires complete target coverage; reuse is preferred through minimum-distinct-tool assignment but one universal tool is not required. Soup requires a global one-to-one assignment of three distinct physical utensils. Cross-function reuse is allowed and neutral.

The oracle uses privileged full instantiated MuJoCo geometry for evaluation only. Its cavity opening is an inner-rim estimate that excludes handles/exterior protrusions. Production uses only visible region-gated RGB-D point-cloud measurements and YOLO-World semantic evidence. An unresolved witness is called task-level `INFEASIBLE` only after the fixed inspection order is exhausted.

The final real result is **15/16 correct**: all six oracle-infeasible variants were rejected and nine of ten oracle-feasible variants were detected. `F0_REUSE_ONE` remains a transparent false-infeasible: two B1 objects have valid five-view geometry but ambiguous YOLO-World evidence under the preserved two-view and winning-margin gates. `P0_LAYOUT_BASE` is now a genuinely distinct layout and passes. We did not weaken the protocol or inject hidden labels to obtain 16/16.

Aggregate metrics: overall feasibility accuracy 0.9375; feasible detection/recall 0.90; infeasible recall 1.00; false feasible 0; false infeasible 1; earliest-stage success over all oracle-feasible 0.90 (1.00 conditional on detected-feasible); minimum-coffee-tool optimality over all oracle-feasible 0.90 (1.00 conditional on detected-feasible); soup distinct-assignment validity 1.00.

This is a controlled Phase-1 result, not a claim of general real-world reliability. No FM/LLM/VLM, PDDL/action planner, robot, navigation, IK, or manipulation is used.

## Results

| Variant | Oracle | Prediction | Oracle stage | Predicted stage | Oracle coffee tools | Predicted tools | Result |
|---|---|---|---|---|---:|---:|---|
| F0_REUSE_ONE | FEASIBLE | INFEASIBLE | C1 | - | 1 | - | FAIL |
| F1_INITIAL_COMPLETE | FEASIBLE | FEASIBLE | INITIAL | INITIAL | 1 | 1 | PASS |
| F2_DISTRIBUTED_COFFEE_TWO | FEASIBLE | FEASIBLE | C1 | C1 | 2 | 2 | PASS |
| F3_DISTRIBUTED_COFFEE_THREE | FEASIBLE | FEASIBLE | C1 | C1 | 3 | 3 | PASS |
| F4_EARLY_RELOCATION | FEASIBLE | FEASIBLE | D2 | D2 | 1 | 1 | PASS |
| F5_LATE_RELOCATION | FEASIBLE | FEASIBLE | C1 | C1 | 1 | 1 | PASS |
| F6_DECOY_HEAVY | FEASIBLE | FEASIBLE | C1 | C1 | 1 | 1 | PASS |
| F7_COUNT_SURPLUS | FEASIBLE | FEASIBLE | C1 | C1 | 1 | 1 | PASS |
| I0_MISSING_COFFEE_VESSEL | INFEASIBLE | INFEASIBLE | - | - | - | - | PASS |
| I1_MISSING_SOUP_VESSEL | INFEASIBLE | INFEASIBLE | - | - | 1 | - | PASS |
| I2_UNCOVERED_COFFEE_TARGET | INFEASIBLE | INFEASIBLE | - | - | - | - | PASS |
| I3_ONLY_TWO_SOUP_TOOLS | INFEASIBLE | INFEASIBLE | - | - | 1 | - | PASS |
| I4_SOUP_MATCHING_TRAP | INFEASIBLE | INFEASIBLE | - | - | 1 | - | PASS |
| I5_SEMANTIC_DECOY_GEOMETRY_FAILURE | INFEASIBLE | INFEASIBLE | - | - | - | - | PASS |
| P0_LAYOUT_BASE | FEASIBLE | FEASIBLE | C1 | C1 | 1 | 1 | PASS |
| P1_LAYOUT_SWAPPED | FEASIBLE | FEASIBLE | C1 | C1 | 1 | 1 | PASS |

## Reproduction

Run `./mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1/reproduction_command.sh <benchmark-id>`. Raw PLY/depth data and model weights are intentionally untracked; compact JSON, initial/terminal overview images, and the F0 diagnostic are included here.
