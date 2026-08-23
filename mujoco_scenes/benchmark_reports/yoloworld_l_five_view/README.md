# YOLO-World L five-view adaptation

Kitchen and Living Room have been restored to their exact pre-three-view
camera rigs. Workshop remains on its independent three-view standard.

Living Room classification is **13/13** with L: feasible recall 1.0,
infeasible recall 1.0, and zero false-feasible decisions. The frozen settings
are detector confidence 0.01, minimum mean confidence 0.01, minimum semantic
support 2, and winning-score margin 0.015. One scientific causal guard remains
failed: I5 is rejected, but the detector does not recover the intended
cross-function-conflict signature, so 13/13 must not yet be described as a
fully causally validated result.

Kitchen uses confidence 0.01, minimum mean confidence 0.01, semantic support
1, and `supporting_views_then_weighted_score` fusion. Its final vocabulary is
`semantic_vocabulary_yoloworld_l_five_view_kitchen_nojar.yaml`; closed
non-cup containers remain rejected by RGB-D open-cavity geometry. The full
post-change Kitchen benchmark is **16/16**: all 10 feasible variants are
predicted feasible and all 6 infeasible variants are predicted infeasible.
The run used the five-view legacy Kitchen rig at 1280x960 with
`yolov8l-worldv2.pt` and task-level feasibility evaluation (`--no-robot`).

| Variant | Oracle | Predicted | Result |
| --- | --- | --- | --- |
| F0_REUSE_ONE | FEASIBLE | FEASIBLE | PASS |
| F1_INITIAL_COMPLETE | FEASIBLE | FEASIBLE | PASS |
| F2_DISTRIBUTED_COFFEE_TWO | FEASIBLE | FEASIBLE | PASS |
| F3_DISTRIBUTED_COFFEE_THREE | FEASIBLE | FEASIBLE | PASS |
| F4_EARLY_RELOCATION | FEASIBLE | FEASIBLE | PASS |
| F5_LATE_RELOCATION | FEASIBLE | FEASIBLE | PASS |
| F6_DECOY_HEAVY | FEASIBLE | FEASIBLE | PASS |
| F7_COUNT_SURPLUS | FEASIBLE | FEASIBLE | PASS |
| I0_MISSING_COFFEE_VESSEL | INFEASIBLE | INFEASIBLE | PASS |
| I1_MISSING_SOUP_VESSEL | INFEASIBLE | INFEASIBLE | PASS |
| I2_UNCOVERED_COFFEE_TARGET | INFEASIBLE | INFEASIBLE | PASS |
| I3_ONLY_TWO_SOUP_TOOLS | INFEASIBLE | INFEASIBLE | PASS |
| I4_SOUP_MATCHING_TRAP | INFEASIBLE | INFEASIBLE | PASS |
| I5_SEMANTIC_DECOY_GEOMETRY_FAILURE | INFEASIBLE | INFEASIBLE | PASS |
| P0_LAYOUT_BASE | FEASIBLE | FEASIBLE | PASS |
| P1_LAYOUT_SWAPPED | FEASIBLE | FEASIBLE | PASS |

The benchmark was completed in fresh processes to prevent renderer/model state
from accumulating across all 16 high-resolution variants. Raw comparisons are
under `runs/yoloworld_l_five_view/kitchen_l_post_changes_full16/` (F0-F6),
`kitchen_l_post_changes_remaining9/` (F7 and I0-I5),
`kitchen_l_post_changes_p0_retry/` (P0), and
`kitchen_l_post_changes_p1_retry_no_overlays/` (P1). P1 disabled only semantic
overlay image export; inference and decision settings are identical.

Raw Living Room evidence is at
`outputs/yolo_world_l_five_view/living_room/l_five_view_c01_s2/`.
