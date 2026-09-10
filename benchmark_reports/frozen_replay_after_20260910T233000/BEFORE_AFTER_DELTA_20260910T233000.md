## A. Stage progression (same 96 frozen responses)

| stage | before | after | delta |
| --- | ---: | ---: | ---: |
| strict_v3_valid | 81 | 88 | +7 |
| task_valid | 81 | 88 | +7 |
| graph_compiled | 75 | 87 | +12 |
| grounding_reached | 75 | 87 | +12 |
| complete_grounding | 9 | 10 | +1 |
| astar_reached | 16 | 49 | +33 |
| success | 9 | 10 | +1 |
| outcome_correct | 36 | 41 | +5 |
| feasible GT-satisfied | 0 | 7 | +7 |
| FALSE completions | 9 | 3 | -6 |
| GT goals satisfied (sum) | 21 | 76 | +55 |

## B. Outcome categories

| outcome | before | after | delta |
| --- | ---: | ---: | ---: |
| FUNCTIONAL_ASSIGNMENT_FAILURE | 39 | 13 | -26 |
| GRAPH_COMPILATION_FAILURE | 0 | 61 | +61 |
| None | 3 | 0 | -3 |
| OBJECT_DISCOVERY_FAILURE | 19 | 4 | -15 |
| PLANNING_FAILURE | 8 | 0 | -8 |
| SUCCESS | 9 | 10 | +1 |
| TASK_SPECIFICATION_FAILURE | 18 | 8 | -10 |

## C. Per domain

| domain | n | strict_v3_v b/a | task_valid b/a | graph_compi b/a | grounding_r b/a | complete_gr b/a | astar_reach b/a | success b/a | outcome_cor b/a | gtgoals b/a |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kitchen | 36 | 33/34 | 33/34 | 30/34 | 30/34 | 3/0 | 4/12 | 3/0 | 18/20 | 7/20 |
| living_room | 30 | 23/26 | 23/26 | 23/26 | 23/26 | 2/6 | 2/17 | 2/6 | 12/15 | 4/28 |
| workshop | 30 | 25/28 | 25/28 | 22/27 | 22/27 | 4/4 | 10/20 | 4/4 | 6/6 | 10/28 |

## D. Per variant (after / before where they differ)

| var | n | task | comp | grnd | cgnd | plan | succ | gt-goals | dominant outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K1 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 0->1 | 0->0 | 0->2 | GRAPH_COMPILATION_FAILURE=2 |
| K2 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 1->2 | 0->0 | 1->1 | GRAPH_COMPILATION_FAILURE=3 |
| K3 | 3 | 3->3 | 1->3 | 1->3 | 0->0 | 0->0 | 0->0 | 0->0 | GRAPH_COMPILATION_FAILURE=3 |
| K4 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 0->1 | 0->0 | 0->1 | GRAPH_COMPILATION_FAILURE=3 |
| K5 | 3 | 3->3 | 3->3 | 3->3 | 2->0 | 2->2 | 2->0 | 4->6 | GRAPH_COMPILATION_FAILURE=3 |
| K6 | 3 | 3->3 | 3->3 | 3->3 | 1->0 | 1->1 | 1->0 | 2->4 | GRAPH_COMPILATION_FAILURE=3 |
| K7 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 0->1 | 0->0 | 0->0 | GRAPH_COMPILATION_FAILURE=2 |
| K8 | 3 | 2->2 | 2->2 | 2->2 | 0->0 | 0->0 | 0->0 | 0->0 | GRAPH_COMPILATION_FAILURE=2 |
| K9 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 0->2 | 0->0 | 0->2 | GRAPH_COMPILATION_FAILURE=2 |
| K10 | 3 | 2->3 | 2->3 | 2->3 | 0->0 | 0->1 | 0->0 | 0->2 | GRAPH_COMPILATION_FAILURE=3 |
| K11 | 3 | 3->3 | 2->3 | 2->3 | 0->0 | 0->1 | 0->0 | 0->2 | GRAPH_COMPILATION_FAILURE=3 |
| K12 | 3 | 2->2 | 2->2 | 2->2 | 0->0 | 0->0 | 0->0 | 0->0 | GRAPH_COMPILATION_FAILURE=2 |
| L1 | 3 | 3->3 | 3->3 | 3->3 | 0->1 | 0->2 | 0->1 | 0->3 | GRAPH_COMPILATION_FAILURE=1 |
| L2 | 3 | 2->2 | 2->2 | 2->2 | 1->1 | 1->2 | 1->1 | 2->4 | FUNCTIONAL_ASSIGNMENT_FAILURE=1 |
| L3 | 3 | 1->1 | 1->1 | 1->1 | 0->0 | 0->1 | 0->0 | 0->2 | TASK_SPECIFICATION_FAILURE=2 |
| L4 | 3 | 3->3 | 3->3 | 3->3 | 1->2 | 1->3 | 1->2 | 2->7 | SUCCESS=2 |
| L5 | 3 | 3->3 | 3->3 | 3->3 | 0->2 | 0->2 | 0->2 | 0->5 | SUCCESS=2 |
| L6 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 0->1 | 0->0 | 0->1 | GRAPH_COMPILATION_FAILURE=3 |
| L7 | 3 | 2->3 | 2->3 | 2->3 | 0->0 | 0->3 | 0->0 | 0->3 | FUNCTIONAL_ASSIGNMENT_FAILURE=2 |
| L8 | 3 | 2->3 | 2->3 | 2->3 | 0->0 | 0->2 | 0->0 | 0->2 | GRAPH_COMPILATION_FAILURE=2 |
| L9 | 3 | 1->2 | 1->2 | 1->2 | 0->0 | 0->1 | 0->0 | 0->1 | GRAPH_COMPILATION_FAILURE=1 |
| L10 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 0->0 | 0->0 | 0->0 | GRAPH_COMPILATION_FAILURE=3 |
| W1 | 3 | 3->3 | 3->3 | 3->3 | 2->0 | 2->1 | 2->0 | 2->1 | GRAPH_COMPILATION_FAILURE=3 |
| W2 | 3 | 3->3 | 1->3 | 1->3 | 0->0 | 0->2 | 0->0 | 0->2 | GRAPH_COMPILATION_FAILURE=2 |
| W3 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 0->1 | 0->0 | 0->1 | GRAPH_COMPILATION_FAILURE=3 |
| W4 | 3 | 3->3 | 2->3 | 2->3 | 0->0 | 0->3 | 0->0 | 0->3 | FUNCTIONAL_ASSIGNMENT_FAILURE=2 |
| W5 | 3 | 1->2 | 1->2 | 1->2 | 0->0 | 1->1 | 0->0 | 1->1 | GRAPH_COMPILATION_FAILURE=1 |
| W6 | 3 | 2->3 | 2->3 | 2->3 | 0->1 | 0->3 | 0->1 | 0->5 | GRAPH_COMPILATION_FAILURE=1 |
| W7 | 3 | 2->3 | 2->2 | 2->2 | 1->1 | 1->2 | 1->1 | 1->4 | GRAPH_COMPILATION_FAILURE=2 |
| W8 | 3 | 3->3 | 3->3 | 3->3 | 1->0 | 1->3 | 1->0 | 1->3 | GRAPH_COMPILATION_FAILURE=2 |
| W9 | 3 | 3->3 | 3->3 | 3->3 | 0->0 | 3->2 | 0->0 | 3->2 | FUNCTIONAL_ASSIGNMENT_FAILURE=2 |
| W10 | 3 | 2->2 | 2->2 | 2->2 | 0->2 | 2->2 | 0->2 | 2->6 | SUCCESS=2 |

## E. Regressions (progressed less far after)

- K5/trial_01: depth 7->4, gt goals 2->0, SUCCESS -> GRAPH_COMPILATION_FAILURE; ('coffee_source',)
- K5/trial_02: depth 7->5, gt goals 2->4, SUCCESS -> GRAPH_COMPILATION_FAILURE; 
- K6/trial_01: depth 7->5, gt goals 2->4, SUCCESS -> GRAPH_COMPILATION_FAILURE; 
- L4/trial_02: depth 7->5, gt goals 2->2, SUCCESS -> GRAPH_COMPILATION_FAILURE; 
- W1/trial_02: depth 7->4, gt goals 1->0, SUCCESS -> GRAPH_COMPILATION_FAILURE; fastener, fastener
- W1/trial_03: depth 7->4, gt goals 1->0, SUCCESS -> GRAPH_COMPILATION_FAILURE; fastener, fastener
- W7/trial_03: depth 7->5, gt goals 1->1, SUCCESS -> GRAPH_COMPILATION_FAILURE; 
- W8/trial_01: depth 7->5, gt goals 1->1, SUCCESS -> GRAPH_COMPILATION_FAILURE; 
- W9/trial_02: depth 5->4, gt goals 1->0, PLANNING_FAILURE -> GRAPH_COMPILATION_FAILURE; REACHES_TARGET, REACHES_TARGET(None, None)
