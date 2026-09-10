# Frozen zero-call replay BEFORE

trials: 96

| stage | count | pct |
| --- | ---: | ---: |
| strict_v3_valid | 81 | 84.4% |
| task_valid | 81 | 84.4% |
| graph_compiled | 75 | 78.1% |
| grounding_reached | 75 | 78.1% |
| complete_grounding | 9 | 9.4% |
| astar_reached | 16 | 16.7% |
| success | 9 | 9.4% |
| outcome_correct | 36 | 37.5% |

| domain | n | strict_v3_valid | task_valid | graph_compiled | grounding_reached | complete_grounding | astar_reached | success | outcome_correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kitchen | 36 | 33 | 33 | 30 | 30 | 3 | 4 | 3 | 18 |
| living_room | 30 | 23 | 23 | 23 | 23 | 2 | 2 | 2 | 12 |
| workshop | 30 | 25 | 25 | 22 | 22 | 4 | 10 | 4 | 6 |

## outcome categories
- FUNCTIONAL_ASSIGNMENT_FAILURE: 39
- OBJECT_DISCOVERY_FAILURE: 19
- TASK_SPECIFICATION_FAILURE: 18
- SUCCESS: 9
- PLANNING_FAILURE: 8
- None: 3

## per domain outcome categories
- **kitchen**: OBJECT_DISCOVERY_FAILURE=16, FUNCTIONAL_ASSIGNMENT_FAILURE=10, TASK_SPECIFICATION_FAILURE=4, SUCCESS=3, None=2, PLANNING_FAILURE=1
- **living_room**: FUNCTIONAL_ASSIGNMENT_FAILURE=18, TASK_SPECIFICATION_FAILURE=7, OBJECT_DISCOVERY_FAILURE=3, SUCCESS=2
- **workshop**: FUNCTIONAL_ASSIGNMENT_FAILURE=11, PLANNING_FAILURE=7, TASK_SPECIFICATION_FAILURE=7, SUCCESS=4, None=1

## top stop reasons
-  17  
-  13  UNINSTANTIABLE_MISSING_RELATION: no expressed verified placement requirement
-  12  NO_COMPLETE_FUNCTIONAL_WITNESS
-   7  fastener, fastener
-   5  No meaningful functional roles recovered
-   4  NO_GLOBAL_REGION_ASSIGNMENT
-   3  ('coffee_source',)
-   3  Role 'MAIN_WORKBENCH_ZONE' is a planner context constant and must not appear as a G_F role in domain 'workshop'
-   2  ('water_source',)
-   2  ('coffee_source', 'water_source')
-   2  ('CUP_SAUCER_SET', 'PERSONAL_CUP_SAUCER_REGION')
-   2  DUPLICATE_RELATION_PARTICIPANT: functional_relations[1]
-   2  DUPLICATE_RELATION_PARTICIPANT: functional_relations[2]
-   1  DUPLICATE_OPERATION_PARTICIPANT: operation_pairings[1]
-   1  Predicate 'INSERTABLE_IN' in domain 'kitchen' expects object entity_kind in ('OBJECT',), got 'REGION'
-   1  ('soup_container', 'coffee_source')
-   1  'Coffee_Powder'
-   1  'soup_bowl'
-   1  ('coffee_container', 'soup_container', 'coffee_source')
-   1  ({'group': 'coffee_stirring', 'tool': 'object_0007', 'target': 'object_0001', 'subject_id': 'object_0007', 'object_id': 
-   1  ('coffee_stirrer',)
-   1  ('coffee_container', 'coffee_source', 'water_source')
-   1  UNDECLARED_PARTICIPANT: operation_pairings[4] references ['Person']
-   1  ('soup_container',)
-   1  ({'group': 'coffee_stirring', 'tool': 'object_0006', 'target': 'object_0001', 'subject_id': 'object_0006', 'object_id': 

## disabled operation codes
-  12  UNINSTANTIABLE_MISSING_ROLE
-   9  UNSUPPORTED_OPERATOR

## Provenance

Zero FM calls.  Every row replays the archived raw semantic response of
`benchmark_reports/v3_qwen_distribution_3x32_20260910T053937` through the
deterministic pipeline at commit `470f7c5814657f66d95fa777b6bf7b67d33d0a7e`.

- intended collection: 32 variants x 3 trials = 96
- archived raw semantic responses: 91
- transport failures (completion truncated, no parseable content): 5
  (kitchen K12/trial_03, living_room L2/trial_02, L3/trial_01, L3/trial_02,
  living_room L9/trial_02)
- wire-invalid archived responses: 10
- feasible trials: 60; feasible trials whose GT goals were actually satisfied: 0

Nine trials terminated in `ACTION_SEQUENCE_READY` while satisfying no GT goal,
four of them from a graph carrying no operation group at all.  Those are false
completions, not successes.

Reproduce with:

    PYTHONPATH=. python3 scripts/replay_frozen_distribution.py \
        --root benchmark_reports/v3_qwen_distribution_3x32_20260910T053937 \
        --out <output-dir> --workers 5 --label BEFORE
