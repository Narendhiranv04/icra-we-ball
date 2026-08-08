# Living-room Region-Function Phase 1

Fixed goal: **Prepare the living room for two people watching television. Place one refreshment set within easy reach of each person's seating position, and place the TV remote and game controller together on a suitable shared surface accessible to both people.**

This frozen benchmark performs one INITIAL five-view RGB-D observation. It
grounds only spatial destination-region functions from RGB semantics, measured
support geometry, two-object set packing, and seat-relative context. Objects
are payload operands; there is no object-function grounding. The production
solver exhaustively allocates two distinct personal regions and one separate
shared-controls region. It emits COMPLETE or controlled-set INFEASIBLE and
stops before planning or execution.

| Variant | Oracle | Production | Semantic-only | Geometry-only |
|---|---:|---:|---:|---:|
| F0_BASE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F1_LAYOUT_SWAPPED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F2_INSTANCE_ORDER_PERMUTED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F3_GLOBAL_MATCHING_REQUIRED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F4_PERSONAL_GEOMETRY_ALTERNATIVE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F5_SHARED_ALTERNATIVE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F6_DECOY_SURPLUS | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| I0_PERSONAL_SEMANTIC_DEFICIT | INFEASIBLE | INFEASIBLE | INFEASIBLE | COMPLETE |
| I1_PERSONAL_GEOMETRY_DEFICIT | INFEASIBLE | INFEASIBLE | INFEASIBLE | COMPLETE |
| I2_PERSONAL_TARGET_COVERAGE_FAILURE | INFEASIBLE | INFEASIBLE | COMPLETE | INFEASIBLE |
| I3_SHARED_FIT_FAILURE | INFEASIBLE | INFEASIBLE | COMPLETE | INFEASIBLE |
| I4_SHARED_CONTEXT_FAILURE | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| I5_CROSS_FUNCTION_CONFLICT | INFEASIBLE | INFEASIBLE | COMPLETE | INFEASIBLE |

Overall accuracy: 1.000. Feasible
recall: 1.000. Infeasible recall:
1.000.

Each variant directory contains the compact witness, compatibility matrix,
oracle comparison, and representative RGB/semantic/mask overviews. Raw RGB-D
and point clouds remain in the corresponding untracked `runs/` directory.

The oracle is marked `PRIVILEGED_ORACLE_EVALUATION_ONLY` and is produced only
after the independent production result. It is never imported by production
grounding.
