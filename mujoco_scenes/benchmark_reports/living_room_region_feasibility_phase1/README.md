# Living-room Region-Function Phase 1

Fixed goal: **Prepare the living room for two people watching television. Place one refreshment set within easy reach of each person's seating position, and place the TV remote and game controller together on a suitable shared surface accessible to both people.**

This frozen benchmark performs one INITIAL five-view RGB-D observation. It
grounds only spatial destination-region functions from RGB semantics, measured
support geometry, two-object set packing, and seat-relative context. Objects
are payload operands; there is no object-function grounding. The production
solver exhaustively allocates two distinct personal regions and one separate
shared-controls region. It emits COMPLETE or controlled-set INFEASIBLE and
stops before planning or execution.

The scene uses documented CC0 Poly Haven furniture visuals at real-world scale
with independent analytic collision and RGB-D measurement proxies. Six payload
objects occupy a separate staging console, preserving a sparse destination
layout and reliable one-to-one instance association. Visual mesh dimensions
are never consumed by production inference.

| Variant | Oracle | Production | Semantic-only | Geometry-only |
|---|---:|---:|---:|---:|
| F0_BASE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F1_LAYOUT_SWAPPED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F2_INSTANCE_ORDER_PERMUTED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F3_GLOBAL_MATCHING_REQUIRED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F4_PERSONAL_GEOMETRY_ALTERNATIVE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F5_SHARED_ALTERNATIVE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F6_DECOY_SURPLUS | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| I0_PERSONAL_SEMANTIC_DEFICIT | INFEASIBLE | INFEASIBLE | COMPLETE | COMPLETE |
| I1_PERSONAL_GEOMETRY_DEFICIT | INFEASIBLE | INFEASIBLE | COMPLETE | INFEASIBLE |
| I2_PERSONAL_TARGET_COVERAGE_FAILURE | INFEASIBLE | INFEASIBLE | COMPLETE | INFEASIBLE |
| I3_SHARED_FIT_FAILURE | INFEASIBLE | INFEASIBLE | COMPLETE | COMPLETE |
| I4_SHARED_CONTEXT_FAILURE | INFEASIBLE | INFEASIBLE | COMPLETE | COMPLETE |
| I5_CROSS_FUNCTION_CONFLICT | INFEASIBLE | INFEASIBLE | COMPLETE | COMPLETE |

Overall accuracy: 1.000. Feasible
recall: 1.000. Infeasible recall:
1.000. Selected allocation validity:
1.000. F3 greedy-fail/global-
succeed diagnostic: 1.000.

Each variant directory contains the compact witness, compatibility matrix,
oracle comparison, and representative RGB/semantic/mask overviews. Raw RGB-D
and point clouds remain in the corresponding untracked `runs/` directory.

The oracle is marked `PRIVILEGED_ORACLE_EVALUATION_ONLY` and is produced only
after the independent production result. It is never imported by production
grounding. Every variant includes `evaluation_order.json`; the benchmark root
contains artifact-derived metrics and `scientific_guard_report.json`.
