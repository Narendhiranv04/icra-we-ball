# Living-room Region-Function Phase 1

Fixed goal: **Prepare the living room for two people watching television. Place one cup and one saucer on each person's fixed individual side table, and place the TV remote on the fixed shared coffee table.**

This frozen benchmark performs one INITIAL five-view RGB-D observation. It
grounds only spatial destination-region functions from RGB semantics, measured
support geometry, two-object set packing, and seat-relative context. Objects
are payload operands; there is no object-function grounding. The production
solver exhaustively allocates the two fixed individual tables and the separate
fixed shared table. It emits COMPLETE or controlled-set INFEASIBLE and
stops before planning or execution.

The scene uses documented CC0 Poly Haven furniture visuals at real-world scale
with independent analytic collision and RGB-D measurement proxies. Five payload
objects occupy a separate staging console, preserving a sparse destination
layout and reliable one-to-one instance association. Visual mesh dimensions
are never consumed by production inference.

Candidate support regions are supplied by neutral simulator-derived spatial
proposal volumes. These proposals provide only region localization/evidence
gating. Functional validity, semantic role, support dimensions, geometry, and
target-relative suitability are inferred from rendered RGB/RGB-D evidence.
Open-world support-region proposal/discovery is outside the scope of this
benchmark.

| Variant | Oracle | Production | Semantic-only | Geometry-only |
|---|---:|---:|---:|---:|
| F0_ALL_OBJECTS_IN_STAGING | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F1_LEFT_SAUCER_PREPLACED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F2_LEFT_SAUCER_ON_SHARED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F3_LEFT_CUP_ON_SHARED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F4_SAUCER_PREPLACED_CUP_ON_SHARED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| F5_LEFT_PAIR_ON_SHARED | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| I0_NO_SHARED_TABLE | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| I1_NO_LEFT_PERSONAL_TABLE | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| I2_NO_PERSONAL_TABLES | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| I3_NO_TABLES | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |

Overall accuracy: 1.000. Feasible
recall: 1.000. Infeasible recall:
1.000. Selected allocation validity:
1.000. F0 production complete-
solution count: 1.

Each variant directory contains the compact witness, compatibility matrix,
oracle comparison, and representative RGB/semantic/mask overviews. Raw RGB-D
and point clouds remain in the corresponding untracked `runs/` directory.

The oracle is marked `PRIVILEGED_ORACLE_EVALUATION_ONLY` and is produced only
after the independent production result. It is never imported by production
grounding. Every variant includes `evaluation_order.json`; the benchmark root
contains artifact-derived metrics and `scientific_guard_report.json`.

Current limitations are manually specified functional requirements,
controlled simulator layouts, neutral privileged spatial proposals rather
than free-space proposal discovery, no robot execution, no task planning, no
FM requirement generation, and no photorealism claim. The included Google
Robot artifact validates only its initial compiled pose and clearance.
