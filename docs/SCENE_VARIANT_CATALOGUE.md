# Kitchen and Living-Room Variant Catalogue

The active benchmark contains 12 Kitchen variants and 10 Living-Room
variants. Short labels are stable paper-facing aliases; internal identifiers
remain in the YAML configuration and output metadata.

## Kitchen

The fixed goal prepares coffee and soup for two people. It requires two coffee
containers, two soup containers, one reusable coffee stirrer, two distinct
soup utensils, a kettle, and a coffee source.

| Label | Internal ID | Result | Controlled change |
| --- | --- | --- | --- |
| K1 | F0_ALL_VISIBLE | Feasible | All required objects are visible |
| K2 | F1_HIDDEN_COFFEE_VESSEL | Feasible | Second coffee vessel is in C2 |
| K3 | F2_HIDDEN_SOUP_BOWL | Feasible | Second soup bowl is in B1 |
| K4 | F3_HIDDEN_VESSELS_MIXED | Feasible | A vessel is in C2 and a bowl is in B1 |
| K5 | F4_TOOLS_IN_DRAWERS | Feasible | Required tools are in D1 and D2 |
| K6 | F5_FULL_DISTRIBUTED_SEARCH | Feasible | Required objects span all storage regions |
| K7 | I0_MISSING_COFFEE_VESSEL | Infeasible | One coffee vessel is absent |
| K8 | I1_MISSING_SOUP_BOWL | Infeasible | One soup bowl is absent |
| K9 | I2_MISSING_COFFEE_SPOON | Infeasible | The coffee stirrer is absent |
| K10 | I3_MISSING_SOUP_UTENSIL | Infeasible | One soup utensil is absent |
| K11 | I4_MISSING_KETTLE | Infeasible | The kettle is absent |
| K12 | I5_MISSING_COFFEE_JAR | Infeasible | The coffee source is absent |

Authoritative layouts:
`mujoco_scenes/configs/kitchen_feasibility_variants.yaml`.

## Living room

The fixed goal places a cup and saucer near each seat and the remote on a
surface accessible from both seats.

| Label | Internal ID | Result |
| --- | --- | --- |
| L1 | F0_ALL_OBJECTS_IN_STAGING | Feasible |
| L2 | F1_LEFT_SAUCER_PREPLACED | Feasible |
| L3 | F2_LEFT_SAUCER_ON_SHARED | Feasible |
| L4 | F3_LEFT_CUP_ON_SHARED | Feasible |
| L5 | F4_SAUCER_PREPLACED_CUP_ON_SHARED | Feasible |
| L6 | F5_LEFT_PAIR_ON_SHARED | Feasible |
| L7 | I0_NO_SHARED_TABLE | Infeasible |
| L8 | I1_NO_LEFT_PERSONAL_TABLE | Infeasible |
| L9 | I2_NO_PERSONAL_TABLES | Infeasible |
| L10 | I3_NO_TABLES | Infeasible |

Authoritative layouts:
`mujoco_scenes/configs/living_room_variants.yaml`.

Workshop variants are intentionally outside this branch's current integration
scope.
