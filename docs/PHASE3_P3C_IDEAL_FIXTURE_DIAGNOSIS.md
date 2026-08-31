# Phase 3 Pass P3-C / P3-C.2: Ideal Raw VLM Fixture Framework & Concept-Exact Canonicalization Diagnosis

**Execution Date**: 2026-09-01  
**Target Branch**: `naren/pipeline_check`  
**Status**: **P3-C / P3-C.2 COMPLETE — CONCEPT-EXACT DIAGNOSTIC CONTROL ESTABLISHED**

---

## 1. Executive Summary & Scientific Purpose

Pass P3-C / P3-C.2 establishes a deterministic, concept-exact offline semantic control to answer:
> *"If the VLM had produced a semantically excellent, schema-valid raw response using neutral model-local identifiers and natural open-vocabulary language, what would the current deterministic software do with it?"*

This isolates:
1. **Model Generation Capacity** (zero model calls were made in this control), from
2. **Deterministic Canonicalizer / Interface Implementation** (`kitchen_vlm_functional_graph.py`, `environment_vlm_requirements.py`, and `workshop_phase1/requirements.py`).

### Invariant & Anti-Leak Hardening
- **Neutral Model-Local Identifiers**: All raw IDs are neutral (`role_1`..`role_6`, `group_1`..`group_2`, `search_1`..`search_3`).
- **Zero Predicate Leakage**: All relation-bearing fields (`functional_relations[].relation`, `interaction_groups[].required_relations[]`, `interaction_groups[].context_relations[]`) are verified to contain zero internal canonical predicate tokens (`INSERTABLE_IN`, `FITS_SET_ON`, `NEAR_SEAT`, `CAN_DRIVE_SCREW`, `COMPATIBLE_WITH`, etc.).
- **Zero Oracle Tokens**: Zero simulator backend handles (`workshop_power_driver`, `workshop_medium_phillips_screw`) and zero benchmark oracle region/object IDs (`LEFT_DRAWER`, `RIGHT_DRAWER`, `TOOL_CABINET`, `D1`, `D2`, `C1`, `C2`, `B1`, `object_0001`, `region_0001`, `K1`, `L1`, `W1`).

---

## 2. Canonicalization Diagnostic & Reference Summary

| Domain | Fixture File | Production Schema | Raw Semantics | Current Canonicalizer Outcome | Role Identity Recall | Role Exact Recall | Relation Recall | OpGroup Identity Recall | Reference Complete | Exact Structural Match | First Failing Layer / Structural Notes |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Kitchen** | `kitchen_K1.json` | `KITCHEN_FUNCTIONAL_GRAPH_SCHEMA` | Complete (6 roles, 4 rels, 2 groups) | **`CANONICALIZED`** | **1.0** (6/6) | **0.833** (5/6) | **1.0** (4/4) | **1.0** (2/2) | **True** | False | No failure on ideal fixture. `exact_structural_match` is False because reusable `coffee_stirrer` candidate specifies point count `[1, 1]`, whereas reference specifies range `[1, 2]` (`cardinality_compatible: True`, but not `cardinality_exact`). |
| **Living Room** | `living_room_L1.json` | `RESPONSE_SCHEMA` | Complete (6 roles, 4 rels, 1 group) | **`CANONICALIZATION_FAILED`** | N/A | N/A | N/A | N/A | N/A | N/A | **Layer B (Canonicalizer)**: `environment_vlm_requirements.py:172` in `map_living_room_relation()` (`UnmappedFunctionalConceptError`). |
| **Workshop** | `workshop_W1.json` | `RESPONSE_SCHEMA` | Complete (3 roles, 3 rels, 1 group, 3 regions) | **`CANONICALIZED`** | **1.0** (3/3) | **1.0** (3/3) | **1.0** (3/3) | 1.0 (vacuous) | **True** | False | No failure on ideal fixture. `exact_structural_match` is False because candidate declares semantically valid `group_1` (`extra_operation_groups: ['group_1']`), which is absent in static GT reference (`operation_groups: ()`). |

---

## 3. Evidence-Based Concept Preservation & Loss Breakdown

### 3.1 Kitchen (K1 Ideal Fixture)

#### Roles
- `role:role_1` $\to$ `PRESERVED -> coffee_container` (count: 2, `DISTINCT`)
- `role:role_2` $\to$ `PRESERVED -> soup_container` (count: 2, `DISTINCT`)
- `role:role_3` $\to$ `PRESERVED -> coffee_stirrer` (count: 1, `REUSABLE`)
- `role:role_4` $\to$ `PRESERVED -> soup_eating_utensil` (count: 2, `DISTINCT`)
- `role:role_5` $\to$ `PRESERVED -> water_source` (count: 1, `DISTINCT`)
- `role:role_6` $\to$ `PRESERVED -> coffee_source` (count: 1, `DISTINCT`)

#### Properties (Evidence-Derived via `map_unary_property`)
| Raw Key | Raw Phrase | Canonical Role | Mapped Predicate | Role Graph Predicate | Trace Classification |
|---|---|---|:---:|:---:|:---:|
| `prop:role_1:open cavity` | "open cavity" | `coffee_container` | `OPEN_CAVITY` | `OPEN_CAVITY` | `PRESERVED -> OPEN_CAVITY` |
| `prop:role_1:capable of holding liquid` | "capable of holding liquid" | `coffee_container` | `OPEN_CAVITY` | `OPEN_CAVITY` | `MERGED_BY_EXPLICIT_RULE -> OPEN_CAVITY` |
| `prop:role_2:open cavity` | "open cavity" | `soup_container` | `OPEN_CAVITY` | `OPEN_CAVITY` | `PRESERVED -> OPEN_CAVITY` |
| `prop:role_2:capable of holding liquid` | "capable of holding liquid" | `soup_container` | `OPEN_CAVITY` | `OPEN_CAVITY` | `MERGED_BY_EXPLICIT_RULE -> OPEN_CAVITY` |
| `prop:role_3:elongated shape` | "elongated shape" | `coffee_stirrer` | `ELONGATED_OBJECT` | `ELONGATED_OBJECT` | `PRESERVED -> ELONGATED_OBJECT` |
| `prop:role_3:slender` | "slender" | `coffee_stirrer` | `ELONGATED_OBJECT` | `ELONGATED_OBJECT` | `MERGED_BY_EXPLICIT_RULE -> ELONGATED_OBJECT` |
| `prop:role_4:elongated shape` | "elongated shape" | `soup_eating_utensil` | `ELONGATED_OBJECT` | `ELONGATED_OBJECT` | `PRESERVED -> ELONGATED_OBJECT` |

#### Relations (Concept-Exact, Predicate-Verified)
| Unique Trace Key | Raw Relation Text | Canonical Subject | Canonical Predicate | Canonical Object | Triple in $G_F$? | Outcome |
|---|---|---|:---:|---|:---:|:---:|
| `rel:0:role_3:fits inside:role_1` | "fits inside" | `coffee_stirrer` | `INSERTABLE_IN` | `coffee_container` | Yes | `PRESERVED -> coffee_stirrer -[INSERTABLE_IN]-> coffee_container` |
| `rel:1:role_3:reaches the bottom:role_1` | "reaches the bottom" | `coffee_stirrer` | `REACHES_BOTTOM` | `coffee_container` | Yes | `PRESERVED -> coffee_stirrer -[REACHES_BOTTOM]-> coffee_container` |
| `rel:2:role_4:fits inside:role_2` | "fits inside" | `soup_eating_utensil` | `INSERTABLE_IN` | `soup_container` | Yes | `PRESERVED -> soup_eating_utensil -[INSERTABLE_IN]-> soup_container` |
| `rel:3:role_4:reaches the bottom:role_2` | "reaches the bottom" | `soup_eating_utensil` | `REACHES_BOTTOM` | `soup_container` | Yes | `PRESERVED -> soup_eating_utensil -[REACHES_BOTTOM]-> soup_container` |

#### Groups
- `group:group_1` $\to$ `PRESERVED -> coffee_stirring` (`SEQUENTIAL_REUSE_ALLOWED`)
- `group:group_2` $\to$ `PRESERVED -> soup_serving` (`DEDICATED_PER_TARGET`)

#### Reference Evaluation Breakdown
- `role_identity_recall`: 1.0 (6/6), `role_identity_precision`: 1.0 (6/6)
- `role_exact_recall`: 0.833 (5/6), `role_exact_precision`: 0.833 (5/6)
- `relation_recall`: 1.0 (4/4), `relation_precision`: 1.0 (4/4)
- `operation_group_identity_recall`: 1.0 (2/2), `operation_group_identity_precision`: 1.0 (2/2)
- `reference_complete`: `True`, `exact_structural_match`: `False`
- **Evaluator Cardinality Diagnostic**:
  ```json
  "role_cardinality_diagnostics": {
    "coffee_stirrer": {
      "reference_binding": "REUSABLE",
      "candidate_binding": "REUSABLE",
      "reference_range": [1, 2],
      "candidate_range": [1, 1],
      "cardinality_compatible": true,
      "cardinality_exact": false,
      "reason": "Candidate reusable count [1, 1] is semantically compatible with reference allowed interval [1, 2]"
    }
  }
  ```
  *Explanation*: Candidate specifies 1 reusable stirrer (`[1, 1]`), which is semantically valid for preparing 2 coffees. The GT reference declares allowed interval `[1, 2]`. This compatible but non-exact interval makes `all_cardinalities_exact = False`, yielding `5/6 = 0.833` exact role recall and `exact_structural_match = False`.

---

### 3.2 Living Room (L1 Ideal Fixture)

#### End-to-End Failure
- **Exception Type**: `UnmappedFunctionalConceptError`
- **Category**: `UNMAPPED_FUNCTIONAL_CONCEPT`
- **Module**: `environment_vlm_requirements.py:172` in `map_living_room_relation()`
- **Message**: `VLM living room relation 'can hold drinkware set' cannot be mapped to any reviewed relation`

#### Granular Sub-Concept Mapping Breakdown (with Contextual Roles)
| Raw Concept Key | Entity Kind & Raw Text | Canonical Role Context | Mapping Target | Sub-Diagnostic Outcome |
|---|---|---|:---:|:---:|
| `role:role_1` | `REGION`: "hold items for viewer" | — | `PERSONAL_CUP_SAUCER_REGION` | **`PRESERVED/MAPPABLE`** |
| `role:role_2` | `REGION`: "hold items for viewers" | — | `SHARED_REMOTE_REGION` | **`PRESERVED/MAPPABLE`** |
| `role:role_3` | `OBJECT`: "contain hot beverage and saucer" | — | `CUP_SAUCER_SET` | **`PRESERVED/MAPPABLE`** |
| `role:role_4` | `OBJECT`: "control television" | — | `REMOTE` | **`PRESERVED/MAPPABLE`** |
| `role:role_5` | `FIXED_TARGET`: "viewer seating position" | — | `SEATING_POSITION` | **`SYSTEM_CONTEXT_COMPILED`** |
| `role:role_6` | `FIXED_TARGET`: "paired viewer seating area" | — | `SEATING_PAIR` | **`SYSTEM_CONTEXT_COMPILED`** |
| `prop:role_1:planar horizontal support` | "planar horizontal support" | `PERSONAL_CUP_SAUCER_REGION` | `PLANAR_SUPPORT` | **`PRESERVED/MAPPABLE -> PLANAR_SUPPORT`** |
| `prop:role_2:planar horizontal support` | "planar horizontal support" | `SHARED_REMOTE_REGION` | `PLANAR_SUPPORT` | **`PRESERVED/MAPPABLE -> PLANAR_SUPPORT`** |
| `rel:0:role_1:can hold drinkware set:role_3` | "can hold drinkware set" | `PERSONAL_CUP_SAUCER_REGION` $\to$ `CUP_SAUCER_SET` | — | **`REJECTED: UnmappedFunctionalConceptError`** |
| `rel:1:role_1:near seat:role_5` | "near seat" | `PERSONAL_CUP_SAUCER_REGION` $\to$ `SEATING_POSITION` | `NEAR_SEAT` | **`PRESERVED/MAPPABLE -> PERSONAL_CUP_SAUCER_REGION -[NEAR_SEAT]-> SEATING_POSITION`** |
| `rel:2:role_2:can hold remote:role_4` | "can hold remote" | `SHARED_REMOTE_REGION` $\to$ `REMOTE` | — | **`REJECTED: UnmappedFunctionalConceptError`** |
| `rel:3:role_2:accessible from both seats:role_6` | "accessible from both seats" | `SHARED_REMOTE_REGION` $\to$ `SEATING_PAIR` | `ACCESSIBLE_FROM_BOTH_SEATS` | **`PRESERVED/MAPPABLE -> SHARED_REMOTE_REGION -[ACCESSIBLE_FROM_BOTH_SEATS]-> SEATING_PAIR`** |
| `group:group_1` | Support drinkware beside seat | `PERSONAL_CUP_SAUCER_REGION` $\to$ `CUP_SAUCER_SET` | — | **`NOT_REACHED_DUE_TO_PRIOR_FAILURE`** |

---

### 3.3 Workshop (W1 Ideal Fixture)

#### Roles & Properties
- `role:role_1` $\to$ `PRESERVED -> driver` (count: 1, `DISTINCT`)
- `role:role_2` $\to$ `PRESERVED -> fastener` (count: 1, `DISTINCT`)
- `role:role_3` $\to$ `PRESERVED -> repair_target` (`SYSTEM_OWNED_FIXED_TARGET_REPRESENTATION`)
- `properties` $\to$ **`Workshop raw unary property count = 0 (no property preservation cases applicable)`**

#### Relations (Concept-Exact, Predicate-Verified via `provider.normalized_relations`)
| Unique Trace Key | Raw Relation Text | Canonical Subject | Canonical Predicate | Canonical Object | Triple in $G_F$? | Outcome |
|---|---|---|:---:|---|:---:|:---:|
| `rel:0:role_1:compatible with:role_2` | "compatible with" | `driver` | `COMPATIBLE_WITH` | `fastener` | Yes | `PRESERVED -> driver -[COMPATIBLE_WITH]-> fastener` |
| `rel:1:role_1:reaches target:role_3` | "reaches target" | `driver` | `REACHES_TARGET` | `repair_target` | Yes | `PRESERVED -> driver -[REACHES_TARGET]-> repair_target` |
| `rel:2:role_2:compatible with target:role_3` | "compatible with target" | `fastener` | `COMPATIBLE_WITH_TARGET` | `repair_target` | Yes | `PRESERVED -> fastener -[COMPATIBLE_WITH_TARGET]-> repair_target` |

#### Regions (1-to-1 Resolution via `resolve_workshop_region_proposal`)
| Raw Proposal Key | Proposal Label & Visual Description | Canonical Target Region | Outcome |
|---|---|:---:|:---:|
| `region:search_1` | "left storage drawer beneath the workbench" | `LEFT_DRAWER` | `PRESERVED -> LEFT_DRAWER` |
| `region:search_2` | "right storage drawer beneath the workbench" | `RIGHT_DRAWER` | `PRESERVED -> RIGHT_DRAWER` |
| `region:search_3` | "tall tool cabinet beside the workbench" | `TOOL_CABINET` | `PRESERVED -> TOOL_CABINET` |

#### Reference Evaluation Breakdown & Metric Disclosure
- `role_identity_recall`: 1.0 (3/3), `role_identity_precision`: 1.0 (3/3)
- `role_exact_recall`: 1.0 (3/3), `role_exact_precision`: 1.0 (3/3)
- `relation_recall`: 1.0 (3/3), `relation_precision`: 1.0 (3/3)
- `operation_group_identity_recall`: 1.0 *(vacuous: reference declares 0 operation groups)*
- `operation_group_identity_precision`: 0.0 *(candidate declares extra `group_1`)*
- `extra_operation_groups`: `['group_1']`
- `reference_complete`: `True`, `exact_structural_match`: `False`
- **Structural Disclosure**: Candidate G_F expresses a valid interaction group for driving the fastener into the joint (`group_1`). The offline GT reference specifies `operation_groups: ()`. This extra representational group causes `operation_group_identity_precision = 0.0` and `exact_structural_match = False`.

---

## 4. Evaluator Diagnostics Summary

```json
{
  "kitchen": {
    "reference_complete": true,
    "exact_structural_match": false,
    "role_attribute_mismatches": {},
    "role_cardinality_diagnostics": {
      "coffee_stirrer": {
        "cardinality_compatible": true,
        "cardinality_exact": false,
        "reference_range": [1, 2],
        "candidate_range": [1, 1]
      }
    },
    "operation_group_representation_diagnostics": {
      "soup_serving": {
        "selection_preference": {
          "reference": "",
          "candidate": "deterministic_rank",
          "grounding_relevant": false
        }
      }
    }
  },
  "workshop": {
    "reference_complete": true,
    "exact_structural_match": false,
    "role_attribute_mismatches": {},
    "role_cardinality_diagnostics": {},
    "operation_groups": {
      "extra": ["group_1"]
    }
  }
}
```
