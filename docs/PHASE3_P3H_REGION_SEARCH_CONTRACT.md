# Phase 3 P3-H: Region Resolution & Search Contract

## 1. Executive Summary

Pass **P3-H** establishes the definitive, typed, and immutable runtime search boundary for the Phase-3 Functional Task and Motion Planning (TAMP) framework. Following the closure of semantic canonicalization across all three benchmark domains (Kitchen in P3-E/E.1/E.2, Living Room in P3-F/F.1/F.2, Workshop in P3-G/G.1/G.2/G.3), P3-H decouples $G_F$ functional requirement graphs from runtime perception and inspection sequencing via `SearchRegionContract`.

---

## 2. Core Invariants & Architecture

```
                 GT Provider / Natural VLM Document
                                 │
                                 ▼
                     G_F Functional Requirement Graph
                                 │
                                 ▼
         [ One-Time Handoff Boundary: freeze_search_region_contract() ]
                                 │
                                 ▼
               SearchRegionContract (Frozen Dataclass)
               - domain: str
               - canonical_region_ids: tuple[str, ...]
               - source: str (GT_ORACLE / VLM / GT_EXPLICIT / NO_SEARCH)
               - policy_version: 'phase3_p3h_v1'
               - proposal_trace: tuple[dict, ...]
               - no_search_required: bool
                                 │
                                 ▼
                   Perception / Inspection Loop
             (Evaluates satisfaction against growing G_O)
```

### Key Invariants Enforced:
1. **Immutability & Isolation**: `SearchRegionContract` is a frozen dataclass whose `canonical_region_ids` and `proposal_trace` are detached, immutable tuples. Mutating original $G_F$ metadata, candidate region lists, or adapter documents after contract creation has zero impact on runtime execution.
2. **Strict Single Provider Invocation**: The provider (GT or VLM) is invoked exactly once during specification acquisition. Subsequent grounding retries and physical inspections never re-query or recall foundation models.
3. **Sole Role-Assignment Authority**: Region ordering determines **WHERE** physical evidence is acquired next when grounding is incomplete. It does not rank candidate role assignments, re-weight $\phi^*$, modify $G_F$, or influence `ground_graph()`.
4. **Fail-Closed Taxonomy**: Missing, unknown, duplicate, or out-of-domain search region proposals fail closed immediately with `SearchRegionContractError`.

---

## 3. Domain Search Policies

| Domain | Allowed Canonical Search Regions | Inspection Policy | Early Termination Behavior |
|---|---|---|---|
| **Kitchen** | `("D1", "D2", "C2", "B1", "C1")` | Ranked sequential inspection of cupboards/drawers | Terminates as soon as $G_O$ satisfies $G_F$ |
| **Workshop** | `("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")` | Ranked sequential exploration of workbench storage | Terminates as soon as required driver/fastener are detected and geometrically verified |
| **Living Room** | `()` | Explicit no-search contract (`no_search_required = True`) | Global grounding on initial observable scene |

### Workshop W1 Inspection Dynamics:
In variant W1, the canonical search contract specifies `("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")`. Initial workbench perception detects the workpiece `repair_target`. Physical opening of `LEFT_DRAWER` in Stage 1 acquires verified observations of `object_0001` (driver) and `object_0002` (fastener) satisfying all functional relations (`COMPATIBLE_WITH`, `COMPATIBLE_WITH_TARGET`, `REACHES_TARGET`). Grounding returns `COMPLETE`, and search halts immediately after 1 inspection (`LEFT_DRAWER`), leaving remaining regions unvisited.

---

## 4. Verification Evidence

- **Unit Test Suite**: `mujoco_scenes/functional_tamp_pipeline/tests/test_search_region_contract.py` (17/17 tests passing) verifying immutability, mutation isolation, single provider invocation, fail-closed validation, and grounding independence.
- **Pipeline Test Suite**: `mujoco_scenes/functional_tamp_pipeline/tests/` (272/272 tests passing).
- **Benchmark GT Controls**:
  - **Workshop W1**: `ACTION_SEQUENCE_READY`, Inspections: 1 (`LEFT_DRAWER`), PlanLen: 5, Combined: 6, Grounding: `COMPLETE`, GroundingAudit: `VALID`, ReplayValid: `VALID`, AccessValid: `VALID`, Condition: `gt_oracle`, Scientifically Valid: 1/1.
  - **Kitchen K1**: `ACTION_SEQUENCE_READY`, Inspections: 0, PlanLen: 24, Combined: 24, Grounding: `COMPLETE`, GroundingAudit: `VALID`, ReplayValid: `VALID`, AccessValid: `VALID`, Condition: `gt_oracle`, Scientifically Valid: 1/1.
  - **Living Room L1**: `ACTION_SEQUENCE_READY`, Inspections: 0, PlanLen: 10, Combined: 10, Grounding: `COMPLETE`, GroundingAudit: `VALID`, ReplayValid: `VALID`, AccessValid: `VALID`, Condition: `gt_living_room`, Scientifically Valid: 1/1.
