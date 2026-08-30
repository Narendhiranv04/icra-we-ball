# VLM Functional Specification Interface Contract

This document defines the authoritative scientific interface contract between the Vision-Language Model (VLM) specification provider and downstream perception, grounding, and task-and-motion planning (TAMP) components.

---

## 1. Architectural Overview

The goal of the functional-grounding TAMP pipeline is to enable a robot to perform complex, multi-stage manipulation tasks without pre-programmed instance identities, ground-truth physical dimensions, or hardcoded action sequences.

```
+---------------------------------------------------------------------------------------------------+
| 1. VLM Specification Acquisition (Zero-Leakage Boundary)                                          |
|                                                                                                   |
|    Natural-Language Goal + Initial Multi-View RGB Observations                                    |
|              ↓                                                                                    |
|    VLM (Qwen / Foundation Model)                                                                  |
|              ↓                                                                                    |
|    Natural-Language Functional Specification (JSON)                                               |
|    - VLM-local role IDs, qualitative properties, qualitative relations, candidate categories     |
|    - Visually proposed inspectable storage regions & search preference                            |
+---------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+---------------------------------------------------------------------------------------------------+
| 2. Deterministic Backend Canonicalizer & Region Resolver                                          |
|                                                                                                   |
|    - Maps natural-language properties → implemented geometric/semantic checkers (reviewed table) |
|    - Maps visually proposed regions → canonical actionable region IDs (reviewed alias table)      |
|    - Strict structural validation; fail-closed on unmapped concepts or internal inconsistencies   |
|    - Constructs immutable Functional Requirement Graph (G_F) & detector vocabulary                |
+---------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+---------------------------------------------------------------------------------------------------+
| 3. Downstream Grounding & Multi-Condition Planning                                                |
|                                                                                                   |
|    - Sequential inspection over resolved candidate regions (G_O acquisition)                      |
|    - Continuous geometric verification & compatibility witness extraction (phi*)                  |
|    - A* Symbolic and Geometric Task Planning → action_plan.json                                  |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. VLM Input Contract (Zero-Leakage)

### Allowed Inputs
The VLM specification prompt receives strictly:
1. **Natural-Language Task Goal**: The human task instruction (e.g. *"Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil. Search closed kitchen storage for anything still required."*).
2. **Initial Multi-View RGB Scene Captures**: 5 unoccluded camera views of the initial environment state (`overhead_camera`, `front_camera`, `side_camera`, `left_shoulder_camera`, `right_shoulder_camera`).
3. **Generic Schema and Instructions**: Explains abstract concept kinds (`OBJECT`, `REGION`, `FIXED_TARGET`), cardinalities, and output JSON structure.

### Strictly Forbidden Inputs
The VLM specification prompt MUST NEVER receive:
- **Ground-Truth (GT) Scene Information**: Object inventories, instance IDs (`soup_spoon_1`, `kettle_1`), object bounding boxes, 3D poses, or scene graphs ($G_O$).
- **Internal Checker APIs / Names**: Predicate names like `OPEN_CAVITY`, `ELONGATED_OBJECT`, `INSERTABLE_IN`, `REACHES_BOTTOM`, `PLANAR_SUPPORT`, `CAN_DRIVE_SCREW`, or numeric property identifiers (`total_length_m`, `cavity_depth_m`).
- **Canonical Search Region Identifiers**: Pre-enumerated lists of simulation region IDs (e.g., `D1`, `D2`, `C2`, `B1`, `C1`, `LEFT_DRAWER`, `RIGHT_DRAWER`, `TOOL_CABINET`).
- **Target Feasibility & Oracle Hints**: Feasibility labels, optimal inspection orders, or ground-truth action sequences.

---

## 3. VLM Output Contract (Natural Language Specification)

The VLM returns a structured JSON document containing qualitative descriptions:

```json
{
  "status": "SUPPORTED",
  "task_summary": "Prepare coffee and soup for two people.",
  "functional_roles": [
    {
      "id": "drink_receptacle",
      "entity_kind": "OBJECT",
      "function": "contain liquid coffee serving",
      "required_count": 2,
      "binding_policy": "DISTINCT",
      "candidate_categories": ["cup", "coffee mug"],
      "required_properties": ["open cavity container", "able to hold liquid"]
    },
    {
      "id": "stirring_tool",
      "entity_kind": "OBJECT",
      "function": "stir coffee in container",
      "required_count": 1,
      "binding_policy": "REUSABLE",
      "candidate_categories": ["spoon", "stirring stick"],
      "required_properties": ["elongated utensil", "long enough to reach bottom"]
    }
  ],
  "functional_relations": [
    {
      "subject_role": "stirring_tool",
      "relation": "must fit inside and reach bottom",
      "object_role": "drink_receptacle"
    }
  ],
  "interaction_groups": [
    {
      "id": "mix_coffee",
      "function": "stir",
      "tool_role": "stirring_tool",
      "target_role": "drink_receptacle",
      "required_target_count": 2,
      "reuse_policy": "SEQUENTIAL_REUSE_ALLOWED",
      "required_relations": ["fits inside", "reaches bottom"]
    }
  ],
  "inspectable_regions": [
    {
      "id": "storage_1",
      "label": "upper wall cupboard",
      "visual_description": "closed wall cabinet above counter",
      "reason": "storage area for drinkware"
    }
  ],
  "inspection_order": ["storage_1"],
  "unsupported_reason": ""
}
```

---

## 4. Deterministic Backend Canonicalization

Downstream backend code compiles the VLM's natural-language output deterministically and fails closed on invalid or ambiguous specifications:

1. **Property & Relation Mapping**:
   - Matches qualitative natural-language phrases to implemented geometric/semantic checkers using reviewed alias tables.
   - Example: `"open cavity"`, `"holds liquid"`, `"container"` $\to$ `OPEN_CAVITY`.
   - Example: `"long enough to reach bottom"`, `"touches bottom"` $\to$ `REACHES_BOTTOM`.
   - If an unmapped property is encountered, compilation fails with a clear diagnostic (`VLM_SPEC_FAILED`).
2. **Region Proposal Resolution**:
   - Matches visually proposed region descriptions (`"upper wall cupboard"`) against domain region alias tables $\to$ `C2`.
   - Unresolved proposals are recorded in diagnostics and excluded from actionable search.
   - The resulting $G_F$ `candidate_regions` contains **ONLY** successfully resolved VLM proposals. No fallback to the full canonical catalog is permitted.
3. **Strict Validation & No Semantic Repairs**:
   - Rejects `status == "UNSUPPORTED"`.
   - Strict equality between role cardinalities and operation target counts (no silent auto-repair).
   - Strict source-role cardinality validation.

---

## 5. Downstream Perception, Grounding, and Planning

1. **Targeted Object Detection**:
   - The semantic detector vocabulary is derived **strictly** from the canonical categories present in $G_F$ and their reviewed detector aliases.
2. **Sequential Search & $G_O$ Construction**:
   - The robot sequentially inspects only the regions proposed by the VLM.
   - Search exhaustion occurs when all VLM-proposed regions have been opened.
3. **Exact Compatibility Grounding ($\phi^*$)**:
   - Evaluates continuous geometric predicates across observed objects.
   - Binds physical witnesses to functional roles.
4. **Task & Motion Planning**:
   - Common A* symbolic and geometric planner solves for the minimal-cost action sequence (`action_plan.json`).
