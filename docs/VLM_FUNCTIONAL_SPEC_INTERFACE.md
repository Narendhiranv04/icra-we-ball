# VLM Functional Specification Interface Contract (Pass 3.6A.3)

**Canonicalization Version**: `phase3_6a3_v1`

This document defines the authoritative scientific interface contract between the Vision-Language Model (VLM) specification provider and downstream perception, grounding, and task-and-motion planning (TAMP) components.

---

## 1. Architectural Overview

The goal of the functional-grounding TAMP pipeline is to enable a robot to perform complex, multi-stage manipulation tasks without pre-programmed instance identities, ground-truth physical dimensions, or hardcoded action sequences.

```
TASK + INITIAL MULTI-VIEW RGB
        ↓
       VLM (Qwen / Foundation Model)
        ↓
complete natural-language functional specification
        ↓
strict deterministic representation & canonicalization (phase3_6a3_v1)
        ↓
canonical G_F
        ↓
══════════════════════════════════════════════════════════════════════════
               VLM HAS NO ROLE BELOW THIS POINT
══════════════════════════════════════════════════════════════════════════
        ↓
G_O (sequential inspection) → grounding (phi*) → symbolic compiler → A*
```

---

## 2. VLM Input Contract (Zero-Leakage)

### Allowed Inputs
The VLM specification prompt receives strictly:
1. **Natural-Language Task Goal**: The human task instruction (e.g. *"Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil. Search closed kitchen storage for anything still required."*).
2. **Initial Multi-View RGB Scene Captures**:
   - Kitchen: 5 unoccluded camera views rendered at **1280×960** resolution (`overhead_camera`, `front_camera`, `side_camera`, `left_shoulder_camera`, `right_shoulder_camera`).
   - Living Room: 5 unoccluded camera views rendered at **1280×960** resolution (`l2_camera_front`, `l2_camera_close`, `l2_camera_top`, `l2_camera_left`, `l2_camera_right`).
   - Workshop: 3 logical unoccluded camera views (`initial_iso_left`, `initial_iso_right`, `initial_detail`).
3. **Generic Schema and Instructions**: Explains abstract concept kinds (`OBJECT`, `REGION`, `FIXED_TARGET`), binding policies (`DISTINCT`, `REUSABLE`, `SHARED`), unary vs binary relations, and output JSON structure.

### Strictly Forbidden Inputs (Zero-Leakage Boundary)
The VLM specification prompt MUST NEVER receive:
- **Ground-Truth (GT) Scene Information**: Object inventories, instance IDs (`soup_spoon_1`, `kettle_1`), object bounding boxes, 3D poses, or scene graphs ($G_O$).
- **Internal Checker APIs / Names**: Predicate names like `OPEN_CAVITY`, `ELONGATED_OBJECT`, `INSERTABLE_IN`, `REACHES_BOTTOM`, `PLANAR_SUPPORT`, `CAN_DRIVE_SCREW`, or numeric property identifiers (`total_length_m`, `cavity_depth_m`).
- **Canonical Search Region Identifiers**: Pre-enumerated lists of simulation region IDs (e.g., `D1`, `D2`, `C2`, `B1`, `C1`, `LEFT_DRAWER`, `RIGHT_DRAWER`, `TOOL_CABINET`).
- **Target Feasibility & Oracle Hints**: Feasibility labels, optimal inspection orders, or ground-truth action sequences.

---

## 3. VLM Output Contract (Natural Language Specification)

The VLM returns a structured JSON document containing qualitative descriptions adhering to the following schema:

```json
{
  "status": "SUPPORTED",
  "task_summary": "Prepare two coffees and two soups using available kitchenware.",
  "functional_roles": [
    {
      "id": "drink_receptacle",
      "entity_kind": "OBJECT",
      "function": "contain liquid coffee serving",
      "description": "receptacle for hot coffee",
      "required_count": 2,
      "binding_policy": "DISTINCT",
      "candidate_categories": ["cup", "coffee mug"],
      "visible_candidates": [],
      "required_properties": ["open cavity container", "able to hold liquid"]
    },
    {
      "id": "stirring_tool",
      "entity_kind": "OBJECT",
      "function": "stir coffee in container",
      "description": "slender tool to mix beverage",
      "required_count": 1,
      "binding_policy": "REUSABLE",
      "candidate_categories": ["spoon", "stirring stick"],
      "visible_candidates": [],
      "required_properties": ["elongated utensil"]
    }
  ],
  "functional_relations": [
    {
      "subject_role": "stirring_tool",
      "relation": "reaches bottom",
      "object_role": "drink_receptacle"
    }
  ],
  "interaction_groups": [
    {
      "id": "mix_drinks",
      "function": "stir",
      "tool_role": "stirring_tool",
      "target_role": "drink_receptacle",
      "required_target_count": 2,
      "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
      "required_relations": ["reaches bottom"]
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

### UNSUPPORTED Status Contract
If the VLM determines that the task cannot be represented with this abstraction:
- `status`: `"UNSUPPORTED"`
- `functional_roles`: `[]`
- `functional_relations`: `[]`
- `interaction_groups`: `[]`
- `inspectable_regions`: `[]`
- `inspection_order`: `[]`
- `unsupported_reason`: Non-empty diagnostic string.

---

## 4. Deterministic Backend Canonicalization (`phase3_6a3_v1`)

Downstream backend code compiles the VLM's natural-language output deterministically and fails closed on invalid or ambiguous specifications:

1. **Role Canonicalization (Local ID Independence)**:
   - Maps role function phrases to reviewed domain canonical roles ignoring local VLM IDs:
     - **Kitchen**: `coffee_container`, `soup_container`, `coffee_stirrer`, `soup_eating_utensil`, `coffee_source`, `water_source`.
     - **Living Room**: `personal_cup_saucer`, `shared_remote`.
     - **Workshop**: `CAN_DRIVE_SCREW`, `CAN_FASTEN`, `repair_target` (`FIXED_TARGET`).
2. **Unary Properties**:
   - `required_properties` contains **UNARY ONLY** physical properties of one role.
   - Evaluated against exact/alias predicate checkers (e.g. `OPEN_CAVITY`, `ELONGATED_OBJECT`, `PLANAR_SUPPORT`).
3. **Explicit Functional Relations**:
   - Top-level `functional_relations` specify explicit `subject_role`, `relation`, `object_role`.
   - Verified that both endpoints exist and relation maps deterministically (e.g. `COMPATIBLE_WITH`, `REACHES_TARGET`, `COMPATIBLE_WITH_TARGET`).
4. **Region Proposal Resolution**:
   - Matches visually proposed region descriptions (`"upper wall cupboard"`) against domain region alias tables $\to$ `C2`.
   - Local VLM IDs (e.g., `"c2"`) are completely ignored; resolution uses only natural language `label` and `visual_description`.
   - The resulting $G_F$ `candidate_regions` contains **ONLY** successfully resolved VLM proposals. No fallback to the full canonical catalog is permitted.
5. **Strict Validation & No Semantic Repairs**:
   - Fails closed on unmapped properties, relations, roles, or mismatched policies.
   - Attaches `vlm_canonicalization_version: "phase3_6a3_v1"` and full transformation provenance to $G_F$ metadata.

---

## 5. Downstream Execution Boundary (Zero VLM Role)

Below $G_F$, the VLM plays zero role:
1. **Targeted Semantic Detection**: Open-vocabulary detector vocabulary is derived from $G_F$ canonical categories.
2. **Sequential Search & $G_O$ Construction**: The robot sequentially inspects only the regions in $G_F$.
3. **Exact Compatibility Grounding ($\phi^*$)**: Evaluates continuous physical geometry across observed candidates.
4. **Symbolic Compilation & A* Planning**: Compiles grounded scene into STRIPS problem and computes executable action plan.
