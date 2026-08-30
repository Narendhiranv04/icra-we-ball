# VLM Functional Specification Interface Contract (Pass 3.6B.0)

**Canonicalization Version**: `phase3_6a7_2_1_v1`
**VLM Interface Implementation Frozen**: YES
**Pass 3.6B.0 Evaluator Preflight**: COMPLETE
**Phase 3 Frozen**: NO (Live audit & evaluation in Pass 3.6B)
**Ready for Phase 4**: NO

This document defines the authoritative scientific interface contract between the Vision-Language Model (VLM) specification provider and downstream perception, grounding, and task-and-motion planning (TAMP) components.

---

## 1. Architectural Overview

The goal of the functional-grounding TAMP pipeline is to enable a robot to perform complex, multi-stage manipulation tasks without pre-programmed instance identities, ground-truth physical dimensions, or hardcoded action sequences.

```
TASK + INITIAL MULTI-VIEW RGB
        ↓
       VLM (Qwen / Foundation Model)
        ↓
raw VLM output (raw_vlm_response)
        ↓
strict generic schema validation (validated_vlm_specification)
        ↓
lossless deterministic canonicalization (phase3_6a7_2_1_v1, canonicalization_trace)
        ↓
generic domain-independent runtime G_F validation (validate_runtime_gf)
        ↓
canonical G_F (with complete roles, relations, and operation groups)
        ↓
══════════════════════════════════════════════════════════════════════════
               VLM HAS NO ROLE BELOW THIS POINT
══════════════════════════════════════════════════════════════════════════
        ↓
G_O (sequential inspection) → grounding (phi*) → symbolic compiler → A*
```

*Note on Evaluation Separation*: Offline task/domain evaluation (e.g. assessing candidate $G_F$ recall/completeness against reference specifications) is strictly decoupled from runtime and performed via `gf_reference_evaluator.py:evaluate_gf_against_reference()`. The proposed-method runtime execution path contains zero task oracle lists or expected benchmark nouns. Representational differences (such as legacy Workshop role-function markers and Kitchen reusable point-count vs interval representation) are normalized strictly in offline evaluation code without modifying runtime representation. See Pass 3.6B.0 verification report for exact local test count.

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

## 4. Deterministic Backend Canonicalization (`phase3_6a7_2_v1`)

Downstream backend code compiles the VLM's natural-language output deterministically and fails closed on invalid or ambiguous specifications:

1. **Role Canonicalization & Entity-Kind Compatibility**:
   - Maps role function phrases to reviewed domain canonical roles:
     - **Kitchen**: `coffee_container` (`OBJECT`), `soup_container` (`OBJECT`), `coffee_stirrer` (`OBJECT`), `soup_eating_utensil` (`OBJECT`), `coffee_source` (`OBJECT`), `water_source` (`OBJECT`).
     - **Living Room**: `PERSONAL_CUP_SAUCER_REGION` (`REGION`), `SHARED_REMOTE_REGION` (`REGION`), `CUP_SAUCER_SET` (`OBJECT`), `REMOTE` (`OBJECT`), `SEATING_POSITION` (`FIXED_TARGET`), `SEATING_PAIR` (`FIXED_TARGET`).
     - **Workshop**: `driver` (`OBJECT`), `fastener` (`OBJECT`), `repair_target` (`FIXED_TARGET`).
   - Rejects mismatched `entity_kind` with `MalformedVLMSpecificationError`.

2. **Living Task-Anchors vs Discoverable Support Regions**:
   - Task anchors (`CUP_SAUCER_SET`, `REMOTE`, `SEATING_POSITION`, `SEATING_PAIR`) contain reviewed graph-anchor canonical categories matching production $G_O$ nodes, while preserving raw candidate categories and canonical graph categories.
   - Discoverable support regions (`PERSONAL_CUP_SAUCER_REGION`, `SHARED_REMOTE_REGION`) preserve strictly VLM-derived open-vocabulary candidate categories.

3. **Domain-Scoped Unary Checker Capabilities**:
   - Living Room supports **ONLY** `PLANAR_SUPPORT`. Kitchen checkers (`OPEN_CAVITY`, `ELONGATED_OBJECT`) fail closed with `UnsupportedCheckerCapabilityError` if referenced in Living Room.
   - All relations map strictly through `binary_relation_aliases`.

4. **Strict Interaction Group Schema Validation**:
   - `required_relations` is mandatory (`minItems=1`).
   - Paired `context_role` and `context_relations`: if `context_role` is present, `context_relations` must be non-empty; if `context_role` is absent, `context_relations` cannot be specified.
   - Zero silent dropping of interaction groups.

5. **Generic Runtime G_F Validation vs Offline Reference Evaluation**:
   - Runtime validation (`validate_runtime_gf`) verifies domain-independent graph structural integrity (unique role names, count $\ge 1$, valid entity kinds, endpoints in nodes, non-empty required relations) with zero expected task nouns or domain oracle lists.
   - Task/domain completeness evaluation is strictly offline via `gf_reference_evaluator.py:evaluate_gf_against_reference()`.

6. **Separate Provenance Layers**:
   - `raw_vlm_response`: Exact decoded JSON before validation.
   - `validated_vlm_specification`: Validated schema document.
   - `canonicalization_trace`: Detailed mappings of role IDs, categories, and predicates.
   - `vlm_canonicalization_version`: Fixed to `phase3_6a7_2_1_v1`.

---

## 5. Downstream Execution Boundary (Zero VLM Role)

Below $G_F$, the VLM plays zero role:
1. **Targeted Semantic Detection**: Open-vocabulary detector vocabulary is derived from $G_F$ canonical categories.
2. **Sequential Search & $G_O$ Construction**: The robot sequentially inspects only the regions in $G_F$.
3. **Exact Compatibility Grounding ($\phi^*$)**: Evaluates continuous physical geometry across observed candidates.
4. **Symbolic Compilation & A* Planning**: Compiles grounded scene into STRIPS problem and computes executable action plan.
