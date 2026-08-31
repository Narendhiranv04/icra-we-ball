"""Regression tests for the natural-language Kitchen VLM functional specification path."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest
from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from mujoco_scenes.kitchen_vlm_functional_graph import (
    compile_vlm_functional_graph,
    resolve_kitchen_region_proposal,
    map_unary_property,
    map_binary_relation,
    map_kitchen_role_function,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMResponseValidationError,
    validate_kitchen_functional_specification,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
REGIONS = ("D1", "D2", "C2", "B1", "C1")


def natural_kitchen_spec() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two coffees and two soups using available kitchenware.",
        "functional_roles": [
            {
                "id": "drink_receptacle",
                "entity_kind": "OBJECT",
                "function": "contain one coffee serving",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup", "coffee mug"],
                "visible_candidates": [],
                "required_properties": ["open cavity", "capable of containing liquid"],
            },
            {
                "id": "soup_receptacle",
                "entity_kind": "OBJECT",
                "function": "contain one soup serving",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["bowl", "soup bowl"],
                "visible_candidates": [],
                "required_properties": ["open cavity", "holds liquid"],
            },
            {
                "id": "mixing_implement",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["spoon", "metal spoon"],
                "visible_candidates": [],
                "required_properties": ["elongated", "slender"],
            },
            {
                "id": "soup_implement",
                "entity_kind": "OBJECT",
                "function": "serve with soup",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["soup_spoon", "soup spoon"],
                "visible_candidates": [],
                "required_properties": ["elongated object"],
            },
            {
                "id": "water_source",
                "entity_kind": "OBJECT",
                "function": "provide water for coffee",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["kettle", "water jug"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "coffee_source",
                "entity_kind": "OBJECT",
                "function": "provide coffee material",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["coffee_jar", "coffee jar"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "mixing_implement",
                "relation": "reaches bottom",
                "object_role": "drink_receptacle",
            },
            {
                "subject_role": "soup_implement",
                "relation": "fits into",
                "object_role": "soup_receptacle",
            },
        ],
        "interaction_groups": [
            {
                "id": "mix_drinks",
                "function": "stir",
                "tool_role": "mixing_implement",
                "target_role": "drink_receptacle",
                "required_target_count": 2,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["fits inside", "reaches bottom"],
            },
            {
                "id": "equip_soups",
                "function": "provide utensil",
                "tool_role": "soup_implement",
                "target_role": "soup_receptacle",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["fits inside"],
            },
        ],
        "cross_group_reuse_allowed": False,
        "inspectable_regions": [
            {
                "id": "reg_1",
                "label": "upper wall cupboard",
                "visual_description": "cupboard above counter",
                "reason": "may hold vessels",
            },
            {
                "id": "reg_2",
                "label": "upper drawer",
                "visual_description": "top drawer below counter",
                "reason": "may hold utensils",
            },
        ],
        "inspection_order": ["reg_1", "reg_2"],
        "unsupported_reason": "",
    }


qwen_graph = natural_kitchen_spec


class FakeTransport:
    def __init__(self, document: dict):
        self.document = document
        self.payloads = []

    def complete(self, payload):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": json.dumps(self.document)}}]}


def test_natural_kitchen_spec_canonicalizes_properties_and_regions():
    contract, vocabularies, trace = compile_vlm_functional_graph(
        natural_kitchen_spec(),
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )

    assert set(contract["roles"]) == {
        "coffee_container", "soup_container", "coffee_stirrer", "soup_eating_utensil",
        "water_source", "coffee_source",
    }
    assert contract["specification_source"] == "qwen_vlm_natural_language_specification"
    assert contract["roles"]["coffee_container"]["unary_geometry"][0]["predicate"] == "OPEN_CAVITY"
    assert contract["roles"]["coffee_stirrer"]["unary_geometry"][0]["predicate"] == "ELONGATED_OBJECT"
    
    # Check relations are canonicalized
    rel_preds = [r["predicate"] for r in contract["relations"]]
    assert "INSERTABLE_IN" in rel_preds
    assert "REACHES_BOTTOM" in rel_preds

    # Check candidate regions contain ONLY resolved regions
    assert trace["candidate_regions"] == ["C2", "D1"]
    assert trace["inspection_order"] == ["C2", "D1"]


def test_local_id_collision_independence():
    """VLM local ID 'c2' must not trick the resolver if visual label says 'upper drawer'."""
    proposal = {
        "id": "c2",
        "label": "upper drawer",
        "visual_description": "top drawer below counter",
        "reason": "storage",
    }
    resolved = resolve_kitchen_region_proposal(proposal)
    assert resolved == "D1", f"Expected D1 (from label), got {resolved} (confused by id: c2)"


def test_binding_policy_preservation():
    """Raw VLM binding_policy must be preserved into G_F without modification."""
    spec = natural_kitchen_spec()
    spec["functional_roles"][0]["binding_policy"] = "DISTINCT"
    spec["functional_roles"][2]["binding_policy"] = "REUSABLE"

    contract, vocabularies, trace = compile_vlm_functional_graph(
        spec,
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )
    assert contract["roles"]["coffee_container"]["vlm_binding_policy"] == "DISTINCT"
    assert contract["roles"]["coffee_stirrer"]["vlm_binding_policy"] == "REUSABLE"


def test_entity_kind_preservation():
    """Raw VLM entity_kind must be preserved without coercion."""
    spec = natural_kitchen_spec()
    contract, vocabularies, trace = compile_vlm_functional_graph(
        spec,
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )
    assert contract["roles"]["coffee_container"]["entity_kind"] == "OBJECT"


def test_object_nouns_do_not_prove_geometry():
    """Object nouns like 'spoon' or 'cup' in required_properties must not map to physical geometry."""
    with pytest.raises(VLMSpecificationError, match="cannot be mapped to any active physical unary property"):
        spec = natural_kitchen_spec()
        spec["functional_roles"][2]["required_properties"] = ["spoon", "metal spoon"]
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_unique_property_mapping():
    """Ambiguous or unmapped properties must fail closed."""
    assert map_unary_property("open cavity") == "OPEN_CAVITY"
    assert map_unary_property("elongated object") == "ELONGATED_OBJECT"
    assert map_unary_property("completely unmapped non-physical concept") is None


def test_inspection_order_resolves_through_local_id_map():
    spec = natural_kitchen_spec()
    spec["inspectable_regions"] = [
        {"id": "loc_cupboard", "label": "upper wall cupboard", "visual_description": "cupboard above counter", "reason": "cups"},
        {"id": "loc_drawer", "label": "upper drawer", "visual_description": "top drawer below counter", "reason": "spoons"},
    ]
    spec["inspection_order"] = ["loc_drawer", "loc_cupboard"]

    contract, vocabularies, trace = compile_vlm_functional_graph(
        spec,
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )
    assert trace["inspection_order"] == ["D1", "C2"]


def test_adapter_outgoing_payload_has_zero_checker_and_region_leaks(tmp_path):
    image = tmp_path / "initial.png"
    image.write_bytes(PNG_1X1)
    transport = FakeTransport(natural_kitchen_spec())
    adapter = FMAdapter(model="qwen", transport=transport)

    result = adapter.generate_kitchen_functional_graph(
        "Prepare two coffees and two soups.", observation_images=[image]
    )

    assert result == natural_kitchen_spec()
    assert adapter.metrics.total_calls == 1
    assert len(transport.payloads) == 1

    payload_json = json.dumps(transport.payloads[0])
    
    # Assert NO checker names in payload
    forbidden_checkers = [
        "OPEN_CAVITY", "ELONGATED_OBJECT", "INSERTABLE_IN", "REACHES_BOTTOM",
        "PLANAR_SUPPORT", "CAN_DRIVE_SCREW", "CAN_FASTEN", "total_length_m", "cavity_depth_m"
    ]
    for checker in forbidden_checkers:
        assert checker not in payload_json, f"Information leak detected: {checker} in payload"

    # Assert NO canonical region IDs in payload
    forbidden_regions = ["D1", "D2", "C2", "B1", "C1", "LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"]
    for reg in forbidden_regions:
        assert f'"{reg}"' not in payload_json, f"Information leak detected: region {reg} in payload"


def test_unsupported_task_fails_closed():
    spec = natural_kitchen_spec()
    spec["status"] = "UNSUPPORTED"
    spec["unsupported_reason"] = "Cannot serve food without ingredients"
    spec["functional_roles"] = []
    spec["functional_relations"] = []
    spec["interaction_groups"] = []
    spec["inspectable_regions"] = []
    spec["inspection_order"] = []

    with pytest.raises(VLMSpecificationError, match="VLM marked task unsupported"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_inconsistent_role_count_fails_closed():
    spec = natural_kitchen_spec()
    # Mismatch: role count is 1, but operation requires 2
    spec["functional_roles"][0]["required_count"] = 1
    spec["interaction_groups"][0]["required_target_count"] = 2

    with pytest.raises(VLMSpecificationError, match="has required_count 1, but group requires 2"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_unresolved_region_proposal_excluded_from_candidate_regions():
    spec = natural_kitchen_spec()
    spec["inspectable_regions"] = [
        {"id": "reg_1", "label": "upper wall cupboard", "visual_description": "cupboard", "reason": "storage"},
        {"id": "reg_2", "label": "bookshelf in bedroom", "visual_description": "bookshelf", "reason": "storage"},
    ]
    spec["inspection_order"] = ["reg_1", "reg_2"]

    contract, vocabularies, trace = compile_vlm_functional_graph(
        spec,
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )

    # Only C2 should be in candidate_regions
    assert trace["candidate_regions"] == ["C2"]
    assert trace["inspection_order"] == ["C2"]
    assert len(trace["unresolved_proposals"]) == 1
    assert trace["unresolved_proposals"][0]["label"] == "bookshelf in bedroom"


def test_no_full_catalog_fallback():
    spec = natural_kitchen_spec()
    # VLM proposes only 1 region
    spec["inspectable_regions"] = [
        {"id": "reg_1", "label": "upper wall cupboard", "visual_description": "cupboard", "reason": "storage"},
    ]
    spec["inspection_order"] = ["reg_1"]

    contract, vocabularies, trace = compile_vlm_functional_graph(
        spec,
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )

    # Must NOT fall back to all 5 regions
    assert trace["candidate_regions"] == ["C2"]
    assert len(trace["candidate_regions"]) == 1


# ============================================================
# P3-E Fail-Closed and Lossless Negative & Invariant Tests
# ============================================================


def test_unmapped_binary_relation_fails_closed():
    """Step 1 & Step 6: Unknown binary relation must raise UnmappedFunctionalConceptError, never fabricate INSERTABLE_IN."""
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError

    spec = natural_kitchen_spec()
    # Replace valid relation with an unmapped phrase
    spec["functional_relations"][0]["relation"] = "must be placed adjacent to"

    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any active Kitchen binary predicate") as exc_info:
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )
    assert exc_info.value.category == "UNMAPPED_FUNCTIONAL_CONCEPT"


def test_broad_keyword_relation_not_fabricated():
    """Step 2: Broad keywords like 'reach', 'require', 'stir with' must not fabricate INSERTABLE_IN."""
    assert map_binary_relation("stir with coffee") is None
    assert map_binary_relation("requires water") is None
    assert map_binary_relation("reaches near table") is None


def test_unmapped_role_fails_closed():
    """Step 3: Unmapped task-required role must raise UnmappedFunctionalConceptError."""
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError

    spec = natural_kitchen_spec()
    # Add an unmapped role
    spec["functional_roles"].append({
        "id": "vacuum_tool",
        "entity_kind": "OBJECT",
        "function": "vacuum the carpet and clean floor",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["vacuum_cleaner"],
        "visible_candidates": [],
        "required_properties": [],
    })

    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any canonical Kitchen role") as exc_info:
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )
    assert exc_info.value.category == "UNMAPPED_FUNCTIONAL_CONCEPT"


def test_duplicate_canonical_role_collision_fails_closed():
    """Step 4 & Step 14D: Multiple raw role IDs mapping to the same canonical role must fail closed, not max()/sum()."""
    from mujoco_scenes.functional_tamp_pipeline.errors import AmbiguousCanonicalizationError

    spec = natural_kitchen_spec()
    # Add duplicate raw role for coffee_container with count=3
    spec["functional_roles"].append({
        "id": "second_coffee_receptacle",
        "entity_kind": "OBJECT",
        "function": "hold coffee serving",
        "required_count": 3,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["coffee_cup"],
        "visible_candidates": [],
        "required_properties": ["open cavity"],
    })

    with pytest.raises(AmbiguousCanonicalizationError, match="Multiple distinct raw roles .* map to the same canonical role 'coffee_container'") as exc_info:
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )
    assert exc_info.value.category == "AMBIGUOUS_CANONICALIZATION"


def test_unmapped_required_property_fails_closed():
    """Step 5: Unmapped required property must raise UnmappedFunctionalConceptError."""
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError

    spec = natural_kitchen_spec()
    spec["functional_roles"][0]["required_properties"].append("heat resistant ceramic material")

    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any active physical unary property") as exc_info:
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )
    assert exc_info.value.category == "UNMAPPED_FUNCTIONAL_CONCEPT"


def test_undeclared_relation_endpoint_fails_closed():
    """Step 6: Relation with undeclared endpoint must raise MalformedVLMSpecificationError."""
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError

    spec = natural_kitchen_spec()
    spec["functional_relations"].append({
        "subject_role": "ghost_role_1",
        "relation": "fits inside",
        "object_role": "drink_receptacle",
    })

    with pytest.raises(MalformedVLMSpecificationError, match="undeclared"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_unsupported_operation_group_pair_fails_closed():
    """Step 7: Operation group with unsupported pair must raise MalformedVLMSpecificationError."""
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError

    spec = natural_kitchen_spec()
    spec["interaction_groups"].append({
        "id": "unsupported_group",
        "function": "stir coffee",
        "tool_role": "water_source",
        "target_role": "coffee_source",
        "required_target_count": 1,
        "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
        "required_relations": ["fits inside"],
    })

    with pytest.raises(MalformedVLMSpecificationError, match="Unsupported Kitchen operation group tool/target pair"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_duplicate_operation_group_collision_fails_closed():
    """Step 7: Multiple raw operation groups mapping to same canonical group must fail closed."""
    from mujoco_scenes.functional_tamp_pipeline.errors import AmbiguousCanonicalizationError

    spec = natural_kitchen_spec()
    spec["interaction_groups"].append({
        "id": "second_mix_drinks",
        "function": "stir coffee again",
        "tool_role": "mixing_implement",
        "target_role": "drink_receptacle",
        "required_target_count": 2,
        "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
        "required_relations": ["fits inside"],
    })

    with pytest.raises(AmbiguousCanonicalizationError, match="Multiple raw operation groups .* map to the same canonical operation group 'coffee_stirring'"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_target_count_clipping_prevented():
    """Step 8 & Step 14C: required_target_count exceeding target role count must fail closed, never clipped."""
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError

    spec = natural_kitchen_spec()
    # Target role 'drink_receptacle' has count 2, group requires 3
    spec["interaction_groups"][0]["required_target_count"] = 3

    with pytest.raises(MalformedVLMSpecificationError, match="has required_count 2, but group requires 3"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_empty_operation_relations_fails_closed():
    """Step 9: Operation group with empty required_relations must fail closed, never default to INSERTABLE_IN."""
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError

    spec = natural_kitchen_spec()
    spec["interaction_groups"][0]["required_relations"] = []

    with pytest.raises(MalformedVLMSpecificationError, match="required_relations"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_unmapped_operation_relation_fails_closed():
    """Step 9: Unmapped relation in operation group must raise UnmappedFunctionalConceptError."""
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError

    spec = natural_kitchen_spec()
    spec["interaction_groups"][0]["required_relations"] = ["heats liquid quickly"]

    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any active Kitchen predicate"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_invalid_raw_counts_and_policies_fail_closed():
    """Step 10: Invalid counts and unknown policies must raise MalformedVLMSpecificationError without self-repair."""
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError

    # 1. Invalid required_count (0)
    spec1 = natural_kitchen_spec()
    spec1["functional_roles"][0]["required_count"] = 0
    with pytest.raises(MalformedVLMSpecificationError, match="required_count"):
        compile_vlm_functional_graph(spec1, task_instruction="Task", observable_regions=REGIONS)

    # 2. Unknown binding policy
    spec2 = natural_kitchen_spec()
    spec2["functional_roles"][0]["binding_policy"] = "UNKNOWN_BINDING"
    with pytest.raises(MalformedVLMSpecificationError, match="binding_policy"):
        compile_vlm_functional_graph(spec2, task_instruction="Task", observable_regions=REGIONS)

    # 3. Unknown usage policy
    spec3 = natural_kitchen_spec()
    spec3["interaction_groups"][0]["usage_policy"] = "UNKNOWN_USAGE"
    with pytest.raises(MalformedVLMSpecificationError, match="usage_policy"):
        compile_vlm_functional_graph(spec3, task_instruction="Task", observable_regions=REGIONS)


def test_cardinality_and_reusable_policy_preservation():
    """Step 14A, 14B, 14E: Raw counts, target counts, and reusable policies are exactly preserved."""
    spec = natural_kitchen_spec()
    spec["functional_roles"][0]["required_count"] = 2
    spec["functional_roles"][2]["required_count"] = 1
    spec["functional_roles"][2]["binding_policy"] = "REUSABLE"
    spec["interaction_groups"][0]["required_target_count"] = 2
    spec["interaction_groups"][0]["usage_policy"] = "SEQUENTIAL_REUSE_ALLOWED"

    contract, _, trace = compile_vlm_functional_graph(
        spec, task_instruction="Task", observable_regions=REGIONS
    )

    assert contract["roles"]["coffee_container"]["count"] == 2
    assert contract["roles"]["coffee_stirrer"]["count"] == 1
    assert contract["roles"]["coffee_stirrer"]["vlm_binding_policy"] == "REUSABLE"
    assert contract["operation_groups"]["coffee_stirring"]["required_target_count"] == 2
    assert contract["operation_groups"]["coffee_stirring"]["usage_policy"]["mode"] == "sequential_reuse_allowed"

    # Concept accounting trace verified
    assert "concept_accounting" in trace
    assert trace["concept_accounting"]["roles"]["drink_receptacle"]["status"] == "PRESERVED"
    assert trace["concept_accounting"]["operation_groups"][0]["status"] == "PRESERVED"


# ============================================================
# P3-E.1 Lexical Precision, Function Semantics & Provenance Tests
# ============================================================


def test_binary_relation_short_fragments_fail_closed():
    """P3-E.1 Step 1: Binary relation reverse short fragments must return None, not match longer aliases."""
    assert map_binary_relation("fit") is None
    assert map_binary_relation("inside") is None
    assert map_binary_relation("bottom") is None
    assert map_binary_relation("reach") is None

    # Valid richer phrases must continue to map accurately
    assert map_binary_relation("the utensil must fit inside the vessel") == "INSERTABLE_IN"
    assert map_binary_relation("the spoon must reach the bottom of the bowl") == "REACHES_BOTTOM"


def test_unary_property_short_fragments_fail_closed():
    """P3-E.1 Step 2: Unary property reverse short fragments must return None."""
    assert map_unary_property("open") is None
    assert map_unary_property("shape") is None

    # Valid richer phrases must continue to map accurately
    assert map_unary_property("must have an open cavity") == "OPEN_CAVITY"
    assert map_unary_property("must have an elongated shape") == "ELONGATED_OBJECT"


def test_role_alias_short_fragments_fail_closed():
    """P3-E.1 Step 3: Generic isolated words must not match a specific role merely via reverse containment."""
    assert map_kitchen_role_function("serving") is None
    assert map_kitchen_role_function("vessel") is None
    assert map_kitchen_role_function("material") is None
    assert map_kitchen_role_function("individual") is None


def test_interaction_group_function_validation_success():
    """P3-E.1 Step 4A & 4B: Valid interaction group function semantics succeed."""
    spec = natural_kitchen_spec()
    contract, _, trace = compile_vlm_functional_graph(
        spec, task_instruction="Task", observable_regions=REGIONS
    )
    assert "coffee_stirring" in contract["operation_groups"]
    assert "soup_serving" in contract["operation_groups"]


def test_interaction_group_unmapped_function_fails_closed():
    """P3-E.1 Step 4C: Unmapped interaction group function fails closed."""
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError

    spec = natural_kitchen_spec()
    spec["interaction_groups"][0]["function"] = "hammer a nail into the table"

    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any active Kitchen operation group"):
        compile_vlm_functional_graph(spec, task_instruction="Task", observable_regions=REGIONS)


def test_interaction_group_contradictory_function_fails_closed():
    """P3-E.1 Step 4D: Contradiction between group function semantics and tool/target endpoints fails closed."""
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError

    spec = natural_kitchen_spec()
    # Coffee stirring function assigned to soup spoon -> soup bowl endpoints
    spec["interaction_groups"][1]["function"] = "stir coffee thoroughly"

    with pytest.raises(MalformedVLMSpecificationError, match="contradicts tool/target endpoint pair"):
        compile_vlm_functional_graph(spec, task_instruction="Task", observable_regions=REGIONS)


def test_concept_accounting_complete_coverage_on_ideal_k1():
    """P3-E.1 Step 5 & Step 8: Full concept accounting coverage on ideal K1 fixture."""
    from mujoco_scenes.functional_tamp_pipeline.tests.test_ideal_fixtures import FIXTURES_DIR

    k1_path = FIXTURES_DIR / "kitchen_K1.json"
    k1_doc = json.loads(k1_path.read_text(encoding="utf-8"))

    contract, _, trace = compile_vlm_functional_graph(
        k1_doc, task_instruction="Task", observable_regions=REGIONS
    )

    accounting = trace["concept_accounting"]
    # 1. Exact raw role count accounting
    assert len(accounting["roles"]) == len(k1_doc["functional_roles"])
    for r in k1_doc["functional_roles"]:
        assert r["id"] in accounting["roles"]
        assert accounting["roles"][r["id"]]["status"] == "PRESERVED"

    # 2. Exact raw property count accounting
    total_raw_props = sum(len(r.get("required_properties", [])) for r in k1_doc["functional_roles"])
    assert len(accounting["properties"]) == total_raw_props

    # 3. Exact raw relation count accounting
    assert len(accounting["relations"]) == len(k1_doc["functional_relations"])

    # 4. Exact raw operation group count and function accounting
    assert len(accounting["operation_groups"]) == len(k1_doc["interaction_groups"])
    for op_row in accounting["operation_groups"]:
        assert op_row["status"] == "PRESERVED"
        assert op_row["function_mapping_status"] == "PRESERVED"
        assert bool(op_row["raw_function"])
        assert bool(op_row["canonical_function"])


def test_kitchen_canonicalizer_version_provenance():
    """P3-E.1 Step 7: Kitchen canonicalizer version is bumped and matches metadata trace."""
    from mujoco_scenes.kitchen_vlm_functional_graph import KITCHEN_VLM_CANONICALIZATION_VERSION
    from mujoco_scenes.functional_tamp_pipeline.tests.test_ideal_fixtures import MockFMAdapter, FIXTURES_DIR

    assert KITCHEN_VLM_CANONICALIZATION_VERSION == "phase3_p3e_1_v1"
    assert KITCHEN_VLM_CANONICALIZATION_VERSION != "phase3_6a7_2_1_v1"

    k1_doc = json.loads((FIXTURES_DIR / "kitchen_K1.json").read_text(encoding="utf-8"))
    adapter = MockFMAdapter(k1_doc)
    gf = VLMSpecProvider._kitchen("Task", [], adapter=adapter)

    assert gf.metadata["vlm_canonicalization_version"] == "phase3_p3e_1_v1"
    assert gf.metadata["canonicalization_trace"]["vlm_canonicalization_version"] == "phase3_p3e_1_v1"
    assert (
        gf.metadata["canonicalization_trace"]["vlm_canonicalization_version"]
        == gf.metadata["vlm_canonicalization_version"]
    )


