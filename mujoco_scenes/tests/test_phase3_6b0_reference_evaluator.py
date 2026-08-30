"""Dedicated unit and contract test suite for Pass 3.6B.0 Fair G_F Reference Evaluator.

Verifies:
1. Workshop legacy role-function marker normalization (evaluation-only).
2. Detection of genuine physical unary requirement mismatches.
3. Kitchen reusable role point-count vs interval semantic cardinality compatibility.
4. Out-of-range cardinality rejection and invalid binding policy rejection.
5. Operation-group errors are strictly caught and not hidden by cardinality compatibility.
6. 4-tuple relation comparison with expected polarity.
7. Extra structure handling (precision < 1.0, reference_complete=True, exact_structural_match=False).
8. Missing structure handling (recalls < 1.0, reference_complete=False).
9. Semantic operation group matching with different group IDs.
10. Ambiguous operation group match detection and reporting.
11. Cross-group reuse mismatch detection.
12. same_tool_must_cover_all_targets comparison.
13. Open-vocabulary category lexical difference is diagnostic-only (does not fail reference_complete).
14. Complete graph non-mutation invariance for candidate and reference graphs.
15. Downstream compilers non-mutation invariance across Kitchen, Living Room, and Workshop.
16. Static check that gf_reference_evaluator is strictly offline and never imported by runtime.
"""

from __future__ import annotations

import copy
import sys
import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import (
    GFReferenceEvaluationResult,
    evaluate_gf_against_reference,
)
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLMSpecProvider,
    VLM_CANONICALIZATION_VERSION,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    validate_kitchen_functional_specification,
    validate_requirement_response,
)
from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider


class MockFMAdapter:
    """Deterministic mock adapter returning a fixed JSON document payload."""

    def __init__(self, doc: dict, raw_doc: dict | None = None):
        self._doc = doc
        self.last_raw_requirement_response = raw_doc or doc
        self.last_validated_requirement_response = doc
        self.last_raw_kitchen_graph_response = raw_doc or doc
        self.last_validated_kitchen_graph_response = doc
        self.last_raw_inspection_response = None
        self.last_raw_response = raw_doc or doc
        self.last_observation_images = []
        self.metrics = type("Metrics", (), {"total_calls": 1, "requirement_calls": 1, "inspection_calls": 0})()

    def generate_task_requirements(self, task_instruction: str, *, observation_images: list | None = None) -> dict:
        return validate_requirement_response(copy.deepcopy(self._doc))

    def generate_kitchen_functional_graph(self, task_instruction: str, *, observation_images: list | None = None) -> dict:
        return validate_kitchen_functional_specification(copy.deepcopy(self._doc))

    def generate_inspection_priors(self, task_instruction: str, search_region_descriptors: Any = None, *, observation_images: list | None = None) -> dict:
        return {"inspectable_regions": [], "inspection_order": [], "confidence": 1.0}


def _valid_workshop_raw_vlm() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Perform assembly with tool and fastener",
        "functional_roles": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive screws into hole",
                "description": "tool for driving screws",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [{"label": "driver", "visual_description": "black tool"}],
                "required_properties": [],
            },
            {
                "id": "fastener_part",
                "entity_kind": "OBJECT",
                "function": "threaded screw to join parts",
                "description": "fastener part",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "visible_candidates": [{"label": "screw", "visual_description": "silver screw"}],
                "required_properties": [],
            },
            {
                "id": "target_joint",
                "entity_kind": "FIXED_TARGET",
                "function": "workbench hole for repair",
                "description": "repair target hole",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["hole"],
                "visible_candidates": [{"label": "hole", "visual_description": "recess"}],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "driver_tool", "relation": "compatible with", "object_role": "fastener_part"},
            {"subject_role": "driver_tool", "relation": "reaches target", "object_role": "target_joint"},
            {"subject_role": "fastener_part", "relation": "threads into target hole", "object_role": "target_joint"},
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


def _valid_kitchen_raw_vlm() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two cups of coffee and two bowls of soup",
        "functional_roles": [
            {
                "id": "coffee_mug",
                "entity_kind": "OBJECT",
                "function": "contain an individual serving of coffee",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee mug", "cup"],
                "visible_candidates": [
                    {"label": "mug_1", "visual_description": "white mug"},
                    {"label": "mug_2", "visual_description": "blue mug"},
                ],
                "required_properties": ["open cavity"],
            },
            {
                "id": "soup_bowl",
                "entity_kind": "OBJECT",
                "function": "contain an individual serving of soup",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["soup bowl", "bowl"],
                "visible_candidates": [
                    {"label": "bowl_1", "visual_description": "ceramic bowl"},
                    {"label": "bowl_2", "visual_description": "glass bowl"},
                ],
                "required_properties": ["open cavity"],
            },
            {
                "id": "spoon_tool",
                "entity_kind": "OBJECT",
                "function": "stir coffee in container",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["spoon", "stirrer"],
                "visible_candidates": [{"label": "spoon", "visual_description": "metal spoon"}],
                "required_properties": ["elongated object"],
            },
            {
                "id": "soup_spoon_tool",
                "entity_kind": "OBJECT",
                "function": "provide a suitable eating utensil for each soup bowl",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["soup spoon", "spoon"],
                "visible_candidates": [
                    {"label": "soup_spoon_1", "visual_description": "large spoon 1"},
                    {"label": "soup_spoon_2", "visual_description": "large spoon 2"},
                ],
                "required_properties": ["elongated object"],
            },
            {
                "id": "coffee_jar",
                "entity_kind": "OBJECT",
                "function": "provide coffee material",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee jar", "coffee container"],
                "visible_candidates": [{"label": "jar", "visual_description": "glass jar with beans"}],
                "required_properties": [],
            },
            {
                "id": "water_kettle",
                "entity_kind": "OBJECT",
                "function": "provide hot water",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["kettle", "water jug"],
                "visible_candidates": [{"label": "kettle", "visual_description": "electric kettle"}],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "spoon_tool", "relation": "insertable in", "object_role": "coffee_mug"},
            {"subject_role": "spoon_tool", "relation": "reaches bottom", "object_role": "coffee_mug"},
            {"subject_role": "soup_spoon_tool", "relation": "insertable in", "object_role": "soup_bowl"},
            {"subject_role": "soup_spoon_tool", "relation": "reaches bottom", "object_role": "soup_bowl"},
        ],
        "interaction_groups": [
            {
                "id": "coffee_stirring",
                "function": "stir coffee",
                "tool_role": "spoon_tool",
                "target_role": "coffee_mug",
                "required_target_count": 2,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["insertable in", "reaches bottom"],
            },
            {
                "id": "soup_serving",
                "function": "serve soup",
                "tool_role": "soup_spoon_tool",
                "target_role": "soup_bowl",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["insertable in", "reaches bottom"],
            },
        ],
        "cross_group_reuse_allowed": False,
        "inspectable_regions": [
            {"id": "drawer_top", "label": "top kitchen drawer", "visual_description": "wooden upper drawer", "reason": "utensil storage"}
        ],
        "inspection_order": ["drawer_top"],
        "unsupported_reason": "",
    }


# ===========================================================================
# 1. Workshop Legacy Marker Normalization & Physical Unary Verification
# ===========================================================================
def test_workshop_perfect_equivalence_with_legacy_marker_normalization():
    """Real GT Workshop reference vs synthetic canonical VLM G_F."""
    ref_gf = GTSpecProvider().provide("workshop", "fasten joint")
    assert "CAN_DRIVE_SCREW" in ref_gf.nodes["driver"].unary_predicates
    assert "CAN_FASTEN" in ref_gf.nodes["fastener"].unary_predicates

    adapter = MockFMAdapter(_valid_workshop_raw_vlm())
    provider = FMRequirementProvider(fm_adapter=adapter)
    cand_gf = VLMSpecProvider()._workshop("fasten joint", [], provider=provider)

    assert cand_gf.nodes["driver"].unary_predicates == ()
    assert cand_gf.nodes["fastener"].unary_predicates == ()

    result = evaluate_gf_against_reference(cand_gf, ref_gf)

    assert result.reference_complete is True
    assert result.exact_structural_match is True
    assert result.structurally_complete is True
    assert "unary_predicates" not in result.role_attribute_mismatches.get("driver", {})
    assert "unary_predicates" not in result.role_attribute_mismatches.get("fastener", {})
    assert "driver" in result.role_normalization_diagnostics
    assert "CAN_DRIVE_SCREW" in result.role_normalization_diagnostics["driver"]["ignored_legacy_role_function_markers"]
    assert "fastener" in result.role_normalization_diagnostics
    assert "CAN_FASTEN" in result.role_normalization_diagnostics["fastener"]["ignored_legacy_role_function_markers"]


def test_workshop_genuine_physical_unary_mismatch_fails():
    """If a candidate lacks a genuine physical unary requirement, it must still fail."""
    ref_gf = GTSpecProvider().provide("workshop", "fasten joint")
    # Add a genuine physical unary predicate to reference (e.g. ELONGATED_OBJECT on driver)
    driver_role = ref_gf.nodes["driver"]
    new_driver = FunctionalRole(
        name=driver_role.name,
        entity_kind=driver_role.entity_kind,
        count=driver_role.count,
        semantic_categories=driver_role.semantic_categories,
        unary_predicates=("CAN_DRIVE_SCREW", "ELONGATED_OBJECT"),
        binding_policy=driver_role.binding_policy,
        verification_mode=driver_role.verification_mode,
    )
    mod_nodes = dict(ref_gf.nodes)
    mod_nodes["driver"] = new_driver
    ref_with_genuine_unary = FunctionalRequirementGraph(
        domain=ref_gf.domain,
        task_instruction=ref_gf.task_instruction,
        nodes=mod_nodes,
        relations=ref_gf.relations,
        operation_groups=ref_gf.operation_groups,
    )

    adapter = MockFMAdapter(_valid_workshop_raw_vlm())
    provider = FMRequirementProvider(fm_adapter=adapter)
    cand_gf = VLMSpecProvider()._workshop("fasten joint", [], provider=provider)

    result = evaluate_gf_against_reference(cand_gf, ref_with_genuine_unary)
    assert result.reference_complete is False
    assert "driver" in result.role_attribute_mismatches
    assert "unary_predicates" in result.role_attribute_mismatches["driver"]
    assert result.role_attribute_mismatches["driver"]["unary_predicates"]["reference"] == ["ELONGATED_OBJECT"]


# ===========================================================================
# 2. Kitchen Cardinality Compatibility (Point-Count vs Interval)
# ===========================================================================
def test_kitchen_reusable_point_count_compatible_with_gt_interval():
    """Candidate coffee_stirrer count=1 [1,1] REUSABLE vs GT [1,2] REUSABLE."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    assert ref_gf.nodes["coffee_stirrer"].minimum_count == 1
    assert ref_gf.nodes["coffee_stirrer"].maximum_count == 2
    assert ref_gf.nodes["coffee_stirrer"].binding_policy == "REUSABLE"

    adapter = MockFMAdapter(_valid_kitchen_raw_vlm())
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    assert cand_gf.nodes["coffee_stirrer"].minimum_count == 1
    assert cand_gf.nodes["coffee_stirrer"].maximum_count == 1
    assert cand_gf.nodes["coffee_stirrer"].binding_policy == "REUSABLE"

    result = evaluate_gf_against_reference(cand_gf, ref_gf)

    assert result.reference_complete is True
    stirrer_diag = result.role_cardinality_diagnostics["coffee_stirrer"]
    assert stirrer_diag["cardinality_compatible"] is True
    assert stirrer_diag["cardinality_exact"] is False  # [1,1] vs [1,2] is compatible but not exact
    assert result.exact_structural_match is False  # Because cardinality is not exact


def test_kitchen_reusable_count_2_compatible():
    """Candidate coffee_stirrer count=2 [2,2] REUSABLE vs GT [1,2] REUSABLE."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    raw_doc = _valid_kitchen_raw_vlm()
    for r in raw_doc["functional_roles"]:
        if r["id"] == "spoon_tool":
            r["required_count"] = 2
    adapter = MockFMAdapter(raw_doc)
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    result = evaluate_gf_against_reference(cand_gf, ref_gf)
    assert result.reference_complete is True
    assert result.role_cardinality_diagnostics["coffee_stirrer"]["cardinality_compatible"] is True
    assert result.role_cardinality_diagnostics["coffee_stirrer"]["cardinality_exact"] is False


def test_kitchen_reusable_count_out_of_range_fails():
    """Candidate coffee_stirrer count=3 [3,3] REUSABLE vs GT [1,2] REUSABLE fails."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    raw_doc = _valid_kitchen_raw_vlm()
    for r in raw_doc["functional_roles"]:
        if r["id"] == "spoon_tool":
            r["required_count"] = 3
    adapter = MockFMAdapter(raw_doc)
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    result = evaluate_gf_against_reference(cand_gf, ref_gf)
    assert result.reference_complete is False
    assert "coffee_stirrer" in result.role_attribute_mismatches
    assert "cardinality" in result.role_attribute_mismatches["coffee_stirrer"]
    assert result.role_cardinality_diagnostics["coffee_stirrer"]["cardinality_compatible"] is False


def test_kitchen_wrong_binding_policy_fails():
    """Candidate coffee_stirrer count=1 DISTINCT vs GT [1,2] REUSABLE fails."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    raw_doc = _valid_kitchen_raw_vlm()
    for r in raw_doc["functional_roles"]:
        if r["id"] == "spoon_tool":
            r["binding_policy"] = "DISTINCT"
    adapter = MockFMAdapter(raw_doc)
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    result = evaluate_gf_against_reference(cand_gf, ref_gf)
    assert result.reference_complete is False
    assert "coffee_stirrer" in result.role_attribute_mismatches
    assert "binding_policy" in result.role_attribute_mismatches["coffee_stirrer"]


def test_operation_group_error_not_hidden_by_cardinality():
    """Operation group error (target count or usage policy) fails even if role count is compatible."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    raw_doc = _valid_kitchen_raw_vlm()
    # Modify coffee_stirring group to DEDICATED_PER_TARGET
    for grp in raw_doc["interaction_groups"]:
        if grp["id"] == "coffee_stirring":
            grp["usage_policy"] = "DEDICATED_PER_TARGET"
    adapter = MockFMAdapter(raw_doc)
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    result = evaluate_gf_against_reference(cand_gf, ref_gf)
    assert result.reference_complete is False
    assert "coffee_stirring" in result.operation_group_mismatches
    assert "usage_policy" in result.operation_group_mismatches["coffee_stirring"]


# ===========================================================================
# 3. Relations and Operation-Group Hardened Semantics
# ===========================================================================
def test_relation_polarity_mismatch_fails():
    """Relation with expected=False when reference expected=True fails."""
    ref_gf = GTSpecProvider().provide("workshop", "fasten joint")
    adapter = MockFMAdapter(_valid_workshop_raw_vlm())
    provider = FMRequirementProvider(fm_adapter=adapter)
    cand_gf = VLMSpecProvider()._workshop("fasten joint", [], provider=provider)

    # Invert expected polarity of a relation in candidate
    inverted_rels = list(cand_gf.relations)
    inverted_rels[0] = FunctionalRelation(
        subject_role=inverted_rels[0].subject_role,
        predicate=inverted_rels[0].predicate,
        object_role=inverted_rels[0].object_role,
        expected=False,
    )
    bad_cand_gf = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=cand_gf.nodes,
        relations=tuple(inverted_rels),
        operation_groups=cand_gf.operation_groups,
    )

    result = evaluate_gf_against_reference(bad_cand_gf, ref_gf)
    assert result.reference_complete is False
    assert ("driver", "COMPATIBLE_WITH", "fastener", True) in result.missing_relations
    assert ("driver", "COMPATIBLE_WITH", "fastener", False) in result.extra_relations


def test_operation_group_different_id_matches_semantically():
    """Candidate group with arbitrary local ID matches reference group by role signature."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    raw_doc = _valid_kitchen_raw_vlm()
    adapter = MockFMAdapter(raw_doc)
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    # Directly replace group id in candidate graph
    mod_ops = []
    for g in cand_gf.operation_groups:
        if g.id == "coffee_stirring":
            mod_ops.append(
                OperationGroup(
                    id="group_custom_abc_123",
                    function=g.function,
                    tool_role=g.tool_role,
                    target_role=g.target_role,
                    required_target_count=g.required_target_count,
                    usage_policy=g.usage_policy,
                    required_relations=g.required_relations,
                    context_role=g.context_role,
                    context_relations=g.context_relations,
                    distinct_within_group=g.distinct_within_group,
                    same_tool_must_cover_all_targets=g.same_tool_must_cover_all_targets,
                )
            )
        else:
            mod_ops.append(g)

    cand_gf_diff_id = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=cand_gf.nodes,
        relations=cand_gf.relations,
        operation_groups=tuple(mod_ops),
        cross_group_reuse_allowed=cand_gf.cross_group_reuse_allowed,
    )

    result = evaluate_gf_against_reference(cand_gf_diff_id, ref_gf)
    assert result.reference_complete is True
    assert ("coffee_stirring", "group_custom_abc_123") in result.matched_operation_groups
    assert len(result.missing_operation_groups) == 0


def test_operation_group_same_id_wrong_semantics_does_not_match():
    """Candidate group with same ID as reference but wrong semantic roles must NOT match."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    # Reference has coffee_stirring with tool_role="coffee_stirrer", target_role="coffee_container"
    # Create candidate where group id "coffee_stirring" has wrong roles (e.g. soup_eating_utensil -> soup_container)
    wrong_role_op = OperationGroup(
        id="coffee_stirring",
        function="wrong roles",
        tool_role="soup_eating_utensil",
        target_role="soup_container",
        required_target_count=2,
        usage_policy="DEDICATED_PER_TARGET",
        required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
    )

    cand_gf = FunctionalRequirementGraph(
        domain=ref_gf.domain,
        task_instruction=ref_gf.task_instruction,
        nodes=ref_gf.nodes,
        relations=ref_gf.relations,
        operation_groups=(wrong_role_op,),
        cross_group_reuse_allowed=ref_gf.cross_group_reuse_allowed,
    )

    result = evaluate_gf_against_reference(cand_gf, ref_gf)
    # coffee_stirring in reference requires (coffee_stirrer, coffee_container), which wrong_role_op does NOT satisfy
    assert result.reference_complete is False
    assert "coffee_stirring" in result.missing_operation_groups
    assert ("coffee_stirring", "coffee_stirring") not in result.matched_operation_groups


def test_operation_group_id_used_only_as_semantic_tiebreak():
    """When multiple candidates are semantically eligible, reference ID tie-breaks if unique."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    adapter = MockFMAdapter(_valid_kitchen_raw_vlm())
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    # Two candidate groups for coffee_stirrer -> coffee_container: one named coffee_stirring, one named alt_stirring
    grp_exact_id = OperationGroup(
        id="coffee_stirring",
        function="stir coffee",
        tool_role="coffee_stirrer",
        target_role="coffee_container",
        required_target_count=2,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
    )
    grp_alt_id = OperationGroup(
        id="alt_stirring",
        function="stir coffee alt",
        tool_role="coffee_stirrer",
        target_role="coffee_container",
        required_target_count=2,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
    )
    soup_op = next(g for g in cand_gf.operation_groups if g.id == "soup_serving")

    cand_with_tiebreak = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=cand_gf.nodes,
        relations=cand_gf.relations,
        operation_groups=(grp_alt_id, grp_exact_id, soup_op),
        cross_group_reuse_allowed=cand_gf.cross_group_reuse_allowed,
    )

    result = evaluate_gf_against_reference(cand_with_tiebreak, ref_gf)
    assert result.reference_complete is True
    # Successfully tie-broken to coffee_stirring
    assert ("coffee_stirring", "coffee_stirring") in result.matched_operation_groups
    assert "alt_stirring" in result.extra_operation_groups


def test_operation_group_ambiguity_reported():
    """Two unmatched candidate groups matching the same roles without ID tiebreak are reported as ambiguous."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    adapter = MockFMAdapter(_valid_kitchen_raw_vlm())
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    grp1 = OperationGroup(
        id="cand_stir_1",
        function="stir coffee 1",
        tool_role="coffee_stirrer",
        target_role="coffee_container",
        required_target_count=2,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
    )
    grp2 = OperationGroup(
        id="cand_stir_2",
        function="stir coffee 2",
        tool_role="coffee_stirrer",
        target_role="coffee_container",
        required_target_count=2,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
    )
    # Filter out coffee_stirring from original cand_gf and replace with grp1 and grp2
    other_ops = tuple(g for g in cand_gf.operation_groups if g.id != "coffee_stirring")
    ambig_cand_gf = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=cand_gf.nodes,
        relations=cand_gf.relations,
        operation_groups=other_ops + (grp1, grp2),
        cross_group_reuse_allowed=cand_gf.cross_group_reuse_allowed,
    )

    result = evaluate_gf_against_reference(ambig_cand_gf, ref_gf)
    assert result.reference_complete is False
    assert "coffee_stirring" in result.ambiguous_operation_groups
    assert set(result.ambiguous_operation_groups["coffee_stirring"]) == {"cand_stir_1", "cand_stir_2"}


def test_distinct_within_group_mismatch_is_diagnostic_only():
    """Mismatch in distinct_within_group is diagnostic-only and does not fail reference_complete or exact_structural_match."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    adapter = MockFMAdapter(_valid_kitchen_raw_vlm())
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    # Invert distinct_within_group on coffee_stirring
    mod_ops = []
    for g in cand_gf.operation_groups:
        if g.id == "coffee_stirring":
            mod_ops.append(
                OperationGroup(
                    id=g.id,
                    function=g.function,
                    tool_role=g.tool_role,
                    target_role=g.target_role,
                    required_target_count=g.required_target_count,
                    usage_policy=g.usage_policy,
                    required_relations=g.required_relations,
                    context_role=g.context_role,
                    context_relations=g.context_relations,
                    distinct_within_group=True,  # Reference is False
                    same_tool_must_cover_all_targets=g.same_tool_must_cover_all_targets,
                )
            )
        else:
            mod_ops.append(g)

    cand_gf_mod = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=cand_gf.nodes,
        relations=cand_gf.relations,
        operation_groups=tuple(mod_ops),
        cross_group_reuse_allowed=cand_gf.cross_group_reuse_allowed,
    )

    result = evaluate_gf_against_reference(cand_gf_mod, ref_gf)
    assert result.reference_complete is True
    assert result.exact_structural_match is False  # Reusable cardinality [1,1] vs [1,2] makes exact False, but not group
    assert result.operation_group_exact_recall == 1.0
    assert result.operation_group_exact_precision == 1.0
    assert "coffee_stirring" in result.operation_group_representation_diagnostics
    assert result.operation_group_representation_diagnostics["coffee_stirring"]["distinct_within_group"]["grounding_relevant"] is False


def test_cross_group_reuse_mismatch_fails():
    """Mismatch in cross_group_reuse_allowed fails."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    adapter = MockFMAdapter(_valid_kitchen_raw_vlm())
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    # Invert cross_group_reuse_allowed
    mod_cand_gf = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=cand_gf.nodes,
        relations=cand_gf.relations,
        operation_groups=cand_gf.operation_groups,
        cross_group_reuse_allowed=True,  # Reference is False
    )

    result = evaluate_gf_against_reference(mod_cand_gf, ref_gf)
    assert result.reference_complete is False
    assert result.cross_group_reuse_mismatch is True
    assert "cross_group_reuse_allowed" in result.graph_attribute_mismatches


def test_same_tool_coverage_mismatch_fails():
    """Mismatch in same_tool_must_cover_all_targets fails."""
    ref_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    adapter = MockFMAdapter(_valid_kitchen_raw_vlm())
    cand_gf = VLMSpecProvider()._kitchen("prepare coffee and soup", [], adapter=adapter)

    # Modify an operation group's same_tool_must_cover_all_targets
    mod_ops = []
    for g in cand_gf.operation_groups:
        if g.id == "coffee_stirring":
            mod_ops.append(
                OperationGroup(
                    id=g.id,
                    function=g.function,
                    tool_role=g.tool_role,
                    target_role=g.target_role,
                    required_target_count=g.required_target_count,
                    usage_policy=g.usage_policy,
                    required_relations=g.required_relations,
                    context_role=g.context_role,
                    context_relations=g.context_relations,
                    distinct_within_group=g.distinct_within_group,
                    same_tool_must_cover_all_targets=True,  # Reference is False
                )
            )
        else:
            mod_ops.append(g)

    mod_cand_gf = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=cand_gf.nodes,
        relations=cand_gf.relations,
        operation_groups=tuple(mod_ops),
        cross_group_reuse_allowed=cand_gf.cross_group_reuse_allowed,
    )

    result = evaluate_gf_against_reference(mod_cand_gf, ref_gf)
    assert result.reference_complete is False
    assert "coffee_stirring" in result.operation_group_mismatches
    assert "same_tool_must_cover_all_targets" in result.operation_group_mismatches["coffee_stirring"]


# ===========================================================================
# 3b. Domain Safety Guards
# ===========================================================================
def test_domain_mismatch_candidate_and_reference_raises():
    """Candidate domain differing from reference domain raises ValueError."""
    cand_workshop = GTSpecProvider().provide("workshop", "fasten joint")
    ref_kitchen = GTSpecProvider().provide("kitchen", "prepare coffee and soup")

    with pytest.raises(ValueError) as exc_info:
        evaluate_gf_against_reference(cand_workshop, ref_kitchen)
    assert "Reference evaluation domain mismatch: candidate='workshop', reference='kitchen'" in str(exc_info.value)


def test_explicit_domain_override_mismatch_raises():
    """Explicit domain differing from candidate domain raises ValueError."""
    cand_workshop = GTSpecProvider().provide("workshop", "fasten joint")

    with pytest.raises(ValueError) as exc_info:
        evaluate_gf_against_reference(cand_workshop, domain="kitchen")
    assert "explicit domain='kitchen' differs from candidate domain='workshop'" in str(exc_info.value)


def test_matching_domain_evaluation_succeeds():
    """Matching candidate and reference domain evaluates successfully."""
    cand_workshop = GTSpecProvider().provide("workshop", "fasten joint")
    ref_workshop = GTSpecProvider().provide("workshop", "fasten joint")

    result = evaluate_gf_against_reference(cand_workshop, ref_workshop)
    assert result.reference_complete is True
    assert result.exact_structural_match is True


# ===========================================================================
# 4. Metrics, Extra Structure, Missing Structure, Category Diagnostics
# ===========================================================================
def test_extra_structure_preserves_reference_complete_but_lowers_precision():
    """Candidate with all required structure plus extra role, relation, group."""
    ref_gf = GTSpecProvider().provide("workshop", "fasten joint")
    adapter = MockFMAdapter(_valid_workshop_raw_vlm())
    provider = FMRequirementProvider(fm_adapter=adapter)
    cand_gf = VLMSpecProvider()._workshop("fasten joint", [], provider=provider)

    extra_node = FunctionalRole(name="extra_magnifier", entity_kind="OBJECT", count=1, binding_policy="DISTINCT")
    extra_rel = FunctionalRelation(subject_role="extra_magnifier", predicate="REACHES_TARGET", object_role="repair_target", expected=True)
    extra_op = OperationGroup(
        id="extra_op",
        function="inspect",
        tool_role="extra_magnifier",
        target_role="repair_target",
        required_target_count=1,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("REACHES_TARGET",),
    )

    nodes = dict(cand_gf.nodes)
    nodes["extra_magnifier"] = extra_node
    rels = cand_gf.relations + (extra_rel,)
    ops = cand_gf.operation_groups + (extra_op,)

    extra_cand_gf = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=nodes,
        relations=rels,
        operation_groups=ops,
    )

    result = evaluate_gf_against_reference(extra_cand_gf, ref_gf)
    assert result.reference_complete is True
    assert result.exact_structural_match is False
    assert result.extra_roles == ("extra_magnifier",)
    assert result.role_identity_recall == 1.0
    assert result.role_identity_precision < 1.0
    assert result.relation_recall == 1.0
    assert result.relation_precision < 1.0
    assert result.operation_group_identity_recall == 1.0
    assert result.operation_group_identity_precision < 1.0


def test_missing_structure_lowers_recall_and_fails_reference_complete():
    """Candidate missing a required role, relation, and group."""
    ref_gf = GTSpecProvider().provide("workshop", "fasten joint")
    adapter = MockFMAdapter(_valid_workshop_raw_vlm())
    provider = FMRequirementProvider(fm_adapter=adapter)
    cand_gf = VLMSpecProvider()._workshop("fasten joint", [], provider=provider)

    # Drop fastener role and its relations
    nodes = {k: v for k, v in cand_gf.nodes.items() if k != "fastener"}
    rels = tuple(r for r in cand_gf.relations if r.subject_role != "fastener" and r.object_role != "fastener")
    ops = ()

    missing_cand_gf = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=nodes,
        relations=rels,
        operation_groups=ops,
    )

    result = evaluate_gf_against_reference(missing_cand_gf, ref_gf)
    assert result.reference_complete is False
    assert "fastener" in result.missing_roles
    assert result.role_identity_recall < 1.0
    assert result.relation_recall < 1.0


def test_semantic_category_lexical_mismatch_is_diagnostic_only():
    """Open-vocabulary novel phrases in candidate_categories do NOT fail reference_complete."""
    ref_gf = GTSpecProvider().provide("workshop", "fasten joint")
    adapter = MockFMAdapter(_valid_workshop_raw_vlm())
    provider = FMRequirementProvider(fm_adapter=adapter)
    cand_gf = VLMSpecProvider()._workshop("fasten joint", [], provider=provider)

    # Overwrite driver semantic categories with a completely novel open-vocabulary description
    novel_driver = FunctionalRole(
        name="driver",
        entity_kind="OBJECT",
        count=1,
        semantic_categories=("manual cross-head torque applicator", "slender metal driving rod"),
        binding_policy="DISTINCT",
    )
    nodes = dict(cand_gf.nodes)
    nodes["driver"] = novel_driver

    novel_cand_gf = FunctionalRequirementGraph(
        domain=cand_gf.domain,
        task_instruction=cand_gf.task_instruction,
        nodes=nodes,
        relations=cand_gf.relations,
        operation_groups=cand_gf.operation_groups,
    )

    result = evaluate_gf_against_reference(novel_cand_gf, ref_gf)
    assert result.reference_complete is True
    diag = result.category_diagnostics["driver"]
    assert diag["exact_overlap"] == []
    assert diag["overlap_count"] == 0
    assert diag["overlap_ratio"] == 0.0


# ===========================================================================
# 5. Non-Mutation Invariance
# ===========================================================================
def test_evaluator_never_mutates_candidate_or_reference():
    """Assert candidate and reference graphs are strictly bit-for-bit unchanged before & after."""
    ref_gf = GTSpecProvider().provide("workshop", "fasten joint")
    adapter = MockFMAdapter(_valid_workshop_raw_vlm())
    provider = FMRequirementProvider(fm_adapter=adapter)
    cand_gf = VLMSpecProvider()._workshop("fasten joint", [], provider=provider)

    ref_copy = copy.deepcopy(ref_gf)
    cand_copy = copy.deepcopy(cand_gf)

    _ = evaluate_gf_against_reference(cand_gf, ref_gf)

    assert cand_gf.nodes == cand_copy.nodes
    assert cand_gf.relations == cand_copy.relations
    assert cand_gf.operation_groups == cand_copy.operation_groups
    assert ref_gf.nodes == ref_copy.nodes
    assert ref_gf.relations == ref_copy.relations
    assert ref_gf.operation_groups == ref_copy.operation_groups


def test_compilers_never_mutate_canonical_gf():
    """All domain compilers must never mutate canonical G_F."""
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import compile_kitchen_contract_from_graph
    from mujoco_scenes.functional_tamp_pipeline.domains.living_room import compile_living_room_task_from_graph
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import compile_workshop_requirements_from_graph

    k_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    k_copy = copy.deepcopy(k_gf)
    _ = compile_kitchen_contract_from_graph(k_gf)
    assert k_gf.nodes == k_copy.nodes
    assert k_gf.relations == k_copy.relations
    assert k_gf.operation_groups == k_copy.operation_groups

    l_gf = GTSpecProvider().provide("living_room", "serve tea")
    l_copy = copy.deepcopy(l_gf)
    _ = compile_living_room_task_from_graph(l_gf)
    assert l_gf.nodes == l_copy.nodes
    assert l_gf.relations == l_copy.relations
    assert l_gf.operation_groups == l_copy.operation_groups

    w_gf = GTSpecProvider().provide("workshop", "fasten joint")
    w_copy = copy.deepcopy(w_gf)
    _ = compile_workshop_requirements_from_graph(w_gf)
    assert w_gf.nodes == w_copy.nodes
    assert w_gf.relations == w_copy.relations
    assert w_gf.operation_groups == w_copy.operation_groups


# ===========================================================================
# 6. Static Offline Isolation
# ===========================================================================
def test_reference_evaluator_is_strictly_offline():
    """Assert gf_reference_evaluator is NEVER imported by runtime modules."""
    forbidden_runtime_modules = [
        "mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider",
        "mujoco_scenes.functional_tamp_pipeline.gt_spec_provider",
        "mujoco_scenes.functional_tamp_pipeline.grounding",
        "mujoco_scenes.functional_tamp_pipeline.search_order",
        "mujoco_scenes.functional_tamp_pipeline.domains.kitchen",
        "mujoco_scenes.functional_tamp_pipeline.domains.living_room",
        "mujoco_scenes.functional_tamp_pipeline.domains.workshop",
    ]
    import importlib
    for mod_name in forbidden_runtime_modules:
        mod = importlib.import_module(mod_name)
        with open(mod.__file__, "r", encoding="utf-8") as f:
            content = f.read()
        assert "gf_reference_evaluator" not in content, f"Forbidden import of gf_reference_evaluator found in {mod_name}"
        assert "evaluate_gf_against_reference" not in content, f"Forbidden call to evaluate_gf_against_reference found in {mod_name}"
