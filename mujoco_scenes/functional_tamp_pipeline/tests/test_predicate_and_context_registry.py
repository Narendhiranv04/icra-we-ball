"""Unit and integration tests for frozen Predicate Registry and System Context Registry (Pass P3-D)."""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.audit import audit_plan_grounding
from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.grounding import GraphGroundingResult
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.predicate_registry import (
    PREDICATE_REGISTRY,
    get_active_predicates,
    get_predicate_signature,
    validate_predicate_signature,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedObject,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.system_context_registry import (
    get_domain_planner_context_constants,
    get_domain_search_regions,
    get_domain_system_fixed_anchors,
    is_valid_planner_argument,
)
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import validate_runtime_gf


def test_predicate_registry_completeness():
    """Verify that all domains have frozen, active predicate signatures."""
    kitchen_preds = {sig.name for sig in get_active_predicates("kitchen")}
    assert kitchen_preds == {"OPEN_CAVITY", "ELONGATED_OBJECT", "INSERTABLE_IN", "REACHES_BOTTOM"}

    living_preds = {sig.name for sig in get_active_predicates("living_room")}
    assert living_preds == {"PLANAR_SUPPORT", "FITS_SET_ON", "FITS_ON", "NEAR_SEAT", "ACCESSIBLE_FROM_BOTH_SEATS"}

    workshop_preds = {sig.name for sig in get_active_predicates("workshop")}
    assert workshop_preds == {"COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET"}

    # Workshop legacy capability markers
    assert get_predicate_signature("workshop", "CAN_DRIVE_SCREW").is_legacy_capability_marker is True
    assert get_predicate_signature("workshop", "CAN_DRIVE_SCREW").active_in_functional_graph is False
    assert get_predicate_signature("workshop", "CAN_FASTEN").is_legacy_capability_marker is True
    assert get_predicate_signature("workshop", "CAN_FASTEN").active_in_functional_graph is False


def test_predicate_validation_enforces_direction_and_roles():
    """Verify that reversed or role-incompatible predicate signatures fail closed."""
    # Kitchen: INSERTABLE_IN must be tool -> container (coffee_stirrer -> coffee_container)
    validate_predicate_signature(
        domain="kitchen",
        predicate="INSERTABLE_IN",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        subject_role="coffee_stirrer",
        object_role="coffee_container",
    )

    # Reversed direction must fail
    with pytest.raises(MalformedVLMSpecificationError, match="expects subject role"):
        validate_predicate_signature(
            domain="kitchen",
            predicate="INSERTABLE_IN",
            subject_kind="OBJECT",
            object_kind="OBJECT",
            subject_role="coffee_container",
            object_role="coffee_stirrer",
        )

    # Workshop: COMPATIBLE_WITH must be driver -> fastener
    validate_predicate_signature(
        domain="workshop",
        predicate="COMPATIBLE_WITH",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        subject_role="driver",
        object_role="fastener",
    )

    with pytest.raises(MalformedVLMSpecificationError, match="expects subject role"):
        validate_predicate_signature(
            domain="workshop",
            predicate="COMPATIBLE_WITH",
            subject_kind="OBJECT",
            object_kind="OBJECT",
            subject_role="fastener",
            object_role="driver",
        )

    # Living Room: FITS_SET_ON must be REGION -> OBJECT
    validate_predicate_signature(
        domain="living_room",
        predicate="FITS_SET_ON",
        subject_kind="REGION",
        object_kind="OBJECT",
        subject_role="PERSONAL_CUP_SAUCER_REGION",
        object_role="CUP_SAUCER_SET",
    )

    with pytest.raises(MalformedVLMSpecificationError, match="expects subject entity_kind"):
        validate_predicate_signature(
            domain="living_room",
            predicate="FITS_SET_ON",
            subject_kind="OBJECT",
            object_kind="REGION",
            subject_role="CUP_SAUCER_SET",
            object_role="PERSONAL_CUP_SAUCER_REGION",
        )


def test_predicate_validation_rejects_unknown_and_inactive_predicates():
    """Verify that unknown predicates and legacy inactive markers fail closed."""
    # Unknown predicate
    with pytest.raises(MalformedVLMSpecificationError, match="Unknown predicate 'MAGIC_FIT'"):
        validate_predicate_signature(
            domain="kitchen",
            predicate="MAGIC_FIT",
            subject_kind="OBJECT",
        )

    # Cross-domain predicate name (e.g. Workshop predicate used in Kitchen)
    with pytest.raises(MalformedVLMSpecificationError, match="Unknown predicate 'COMPATIBLE_WITH' for domain 'kitchen'"):
        validate_predicate_signature(
            domain="kitchen",
            predicate="COMPATIBLE_WITH",
            subject_kind="OBJECT",
            object_kind="OBJECT",
        )

    # Legacy inactive marker in canonical graph
    with pytest.raises(MalformedVLMSpecificationError, match="legacy capability marker and not active"):
        validate_predicate_signature(
            domain="workshop",
            predicate="CAN_DRIVE_SCREW",
            subject_kind="OBJECT",
            subject_role="driver",
        )


def test_runtime_gf_validator_rejects_invalid_operation_group_relations():
    """Verify that validate_runtime_gf catches invalid operation group predicate signatures."""
    # Build a graph with wrong operation group required relation
    invalid_graph = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Test task",
        nodes={
            "coffee_container": FunctionalRole(name="coffee_container", entity_kind="OBJECT", count=1, semantic_categories=("cup",)),
            "coffee_stirrer": FunctionalRole(name="coffee_stirrer", entity_kind="OBJECT", count=1, semantic_categories=("spoon",)),
        },
        relations=(
            FunctionalRelation(subject_role="coffee_stirrer", predicate="INSERTABLE_IN", object_role="coffee_container"),
        ),
        operation_groups=(
            OperationGroup(
                id="test_group",
                function="test",
                tool_role="coffee_container",  # Invalid tool role for INSERTABLE_IN
                target_role="coffee_stirrer",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=("INSERTABLE_IN",),
            ),
        ),
        source="GT_FUNCTIONAL_SPEC_ONLY",
    )

    with pytest.raises(MalformedVLMSpecificationError, match="expects subject role"):
        validate_runtime_gf(invalid_graph)


def test_system_context_registry_domain_isolation():
    """Verify that planner context constants are strictly domain-isolated."""
    kitchen_consts = get_domain_planner_context_constants("kitchen")
    assert kitchen_consts == frozenset({"countertop", "serving_area"})

    living_consts = get_domain_planner_context_constants("living_room")
    assert living_consts == frozenset({"staging_tray"})

    workshop_consts = get_domain_planner_context_constants("workshop")
    assert workshop_consts == frozenset({"MAIN_WORKBENCH_ZONE", "workshop_frame_joint", "work_surface"})


def test_audit_plan_grounding_rejects_cross_domain_constants():
    """Verify that audit_plan_grounding rejects constants from other domains."""
    # Kitchen plan using Workshop constant
    graph_f_k = GTSpecProvider().provide("kitchen", "Prepare coffee")
    graph_o_k = ObservedSceneGraph()
    graph_o_k.add_node(ObservedObject(instance_id="mug_1", entity_kind="OBJECT", canonical_category="mug"))
    ground_res_k = GraphGroundingResult(status="COMPLETE", complete=True, assignment={"coffee_container": "mug_1"})

    cross_domain_plan = [
        {"action_index": 1, "operator": "PLACE", "arguments": ["mug_1", "workshop_frame_joint"]},
    ]
    audit = audit_plan_grounding(graph_f_k, graph_o_k, ground_res_k, cross_domain_plan, home_region="countertop")
    assert audit["plan_uses_only_grounded_task_objects"] is False
    assert any("workshop_frame_joint" in v for v in audit["violations"])

    # Living Room plan using Kitchen constant
    graph_f_l = GTSpecProvider().provide("living_room", "Prepare living room")
    graph_o_l = ObservedSceneGraph()
    graph_o_l.add_node(ObservedObject(instance_id="remote_1", entity_kind="OBJECT", canonical_category="remote_control"))
    ground_res_l = GraphGroundingResult(status="COMPLETE", complete=True, assignment={"REMOTE": "remote_1"})

    kitchen_constant_in_living_plan = [
        {"action_index": 1, "operator": "PLACE", "arguments": ["remote_1", "serving_area"]},
    ]
    audit_l = audit_plan_grounding(graph_f_l, graph_o_l, ground_res_l, kitchen_constant_in_living_plan, home_region="staging_tray")
    assert audit_l["plan_uses_only_grounded_task_objects"] is False
    assert any("serving_area" in v for v in audit_l["violations"])


def test_audit_plan_grounding_rejects_unknown_operation_groups():
    """Verify that audit_plan_grounding fails closed if an operation binding references an undeclared group."""
    graph_f = GTSpecProvider().provide("kitchen", "Prepare coffee")
    graph_o = ObservedSceneGraph()
    ground_res = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"coffee_container": "mug_1"},
        operation_bindings={"fabricated_group_xyz": [{"tool_id": "spoon_1", "target_id": "mug_1"}]},
    )
    plan = []
    audit = audit_plan_grounding(graph_f, graph_o, ground_res, plan, home_region="countertop")
    assert audit["all_required_relations_true"] is False
    assert any("fabricated_group_xyz" in v for v in audit["violations"])


def test_gt_spec_provider_all_domains_pass_predicate_and_context_validation():
    """Verify that GT specifications across all 3 domains pass runtime G_F validation against frozen registry."""
    gt_provider = GTSpecProvider()
    gt_k = gt_provider.provide("kitchen", "Prepare and serve coffee and soup for two people")
    validate_runtime_gf(gt_k)

    gt_l = gt_provider.provide("living_room", "Prepare the living room for two people watching television")
    validate_runtime_gf(gt_l)

    gt_w = gt_provider.provide("workshop", "Repair the frame by securing the loose joint")
    validate_runtime_gf(gt_w)
