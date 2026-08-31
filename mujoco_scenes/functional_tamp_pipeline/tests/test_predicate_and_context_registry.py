"""Unit and integration tests for frozen Predicate Registry and System Context Registry (Pass P3-D / P3-D.1)."""

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
    PredicateStatus,
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
    PLANNER_CONTEXT_CONSTANTS,
    SEARCH_REGIONS,
    SELECTABLE_ROLES,
    SYSTEM_FIXED_ANCHORS,
    get_domain_planner_context_constants,
    get_domain_search_regions,
    get_domain_selectable_roles,
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


def test_registries_are_read_only():
    """Verify that exported registry mappings are read-only MappingProxyType and raise TypeError on mutation."""
    with pytest.raises(TypeError):
        PREDICATE_REGISTRY[("kitchen", "FAKE")] = None  # type: ignore

    with pytest.raises(TypeError):
        PLANNER_CONTEXT_CONSTANTS["workshop"] = frozenset()  # type: ignore

    with pytest.raises(TypeError):
        SYSTEM_FIXED_ANCHORS["workshop"] = frozenset()  # type: ignore

    with pytest.raises(TypeError):
        SELECTABLE_ROLES["workshop"] = frozenset()  # type: ignore

    with pytest.raises(TypeError):
        SEARCH_REGIONS["workshop"] = frozenset()  # type: ignore


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
    """Verify that unknown predicates, legacy markers, and unsupported emittables fail closed."""
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

    # Unsupported emittable predicate in Workshop
    with pytest.raises(MalformedVLMSpecificationError, match="canonicalizer-emittable but unsupported"):
        validate_predicate_signature(
            domain="workshop",
            predicate="LOCATED_ON",
            subject_kind="FIXED_TARGET",
            object_kind="REGION",
            subject_role="repair_target",
            object_role="workbench_surface",
        )


def test_runtime_gf_validator_rejects_invalid_operation_group_relations():
    """Verify that validate_runtime_gf catches invalid operation group predicate signatures."""
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


def test_operation_group_context_relation_direction_enforced():
    """Verify that operation group context_relations enforce directional predicate signatures."""
    # Valid living room operation group: PERSONAL_CUP_SAUCER_REGION -> SEATING_POSITION with NEAR_SEAT
    valid_graph = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="Set up cup and saucer",
        nodes={
            "PERSONAL_CUP_SAUCER_REGION": FunctionalRole(name="PERSONAL_CUP_SAUCER_REGION", entity_kind="REGION", count=1),
            "CUP_SAUCER_SET": FunctionalRole(name="CUP_SAUCER_SET", entity_kind="OBJECT", count=1),
            "SEATING_POSITION": FunctionalRole(name="SEATING_POSITION", entity_kind="FIXED_TARGET", count=1),
        },
        relations=(
            FunctionalRelation(subject_role="PERSONAL_CUP_SAUCER_REGION", predicate="FITS_SET_ON", object_role="CUP_SAUCER_SET"),
            FunctionalRelation(subject_role="PERSONAL_CUP_SAUCER_REGION", predicate="NEAR_SEAT", object_role="SEATING_POSITION"),
        ),
        operation_groups=(
            OperationGroup(
                id="place_cup_saucer",
                function="PLACE_DRINKWARE",
                tool_role="PERSONAL_CUP_SAUCER_REGION",
                target_role="CUP_SAUCER_SET",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=("FITS_SET_ON",),
                context_role="SEATING_POSITION",
                context_relations=("NEAR_SEAT",),
            ),
        ),
        source="GT_LIVING_ROOM_SPEC",
    )
    validate_runtime_gf(valid_graph)

    # Invalid: context_role SEATING_POSITION used with ACCESSIBLE_FROM_BOTH_SEATS (which requires SEATING_PAIR)
    invalid_graph = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="Set up remote",
        nodes={
            "SHARED_REMOTE_REGION": FunctionalRole(name="SHARED_REMOTE_REGION", entity_kind="REGION", count=1),
            "REMOTE": FunctionalRole(name="REMOTE", entity_kind="OBJECT", count=1),
            "SEATING_POSITION": FunctionalRole(name="SEATING_POSITION", entity_kind="FIXED_TARGET", count=1),
        },
        relations=(
            FunctionalRelation(subject_role="SHARED_REMOTE_REGION", predicate="FITS_ON", object_role="REMOTE"),
        ),
        operation_groups=(
            OperationGroup(
                id="place_remote",
                function="PLACE_DEVICE",
                tool_role="SHARED_REMOTE_REGION",
                target_role="REMOTE",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=("FITS_ON",),
                context_role="SEATING_POSITION",
                context_relations=("ACCESSIBLE_FROM_BOTH_SEATS",),  # Invalid object for ACCESSIBLE_FROM_BOTH_SEATS (requires SEATING_PAIR)
            ),
        ),
        source="GT_LIVING_ROOM_SPEC",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="expects object role in"):
        validate_runtime_gf(invalid_graph)


def test_system_context_registry_domain_isolation():
    """Verify that planner context constants are strictly domain-isolated and exclude false symbols."""
    kitchen_consts = get_domain_planner_context_constants("kitchen")
    assert kitchen_consts == frozenset({"countertop", "serving_area"})

    living_consts = get_domain_planner_context_constants("living_room")
    assert living_consts == frozenset({"staging_tray"})

    workshop_consts = get_domain_planner_context_constants("workshop")
    # Must NOT include "work_surface" (which is a context dictionary key, not a planner symbol)
    assert workshop_consts == frozenset({"MAIN_WORKBENCH_ZONE", "workshop_frame_joint"})
    assert "work_surface" not in workshop_consts


def test_workshop_plan_argument_work_surface_fails():
    """Verify that false planner constant 'work_surface' fails closed in Workshop audit."""
    graph_f_w = GTSpecProvider().provide("workshop", "Repair frame")
    graph_o_w = ObservedSceneGraph()
    graph_o_w.add_node(ObservedObject(instance_id="driver_1", entity_kind="OBJECT", canonical_category="screwdriver"))
    graph_o_w.add_node(ObservedObject(instance_id="fastener_1", entity_kind="OBJECT", canonical_category="screw"))
    graph_o_w.add_node(ObservedNode(instance_id="repair_target", entity_kind="FIXED_TARGET", canonical_category="repair_target"))
    ground_res_w = GraphGroundingResult(status="COMPLETE", complete=True, assignment={"driver": "driver_1", "fastener": "fastener_1", "repair_target": "repair_target"})

    # Action using "work_surface" instead of "MAIN_WORKBENCH_ZONE"
    bad_plan = [
        {"action_index": 1, "operator": "PLACE", "arguments": ["driver_1", "work_surface"]},
    ]
    audit = audit_plan_grounding(graph_f_w, graph_o_w, ground_res_w, bad_plan, home_region="MAIN_WORKBENCH_ZONE")
    assert audit["plan_uses_only_grounded_task_objects"] is False
    assert any("work_surface" in v for v in audit["violations"])


def test_home_region_cannot_bypass_validation():
    """Verify that audit_plan_grounding does not allow an arbitrary home_region string to bypass validation."""
    graph_f_k = GTSpecProvider().provide("kitchen", "Prepare coffee")
    graph_o_k = ObservedSceneGraph()
    graph_o_k.add_node(ObservedObject(instance_id="mug_1", entity_kind="OBJECT", canonical_category="mug"))
    ground_res_k = GraphGroundingResult(status="COMPLETE", complete=True, assignment={"coffee_container": "mug_1"})

    # Caller passes a foreign/illegal home_region string and uses it in plan
    illegal_home_plan = [
        {"action_index": 1, "operator": "PLACE", "arguments": ["mug_1", "workshop_frame_joint"]},
    ]
    # Passing home_region="workshop_frame_joint" must NOT bypass argument validation
    audit = audit_plan_grounding(
        graph_f_k,
        graph_o_k,
        ground_res_k,
        illegal_home_plan,
        home_region="workshop_frame_joint",
    )
    assert audit["plan_uses_only_grounded_task_objects"] is False
    assert any("workshop_frame_joint" in v for v in audit["violations"])


def test_allowed_context_ids_cannot_authorize_unassigned_object():
    """Verify that allowed_context_ids cannot authorize an unassigned OBJECT on standard domains."""
    graph_f_k = GTSpecProvider().provide("kitchen", "Prepare coffee")
    graph_o_k = ObservedSceneGraph()
    graph_o_k.add_node(ObservedObject(instance_id="mug_1", entity_kind="OBJECT", canonical_category="mug"))
    graph_o_k.add_node(ObservedObject(instance_id="object_9999", entity_kind="OBJECT", canonical_category="cup"))
    # object_9999 is in G_O, but NOT in phi* (assigned_object_ids)
    ground_res_k = GraphGroundingResult(status="COMPLETE", complete=True, assignment={"coffee_container": "mug_1"})

    plan_using_unassigned_obj = [
        {"action_index": 1, "operator": "PLACE", "arguments": ["object_9999", "countertop"]},
    ]
    # Attempting to bypass phi* via allowed_context_ids must FAIL
    audit = audit_plan_grounding(
        graph_f_k,
        graph_o_k,
        ground_res_k,
        plan_using_unassigned_obj,
        home_region="countertop",
        allowed_context_ids=["object_9999"],
    )
    assert audit["plan_uses_only_grounded_task_objects"] is False
    assert any("object_9999" in v for v in audit["violations"])


def test_role_ownership_validation_in_runtime_gf():
    """Verify that validate_runtime_gf enforces strict role ownership per domain."""
    # 1. Fixed anchor with wrong entity_kind (repair_target must be FIXED_TARGET, not OBJECT)
    bad_entity_kind_graph = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Repair frame",
        nodes={
            "driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1, semantic_categories=("screwdriver",)),
            "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1, semantic_categories=("screw",)),
            "repair_target": FunctionalRole(name="repair_target", entity_kind="OBJECT", count=1),  # Wrong! Must be FIXED_TARGET
        },
        relations=(
            FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener"),
        ),
        source="TEST",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="must have entity_kind 'FIXED_TARGET'"):
        validate_runtime_gf(bad_entity_kind_graph)

    # 2. Planner context constant as a G_F role (countertop is planner constant, not G_F role)
    planner_const_as_role_graph = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Prepare coffee",
        nodes={
            "coffee_container": FunctionalRole(name="coffee_container", entity_kind="OBJECT", count=1, semantic_categories=("mug",)),
            "countertop": FunctionalRole(name="countertop", entity_kind="REGION", count=1),  # Wrong!
        },
        source="TEST",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="is a planner context constant"):
        validate_runtime_gf(planner_const_as_role_graph)

    # 3. Search region as a G_F role (LEFT_DRAWER is search region, not selectable role)
    search_region_as_role_graph = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Repair frame",
        nodes={
            "driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1, semantic_categories=("screwdriver",)),
            "LEFT_DRAWER": FunctionalRole(name="LEFT_DRAWER", entity_kind="REGION", count=1),  # Wrong!
        },
        source="TEST",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="is a search region"):
        validate_runtime_gf(search_region_as_role_graph)

    # 4. workbench_surface emitted by workshop canonicalizer must fail closed
    workbench_surface_graph = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Repair frame",
        nodes={
            "driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1, semantic_categories=("screwdriver",)),
            "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1, semantic_categories=("screw",)),
            "workbench_surface": FunctionalRole(name="workbench_surface", entity_kind="REGION", count=1),
        },
        source="TEST",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="Unknown or unauthorized role 'workbench_surface'"):
        validate_runtime_gf(workbench_surface_graph)


def test_search_region_validation_in_runtime_gf():
    """Verify that candidate_regions and region_ranking comply with domain search region ontology."""
    # Unknown search region in kitchen
    bad_search_region_graph = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Prepare coffee",
        nodes={
            "coffee_container": FunctionalRole(name="coffee_container", entity_kind="OBJECT", count=1, semantic_categories=("mug",)),
        },
        candidate_regions=("D1", "NON_EXISTENT_CABINET"),
        region_ranking=("D1", "NON_EXISTENT_CABINET"),
        source="TEST",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="Candidate search region 'NON_EXISTENT_CABINET' is not a registered search region"):
        validate_runtime_gf(bad_search_region_graph)

    # Living room with non-empty candidate regions (living room has no search regions)
    living_with_search_regions = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="Prepare living room",
        nodes={
            "REMOTE": FunctionalRole(name="REMOTE", entity_kind="OBJECT", count=1, semantic_categories=("remote",)),
        },
        candidate_regions=("CABINET_1",),
        region_ranking=("CABINET_1",),
        source="TEST",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="Candidate search region 'CABINET_1' is not a registered search region"):
        validate_runtime_gf(living_with_search_regions)

    # region_ranking mismatch with candidate_regions
    ranking_mismatch_graph = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Prepare coffee",
        nodes={
            "coffee_container": FunctionalRole(name="coffee_container", entity_kind="OBJECT", count=1, semantic_categories=("mug",)),
        },
        candidate_regions=("D1", "D2"),
        region_ranking=("D1", "C1"),
        source="TEST",
    )
    with pytest.raises(MalformedVLMSpecificationError, match="region_ranking .* must match candidate_regions"):
        validate_runtime_gf(ranking_mismatch_graph)


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
