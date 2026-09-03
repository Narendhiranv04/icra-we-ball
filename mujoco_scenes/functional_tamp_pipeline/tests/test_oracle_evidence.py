from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.oracle_evidence import (
    _workshop_driver_fastener_compatibility,
    _workshop_fastener_target_compatibility,
    _workshop_reaches_target,
    build_oracle_graph,
)
from mujoco_scenes.workshop_scene import PRIVILEGED_WORKSHOP_ORACLE_SPECS


def test_kitchen_oracle_graph_is_complete_and_joint_grounding_is_feasible():
    specification = GTSpecProvider().provide("kitchen", "")
    graph = build_oracle_graph("kitchen", "F0_ALL_VISIBLE", specification)
    result = ground_graph(specification, graph, {"search_exhausted": True, "evidence_mode": "joint"})
    assert result.complete
    assert graph.get_relation("INSERTABLE_IN", "s1i_final_long_narrow_spoon", "ab3_narrow_deep_cup").status == "TRUE"


def test_kitchen_geometric_only_can_complete_a_semantic_trap():
    specification = GTSpecProvider().provide("kitchen", "")
    graph = build_oracle_graph("kitchen", "I5_MISSING_COFFEE_JAR", specification)
    geometric = ground_graph(specification, graph, {"search_exhausted": True, "evidence_mode": "geometric_only"})
    joint = ground_graph(specification, graph, {"search_exhausted": True, "evidence_mode": "joint"})
    assert geometric.complete
    assert not joint.complete


def test_living_room_oracle_graph_matches_joint_feasibility():
    specification = GTSpecProvider().provide("living_room", "")
    graph = build_oracle_graph("living_room", "F0_ALL_OBJECTS_IN_STAGING", specification)
    result = ground_graph(specification, graph, {"search_exhausted": True, "evidence_mode": "joint"})
    assert result.complete


def test_workshop_oracle_relations_use_exact_geometry_evidence():
    specification = GTSpecProvider().provide("workshop", "")
    graph = build_oracle_graph(
        "workshop", "F0_MANUAL_FIRST_ONE_REGION", specification,
    )
    driver = "workshop_long_phillips_driver"
    fastener = "workshop_medium_phillips_screw"

    for predicate, subject, target in (
        ("REACHES_TARGET", driver, "repair_target"),
        ("COMPATIBLE_WITH", driver, fastener),
        ("COMPATIBLE_WITH_TARGET", fastener, "repair_target"),
    ):
        relation = graph.get_relation(predicate, subject, target)
        assert relation is not None
        assert relation.status == "TRUE"
        assert relation.evidence["semantic_category_used"] is False
        assert relation.evidence["measurements"]
        assert relation.evidence["method"] == "PRIVILEGED_WORKSHOP_CONSTRUCTION_GEOMETRY_V1"

    assert graph.nodes[driver].unary_predicates == {}
    assert graph.nodes[fastener].unary_predicates == {}


def test_workshop_exact_geometry_rejects_semantically_plausible_decoys():
    specs = PRIVILEGED_WORKSHOP_ORACLE_SPECS
    target = {
        "target_hole_diameter_m": 0.007,
        "target_hole_depth_m": 0.030,
        "target_radial_clearance_m": 0.0005,
    }

    assert _workshop_reaches_target(
        specs["workshop_stubby_phillips_driver"], target,
    )["status"] == "FALSE"
    assert _workshop_driver_fastener_compatibility(
        specs["workshop_flathead_screwdriver"],
        specs["workshop_medium_phillips_screw"],
    )["status"] == "FALSE"
    assert _workshop_fastener_target_compatibility(
        specs["workshop_short_phillips_screw"], target,
    )["status"] == "FALSE"
    assert _workshop_fastener_target_compatibility(
        specs["workshop_hex_bolt"], target,
    )["status"] == "FALSE"
