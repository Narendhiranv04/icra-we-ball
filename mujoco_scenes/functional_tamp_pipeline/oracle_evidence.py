"""Privileged observed-graph builders used only for evidence ablations.

These helpers deliberately keep simulator ground truth separate from the
normal RGB-D/semantic pipeline.  They are for measuring what each grounding
mode would do if its available evidence were perfect; they are not an input to
the production planner.
"""

from __future__ import annotations

from typing import Any

from mujoco_scenes.exact_scene_geometry import extract_exact_object_geometry
from mujoco_scenes.geometry_properties import load_geometry_config
from mujoco_scenes.geometry_relations import evaluate_insertable_in, evaluate_reaches_bottom
from mujoco_scenes.kitchen_feasibility_oracle import load_feasibility_benchmark_config
from mujoco_scenes.living_room_region_oracle import evaluate_privileged_oracle
from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene
from mujoco_scenes.living_room_variants import load_living_room_variants, scene_name
from mujoco_scenes.scene_loader import KitchenScene
from mujoco_scenes.workshop_ground_truth_planner import load_variant_specs
from mujoco_scenes.workshop_scene import (
    PRIVILEGED_WORKSHOP_ORACLE_SPECS,
    WorkshopScene,
)

from .domains.living_room import compile_living_room_task_from_graph
from .models import FunctionalRequirementGraph
from .scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph


def kitchen_variants() -> dict[str, dict[str, Any]]:
    return dict(load_feasibility_benchmark_config()["variants"])


def living_room_variants() -> dict[str, dict[str, Any]]:
    return dict(load_living_room_variants())


def workshop_variants() -> dict[str, dict[str, Any]]:
    return dict(load_variant_specs())


def _kitchen_category(kind: str) -> str:
    value = kind.lower()
    if "coffee" in value and ("jar" in value or "can" in value):
        return "coffee_source"
    if "kettle" in value:
        return "kettle"
    if "mug" in value:
        return "mug"
    if "cup" in value:
        return "cup"
    if "bowl" in value:
        return "bowl"
    if "spoon" in value:
        return "spoon"
    if "fork" in value:
        return "fork"
    return value


def _kitchen_source_region(region: str | None) -> str:
    return "INITIAL" if region is None else str(region)


def _kitchen_predicates(geometry: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    elongated = geometry.get("elongation_ratio")
    return {
        "OPEN_CAVITY": "TRUE" if geometry.get("open_cavity") else "FALSE",
        "ELONGATED_OBJECT": (
            "TRUE"
            if elongated is not None
            and float(elongated) >= float(config["elongated_object"]["minimum_dominant_axis_ratio"])
            else "FALSE"
        ),
    }


def build_kitchen_oracle_graph(variant_id: str) -> ObservedSceneGraph:
    """Return the complete GT graph for one Kitchen feasibility variant."""
    variant = kitchen_variants()[variant_id]
    scene = KitchenScene(variant["scene_name"], include_robot=False, robot="none")
    config = load_geometry_config()
    graph = ObservedSceneGraph(stage_index=0)
    geometry_by_id: dict[str, dict[str, Any]] = {}

    for instance_id, kind, region in scene._object_instance_records:
        try:
            geometry = extract_exact_object_geometry(scene, instance_id, kind, geometry_config=config).as_dict()
        except (ValueError, RuntimeError):
            geometry = {}
        geometry_by_id[instance_id] = geometry
        graph.add_node(ObservedNode(
            instance_id=instance_id,
            entity_kind="OBJECT",
            canonical_category=_kitchen_category(kind),
            source_region=_kitchen_source_region(region),
            geometry=geometry,
            unary_properties=geometry,
            unary_predicates=_kitchen_predicates(geometry, config),
        ))

    clearance = float(config["pairwise_relations"]["clearance_margin_m"])
    grip = float(config["pairwise_relations"]["grip_allowance_m"])
    for tool_id, tool in graph.nodes.items():
        for target_id, target in graph.nodes.items():
            if tool_id == target_id:
                continue
            tool_geometry = geometry_by_id[tool_id]
            target_geometry = geometry_by_id[target_id]
            cross_section = tool_geometry.get("maximum_cross_section_m")
            length = tool_geometry.get("total_length_m")
            opening = target_geometry.get("opening_width_m")
            depth = target_geometry.get("cavity_depth_m")
            if None in (cross_section, length, opening, depth):
                insert = {"status": "UNKNOWN", "reason": "NOT_A_TOOL_CONTAINER_PAIR"}
                reaches = {"status": "UNKNOWN", "reason": "NOT_A_TOOL_CONTAINER_PAIR"}
            else:
                insert = evaluate_insertable_in(float(cross_section), float(opening), clearance)
                reaches = evaluate_reaches_bottom(float(length), float(depth), grip)
            graph.add_relation(ObservedRelation(
                subject_id=tool_id, predicate="INSERTABLE_IN", object_id=target_id,
                status=str(insert["status"]), evidence=dict(insert),
            ))
            graph.add_relation(ObservedRelation(
                subject_id=tool_id, predicate="REACHES_BOTTOM", object_id=target_id,
                status=str(reaches["status"]), evidence=dict(reaches),
            ))
    return graph


def build_living_room_oracle_graph(
    variant_id: str,
    specification: FunctionalRequirementGraph,
) -> ObservedSceneGraph:
    """Return a GT region/payload graph derived from the instantiated room."""
    scene = L2LivingRoomRegionScene(scene_name(variant_id), robot="none")
    task = compile_living_room_task_from_graph(specification)
    oracle = evaluate_privileged_oracle(scene, task)
    graph = ObservedSceneGraph(stage_index=0)

    for region_id, row in oracle["regions"].items():
        graph.add_node(ObservedNode(
            instance_id=region_id,
            entity_kind="REGION",
            canonical_category=str(row["category"]),
            geometry=dict(row),
            unary_predicates={"PLANAR_SUPPORT": "TRUE"},
        ))

    for index in (1, 2):
        graph.add_node(ObservedNode(
            instance_id=f"CUP_SAUCER_SET_{index}",
            entity_kind="OBJECT",
            canonical_category="cup_saucer_set",
        ))
        graph.add_node(ObservedNode(
            instance_id=f"SEATING_POSITION_{index}",
            entity_kind="FIXED_TARGET",
            canonical_category="seating_position",
        ))
    graph.add_node(ObservedNode(
        instance_id="REMOTE", entity_kind="OBJECT", canonical_category="tv_remote",
    ))
    graph.add_node(ObservedNode(
        instance_id="SEATING_PAIR", entity_kind="FIXED_TARGET", canonical_category="seating_pair",
    ))

    for row in oracle["personal_compatibility"]:
        slot_number = int(str(row["slot_id"]).rsplit("_", 1)[-1])
        slot_id = f"CUP_SAUCER_SET_{slot_number}"
        seat_id = f"SEATING_POSITION_{slot_number}"
        graph.add_relation(ObservedRelation(
            subject_id=row["region_id"], predicate="FITS_SET_ON", object_id=slot_id,
            status="TRUE" if row["fits"] else "FALSE", evidence=dict(row),
        ))
        graph.add_relation(ObservedRelation(
            subject_id=row["region_id"], predicate="NEAR_SEAT", object_id=seat_id,
            status="TRUE" if row["distance_m"] <= 1.20 else "FALSE", evidence=dict(row),
        ))
    for row in oracle["shared_compatibility"]:
        graph.add_relation(ObservedRelation(
            subject_id=row["region_id"], predicate="FITS_ON", object_id="REMOTE",
            status="TRUE" if row["fits"] else "FALSE", evidence=dict(row),
        ))
        graph.add_relation(ObservedRelation(
            subject_id=row["region_id"], predicate="ACCESSIBLE_FROM_BOTH_SEATS", object_id="SEATING_PAIR",
            status="TRUE" if max(row["distances_to_seats_m"]) <= 1.65 else "FALSE", evidence=dict(row),
        ))
    return graph


def _workshop_category(instance_id: str) -> str:
    if "power_driver" in instance_id:
        return "power_driver"
    if "phillips_driver" in instance_id:
        return "screwdriver"
    if "screw" in instance_id:
        return "screw"
    if "hammer" in instance_id:
        return "hammer"
    return "unknown"


def _workshop_relation_result(
    *,
    predicate: str,
    required: dict[str, Any],
    checks: dict[str, bool],
) -> dict[str, Any]:
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        status = "UNKNOWN"
        reason = "REQUIRED_EXACT_GEOMETRY_MISSING"
    else:
        status = "TRUE" if all(checks.values()) else "FALSE"
        reason = "ALL_GEOMETRIC_CHECKS_PASS" if status == "TRUE" else "GEOMETRIC_CHECK_FAILED"
    return {
        "status": status,
        "reason": reason,
        "predicate": predicate,
        "measurements": required,
        "checks": checks,
        "missing_measurements": missing,
        "method": "PRIVILEGED_WORKSHOP_CONSTRUCTION_GEOMETRY_V1",
        "semantic_category_used": False,
    }


def _workshop_reaches_target(
    driver_geometry: dict[str, Any],
    target_geometry: dict[str, Any],
) -> dict[str, Any]:
    reach = driver_geometry.get("reach_m")
    depth = target_geometry.get("target_hole_depth_m")
    required = {
        "driver_reach_m": reach,
        "target_hole_depth_m": depth,
    }
    checks = {
        "reach_covers_target_depth": (
            float(reach) >= float(depth)
            if reach is not None and depth is not None else False
        ),
    }
    result = _workshop_relation_result(
        predicate="REACHES_TARGET", required=required, checks=checks,
    )
    result["signed_reach_margin_m"] = (
        float(reach) - float(depth)
        if reach is not None and depth is not None else None
    )
    return result


def _workshop_driver_fastener_compatibility(
    driver_geometry: dict[str, Any],
    fastener_geometry: dict[str, Any],
) -> dict[str, Any]:
    tip_profile = driver_geometry.get("tip_profile")
    tip_width = driver_geometry.get("tip_width_m")
    recess_profile = fastener_geometry.get("recess_profile")
    recess_width = fastener_geometry.get("recess_width_m")
    required = {
        "driver_tip_profile": tip_profile,
        "driver_tip_width_m": tip_width,
        "fastener_recess_profile": recess_profile,
        "fastener_recess_width_m": recess_width,
    }
    checks = {
        "interface_profile_matches": (
            str(tip_profile).upper() == str(recess_profile).upper()
            if tip_profile is not None and recess_profile is not None else False
        ),
        "tip_fits_recess": (
            float(tip_width) <= float(recess_width)
            if tip_width is not None and recess_width is not None else False
        ),
    }
    result = _workshop_relation_result(
        predicate="COMPATIBLE_WITH", required=required, checks=checks,
    )
    result["signed_interface_clearance_m"] = (
        float(recess_width) - float(tip_width)
        if tip_width is not None and recess_width is not None else None
    )
    return result


def _workshop_fastener_target_compatibility(
    fastener_geometry: dict[str, Any],
    target_geometry: dict[str, Any],
) -> dict[str, Any]:
    length = fastener_geometry.get("length_m")
    shaft = fastener_geometry.get("shaft_diameter_m")
    opening = target_geometry.get("target_hole_diameter_m")
    depth = target_geometry.get("target_hole_depth_m")
    radial_clearance = target_geometry.get("target_radial_clearance_m")
    required = {
        "fastener_length_m": length,
        "fastener_shaft_diameter_m": shaft,
        "target_hole_diameter_m": opening,
        "target_hole_depth_m": depth,
        "target_radial_clearance_m": radial_clearance,
    }
    checks = {
        "shaft_fits_target_opening": (
            float(shaft) + 2.0 * float(radial_clearance) <= float(opening)
            if shaft is not None and radial_clearance is not None and opening is not None
            else False
        ),
        "fastener_reaches_target_depth": (
            float(length) >= float(depth)
            if length is not None and depth is not None else False
        ),
    }
    result = _workshop_relation_result(
        predicate="COMPATIBLE_WITH_TARGET", required=required, checks=checks,
    )
    result["signed_radial_fit_margin_m"] = (
        float(opening) - (float(shaft) + 2.0 * float(radial_clearance))
        if shaft is not None and radial_clearance is not None and opening is not None
        else None
    )
    result["signed_engagement_margin_m"] = (
        float(length) - float(depth)
        if length is not None and depth is not None else None
    )
    return result


def build_workshop_oracle_graph(variant_id: str) -> ObservedSceneGraph:
    """Return GT semantics plus independently computed exact Workshop geometry."""
    variant = workshop_variants()[variant_id]
    scene = WorkshopScene(robot="none", variant=variant_id)
    target_geometry = scene.privileged_get_target_joint_specification()
    graph = ObservedSceneGraph(stage_index=0)
    graph.add_node(ObservedNode(
        instance_id="repair_target", entity_kind="FIXED_TARGET",
        canonical_category="repair_target",
        geometry=dict(target_geometry),
        unary_properties=dict(target_geometry),
    ))
    object_ids = sorted({
        object_id
        for objects in variant["storage_contents"].values()
        for object_id in objects
    })
    for object_id in object_ids:
        category = _workshop_category(object_id)
        geometry = dict(PRIVILEGED_WORKSHOP_ORACLE_SPECS.get(object_id, {}))
        graph.add_node(ObservedNode(
            instance_id=object_id,
            entity_kind="OBJECT",
            canonical_category=category,
            geometry=geometry,
            unary_properties=geometry,
            source_region=next(
                region for region, contents in variant["storage_contents"].items()
                if object_id in contents
            ),
        ))
    for driver_id in object_ids:
        driver = graph.nodes[driver_id]
        reach = _workshop_reaches_target(driver.geometry, target_geometry)
        graph.add_relation(ObservedRelation(
            subject_id=driver_id, predicate="REACHES_TARGET", object_id="repair_target",
            status=str(reach["status"]), evidence=reach,
        ))
        for fastener_id in object_ids:
            fastener = graph.nodes[fastener_id]
            compatibility = _workshop_driver_fastener_compatibility(
                driver.geometry, fastener.geometry,
            )
            graph.add_relation(ObservedRelation(
                subject_id=driver_id, predicate="COMPATIBLE_WITH", object_id=fastener_id,
                status=str(compatibility["status"]), evidence=compatibility,
            ))
    for fastener_id in object_ids:
        fastener = graph.nodes[fastener_id]
        target_compatibility = _workshop_fastener_target_compatibility(
            fastener.geometry, target_geometry,
        )
        graph.add_relation(ObservedRelation(
            subject_id=fastener_id, predicate="COMPATIBLE_WITH_TARGET", object_id="repair_target",
            status=str(target_compatibility["status"]), evidence=target_compatibility,
        ))
    return graph


def build_oracle_graph(
    domain: str,
    variant_id: str,
    specification: FunctionalRequirementGraph,
) -> ObservedSceneGraph:
    if domain == "kitchen":
        return build_kitchen_oracle_graph(variant_id)
    if domain == "living_room":
        return build_living_room_oracle_graph(variant_id, specification)
    if domain == "workshop":
        return build_workshop_oracle_graph(variant_id)
    raise ValueError(f"Unsupported domain: {domain}")


def intended_outcome(domain: str, variant_id: str) -> str:
    if domain == "kitchen":
        return str(kitchen_variants()[variant_id]["intended_outcome"])
    if domain == "living_room":
        return str(living_room_variants()[variant_id]["intended_outcome"])
    if domain == "workshop":
        return str(workshop_variants()[variant_id]["intended_outcome"])
    raise ValueError(f"Unsupported domain: {domain}")
