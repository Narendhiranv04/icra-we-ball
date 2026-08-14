"""Run the remaining physical evidence required by the Phase-B freeze."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np

from .kitchen_execution_entities import (
    KitchenExecutionEntityResolver,
    build_phase_b_inventory,
)
from .kitchen_execution_policy import KitchenWorkspace
from .kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from .kitchen_object_manipulation import (
    _body_geom_ids,
    _body_yaw,
    oriented_rectangle_corners,
    oriented_rectangles_clearance,
    rectangle_inside_observed_support,
)
from .scene_loader import KitchenScene


ROOT = Path(__file__).resolve().parents[1]
F1_PHASE1 = ROOT / "runs/feasibility_benchmarks/kitchen_feasibility_phase1_closure_20260809/F1_INITIAL_COMPLETE"
F1_PHASE2 = ROOT / "mujoco_scenes/benchmark_reports/kitchen_symbolic_phase2/variants/F1_INITIAL_COMPLETE"
F2_PHASE1 = ROOT / "runs/feasibility_benchmarks/kitchen_feasibility_phase1_closure_20260809/F2_DISTRIBUTED_COFFEE_TWO"
F2_PHASE2 = ROOT / "mujoco_scenes/benchmark_reports/kitchen_symbolic_phase2/variants/F2_DISTRIBUTED_COFFEE_TWO"
PRIMARY_CURRENT = ROOT / "runs/phaseB_freeze_observed_current"
PRIMARY_C1_CURRENT = ROOT / "runs/phaseB_freeze_observed_swapped_c1"
PRIMARY_B1_CORRECTED = ROOT / "runs/phaseB_freeze_observed_b1_corrected"
PRIMARY_FROZEN = ROOT / "runs/integrated_no_pot_clearance_seed19_20260807"


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _semantic_label(record: dict) -> str | None:
    return (
        record.get("semantics", {}).get("validated", {})
        .get("canonical_label")
    )


def _dimension_vector(record: dict) -> np.ndarray:
    return np.asarray([
        record.get("dimensions_m", {}).get(axis, {}).get("value", np.nan)
        for axis in ("length", "width", "height")
    ], dtype=float)


_CURRENT_MEASUREMENT_FIELDS = (
    "centroid_world_m", "dimensions_m", "principal_axis_world",
    "property_status", "point_count", "contributing_camera_count",
    "geometric_properties", "geometric_predicates", "measurement_quality",
    "measurement_cloud_path", "geometry", "last_evidence_stage",
    "last_evidence_source_region", "last_property_update_stage",
    "last_property_source_region",
)


def _physical_family(record: dict) -> str | None:
    """Return a coarse observable family, including geometry-only fallback."""
    label = _semantic_label(record)
    if label in {"bowl", "cup", "mug", "glass"}:
        return "OPEN_VESSEL"
    if label in {"spoon", "fork", "knife", "stirrer", "utensil"}:
        return "ELONGATED_UTENSIL"
    dimensions = _dimension_vector(record)
    finite = dimensions[np.isfinite(dimensions) & (dimensions > 0.0)]
    if len(finite) >= 2:
        ratio = float(np.max(finite) / np.partition(finite, -2)[-2])
        if ratio >= 2.0:
            return "ELONGATED_UTENSIL"
    return None


def _transfer_current_measurements(
    frozen_registry: dict,
    current_registry: dict,
    *,
    regions: set[str] | None = None,
    excluded_object_ids: set[str] | None = None,
) -> dict:
    """Transfer pose/geometry by episode ID while preserving frozen evidence."""
    result = deepcopy(frozen_registry)
    for object_id, target in result["objects"].items():
        if object_id in (excluded_object_ids or set()):
            continue
        current = current_registry["objects"].get(object_id)
        if current is None:
            continue
        if regions is not None and target.get("source_region") not in regions:
            continue
        for field in _CURRENT_MEASUREMENT_FIELDS:
            if field in current:
                target[field] = deepcopy(current[field])
    return result


def apply_current_target_vessel_grounding(
    frozen_registry: dict,
    current_registry: dict,
    assignments: dict,
    *,
    region: str,
) -> tuple[dict, list[dict]]:
    """Ground frozen target roles to current vessels without hidden identity."""
    result = deepcopy(frozen_registry)
    role_by_id = {
        **{object_id: "COFFEE_VESSEL" for object_id in assignments["coffee_targets"]},
        **{object_id: "SOUP_BOWL" for object_id in assignments["soup_targets"]},
    }
    allowed_labels = {
        "COFFEE_VESSEL": {"cup", "mug"},
        "SOUP_BOWL": {"bowl"},
    }
    frozen_rows = [
        (object_id, result["objects"][object_id], role)
        for object_id, role in role_by_id.items()
        if result["objects"][object_id].get("source_region") == region
    ]
    current_rows = [
        (object_id, row)
        for object_id, row in current_registry["objects"].items()
        if row.get("source_region") == region
    ]
    edges = []
    for frozen_id, frozen_row, role in frozen_rows:
        frozen_dimensions = _dimension_vector(frozen_row)
        for current_id, current_row in current_rows:
            current_label = _semantic_label(current_row)
            if current_label not in allowed_labels[role]:
                continue
            current_dimensions = _dimension_vector(current_row)
            finite = np.isfinite(frozen_dimensions) & np.isfinite(current_dimensions)
            if not np.any(finite):
                continue
            error = float(np.linalg.norm(
                frozen_dimensions[finite] - current_dimensions[finite]
            ))
            edges.append((error, frozen_id, current_id, current_row, role))
    assigned_frozen = set()
    assigned_current = set()
    audit = []
    for error, frozen_id, current_id, current_row, role in sorted(edges):
        if frozen_id in assigned_frozen or current_id in assigned_current:
            continue
        assigned_frozen.add(frozen_id)
        assigned_current.add(current_id)
        target = result["objects"][frozen_id]
        for field in _CURRENT_MEASUREMENT_FIELDS:
            if field in current_row:
                target[field] = deepcopy(current_row[field])
        target["execution_scene_calibration"] = {
            "classification": "CURRENT_EXECUTION_TARGET_GROUNDING",
            "region": region,
            "frozen_functional_family": role,
            "association_inputs": [
                "frozen_source_region", "frozen_functional_family",
                "current_observed_semantic_family", "observed_dimensions",
            ],
            "instance_token_used": False,
            "backend_body_name_used": False,
            "current_observation_generic_id_not_planner_visible": current_id,
            "dimension_error_m": error,
        }
        audit.append({
            "frozen_generic_id": frozen_id,
            "frozen_functional_family": role,
            "current_observation_generic_id": current_id,
            "current_observed_semantic_label": _semantic_label(current_row),
            "source_region": region,
            "dimension_error_m": error,
            "instance_token_used": False,
            "backend_body_name_used": False,
            "accepted": True,
        })
    missing = sorted(set(row[0] for row in frozen_rows) - assigned_frozen)
    if missing:
        raise RuntimeError(
            f"Current {region} vessel grounding did not resolve frozen IDs: {missing}"
        )
    return result, audit


def apply_approved_within_region_execution_calibration(
    frozen_registry: dict,
    current_registry: dict,
    *,
    region: str,
) -> tuple[dict, list[dict]]:
    """Transfer current geometry without token or backend-name association.

    Generic IDs and frozen semantics stay fixed. Candidates are gated by the
    same observed source and semantic family, then ranked one-to-one by
    measured dimension distance. This is intentionally limited to an
    explicitly approved within-region physical calibration.
    """
    result = deepcopy(frozen_registry)
    frozen_rows = [
        (object_id, row)
        for object_id, row in result["objects"].items()
        if row.get("source_region") == region
    ]
    current_rows = [
        (object_id, row)
        for object_id, row in current_registry["objects"].items()
        if row.get("source_region") == region
    ]
    edges = []
    rejected = []
    for frozen_id, frozen_row in frozen_rows:
        frozen_label = _semantic_label(frozen_row)
        frozen_family = _physical_family(frozen_row)
        frozen_dimensions = _dimension_vector(frozen_row)
        for current_id, current_row in current_rows:
            current_label = _semantic_label(current_row)
            current_family = _physical_family(current_row)
            semantic_ok = (
                frozen_label is not None and frozen_label == current_label
            )
            family_ok = frozen_family is not None and frozen_family == current_family
            current_dimensions = _dimension_vector(current_row)
            finite = np.isfinite(frozen_dimensions) & np.isfinite(current_dimensions)
            dimension_error = (
                float(np.linalg.norm(
                    frozen_dimensions[finite] - current_dimensions[finite]
                )) if np.any(finite) else float("inf")
            )
            edge = {
                "frozen_generic_id": frozen_id,
                "current_observation_generic_id": current_id,
                "source_region": region,
                "frozen_semantic_label": frozen_label,
                "current_semantic_label": current_label,
                "semantic_consistent": semantic_ok,
                "frozen_physical_family": frozen_family,
                "current_physical_family": current_family,
                "physical_family_consistent": family_ok,
                "dimension_error_m": dimension_error,
            }
            if (semantic_ok or family_ok) and np.isfinite(dimension_error):
                edges.append((dimension_error, frozen_id, current_id, current_row, edge))
            else:
                rejected.append(edge)
    assigned_frozen = set()
    assigned_current = set()
    accepted = []
    for _, frozen_id, current_id, current_row, edge in sorted(
        edges, key=lambda item: (item[0], item[1], item[2])
    ):
        if frozen_id in assigned_frozen or current_id in assigned_current:
            continue
        assigned_frozen.add(frozen_id)
        assigned_current.add(current_id)
        target = result["objects"][frozen_id]
        # Keep frozen semantic/functional evidence. Transfer only typed,
        # current physical localization and geometry measurements.
        for field in _CURRENT_MEASUREMENT_FIELDS:
            if field in current_row:
                target[field] = deepcopy(current_row[field])
        target["execution_scene_calibration"] = {
            "classification": "APPROVED_WITHIN_REGION_EXECUTION_CALIBRATION",
            "region": region,
            "association_inputs": [
                "observed_source_region", "frozen_semantic_or_shape_family",
                "current_observed_semantic_or_shape_family",
                "observed_dimensions",
            ],
            "instance_token_used": False,
            "backend_body_name_used": False,
            "current_observation_generic_id_not_planner_visible": current_id,
            "current_observed_semantic_label": edge["current_semantic_label"],
            "frozen_physical_family": edge["frozen_physical_family"],
            "current_physical_family": edge["current_physical_family"],
            "dimension_error_m": edge["dimension_error_m"],
        }
        accepted.append({**edge, "accepted": True})
    if len(assigned_frozen) != len(frozen_rows):
        missing = sorted(set(row[0] for row in frozen_rows) - assigned_frozen)
        raise RuntimeError(
            f"Approved {region} calibration did not resolve frozen IDs: {missing}"
        )
    return result, accepted + [{**row, "accepted": False} for row in rejected]


def dispatcher(phase1: Path = F1_PHASE1, phase2: Path = F1_PHASE2):
    registry = read(phase1 / "object_registry.json")
    assignments = read(phase2 / "grounded_role_assignments.json")
    plan = read(phase2 / "generated_plan.json")
    inventory = build_phase_b_inventory(registry, assignments, plan)
    scene = KitchenScene(inventory["scene_name"], robot="google")
    resolver = KitchenExecutionEntityResolver()
    resolution = resolver.resolve(
        inventory, resolver.candidates_from_scene(scene, observed_regions=set())
    )
    return scene, inventory, resolution, KitchenPhaseBExecutionDispatcher(
        scene, inventory, resolution
    )


def primary_validation_dispatcher():
    """Bind current measurements to the authoritative frozen symbolic input."""
    assignments = read(PRIMARY_FROZEN / "grounded_role_assignments.json")
    registry = read(PRIMARY_FROZEN / "object_registry.json")
    registry, target_grounding = apply_current_target_vessel_grounding(
        registry,
        read(PRIMARY_CURRENT / "object_registry.json"),
        assignments,
        region="countertop",
    )
    registry = _transfer_current_measurements(
        registry,
        read(PRIMARY_CURRENT / "object_registry.json"),
        excluded_object_ids={
            row["frozen_generic_id"] for row in target_grounding
        },
    )
    c1_registry = read(PRIMARY_C1_CURRENT / "object_registry.json")
    registry = _transfer_current_measurements(
        registry, c1_registry, regions={"C1"}
    )
    registry, b1_calibration = apply_approved_within_region_execution_calibration(
        registry,
        read(PRIMARY_B1_CORRECTED / "object_registry.json"),
        region="B1",
    )
    plan = read(PRIMARY_FROZEN / "plan.json")
    inventory = build_phase_b_inventory(registry, assignments, plan)
    scene = KitchenScene(inventory["scene_name"], robot="google")
    observed_regions = {
        row["source_context"]["source_container"]
        for row in inventory["objects"]
        if row["source_context"]["source_container"]
    }
    resolver = KitchenExecutionEntityResolver()
    resolution = resolver.resolve(
        inventory,
        resolver.candidates_from_scene(
            scene, observed_regions=observed_regions
        ),
    )
    resolution["approved_within_region_execution_calibration"] = b1_calibration
    resolution["current_target_vessel_grounding"] = target_grounding
    if not resolution["all_resolved"] or not resolution["one_to_one"]:
        raise RuntimeError(
            "Fresh primary execution IDs did not resolve: "
            f"{resolution['unresolved_object_ids']}"
        )
    return scene, inventory, resolution, KitchenPhaseBExecutionDispatcher(
        scene, inventory, resolution
    )


def carry(family: str, output: Path) -> None:
    ids = {
        "VESSEL": "object_0001",
        "BOWL": "object_0004",
        "UTENSIL": "object_0008",
        "KETTLE": "object_0010",
        "JAR_SOURCE": "object_0011",
    }
    _, inventory, resolution, execution = dispatcher()
    object_id = ids[family]
    picked = execution.pick(object_id)
    # RIGHT_SIDE is the validated payload-clear route from the main table for
    # every family.  In particular, rotating toward LEFT_SIDE while holding
    # the coffee jar forward of the wrist sweeps the payload through the
    # counter envelope; that is correctly rejected by base collision checking.
    destination = KitchenWorkspace.RIGHT_SIDE
    moved = None
    returned = None
    if picked["success"]:
        moved = execution.move(destination, carrying_object_id=object_id)
    if moved and moved["success"]:
        returned = execution.move(KitchenWorkspace.HOME, carrying_object_id=object_id)
    passed = bool(
        picked["success"] and moved and moved["success"]
        and returned and returned["success"]
    )
    write(output / "carried_move_result.json", {
        "family": family,
        "generic_object_id": object_id,
        "inventory_backend_free": not inventory["planner_received_backend_names"],
        "resolution_one_to_one": resolution["one_to_one"],
        "pick": picked,
        "outbound_move": moved,
        "return_move": returned,
        "success": passed,
    })
    if not passed:
        raise SystemExit(1)


def b1_repeatability(output: Path, trials: int = 3) -> None:
    records = []
    for trial in range(1, trials + 1):
        _, inventory, resolution, execution = primary_validation_dispatcher()
        binding = next(
            row for row in resolution["accepted"]
            if row["generic_object_id"] == "object_0018"
        )
        pick = execution.pick("object_0018")
        records.append({
            "trial": trial,
            "fresh_scene_reset": True,
            "frozen_generic_object_id": "object_0018",
            "frozen_selected_functions": next(
                row["selected_functions"] for row in inventory["objects"]
                if row["generic_object_id"] == "object_0018"
            ),
            "execution_binding": binding,
            "pick": pick,
            "success": bool(pick["success"]),
        })
    passed = len(records) == trials and all(row["success"] for row in records)
    write(output / "b1_primary_corrected_layout_validation.json", {
        "requested_trials": trials,
        "fresh_reset_trials": len(records),
        "successful_trials": sum(row["success"] for row in records),
        "frozen_generic_object_id": "object_0018",
        "expected_physical_family": "BOWL",
        "instance_token_used_for_runtime_resolution": False,
        "backend_name_exposed_to_planner": False,
        "trials": records,
        "success": passed,
    })
    if not passed:
        raise SystemExit(1)


def c2_open_b1_diagnostic(output: Path) -> None:
    _, inventory, resolution, execution = primary_validation_dispatcher()
    opened_c2 = execution.phase_a.request("OPEN", "C2", execute=True)
    pick = execution.pick("object_0018") if opened_c2["success"] else None
    passed = bool(opened_c2["success"] and pick and pick["success"])
    write(output / "c2_open_b1_diagnostic.json", {
        "sequence": ["OPEN C2", "OPEN B1", "PICK object_0018"],
        "frozen_generic_object_id": "object_0018",
        "inventory_backend_free": not inventory["planner_received_backend_names"],
        "resolution_one_to_one": resolution["one_to_one"],
        "open_c2": opened_c2,
        "b1_pick": pick,
        "close_c2_required": False if passed else None,
        "success": passed,
    })
    if not passed:
        raise SystemExit(1)


def multi_object(output: Path) -> None:
    scene, inventory, resolution, execution = primary_validation_dispatcher()
    # This deliberately contains only Phase-B operations. Storage retrievals
    # and placements are interleaved so that the final state tests composition,
    # not merely a collection of isolated successes.
    requests = (
        ("PICK_PLACE", "object_0013", "object_0003"),  # selected D2 utensil
        ("PICK_PLACE", "object_0020", "serving_area"),  # selected C1 bowl
        ("PICK_PLACE", "object_0016", "serving_area"),  # C2 vessel
        ("PICK_PLACE", "object_0018", "serving_area"),  # B1 bowl
        ("PICK_PLACE", "object_0009", "countertop"),    # kettle return
        ("PICK_PLACE", "object_0010", "countertop"),    # source return
    )
    actions = []
    passed = True
    for _, object_id, destination in requests:
        pick = execution.pick(object_id)
        actions.append(pick)
        if not pick["success"]:
            passed = False
            break
        place = execution.place(object_id, destination)
        actions.append(place)
        if not place["success"]:
            passed = False
            break
    # Settle once, then recompute every relation from the *current* MuJoCo
    # state.  Individual PLACE success records are provenance only and are not
    # accepted as final-state truth.
    for _ in range(400):
        mujoco.mj_step(scene.model, scene.data)
    mujoco.mj_forward(scene.model, scene.data)
    placed_requests = [
        action for action in actions
        if action.get("request", {}).get("action") == "PLACE"
        and action.get("success")
    ]
    binding_by_id = {
        row["generic_object_id"]: row for row in resolution["accepted"]
    }
    inventory_by_id = {
        row["generic_object_id"]: row for row in inventory["objects"]
    }
    floor_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    serving_rows = []
    final_rows = []
    for action in placed_requests:
        post = action["post_place"]
        object_id = post["generic_object_id"]
        backend = binding_by_id[object_id]["physical_backend_body"]
        body_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
        )
        object_geoms = _body_geom_ids(scene.model, body_id)
        target = post["placement_target"]
        support_id = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            target.get("support_backend") or "",
        )
        support_contact = False
        floor_contact = False
        invalid_contacts = []
        other_payload_geoms = {
            geom_id
            for other_id, binding in binding_by_id.items()
            if other_id != object_id
            for geom_id in _body_geom_ids(
                scene.model,
                mujoco.mj_name2id(
                    scene.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    binding["physical_backend_body"],
                ),
            )
        }
        for contact_index in range(scene.data.ncon):
            pair = {
                int(scene.data.contact[contact_index].geom1),
                int(scene.data.contact[contact_index].geom2),
            }
            if not object_geoms & pair:
                continue
            support_contact |= support_id in pair
            floor_contact |= floor_id in pair
            if other_payload_geoms & pair:
                invalid_contacts.append(sorted(pair))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            scene.model,
            scene.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        stable = bool(
            np.linalg.norm(velocity[:3]) <= 0.10
            and np.linalg.norm(velocity[3:]) <= 0.02
        )
        row = {
            "generic_object_id": object_id,
            "destination_kind": target["destination_kind"],
            "current_body_position_world_m": scene.data.xpos[body_id].tolist(),
            "support_contact": support_contact,
            "floor_contact": floor_contact,
            "stable": stable,
            "linear_speed_m_s": float(np.linalg.norm(velocity[3:])),
            "angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
            "invalid_object_contacts": invalid_contacts,
        }
        relation_ok = support_contact and not floor_contact and stable
        if target["destination_kind"] == "SERVING_SUPPORT":
            length, width = execution.manipulation.placement_resolver.footprint(
                object_id
            )
            corners = oriented_rectangle_corners(
                scene.data.xpos[body_id, :2],
                length,
                width,
                _body_yaw(scene.data, body_id),
            )
            support_axis = scene.data.geom_xmat[support_id].reshape(3, 3)[:2, 0]
            containment = rectangle_inside_observed_support(
                corners,
                scene.data.geom_xpos[support_id, :2],
                support_axis,
                float(scene.model.geom_size[support_id, 0] * 2.0),
                float(scene.model.geom_size[support_id, 1] * 2.0),
            )
            row["footprint_corners_world_m"] = corners.tolist()
            row["minimum_edge_margin_m"] = containment["minimum_edge_margin_m"]
            relation_ok &= bool(
                containment["minimum_edge_margin_m"] >= target["edge_margin_m"]
            )
            serving_rows.append((object_id, corners))
        elif target["destination_kind"] == "OBJECT_RELATIVE_DESTINATION":
            intended = target["target_object_id"]
            intended_body = mujoco.mj_name2id(
                scene.model,
                mujoco.mjtObj.mjOBJ_BODY,
                binding_by_id[intended]["physical_backend_body"],
            )
            intended_distance = float(np.linalg.norm(
                scene.data.xpos[body_id, :2]
                - scene.data.xpos[intended_body, :2]
            ))
            competing = {}
            for candidate_id, candidate in inventory_by_id.items():
                if "soup_bowl" not in candidate.get("selected_functions", []):
                    continue
                candidate_body = mujoco.mj_name2id(
                    scene.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    binding_by_id[candidate_id]["physical_backend_body"],
                )
                competing[candidate_id] = float(np.linalg.norm(
                    scene.data.xpos[body_id, :2]
                    - scene.data.xpos[candidate_body, :2]
                ))
            uniquely_closest = all(
                intended_distance + 0.005 < distance
                for candidate_id, distance in competing.items()
                if candidate_id != intended
            )
            row.update(
                intended_target_object_id=intended,
                intended_target_distance_m=intended_distance,
                competing_target_distances_m=competing,
                intended_target_uniquely_closest=uniquely_closest,
            )
            relation_ok &= 0.06 <= intended_distance <= 0.18 and uniquely_closest
        elif target["destination_kind"] == "SOURCE_RETURN":
            source = np.asarray(
                inventory_by_id[object_id]["observed_centroid_world_m"], float
            )
            source_error = float(np.linalg.norm(
                scene.data.xpos[body_id, :2] - source[:2]
            ))
            upright = float(scene.data.xmat[body_id].reshape(3, 3)[2, 2])
            row.update(
                source_return_xy_error_m=source_error,
                upright_alignment=upright,
            )
            relation_ok &= source_error <= 0.04 and upright >= np.cos(np.deg2rad(15.0))
        row["current_physical_relation_verified"] = bool(
            relation_ok and not invalid_contacts
        )
        final_rows.append(row)

    pairwise_serving = []
    for index, (first_id, first) in enumerate(serving_rows):
        for second_id, second in serving_rows[index + 1:]:
            check = oriented_rectangles_clearance(first, second)
            check.update(first_object_id=first_id, second_object_id=second_id)
            pairwise_serving.append(check)
    pairwise_clear = all(
        not row["overlap"] and row["signed_clearance_m"] >= 0.012
        for row in pairwise_serving
    )
    final = {
        "active_payload_welds": [],
        "active_storage_fixtures": [],
        "unexpected_active_storage_fixtures": [],
        "placed_relations": final_rows,
        "pairwise_serving_clearance": pairwise_serving,
        "pairwise_serving_clear": pairwise_clear,
    }
    for equality_id in range(scene.model.neq):
        if not scene.data.eq_active[equality_id]:
            continue
        name = mujoco.mj_id2name(
            scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id
        ) or ""
        if ":pick_weld_" in name:
            final["active_payload_welds"].append(name)
        if name.startswith("storage_fixture_"):
            final["active_storage_fixtures"].append(name)
            if any(
                region in name
                for region in scene.state.opened_containers
            ):
                final["unexpected_active_storage_fixtures"].append(name)
    final["all_relations_verified"] = all(
        row["current_physical_relation_verified"]
        for row in final["placed_relations"]
    )
    final["success"] = bool(
        passed and final["all_relations_verified"]
        and pairwise_clear
        and not final["active_payload_welds"]
        and not final["unexpected_active_storage_fixtures"]
    )
    write(output / "multi_object_validation.json", {
        "inventory_backend_free": not inventory["planner_received_backend_names"],
        "resolution_one_to_one": resolution["one_to_one"],
        "actions": actions,
        "success": passed,
    })
    write(output / "final_physical_relation_validation.json", final)
    if not final["success"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carry-family", choices=(
        "VESSEL", "BOWL", "UTENSIL", "KETTLE", "JAR_SOURCE"
    ))
    parser.add_argument("--multi-object", action="store_true")
    parser.add_argument("--b1-repeatability", action="store_true")
    parser.add_argument("--c2-open-b1-diagnostic", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.multi_object:
        multi_object(args.output_dir)
    elif args.b1_repeatability:
        b1_repeatability(args.output_dir)
    elif args.c2_open_b1_diagnostic:
        c2_open_b1_diagnostic(args.output_dir)
    elif args.carry_family:
        carry(args.carry_family, args.output_dir)
    else:
        parser.error(
            "choose --carry-family, --multi-object, --b1-repeatability, "
            "or --c2-open-b1-diagnostic"
        )


if __name__ == "__main__":
    main()
