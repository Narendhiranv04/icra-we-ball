"""Run the remaining physical evidence required by the Phase-B freeze."""

from __future__ import annotations

import argparse
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
PRIMARY_FROZEN = ROOT / "runs/integrated_no_pot_clearance_seed19_20260807"


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
    """Bind fresh primary-scene observations to frozen functional IDs.

    The current registry supplies all centroids, semantics, and source-region
    evidence.  The previously frozen primary witness/plan supplies only role
    selection and usage; it never supplies backend names or current poses.
    """
    registry = read(PRIMARY_CURRENT / "object_registry.json")
    c1_registry = read(PRIMARY_C1_CURRENT / "object_registry.json")
    for object_id, record in c1_registry["objects"].items():
        if record.get("source_region") == "C1":
            registry["objects"][object_id] = record
    assignments = read(PRIMARY_FROZEN / "grounded_role_assignments.json")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.multi_object:
        multi_object(args.output_dir)
    elif args.carry_family:
        carry(args.carry_family, args.output_dir)
    else:
        parser.error("choose --carry-family or --multi-object")


if __name__ == "__main__":
    main()
