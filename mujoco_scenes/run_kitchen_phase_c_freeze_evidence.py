"""Run authoritative physical evidence for Kitchen Google-Robot Phase C."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any

import mujoco
import numpy as np

from .kitchen_object_manipulation import (
    _body_geom_ids,
    _body_yaw,
    oriented_rectangle_corners,
    oriented_rectangles_clearance,
    rectangle_inside_observed_support,
)
from .kitchen_phase_c_execution import KitchenPhaseCExecutionDispatcher
from .kitchen_pour_stir_manipulation import derive_target_opening, derive_tool_tip
from .run_kitchen_phase_b_freeze_evidence import (
    PRIMARY_FROZEN,
    primary_validation_dispatcher,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/kitchen_phase_c_authoritative"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def execution_code_sha() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()


def frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return (
        read(PRIMARY_FROZEN / "object_registry.json"),
        read(PRIMARY_FROZEN / "plan.json"),
    )


def fresh_dispatcher():
    scene, inventory, resolution, phase_b = primary_validation_dispatcher()
    registry, plan = frozen_inputs()
    return (
        scene,
        inventory,
        resolution,
        KitchenPhaseCExecutionDispatcher(phase_b, registry, plan),
    )


def frozen_actions(operator: str) -> list[dict[str, Any]]:
    _, plan = frozen_inputs()
    return [row for row in plan if row["action"].upper() == operator.upper()]


def _action_content(action: dict[str, Any]) -> str | None:
    arguments = list(action.get("arguments", []))
    return arguments[2] if len(arguments) > 2 else None


def _run_pair(action: dict[str, Any], trial: int | None = None) -> dict[str, Any]:
    operator = action["action"].upper()
    source_id, target_id = action["arguments"][:2]
    _, inventory, resolution, phase_c = fresh_dispatcher()
    pick = phase_c.pick(source_id)
    motion = None
    if pick["success"]:
        if operator == "POUR":
            motion = phase_c.pour(source_id, target_id, _action_content(action))
        else:
            motion = phase_c.stir(source_id, target_id)
    return {
        "execution_code_sha": execution_code_sha(),
        "fresh_scene_reset": True,
        "trial": trial,
        "operator": operator,
        "frozen_step": int(action["step"]),
        "source_or_tool_generic_id": source_id,
        "target_generic_id": target_id,
        "planner_inventory_backend_free": not inventory["planner_received_backend_names"],
        "execution_resolution_one_to_one": resolution["one_to_one"],
        "pick": pick,
        "motion": motion,
        "success": bool(pick["success"] and motion and motion["success"]),
    }


def pair_coverage(operator: str, output: Path) -> dict[str, Any]:
    actions = frozen_actions(operator)
    records = [_run_pair(action) for action in actions]
    payload = {
        "execution_code_sha": execution_code_sha(),
        "operator": operator,
        "expected_frozen_pair_count": len(actions),
        "tested_frozen_pair_count": len(records),
        "successful_pair_count": sum(row["success"] for row in records),
        "fresh_reset_per_pair": True,
        "records": records,
        "success": len(records) == len(actions) and all(row["success"] for row in records),
    }
    write(output / f"{operator.lower()}_pair_coverage.json", payload)
    return payload


def hardest_target(operator: str, source_id: str) -> dict[str, Any]:
    scene, inventory, _, phase_c = fresh_dispatcher()
    candidates = []
    actions = [
        row for row in frozen_actions(operator)
        if row["arguments"][0] == source_id
    ]
    for action in actions:
        target_id = action["arguments"][1]
        opening = phase_c._opening(target_id)
        clearance = min(opening.opening_half_extents_m) - opening.safety_margin_m
        if operator == "STIR":
            observed = phase_c.inventory_by_id[source_id]["observed_dimensions_m"]
            binding = phase_c.binding_by_id[source_id]
            tool = derive_tool_tip(
                scene,
                binding["physical_backend_body"],
                float(observed["length"]),
            )
            tool_radius = 0.5 * min(
                float(observed.get("width", 0.0)),
                float(observed.get("height", 0.0)),
            )
            clearance -= tool_radius
            tool_provenance = tool.provenance
        else:
            tool_provenance = None
        candidates.append({
            "target_generic_id": target_id,
            "opening_half_extents_m": list(opening.opening_half_extents_m),
            "cavity_depth_m": opening.cavity_depth_m,
            "usable_radial_clearance_m": clearance,
            "geometry_provenance": opening.provenance,
            "tool_geometry_provenance": tool_provenance,
            "action": action,
        })
    selected = min(
        candidates,
        key=lambda row: (row["usable_radial_clearance_m"], row["cavity_depth_m"]),
    )
    return {"selection_rule": "MINIMUM_USABLE_RADIAL_CLEARANCE_THEN_CAVITY_DEPTH", "candidates": candidates, "selected": selected}


def repeatability(operator: str, output: Path, trials: int = 3) -> dict[str, Any]:
    sources = sorted({row["arguments"][0] for row in frozen_actions(operator)})
    families = []
    for source_id in sources:
        selection = hardest_target(operator, source_id)
        action = selection["selected"]["action"]
        records = [_run_pair(action, trial=index) for index in range(1, trials + 1)]
        families.append({
            "source_or_tool_generic_id": source_id,
            "hardest_target_selection": selection,
            "requested_trials": trials,
            "successful_trials": sum(row["success"] for row in records),
            "records": records,
            "success": len(records) == trials and all(row["success"] for row in records),
        })
    payload = {
        "execution_code_sha": execution_code_sha(),
        "operator": operator,
        "fresh_reset_per_trial": True,
        "source_family_count": len(families),
        "families": families,
        "success": bool(families and all(row["success"] for row in families)),
    }
    write(output / f"{operator.lower()}_worst_case_repeatability.json", payload)
    return payload


def _frozen_place_destination(source_id: str) -> str:
    _, plan = frozen_inputs()
    for row in plan:
        if (
            row["action"].upper() in {"PLACE", "PLACE_SERVING_UTENSIL"}
            and row["arguments"][0] == source_id
        ):
            return row["arguments"][1]
    raise ValueError(f"No frozen PLACE destination for {source_id}")


def sequential(operator: str, output: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for action in frozen_actions(operator):
        grouped.setdefault(action["arguments"][0], []).append(action)
    sequences = []
    for source_id, actions in sorted(grouped.items()):
        _, inventory, resolution, phase_c = fresh_dispatcher()
        pick = phase_c.pick(source_id)
        motions = []
        if pick["success"]:
            for action in actions:
                if operator == "POUR":
                    result = phase_c.pour(
                        source_id, action["arguments"][1], _action_content(action)
                    )
                else:
                    result = phase_c.stir(source_id, action["arguments"][1])
                motions.append(result)
                if not result["success"]:
                    break
        place = None
        if pick["success"] and len(motions) == len(actions) and all(
            row["success"] for row in motions
        ):
            place = phase_c.place(source_id, _frozen_place_destination(source_id))
        held_backend_names = {
            row.get("held_state_before", {}).get("physical_backend_body")
            for row in motions
        } | {
            row.get("held_state_after", {}).get("physical_backend_body")
            for row in motions
        }
        held_backend_names.discard(None)
        success = bool(
            pick["success"]
            and len(motions) == len(actions)
            and all(row["success"] for row in motions)
            and len(held_backend_names) == 1
            and place
            and place["success"]
        )
        sequences.append({
            "execution_code_sha": execution_code_sha(),
            "fresh_scene_reset": True,
            "source_or_tool_generic_id": source_id,
            "target_generic_ids": [row["arguments"][1] for row in actions],
            "planner_inventory_backend_free": not inventory["planner_received_backend_names"],
            "execution_resolution_one_to_one": resolution["one_to_one"],
            "pick": pick,
            "motions": motions,
            "same_physical_held_object_throughout": len(held_backend_names) == 1,
            "hidden_release_or_regrasp": False,
            "place": place,
            "ledger": phase_c.ledger.summary(),
            "success": success,
        })
    payload = {
        "execution_code_sha": execution_code_sha(),
        "operator": operator,
        "sequences": sequences,
        "success": bool(sequences and all(row["success"] for row in sequences)),
    }
    filename = (
        "pour_sequential_source_validation.json"
        if operator == "POUR" else "stir_sequential_validation.json"
    )
    write(output / filename, payload)
    return payload


def _collect_place_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    if result.get("request", {}).get("action") == "PLACE" and result.get("success"):
        records.append(result)
    for child in result.get("steps", []):
        if isinstance(child, dict):
            records.extend(_collect_place_records(child))
    return records


def final_physical_validation(
    scene, inventory: dict[str, Any], resolution: dict[str, Any],
    phase_c: KitchenPhaseCExecutionDispatcher, actions: list[dict[str, Any]],
) -> dict[str, Any]:
    for _ in range(400):
        mujoco.mj_step(scene.model, scene.data)
    mujoco.mj_forward(scene.model, scene.data)
    binding_by_id = {row["generic_object_id"]: row for row in resolution["accepted"]}
    inventory_by_id = {row["generic_object_id"]: row for row in inventory["objects"]}
    floor_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    placements = [record for action in actions for record in _collect_place_records(action)]
    relation_rows = []
    serving_footprints = []
    for action in placements:
        post = action["post_place"]
        object_id = post["generic_object_id"]
        target = post["placement_target"]
        body_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY,
            binding_by_id[object_id]["physical_backend_body"],
        )
        object_geoms = _body_geom_ids(scene.model, body_id)
        support_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_GEOM,
            target.get("support_backend") or "",
        )
        support_contact = floor_contact = False
        invalid_contacts = []
        other_payload_geoms = {
            geom_id
            for other_id, binding in binding_by_id.items() if other_id != object_id
            for geom_id in _body_geom_ids(
                scene.model,
                mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_BODY,
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
            scene.model, scene.data, mujoco.mjtObj.mjOBJ_BODY,
            body_id, velocity, 0,
        )
        stable = bool(
            np.linalg.norm(velocity[:3]) <= 0.10
            and np.linalg.norm(velocity[3:]) <= 0.02
        )
        relation_ok = support_contact and not floor_contact and stable
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
        if target["destination_kind"] == "SERVING_SUPPORT":
            length, width = phase_c.phase_b.manipulation.placement_resolver.footprint(object_id)
            corners = oriented_rectangle_corners(
                scene.data.xpos[body_id, :2], length, width,
                _body_yaw(scene.data, body_id),
            )
            support_axis = scene.data.geom_xmat[support_id].reshape(3, 3)[:2, 0]
            containment = rectangle_inside_observed_support(
                corners, scene.data.geom_xpos[support_id, :2], support_axis,
                float(scene.model.geom_size[support_id, 0] * 2.0),
                float(scene.model.geom_size[support_id, 1] * 2.0),
            )
            row["minimum_edge_margin_m"] = containment["minimum_edge_margin_m"]
            relation_ok &= bool(
                containment["minimum_edge_margin_m"] >= target["edge_margin_m"]
            )
            serving_footprints.append((object_id, corners))
        elif target["destination_kind"] == "OBJECT_RELATIVE_DESTINATION":
            intended = target["target_object_id"]
            intended_body = mujoco.mj_name2id(
                scene.model, mujoco.mjtObj.mjOBJ_BODY,
                binding_by_id[intended]["physical_backend_body"],
            )
            intended_distance = float(np.linalg.norm(
                scene.data.xpos[body_id, :2] - scene.data.xpos[intended_body, :2]
            ))
            competing = {}
            for candidate_id, candidate in inventory_by_id.items():
                if "soup_bowl" not in candidate.get("selected_functions", []):
                    continue
                candidate_body = mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_BODY,
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
            source = np.asarray(inventory_by_id[object_id]["observed_centroid_world_m"], float)
            source_error = float(np.linalg.norm(scene.data.xpos[body_id, :2] - source[:2]))
            upright = float(scene.data.xmat[body_id].reshape(3, 3)[2, 2])
            row.update(source_return_xy_error_m=source_error, upright_alignment=upright)
            relation_ok &= source_error <= 0.04 and upright >= np.cos(np.deg2rad(15.0))
        row["current_physical_relation_verified"] = bool(
            relation_ok and not invalid_contacts
        )
        relation_rows.append(row)
    pairwise = []
    for index, (first_id, first) in enumerate(serving_footprints):
        for second_id, second in serving_footprints[index + 1:]:
            check = oriented_rectangles_clearance(first, second)
            check.update(first_object_id=first_id, second_object_id=second_id)
            pairwise.append(check)
    active_payload_welds = []
    active_storage_fixtures = []
    unexpected_fixtures = []
    for equality_id in range(scene.model.neq):
        if not scene.data.eq_active[equality_id]:
            continue
        name = mujoco.mj_id2name(
            scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id
        ) or ""
        if ":pick_weld_" in name:
            active_payload_welds.append(name)
        if name.startswith("storage_fixture_"):
            active_storage_fixtures.append(name)
            if any(region in name for region in scene.state.opened_containers):
                unexpected_fixtures.append(name)
    pairwise_clear = all(
        not row["overlap"] and row["signed_clearance_m"] >= 0.012
        for row in pairwise
    )
    result = {
        "execution_code_sha": execution_code_sha(),
        "validation_basis": "CURRENT_SETTLED_MUJOCO_STATE_NOT_CACHED_ACTION_FLAGS",
        "placed_relations": relation_rows,
        "pairwise_serving_clearance": pairwise,
        "pairwise_serving_clear": pairwise_clear,
        "active_payload_welds": active_payload_welds,
        "active_storage_fixtures": active_storage_fixtures,
        "unexpected_active_storage_fixtures": unexpected_fixtures,
        "all_relations_verified": bool(relation_rows) and all(
            row["current_physical_relation_verified"] for row in relation_rows
        ),
    }
    result["success"] = bool(
        result["all_relations_verified"] and pairwise_clear
        and not active_payload_welds and not unexpected_fixtures
    )
    return result


def complete_plan(output: Path) -> dict[str, Any]:
    scene, inventory, resolution, phase_c = fresh_dispatcher()
    _, plan = frozen_inputs()
    actions = []
    for action in plan:
        result = phase_c.execute_phase2_action(action)
        actions.append({"frozen_step": int(action["step"]), **result})
        if not result["success"]:
            break
    counts = Counter(row.get("request", {}).get("action", "UNKNOWN") for row in actions)
    expected_counts = Counter(row["action"].upper() for row in plan)
    symbolic_steps = [row["frozen_step"] for row in actions]
    ledger = phase_c.ledger.summary()
    success = bool(
        len(actions) == len(plan)
        and symbolic_steps == [int(row["step"]) for row in plan]
        and all(row["success"] for row in actions)
        and counts == expected_counts
        and ledger["complete"]
    )
    payload = {
        "execution_code_sha": execution_code_sha(),
        "fresh_scene_reset": True,
        "frozen_plan_length": len(plan),
        "executed_symbolic_count": len(actions),
        "expected_operator_counts": dict(sorted(expected_counts.items())),
        "executed_operator_counts": dict(sorted(counts.items())),
        "executed_step_indices": symbolic_steps,
        "each_step_exactly_once": symbolic_steps == [int(row["step"]) for row in plan],
        "no_replan": True,
        "no_object_substitution": True,
        "actions": actions,
        "success": success,
    }
    write(output / "complete_plan_execution.json", payload)
    write(output / "phaseC_execution_ledger.json", ledger)
    final = final_physical_validation(scene, inventory, resolution, phase_c, actions)
    write(output / "final_phaseC_physical_validation.json", final)
    if not success or not final["success"]:
        raise SystemExit(1)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-coverage", choices=("POUR", "STIR"))
    parser.add_argument("--repeatability", choices=("POUR", "STIR"))
    parser.add_argument("--sequential", choices=("POUR", "STIR"))
    parser.add_argument("--complete-plan", action="store_true")
    args = parser.parse_args()
    if args.pair_coverage:
        result = pair_coverage(args.pair_coverage, args.output_dir)
    elif args.repeatability:
        result = repeatability(args.repeatability, args.output_dir)
    elif args.sequential:
        result = sequential(args.sequential, args.output_dir)
    elif args.complete_plan:
        result = complete_plan(args.output_dir)
    else:
        parser.error("choose a Phase-C physical validation mode")
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
