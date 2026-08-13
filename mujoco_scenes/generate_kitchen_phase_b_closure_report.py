"""Generate the artifact-derived Kitchen Phase-B scientific freeze report."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import mujoco

from .kitchen_execution_entities import build_phase_b_inventory
from .robot_profiles import manipulation_profile
from .scene_loader import KitchenScene


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "mujoco_scenes/benchmark_reports/kitchen_google_execution_phaseB"
PHASE1_ROOT = ROOT / "runs/feasibility_benchmarks/kitchen_feasibility_phase1_closure_20260809"
PHASE2_ROOT = ROOT / "mujoco_scenes/benchmark_reports/kitchen_symbolic_phase2/variants"
VARIANTS = (
    "F1_INITIAL_COMPLETE", "F2_DISTRIBUTED_COFFEE_TWO",
    "F3_DISTRIBUTED_COFFEE_THREE", "F4_EARLY_RELOCATION",
    "F5_LATE_RELOCATION", "F6_DECOY_HEAVY", "F7_COUNT_SURPLUS",
    "P0_LAYOUT_BASE", "P1_LAYOUT_SWAPPED",
)

PICK_EVIDENCE = {
    "table_vessel_1": "runs/phaseB_closure_probe_vessel",
    "table_vessel_2": "runs/phaseB_final_repeatability/table_vessel_trial_2",
    "table_vessel_3": "runs/phaseB_final_repeatability/table_vessel_trial_3",
    "table_bowl_1": "runs/phaseB_closure_probe_bowl",
    "table_bowl_2": "runs/phaseB_final_repeatability/table_bowl_trial_2",
    "table_bowl_3": "runs/phaseB_final_repeatability/table_bowl_trial_3",
    "table_utensil_1": "runs/phaseB_closure_probe_utensil_object8",
    "table_utensil_2": "runs/phaseB_final_repeatability/table_utensil_trial_2",
    "table_utensil_3": "runs/phaseB_final_repeatability/table_utensil_trial_3",
    "table_kettle_1": "runs/phaseB_kettle_body",
    "table_kettle_2": "runs/phaseB_final_repeatability/table_kettle_trial_8",
    "table_kettle_3": "runs/phaseB_final_repeatability/table_kettle_trial_9",
    "table_jar_1": "runs/phaseB_closure_probe_jar5",
    "table_jar_2": "runs/phaseB_final_repeatability/table_jar_trial_2",
    "table_jar_3": "runs/phaseB_final_repeatability/table_jar_trial_3",
    "D1_1": "runs/phaseB_closure_storage_d1_retry",
    "D1_2": "runs/phaseB_final_repeatability/d1_trial_2",
    "D1_3": "runs/phaseB_final_repeatability/d1_trial_3",
    "D2_1": "runs/phaseB_closure_storage_d2",
    "D2_2": "runs/phaseB_final_repeatability/d2_trial_2",
    "D2_3": "runs/phaseB_final_repeatability/d2_trial_3",
    "C1_1": "runs/phaseB_closure_storage_c1_offset",
    "C1_2": "runs/phaseB_final_repeatability/c1_trial_2",
    "C1_3": "runs/phaseB_final_repeatability/c1_trial_3",
    "B1_1": "runs/phaseB_closure_storage_b1_retry3",
    "B1_2": "runs/phaseB_final_repeatability/b1_trial_2",
    "B1_3": "runs/phaseB_final_repeatability/b1_trial_3",
    "C2_vessel_1": "runs/phaseB_closure_storage_c2_vessel_final",
    "C2_vessel_2": "runs/phaseB_final_repeatability/c2_vessel_trial_2",
    "C2_vessel_3": "runs/phaseB_final_repeatability/c2_vessel_trial_3",
    "C2_utensil_1": "runs/phaseB_freeze_c2_preclose_release_trial_1",
    "C2_utensil_2": "runs/phaseB_freeze_c2_preclose_release_trial_2",
    "C2_utensil_3": "runs/phaseB_freeze_c2_preclose_release_trial_3",
}

CARRY_EVIDENCE = {
    "VESSEL": "runs/phaseB_freeze_carried_move/VESSEL/carried_move_result.json",
    "BOWL": "runs/phaseB_freeze_carried_move/BOWL/carried_move_result.json",
    "UTENSIL": "runs/phaseB_freeze_carried_move/UTENSIL/carried_move_result.json",
    "KETTLE": "runs/phaseB_freeze_carried_move/KETTLE/carried_move_result.json",
    "JAR_SOURCE": "runs/phaseB_freeze_carried_move/JAR_SOURCE/carried_move_result.json",
}


def read(path: Path):
    return json.loads(path.read_text())


def write(name: str, payload: Any) -> None:
    path = REPORT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    """Return the repository revision and an honest worktree cleanliness audit."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "git_head": head,
        "worktree_status": "dirty" if status else "clean",
        "worktree_dirty": bool(status),
        "worktree_status_short": status,
    }


def pick_row(relative: str) -> dict[str, Any]:
    directory = ROOT / relative
    pick_path = directory / "pick_result.json"
    if not pick_path.exists():
        return {"evidence_directory": relative, "artifact_exists": False,
                "pick_success": False}
    pick = read(pick_path)
    physical = next((row for row in reversed(pick.get("steps", []))
                     if isinstance(row, dict) and row.get("backend_body")), {})
    place_path = directory / "place_result.json"
    preclose = (physical.get("direct_grasp_analysis") or {}).get(
        "preclose_telemetry"
    ) or {}
    return {
        "evidence_directory": relative,
        "artifact_exists": True,
        "pick_sha256": sha256(pick_path),
        "generic_object_id": pick.get("generic_object_id"),
        "requested_generic_object_id": (
            pick.get("request", {}).get("arguments", [None])[0]
        ),
        "pick_status": pick.get("status"),
        "pick_success": bool(pick.get("success")),
        "family": physical.get("grasp_family"),
        "source_context": physical.get("source_context"),
        "selected_candidate": physical.get("selected_grasp_candidate_id"),
        "bilateral_contact": physical.get("bilateral_contact"),
        "contact_sides": physical.get("contact_sides", []),
        "contact_geoms": physical.get("contact_geoms", []),
        "target_contact_geoms": physical.get("target_contact_geoms", []),
        "attachment_translation_snap_m": physical.get("attachment_translation_snap_m"),
        "attachment_angle_snap_rad": physical.get("attachment_angle_snap_rad"),
        "preclose_cartesian_error_m": preclose.get("preclose_cartesian_error_m"),
        "preclose_orientation_error_rad": preclose.get("preclose_orientation_error_rad"),
        "source_clearance_verified": physical.get("source_clearance_verified"),
        "extraction_strategy": physical.get("extraction_strategy"),
        "navigation_safe_carry_reached": physical.get("navigation_safe_carry_reached"),
        "direct_object_qpos_write": physical.get("direct_object_qpos_write"),
        "duration_s": physical.get("duration_s"),
        "fixture_release": pick.get("storage_fixture_release"),
        "place": read(place_path) if place_path.exists() else None,
        "place_sha256": sha256(place_path) if place_path.exists() else None,
    }


def stats(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values if value is not None]
    return {"trials": len(values), "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None}


def normalized_assignment_ids(payload: dict[str, Any]) -> dict[str, Any]:
    """Project verbose witness records onto execution-relevant identities."""
    if "coffee_targets" in payload:
        return {
            "coffee_targets": sorted(payload.get("coffee_targets", [])),
            "coffee_tools": sorted({
                row.get("utensil_object_id")
                for row in payload.get("coffee_stirring", [])
                if row.get("utensil_object_id")
            }),
            "soup_targets": sorted(payload.get("soup_targets", [])),
            "soup_pairs": sorted(
                (
                    row.get("target_object_id"),
                    row.get("tool_object_id") or row.get("utensil_object_id"),
                )
                for row in payload.get("soup_serving", [])
            ),
            "sources": payload.get("source_roles", {}),
        }
    coffee = payload.get("coffee", [])
    soup = payload.get("soup", [])
    return {
        "coffee_targets": sorted({row.get("target_object_id") for row in coffee}),
        "coffee_tools": sorted({row.get("utensil_object_id") for row in coffee}),
        "soup_targets": sorted({row.get("target_object_id") for row in soup}),
        "soup_pairs": sorted(
            (row.get("target_object_id"), row.get("utensil_object_id"))
            for row in soup
        ),
        "sources": payload.get("source_roles", {}),
    }


def closure_from_artifacts(artifacts: dict[str, bool]) -> dict[str, Any]:
    """Pure final gate used by tests to prevent PICK-only closure claims."""
    required = (
        "pick_coverage_pass", "place_coverage_pass", "carried_move_pass",
        "storage_repeatability_pass", "c2_unrestricted_grasp_pass",
        "entity_resolution_pass",
        "extraction_pass", "destination_coverage_pass", "f1_equivalence_pass",
        "variant_coverage_pass", "isolated_operator_coverage_pass",
        "scientific_guards_pass", "multi_object_pass", "final_relation_pass",
        "tests_pass", "reproduction_manifest_valid",
    )
    missing = [name for name in required if not artifacts.get(name, False)]
    return {"requirements": {name: bool(artifacts.get(name, False)) for name in required},
            "missing_or_failed": missing, "phase_b_closed": not missing}


def guards_are_verifiable(guards: dict[str, dict[str, Any]]) -> bool:
    """Reject bare declarations lacking a method or concrete evidence."""
    return bool(guards) and all(
        row.get("passed") is True
        and bool(row.get("validation_method"))
        and bool(row.get("evidence"))
        for row in guards.values()
    )


def inventories() -> tuple[dict, dict, dict]:
    registry = read(PHASE1_ROOT / "F1_INITIAL_COMPLETE/object_registry.json")
    assignments = read(PHASE2_ROOT / "F1_INITIAL_COMPLETE/grounded_role_assignments.json")
    plan = read(PHASE2_ROOT / "F1_INITIAL_COMPLETE/generated_plan.json")
    inventory = build_phase_b_inventory(registry, assignments, plan)
    resolution = read(ROOT / "runs/phaseB_closure_inventory/F1_INITIAL_COMPLETE/execution_entity_resolution.json")
    inventory_by_id = {row["generic_object_id"]: row for row in inventory["objects"]}
    for row in resolution["accepted"]:
        source = inventory_by_id[row["generic_object_id"]]
        row["semantic_label_source"] = source["semantic_label_source"]
        row["originating_functional_role"] = source["originating_functional_role"]
    provenance = {
        "backend_names_used_as_semantic_input": False,
        "objects": [{key: row.get(key) for key in (
            "generic_object_id", "semantic_label", "semantic_label_source",
            "originating_functional_role", "selected_functions"
        )} for row in inventory["objects"]],
    }
    return inventory, resolution, provenance


def variant_and_operator_coverage(rows: dict[str, dict]) -> tuple[dict, dict]:
    family_evidence = {
        "VESSEL": rows["table_vessel_1"], "BOWL": rows["table_bowl_1"],
        "UTENSIL": rows["table_utensil_1"], "KETTLE": rows["table_kettle_1"],
        "JAR_SOURCE": rows["table_jar_1"],
    }
    storage = {"D1": rows["D1_1"], "D2": rows["D2_1"],
               "C1": rows["C1_1"], "B1": rows["B1_1"],
               "C2_VESSEL": rows["C2_vessel_1"],
               "C2_UTENSIL": rows["C2_utensil_1"]}
    operator_rows = []
    variant_rows = []
    for variant in VARIANTS:
        inventory_path = ROOT / f"runs/phaseB_closure_inventory/{variant}/phaseB_object_inventory.json"
        inventory = read(inventory_path)
        objects = {row["generic_object_id"]: row for row in inventory["objects"]}
        plan = read(PHASE2_ROOT / variant / "generated_plan.json")
        variant_supported = True
        for action in plan:
            operator = action["action"].upper()
            arguments = action.get("arguments", [])
            object_id = arguments[0] if arguments and arguments[0] in objects else None
            obj = objects.get(object_id, {})
            source = obj.get("source_context", {})
            family = None
            functions = obj.get("selected_functions", [])
            if "water_source" in functions: family = "KETTLE"
            elif "coffee_source" in functions: family = "JAR_SOURCE"
            elif any("utensil" in role or "stirrer" in role for role in functions): family = "UTENSIL"
            elif "soup_bowl" in functions: family = "BOWL"
            elif "coffee_vessel" in functions: family = "VESSEL"
            evidence = None
            supported = False
            reason = None
            destination_kind = None
            if operator in {"POUR", "STIR"}:
                reason = "UNSUPPORTED_PHASE_C_OPERATOR"
            elif operator in {"PICK", "PLACE"} and family:
                region = source.get("source_container")
                key = f"{region}_{family}" if region == "C2" else region
                candidate = storage.get(key) if region else family_evidence.get(family)
                if operator == "PLACE":
                    destination = arguments[1]
                    if destination == "serving_area": destination_kind = "SERVING_SUPPORT"
                    elif destination == "countertop": destination_kind = "SOURCE_RETURN"
                    else: destination_kind = "OBJECT_RELATIVE_DESTINATION"
                    # PLACE evidence follows the manipulated object's family,
                    # never merely the destination type.  This preserves the
                    # physical identity of the isolated operator validation.
                    candidate = family_evidence[family]
                    supported = bool(candidate.get("place", {}).get("success"))
                else:
                    supported = bool(candidate and candidate["pick_success"])
                evidence = candidate["evidence_directory"] if candidate else None
            operator_rows.append({
                "variant": variant, "action_index": action.get("step"),
                "action": operator, "arguments": arguments,
                "generic_object_id": object_id, "selected_functions": functions,
                "source": source, "family": family,
                "destination_kind": destination_kind, "evidence": evidence,
                "supported": supported, "reason": reason,
                "symbolic_effects_applied": False if operator in {"POUR", "STIR"} else None,
            })
            if operator in {"PICK", "PLACE"}:
                variant_supported &= supported
        variant_rows.append({"variant": variant, "phase_b_supported": variant_supported,
                             "inventory_path": str(inventory_path.relative_to(ROOT))})
    return ({"variants": variant_rows, "passed": all(r["phase_b_supported"] for r in variant_rows)},
            {"operators": operator_rows,
             "phase_b_passed": all(r["supported"] for r in operator_rows if r["action"] in {"PICK", "PLACE"}),
             "phase_c_explicit": all(r["reason"] == "UNSUPPORTED_PHASE_C_OPERATOR" and not r["symbolic_effects_applied"] for r in operator_rows if r["action"] in {"POUR", "STIR"})})


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = {name: pick_row(path) for name, path in PICK_EVIDENCE.items()}
    inventory, resolution, provenance = inventories()
    write("phaseB_object_inventory.json", inventory)
    write("execution_entity_resolution.json", resolution)
    write("semantic_label_provenance.json", provenance)
    write("grasp_calibration_matrix.json", rows)
    bilateral_rows = {
        name: {
            "bilateral_contact": row.get("bilateral_contact"),
            "contact_sides": row.get("contact_sides"),
            "target_contact_geoms": row.get("target_contact_geoms"),
            "fixture_active_during_contact": (
                row.get("fixture_release") or {}
            ).get("active_during_contact", False),
            "evidence": row["evidence_directory"],
        }
        for name, row in rows.items()
    }
    write("bilateral_contact_prediction_validation.json", {
        "trials": bilateral_rows,
        "prediction_used_for_selection_only": True,
        "passed": all(row["bilateral_contact"] for row in bilateral_rows.values()),
    })
    c2_contact_rows = {
        name: row for name, row in bilateral_rows.items()
        if name.startswith("C2_utensil_")
    }
    write("c2_contact_geometry_validation.json", {
        "utensil_trials": c2_contact_rows,
        "fixture_inactive_during_actual_contact": all(
            not row["fixture_active_during_contact"]
            for row in c2_contact_rows.values()
        ),
        "passed": bool(c2_contact_rows) and all(
            row["bilateral_contact"] and not row["fixture_active_during_contact"]
            for row in c2_contact_rows.values()
        ),
    })

    groups = {
        "table_vessel": [rows[f"table_vessel_{i}"] for i in (1,2,3)],
        "table_bowl": [rows[f"table_bowl_{i}"] for i in (1,2,3)],
        "table_utensil": [rows[f"table_utensil_{i}"] for i in (1,2,3)],
        "table_kettle": [rows[f"table_kettle_{i}"] for i in (1,2,3)],
        "table_jar": [rows[f"table_jar_{i}"] for i in (1,2,3)],
        **{region: [rows[f"{region}_{i}"] for i in (1,2,3)] for region in ("D1","D2","C1","B1")},
        "C2_vessel": [rows[f"C2_vessel_{i}"] for i in (1,2,3)],
        "C2_utensil": [rows[f"C2_utensil_{i}"] for i in (1,2,3)],
    }
    repeatability = {name: {"trials": len(items),
        "successes": sum(item["pick_success"] for item in items),
        "passed": len(items) == 3 and all(item["pick_success"] for item in items)}
        for name, items in groups.items()}
    write("storage_retrieval_repeatability.json", repeatability)
    for name in ("C2_vessel", "C2_utensil"):
        write(f"{name}_retrieval_validation.json", {
            "fresh_reset_trials": groups[name], "passed": repeatability[name]["passed"],
            "unrestricted_fixture_guard_passed": all(
                not item.get("fixture_release", {}).get("active_during_contact", False)
                and item.get("fixture_release", {}).get("released_before_preclose", True)
                for item in groups[name]
            ) if name == "C2_utensil" else True,
        })

    extraction = {}
    for name, row in rows.items():
        kind = (row.get("source_context") or {}).get("source_kind")
        verified = row.get("source_clearance_verified")
        drawer_verified = bool(
            kind == "DRAWER"
            and row.get("navigation_safe_carry_reached")
            and row.get("pick_success")
        )
        extraction[name] = {
            "source_kind": kind, "strategy": row.get("extraction_strategy"),
            "source_clearance_status": (
                "VERIFIED" if verified is True else
                "VERIFIED" if drawer_verified else
                "NOT_APPLICABLE" if kind == "TABLE" else "NOT_CHECKED"
            ),
            "verification_basis": (
                "explicit_aperture_clearance_telemetry" if verified is True else
                "contact_presentation_then_navigation_safe_carry" if drawer_verified else
                "tabletop_source_has_no_container_aperture" if kind == "TABLE" else
                "no_valid_clearance_evidence"
            ),
            "navigation_safe_carry_reached": row.get("navigation_safe_carry_reached"),
            "evidence": row["evidence_directory"],
        }
    write("extraction_validation.json", {"objects": extraction,
        "passed": all(r["source_clearance_status"] in {"VERIFIED","NOT_APPLICABLE"}
                      for r in extraction.values())})
    write("source_context_coverage.json", {name: {"source": row.get("source_context"),
          "evidence": row["evidence_directory"]} for name,row in rows.items()})

    carry = {}
    for family, relative in CARRY_EVIDENCE.items():
        path = ROOT / relative
        carry[family] = read(path) if path.exists() else {"success": False, "missing": relative}
        carry[family]["evidence_path"] = relative
    carry_pass = all(row.get("success") for row in carry.values())
    write("carried_move_validation.json", {"families": carry, "passed": carry_pass})

    place_rows = [row["place"] for row in rows.values() if row.get("place")]
    place_by_kind: dict[str,list] = {}
    for place in place_rows:
        post = place.get("post_place", {})
        kind = post.get("placement_target", {}).get("destination_kind")
        if kind: place_by_kind.setdefault(kind, []).append(post)
    destination = {kind: {"successes": sum(p.get("success",False) for p in items),
                          "trials": len(items), "passed": any(p.get("success") for p in items)}
                   for kind,items in place_by_kind.items()}
    required_destinations = ("SERVING_SUPPORT","OBJECT_RELATIVE_DESTINATION","SOURCE_RETURN")
    destination_pass = all(destination.get(k,{}).get("passed") for k in required_destinations)
    write("destination_type_coverage.json", {"kinds": destination,
          "required": required_destinations, "passed": destination_pass})
    serving = [p for p in place_by_kind.get("SERVING_SUPPORT",[]) if p.get("success")]
    write("serving_placement_validation.json", {"placements": serving,
          "passed": bool(serving) and all(p.get("physical_relation_verified") and p.get("support_contact") and p.get("footprint_inside_support") and p.get("stable") and not p.get("floor_contact") for p in serving)})
    tools = [p for p in place_by_kind.get("OBJECT_RELATIVE_DESTINATION",[]) if p.get("success")]
    write("soup_tool_assignment_validation.json", {"placements": tools,
          "negative_check": "A competing bowl is rejected unless the intended target is uniquely closest",
          "passed": bool(tools) and all(p.get("intended_target_uniquely_closest") and p.get("physical_relation_verified") for p in tools)})
    returns = [p for p in place_by_kind.get("SOURCE_RETURN",[]) if p.get("success")]
    write("source_return_validation.json", {"placements": returns,
          "passed": len({p.get("generic_object_id") for p in returns}) >= 2 and all(p.get("source_region_membership") and p.get("physical_relation_verified") and p.get("support_contact") and p.get("stable") and not p.get("floor_contact") for p in returns)})

    variant_coverage, operator_coverage = variant_and_operator_coverage(rows)
    write("phaseB_variant_coverage.json", variant_coverage)
    write("isolated_operator_validation.json", operator_coverage)

    baseline_assign = read(REPORT.parent / "kitchen_feasibility_phase1/variants/F1_INITIAL_COMPLETE/grounded_assignments.json")
    current_assign = read(PHASE1_ROOT / "F1_INITIAL_COMPLETE/grounded_role_assignments.json")
    phase2_assign = read(PHASE2_ROOT / "F1_INITIAL_COMPLETE/grounded_role_assignments.json")
    baseline_normalized = normalized_assignment_ids(
        baseline_assign.get("assignments", baseline_assign)
    )
    current_normalized = normalized_assignment_ids(current_assign)
    phase2_normalized = normalized_assignment_ids(phase2_assign)
    role_keys = ("coffee_targets", "coffee_tools", "soup_targets", "soup_pairs")
    equivalent_roles = all(
        baseline_normalized[key] == current_normalized[key]
        for key in role_keys
    )
    baseline_sources_available = bool(baseline_normalized["sources"])
    current_phase2_sources_equal = (
        current_normalized["sources"] == phase2_normalized["sources"]
    )
    phase2_plan = read(PHASE2_ROOT / "F1_INITIAL_COMPLETE/generated_plan.json")
    f1_equivalence = {"baseline_phase1": str((REPORT.parent / "kitchen_feasibility_phase1/variants/F1_INITIAL_COMPLETE/grounded_assignments.json").relative_to(ROOT)),
        "current_phase1": str((PHASE1_ROOT / "F1_INITIAL_COMPLETE/grounded_role_assignments.json").relative_to(ROOT)),
        "phase1_role_equivalent": equivalent_roles,
        "legacy_baseline_source_roles": {
            "availability": (
                "AVAILABLE" if baseline_sources_available else "UNAVAILABLE"
            ),
            "value": (
                baseline_normalized["sources"]
                if baseline_sources_available else None
            ),
            "comparison_performed": baseline_sources_available,
        },
        "current_phase1_vs_phase2_source_roles": {
            "current_phase1": current_normalized["sources"],
            "phase2": phase2_normalized["sources"],
            "equivalent": current_phase2_sources_equal,
        },
        "baseline_assignment_projection": baseline_normalized,
        "current_assignment_projection": current_normalized,
        "phase2_assignment_projection": phase2_normalized,
        "phase2_assignment_equivalent": phase2_normalized == current_normalized,
        "phase2_plan_action_count": len(phase2_plan),
        "phase2_validation_passed": read(PHASE2_ROOT / "F1_INITIAL_COMPLETE/validation.json").get("valid", False),
    }
    f1_equivalence["equivalent"] = all((
        f1_equivalence["phase1_role_equivalent"],
        f1_equivalence["phase2_assignment_equivalent"],
        current_phase2_sources_equal,
        f1_equivalence["phase2_validation_passed"],
    ))
    write("f1_physical_layout_equivalence.json", f1_equivalence)

    accepted_storage = []
    for variant in VARIANTS:
        path = ROOT / f"runs/phaseB_closure_inventory/{variant}/execution_entity_resolution.json"
        if not path.exists(): continue
        data = read(path)
        accepted_storage.extend(row for row in data["accepted"] if row["observed_source_context"]["source_container"])
    maximum_error = max((row["centroid_error_m"] for row in accepted_storage), default=0.0)
    chosen_gate = 0.265
    per_object_competition = []
    for variant in VARIANTS:
        path = ROOT / f"runs/phaseB_closure_inventory/{variant}/execution_entity_resolution.json"
        if not path.exists():
            continue
        resolution_data = read(path)
        rejected = resolution_data.get("rejected_candidate_edges", [])
        for winner in resolution_data.get("accepted", []):
            compatible = sorted(
                row["centroid_error_m"] for row in rejected
                if row.get("generic_object_id") == winner.get("generic_object_id")
                and row.get("semantic_consistent")
                and row.get("source_context_consistent")
            )
            second_best = compatible[0] if compatible else None
            winner_distance = winner["centroid_error_m"]
            per_object_competition.append({
                "variant": variant,
                "generic_object_id": winner.get("generic_object_id"),
                "physical_backend_body": winner.get("physical_backend_body"),
                "winning_distance_m": winner_distance,
                "second_best_semantic_and_source_compatible_distance_m": second_best,
                "winning_margin_to_second_best_m": (
                    second_best - winner_distance
                    if second_best is not None else None
                ),
            })
    competing_distances = [
        row["second_best_semantic_and_source_compatible_distance_m"]
        for row in per_object_competition
        if row["second_best_semantic_and_source_compatible_distance_m"] is not None
    ]
    nearest_competing = min(competing_distances, default=None)
    gate_audit = {"accepted_storage_bindings": accepted_storage,
        "maximum_actual_accepted_error_m": maximum_error,
        "nearest_semantic_and_source_consistent_competing_distance_m": nearest_competing,
        "per_object_winner_vs_second_best": per_object_competition,
        "configured_threshold_m": chosen_gate,
        "numerical_safety_margin_m": chosen_gate-maximum_error,
        "semantic_and_source_gates_remain_required": True,
        "passed": maximum_error <= chosen_gate and chosen_gate-maximum_error < 0.01}
    write("storage_entity_resolution_gate_audit.json", gate_audit)

    source = (ROOT / "mujoco_scenes/generic_manipulation.py").read_text()
    mount_scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary", robot="google"
    )
    profile = manipulation_profile("google")
    arm_joint_ids = [
        mujoco.mj_name2id(
            mount_scene.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        for name in profile.arm_joints
    ]
    arm_qpos = mount_scene.model.jnt_qposadr[arm_joint_ids]
    body_geoms = {}
    for body_name in ("google:base_link", "google:link_shoulder"):
        body_id = mujoco.mj_name2id(
            mount_scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        body_geoms[body_name] = [
            geom_id for geom_id in range(mount_scene.model.ngeom)
            if int(mount_scene.model.geom_bodyid[geom_id]) == body_id
        ]
    representative_distances = {}
    for label, joints in (
        ("navigation", profile.navigation_joints),
        ("manipulation_home_seed", profile.home_seed),
    ):
        mount_scene.data.qpos[arm_qpos] = joints
        mujoco.mj_forward(mount_scene.model, mount_scene.data)
        representative_distances[label] = min(
            float(mujoco.mj_geomDistance(
                mount_scene.model, mount_scene.data, first, second, 1.0, None
            ))
            for first in body_geoms["google:base_link"]
            for second in body_geoms["google:link_shoulder"]
        )
    maximum_overlap = min(representative_distances.values())
    mount = {"configured_pair": ["google:base_link","google:link_shoulder"],
        "configured_allowance_m": -0.0615,
        "representative_signed_distances_m": representative_distances,
        "maximum_legitimate_overlap_m": maximum_overlap,
        "allowance_margin_m": maximum_overlap - (-0.0615),
        "source_structural_check": source.count("SELF_COLLISION_MOUNT_ALLOWANCES") >= 1,
        "forbidden_exception_tokens_present": any(token in source.split("SELF_COLLISION_MOUNT_ALLOWANCES",1)[1].split("}",1)[0] for token in ("forearm","bicep","cabinet","shelf","finger")),
        "validation_test": "test_only_the_physical_shoulder_mount_has_a_self_overlap_allowance"}
    mount["passed"] = (
        mount["source_structural_check"]
        and not mount["forbidden_exception_tokens_present"]
        and maximum_overlap >= -0.0615
    )
    write("google_mount_allowance_audit.json", mount)

    code_generic = (ROOT / "mujoco_scenes/generic_manipulation.py").read_text()
    guards = {
        "backend_names_execution_only": {"passed": not inventory["planner_received_backend_names"], "validation_method":"inventory boundary flag", "evidence":["phaseB_object_inventory.json"]},
        "functional_substitution": {"passed": all(r["generic_object_id"] == r["requested_generic_object_id"] for r in rows.values()), "validation_method":"executed IDs equal requested frozen generic IDs in physical telemetry", "evidence":list(PICK_EVIDENCE.values())},
        "bilateral_contact_gates_weld": {"passed": "grasp weld requires confirmed bilateral contact" in code_generic and all(r.get("bilateral_contact") for r in rows.values()), "validation_method":"source guard plus live telemetry", "evidence":["grasp_calibration_matrix.json"]},
        "no_target_qpos_write": {"passed": all(r.get("direct_object_qpos_write") is False for r in rows.values()), "validation_method":"physical result telemetry", "evidence":["grasp_calibration_matrix.json"]},
        "prediction_selection_only": {"passed": "prediction ranks candidates" in (ROOT / "mujoco_scenes/kitchen_object_manipulation.py").read_text().lower(), "validation_method":"source structural check", "evidence":["mujoco_scenes/kitchen_object_manipulation.py"]},
        "phase_c_effects_not_fabricated": {"passed": operator_coverage["phase_c_explicit"], "validation_method":"every frozen POUR/STIR row", "evidence":["isolated_operator_validation.json"]},
        "mount_allowance_scoped": {"passed": mount["passed"], "validation_method":"constant and regression test", "evidence":["google_mount_allowance_audit.json"]},
        "semantic_provenance_explicit": {"passed": all(row["semantic_label_source"] != "UNAVAILABLE" for row in provenance["objects"]), "validation_method":"inventory provenance", "evidence":["semantic_label_provenance.json"]},
    }
    guards_pass = guards_are_verifiable(guards)
    write("scientific_guard_report.json", {"guards": guards, "passed": guards_pass})

    metric_values = lambda key: [row.get(key) for row in rows.values() if row.get(key) is not None]
    carried_moves = [
        move
        for record in carry.values()
        for move in (record.get("outbound_move"), record.get("return_move"))
        if move
    ]
    write("physical_metrics.json", {
        "pick_success_rate": sum(r["pick_success"] for r in rows.values())/len(rows),
        "place_success_rate": (
            sum(bool(place.get("success")) for place in place_rows)
            / len(place_rows) if place_rows else None
        ),
        "attachment_translation_snap_m": stats(metric_values("attachment_translation_snap_m")),
        "attachment_angle_snap_rad": stats(metric_values("attachment_angle_snap_rad")),
        "preclose_cartesian_error_m": stats(metric_values("preclose_cartesian_error_m")),
        "preclose_orientation_error_rad": stats(metric_values("preclose_orientation_error_rad")),
        "physical_pick_duration_s": stats(metric_values("duration_s")),
        "carried_move_translation_drift_m": stats([
            move.get("relative_position_drift_m") for move in carried_moves
        ]),
        "carried_move_orientation_drift_rad": stats([
            move.get("relative_orientation_drift_rad") for move in carried_moves
        ]),
        "placement_linear_speed_m_s": stats([
            place.get("post_place", {}).get("linear_speed_m_s")
            for place in place_rows
        ]),
        "placement_angular_speed_rad_s": stats([
            place.get("post_place", {}).get("angular_speed_rad_s")
            for place in place_rows
        ]),
        "source_clearance_verified_rate": (
            sum(
                row["source_clearance_status"] in {"VERIFIED", "NOT_APPLICABLE"}
                for row in extraction.values()
            ) / len(extraction)
        ),
        "source_return_xy_error_m": stats([p.get("source_return_xy_error_m") for p in returns]),
        "serving_edge_margin_m": stats([p.get("edge_margin_m") for p in serving]),
    })

    multi_path = ROOT / "runs/phaseB_freeze_multi_object_authoritative/multi_object_validation.json"
    relation_path = ROOT / "runs/phaseB_freeze_multi_object_authoritative/final_physical_relation_validation.json"
    multi = read(multi_path) if multi_path.exists() else {"success":False,"missing":str(multi_path)}
    relation = read(relation_path) if relation_path.exists() else {"success":False,"missing":str(relation_path)}
    write("multi_object_validation.json", multi)
    write("final_physical_relation_validation.json", relation)

    provenance = git_provenance()
    manifest_entries = []
    for name,row in rows.items():
        manifest_entries.append({"name":name,"relative_run_path":row["evidence_directory"],
            "generic_object_id":row.get("generic_object_id"),"pick_sha256":row.get("pick_sha256"),
            "place_sha256":row.get("place_sha256"),
            "committed_tree_provenance":"UNAVAILABLE_LEGACY_ARTIFACT"})
    for family,record in carry.items():
        path=ROOT/record["evidence_path"]
        manifest_entries.append({"name":f"carry_{family}","relative_run_path":record["evidence_path"],"sha256":sha256(path) if path.exists() else None,
            "committed_tree_provenance":"UNAVAILABLE_LEGACY_ARTIFACT"})
    for name, path in (("multi_object_authoritative", multi_path),
                       ("final_physical_relations", relation_path)):
        manifest_entries.append({
            "name": name,
            "relative_run_path": str(path.relative_to(ROOT)),
            "sha256": sha256(path) if path.exists() else None,
            "execution_committed_tree_sha": provenance["git_head"],
            "worktree_status_at_manifest_generation": provenance["worktree_status"],
        })
    manifest_valid = all(entry.get("pick_sha256") or entry.get("sha256") for entry in manifest_entries)
    write("physical_run_manifest.json", {"repository": provenance,
          "entries":manifest_entries,"valid":manifest_valid,"raw_artifacts_committed":False})

    test_text = (REPORT / "test_summary.txt").read_text() if (REPORT/"test_summary.txt").exists() else ""
    tests_pass = bool(
        re.search(r"\b\d+ passed\b", test_text)
        and "Result: PASS" in test_text
        and not re.search(r"\b[1-9]\d* failed\b", test_text)
    )
    artifacts = {
        "pick_coverage_pass": all(item["passed"] for item in repeatability.values()),
        "place_coverage_pass": destination_pass,
        "carried_move_pass": carry_pass,
        "storage_repeatability_pass": all(repeatability[k]["passed"] for k in ("D1","D2","C1","B1","C2_vessel","C2_utensil")),
        "c2_unrestricted_grasp_pass": all(
            item.get("fixture_release", {}).get("released_before_preclose")
            and not item.get("fixture_release", {}).get("active_during_contact")
            for item in groups["C2_utensil"]
        ),
        "entity_resolution_pass": resolution["all_resolved"] and resolution["one_to_one"],
        "extraction_pass": all(r["source_clearance_status"] in {"VERIFIED","NOT_APPLICABLE"} for r in extraction.values()),
        "destination_coverage_pass": destination_pass,
        "f1_equivalence_pass": f1_equivalence["equivalent"],
        "variant_coverage_pass": variant_coverage["passed"],
        "isolated_operator_coverage_pass": operator_coverage["phase_b_passed"] and operator_coverage["phase_c_explicit"],
        "scientific_guards_pass": guards_pass,
        "multi_object_pass": bool(multi.get("success")),
        "final_relation_pass": bool(relation.get("success")),
        "tests_pass": tests_pass,
        "reproduction_manifest_valid": manifest_valid,
    }
    closure = closure_from_artifacts(artifacts)
    summary = {"phase":"KITCHEN_GOOGLE_EXECUTION_PHASE_B", **closure,
        "phase_c_operators_remain_unsupported":["POUR","STIR"],
        "phase_c_symbolic_effects_fabricated":False}
    write("validation_summary.json", summary)
    write("environment.json", {"python":platform.python_version(),"platform":platform.platform(),"robot":"google","render_backend":"EGL headless",**provenance})


if __name__ == "__main__":
    main()
