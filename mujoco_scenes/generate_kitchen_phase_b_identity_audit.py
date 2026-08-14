"""Audit corrected-layout Phase-1 identity without using it for planning.

``instance_token`` is deliberately treated as simulator-backed evaluation
metadata.  It may establish cross-run ground truth for this benchmark audit,
but it is never an input to functional grounding, symbolic planning, or the
execution inventory.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "runs/phaseB_freeze_observed_current"
FRESH = ROOT / "runs/phaseB_freeze_observed_b1_corrected"
REPORT = ROOT / "mujoco_scenes/benchmark_reports/kitchen_google_execution_phaseB"
PRIMARY_FROZEN = ROOT / "runs/integrated_no_pot_clearance_seed19_20260807"
EXPECTED_PRIMARY_SHA256 = {
    "object_registry.json": "22dce4b63df5dc38caed122fd87d4e485ae15c43d9eca527a9e085b2239f0817",
    "latest_witness.json": "68533aaf8b234f9cbb187ed4b6a9496691cab838a369257d86447f7ba13726a4",
    "grounded_role_assignments.json": "06b98327fe256942b6c49d22e4d5cd01802925115e9d1b0aba967a70b534af29",
    "plan.json": "de0c488070fc3a0e92dca8bfd1c09e3a007555dcbf452cf9b659f4d99bcefc92",
    "plan_validation.json": "9f99b4eac02758739fd243091cfd109d5f1991e58222290e636d308e8599b078",
    "task_requirements.json": "04fa1f5500106447f364001be4b47cd9a152e9d185602dd1337f42367ca5d2a3",
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, payload: Any) -> None:
    path = REPORT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(record: dict[str, Any]) -> str | None:
    return record.get("semantics", {}).get("validated", {}).get(
        "canonical_label"
    )


def dimensions(record: dict[str, Any]) -> dict[str, float | None]:
    return {
        axis: record.get("dimensions_m", {}).get(axis, {}).get("value")
        for axis in ("length", "width", "height")
    }


def physical_family(record: dict[str, Any]) -> str | None:
    label = semantic(record)
    if label in {"bowl", "cup", "mug", "glass"}:
        return "OPEN_VESSEL"
    if label in {"spoon", "fork", "knife", "stirrer", "utensil"}:
        return "ELONGATED_UTENSIL"
    values = [value for value in dimensions(record).values() if value]
    if len(values) >= 2:
        ordered = sorted(values, reverse=True)
        if ordered[0] / ordered[1] >= 2.0:
            return "ELONGATED_UTENSIL"
    return None


def decision_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        role: {
            "status": row.get("status"),
            "decision": row.get("decision"),
            "semantic_status": row.get("semantic_gate_status"),
            "geometry_status": row.get("geometry_gate_status"),
        }
        for role, row in sorted(
            record.get("functional_role_evaluations", {}).items()
        )
    }


def token_roles(
    witness: dict[str, Any], registry: dict[str, Any]
) -> dict[str, list[str]]:
    result = {}
    for role, object_ids in witness["selected_witness"].items():
        result[role] = sorted(
            registry["objects"][object_id]["instance_token"]
            for object_id in object_ids
        )
    return result


def candidate_by_token_and_role(
    run: Path, registry: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read(run / "candidate_evaluations.json")["candidate_evaluations"]
    return {
        (registry["objects"][row["object_id"]]["instance_token"], row["role"]): {
            "generic_object_id": row["object_id"],
            "status": row.get("status"),
            "decision": row.get("decision"),
            "semantic_label": row.get("semantic", {}).get("canonical_label"),
            "semantic_status": row.get("semantic", {}).get("status"),
            "semantic_confidence": row.get("semantic", {}).get("confidence"),
            "geometry_status": row.get("unary_geometry", {}).get("status"),
        }
        for row in rows
    }


def main() -> None:
    frozen_registry = read(FROZEN / "object_registry.json")
    fresh_registry = read(FRESH / "object_registry.json")
    frozen_witness = read(FROZEN / "latest_witness.json")
    fresh_witness = read(FRESH / "latest_witness.json")
    frozen_config = read(FROZEN / "run_config.json")
    fresh_config = read(FRESH / "run_config.json")

    provenance = {
        "identifiers": [
            {
                "identifier": "instance_token",
                "generated_by": (
                    "ObservedStateRun._association_token in "
                    "mujoco_scenes/observed_state.py"
                ),
                "input_fields": [
                    "scene_name",
                    "MuJoCo object instance body name (instance_id)",
                ],
                "algorithm": "sha256(f'{scene_name}:{instance_id}')[:24]",
                "planner_visible": False,
                "perception_visible": (
                    "INTERNAL_ASSOCIATION_ONLY; stored in evaluation registry"
                ),
                "simulator_ground_truth_dependency": True,
                "backend_or_hidden_identity_leak": True,
                "intended_role": (
                    "within-run association and evaluation-only cross-run "
                    "physical-instance correspondence"
                ),
            },
            {
                "identifier": "generic_object_id (object_XXXX)",
                "generated_by": (
                    "ObservedStateRun._new_object_id in "
                    "mujoco_scenes/observed_state.py"
                ),
                "input_fields": [
                    "current registry object count",
                    "first-acceptance iteration order",
                ],
                "algorithm": "object_{len(registry.objects)+1:04d}",
                "planner_visible": True,
                "perception_visible": True,
                "simulator_ground_truth_dependency": False,
                "backend_or_hidden_identity_leak": False,
                "intended_role": "episode-local backend-free symbolic label",
            },
            {
                "identifier": "physical_backend_body",
                "generated_by": (
                    "scene_loader._inject_object / MuJoCo model composition"
                ),
                "input_fields": [
                    "configured object kind",
                    "duplicate-instance counter",
                ],
                "planner_visible": False,
                "perception_visible": (
                    "SIMULATOR SEGMENTATION IMPLEMENTATION ONLY"
                ),
                "simulator_ground_truth_dependency": True,
                "backend_or_hidden_identity_leak": True,
                "intended_role": "execution-only physical backend binding",
            },
        ],
        "runtime_boundary_correction": {
            "instance_token_removed_from_phase_b_execution_inventory": True,
            "planner_and_functional_grounding_use_instance_token": False,
            "audit_use": "EVALUATION_ONLY_GROUND_TRUTH",
        },
    }
    write("phaseB_identity_provenance.json", provenance)

    fresh_by_token = {
        row["instance_token"]: (object_id, row)
        for object_id, row in fresh_registry["objects"].items()
    }
    correspondence = []
    phi = {}
    decision_mismatches = []
    for frozen_id, frozen_row in sorted(frozen_registry["objects"].items()):
        token = frozen_row["instance_token"]
        matched = fresh_by_token.get(token)
        fresh_id, fresh_row = matched if matched else (None, {})
        if fresh_id is not None:
            phi[frozen_id] = fresh_id
        frozen_decisions = decision_projection(frozen_row)
        fresh_decisions = decision_projection(fresh_row)
        decisions_equal = frozen_decisions == fresh_decisions
        if not decisions_equal:
            decision_mismatches.append({
                "frozen_generic_id": frozen_id,
                "fresh_generic_id": fresh_id,
                "instance_token": token,
                "frozen": frozen_decisions,
                "fresh": fresh_decisions,
            })
        correspondence.append({
            "frozen_generic_id": frozen_id,
            "frozen_instance_token": token,
            "fresh_generic_id": fresh_id,
            "fresh_instance_token": (
                fresh_row.get("instance_token") if matched else None
            ),
            "token_equal": bool(
                matched and token == fresh_row.get("instance_token")
            ),
            "semantic_label_frozen": semantic(frozen_row),
            "semantic_label_fresh": semantic(fresh_row),
            "source_region_frozen": frozen_row.get("source_region"),
            "source_region_fresh": fresh_row.get("source_region"),
            "observed_dimensions_frozen_m": dimensions(frozen_row),
            "observed_dimensions_fresh_m": dimensions(fresh_row),
            "functional_decisions_equal": decisions_equal,
            "correspondence_status": (
                "EXACT_EVALUATION_TOKEN_MATCH"
                if matched else "NO_TOKEN_MATCH"
            ),
        })

    complete = len(phi) == len(frozen_registry["objects"])
    one_to_one = len(set(phi.values())) == len(phi)
    bijection = {
        "mapping_basis": (
            "evaluation-only exact instance_token equality; tokens are not "
            "planner inputs"
        ),
        "phi_frozen_to_fresh": phi,
        "one_to_one": one_to_one,
        "complete_for_discovered_frozen_objects": complete,
        "complete_for_frozen_assignment": all(
            object_id in phi
            for ids in frozen_witness["selected_witness"].values()
            for object_id in ids
        ),
        "correspondence": correspondence,
        "passed": one_to_one and complete,
    }
    write("phaseB_generic_identity_bijection.json", bijection)
    write("phaseB_instance_correspondence.json", {
        "objects": correspondence,
        "all_frozen_instances_present": complete,
        "all_fresh_instances_accounted_for": (
            set(phi.values()) == set(fresh_registry["objects"])
        ),
    })

    frozen_role_tokens = token_roles(frozen_witness, frozen_registry)
    fresh_role_tokens = token_roles(fresh_witness, fresh_registry)
    frozen_candidates = candidate_by_token_and_role(FROZEN, frozen_registry)
    fresh_candidates = candidate_by_token_and_role(FRESH, fresh_registry)
    role_comparison = []
    for role in sorted(set(frozen_role_tokens) | set(fresh_role_tokens)):
        frozen_ids = frozen_witness["selected_witness"].get(role, [])
        fresh_ids = fresh_witness["selected_witness"].get(role, [])
        mapped = sorted(phi[object_id] for object_id in frozen_ids)
        role_comparison.append({
            "role": role,
            "frozen_generic_ids": frozen_ids,
            "fresh_generic_ids": fresh_ids,
            "phi_frozen_generic_ids": mapped,
            "literal_generic_id_equivalent": sorted(frozen_ids) == sorted(fresh_ids),
            "frozen_instance_tokens": frozen_role_tokens.get(role, []),
            "fresh_instance_tokens": fresh_role_tokens.get(role, []),
            "same_physical_instances": (
                frozen_role_tokens.get(role, []) == fresh_role_tokens.get(role, [])
            ),
            "functional_assignment_equivalent": mapped == sorted(fresh_ids),
        })

    status_equal = frozen_witness.get("status") == fresh_witness.get("status")
    order_equal = (
        frozen_config.get("inspection_sequence")
        == fresh_config.get("inspection_sequence")
    )
    count_equal = frozen_witness.get("stage") == fresh_witness.get("stage")
    instances_equal = complete and set(phi.values()) == set(
        fresh_registry["objects"]
    )
    roles_equal = all(row["same_physical_instances"] for row in role_comparison)
    literal_equal = frozen_witness["selected_witness"] == fresh_witness["selected_witness"]
    old_coffee = set(frozen_role_tokens["coffee_stirrer"])
    old_soup = set(frozen_role_tokens["soup_eating_utensil"])
    new_coffee = set(fresh_role_tokens["coffee_stirrer"])
    new_soup = set(fresh_role_tokens["soup_eating_utensil"])
    reuse_equal = (
        len(old_coffee) == len(new_coffee) == 1
        and len(old_soup) == len(new_soup) == 3
        and len(old_coffee & old_soup) == len(new_coffee & new_soup) == 1
    )
    phase1_pass = all((
        status_equal, order_equal, count_equal, instances_equal,
        roles_equal, not decision_mismatches, reuse_equal,
    ))
    phase1 = {
        "classification": "UPSTREAM_PERCEPTION_DIAGNOSTIC",
        "phase_b_execution_closure_gate": False,
        "literal_generic_id_equivalent": literal_equal,
        "phase1_equivalent_modulo_identity_renaming": phase1_pass,
        "gates": {
            "task_completion_status_identical": status_equal,
            "inspection_order_identical": order_equal,
            "inspection_count_identical": count_equal,
            "same_physical_instances_discovered": instances_equal,
            "same_physical_instances_satisfy_each_role": roles_equal,
            "semantic_geometric_decisions_equivalent": not decision_mismatches,
            "reuse_distinctness_structure_equivalent": reuse_equal,
            "backend_names_used_by_functional_solver": False,
        },
        "role_comparison": role_comparison,
        "functional_decision_mismatches": decision_mismatches,
        "specific_checks": {
            "coffee_stirrer": next(
                row for row in role_comparison if row["role"] == "coffee_stirrer"
            ),
            "soup_utensils": next(
                row for row in role_comparison
                if row["role"] == "soup_eating_utensil"
            ),
            "b1_bowl": {
                "frozen_generic_id": "object_0018",
                "fresh_generic_id": phi.get("object_0018"),
                "instance_token": frozen_registry["objects"]["object_0018"]["instance_token"],
                "same_physical_instance": phi.get("object_0018") == "object_0017",
                "literal_id_equal": phi.get("object_0018") == "object_0018",
            },
        },
        "selection_change_analysis": {
            "selection_policy_frozen": frozen_witness.get("selection_policy"),
            "selection_policy_fresh": fresh_witness.get("selection_policy"),
            "valid_assignment_count_frozen": frozen_witness.get(
                "valid_assignment_count"
            ),
            "valid_assignment_count_fresh": fresh_witness.get(
                "valid_assignment_count"
            ),
            "changed_instance_candidates": [
                {
                    "instance_token": token,
                    "frozen_generic_id": next(
                        object_id for object_id, row in frozen_registry["objects"].items()
                        if row["instance_token"] == token
                    ),
                    "fresh_generic_id": next(
                        object_id for object_id, row in fresh_registry["objects"].items()
                        if row["instance_token"] == token
                    ),
                    "role": role,
                    "frozen_candidate": frozen_candidates.get((token, role)),
                    "fresh_candidate": fresh_candidates.get((token, role)),
                }
                for role in ("coffee_stirrer", "soup_eating_utensil")
                for token in sorted(
                    set(frozen_role_tokens.get(role, ()))
                    ^ set(fresh_role_tokens.get(role, ()))
                )
            ],
            "cause": (
                "Fresh RGB semantic evidence promoted tokens "
                "11efa15f4a8001764dbdf8f0 (knife->spoon) and "
                "04430c365dc26f6397809590 (UNKNOWN->spoon) from rejected/"
                "indeterminate to TRUE while their unary geometry remained "
                "TRUE. The unchanged deterministic policy orders valid "
                "assignments by group preferences then semantic rank and "
                "generic ID (confidence is gate-only), so the newly eligible "
                "lower episode IDs displaced tokens 5e77c284f572471e67ed77d7 "
                "and 6fc36a1e8be51572abd233c0. This is a genuine perception/"
                "global-matching witness change, not alpha-renaming."
            ),
        },
        "source_role_identity_check": {
            "basis": (
                "Frozen source roles are upstream symbolic-source assignments, "
                "not Phase-1 target/tool selections. Their generic IDs and "
                "evaluation tokens are unchanged under phi."
            ),
            "water_source": {
                "frozen_generic_id": "object_0009",
                "fresh_generic_id": phi.get("object_0009"),
                "instance_token": frozen_registry["objects"]["object_0009"]["instance_token"],
                "same_physical_instance": phi.get("object_0009") == "object_0009",
            },
            "coffee_source": {
                "frozen_generic_id": "object_0010",
                "fresh_generic_id": phi.get("object_0010"),
                "instance_token": frozen_registry["objects"]["object_0010"]["instance_token"],
                "same_physical_instance": phi.get("object_0010") == "object_0010",
            },
        },
        "fresh_phase2_recompilation_permitted": False,
        "physical_execution_permitted": "GATED_BY_FROZEN_INPUT_INTEGRITY_AND_EXECUTION_CALIBRATION",
        "stop_reason": (
            None if phase1_pass else
            "GENUINE_FUNCTIONAL_WITNESS_CHANGE_AFTER_FRESH_PERCEPTION"
        ),
    }
    write("phase1_alpha_equivalence.json", phase1)

    # No fresh Phase 2 is compiled. The authoritative frozen plan remains the
    # conditional execution contract regardless of this perception diagnostic.
    write("phase2_alpha_equivalence.json", {
        "run": False,
        "classification": "FRESH_RECOMPILATION_NOT_APPLICABLE",
        "phase_b_execution_closure_gate": False,
        "status": "FROZEN_PHASE2_PLAN_RETAINED",
        "fresh_phase1_diagnostic_passed": phase1_pass,
    })

    # This is the Phase-B boundary audit. The fresh Phase-1 witness above is
    # deliberately diagnostic and is not substituted for the frozen inputs.
    from .run_kitchen_phase_b_freeze_evidence import primary_validation_dispatcher

    _scene, inventory, resolution, _execution = primary_validation_dispatcher()
    authoritative_registry = read(PRIMARY_FROZEN / "object_registry.json")
    authoritative_witness = read(PRIMARY_FROZEN / "latest_witness.json")
    original_b1 = frozen_config["scene_layout"]["closed_region_contents"]["B1"]
    corrected_b1 = fresh_config["scene_layout"]["closed_region_contents"]["B1"]
    frozen_b1_tokens = sorted(
        row["instance_token"] for row in authoritative_registry["objects"].values()
        if row.get("source_region") == "B1"
    )
    fresh_b1_tokens = sorted(
        row["instance_token"] for row in fresh_registry["objects"].values()
        if row.get("source_region") == "B1"
    )
    binding_by_id = {
        row["generic_object_id"]: row for row in resolution["accepted"]
    }
    inventory_by_id = {
        row["generic_object_id"]: row for row in inventory["objects"]
    }
    b1_objects = []
    for frozen_id in ("object_0017", "object_0018"):
        frozen_row = authoritative_registry["objects"][frozen_id]
        current_id = phi[frozen_id]
        current_row = fresh_registry["objects"][current_id]
        token = frozen_row["instance_token"]
        backend = binding_by_id.get(frozen_id, {}).get("physical_backend_body")
        configured_kind = (
            "s1i_coffee_near_miss_spoon"
            if physical_family(frozen_row) == "ELONGATED_UTENSIL"
            else "ab3_deep_bowl"
        )
        b1_objects.append({
            "evaluation_only_instance_token": token,
            "frozen_generic_id": frozen_id,
            "original_b1_slot": original_b1.index(configured_kind),
            "corrected_b1_slot": corrected_b1.index(configured_kind),
            "frozen_semantic_evidence": frozen_row.get("semantics", {}).get("validated"),
            "corrected_scene_observed_semantic_evidence": current_row.get("semantics", {}).get("validated"),
            "frozen_measured_dimensions_m": dimensions(frozen_row),
            "corrected_measured_dimensions_m": dimensions(current_row),
            "frozen_observable_physical_family": physical_family(frozen_row),
            "corrected_observable_physical_family": physical_family(current_row),
            "source_region": current_row.get("source_region"),
            "functional_role_in_frozen_witness": [
                role for role, ids in authoritative_witness["selected_witness"].items()
                if frozen_id in ids
            ],
            "execution_binding": (
                {
                    "physical_backend_body": backend,
                    "resolution_method": binding_by_id[frozen_id]["resolution_method"],
                    "centroid_error_m": binding_by_id[frozen_id]["centroid_error_m"],
                } if backend else "UNRESOLVED"
            ),
        })
    phase1_files = (
        "object_registry.json", "grounded_role_assignments.json",
        "latest_witness.json",
    )
    phase2_files = ("plan.json", "grounded_role_assignments.json", "plan_validation.json")
    primary_hashes = {
        name: sha256(PRIMARY_FROZEN / name)
        for name in EXPECTED_PRIMARY_SHA256
    }
    phase1_integrity = all(
        primary_hashes[name] == EXPECTED_PRIMARY_SHA256[name]
        for name in phase1_files
    )
    phase2_integrity = all(
        primary_hashes[name] == EXPECTED_PRIMARY_SHA256[name]
        for name in phase2_files
    )
    task_integrity = (
        primary_hashes["task_requirements.json"]
        == EXPECTED_PRIMARY_SHA256["task_requirements.json"]
    )
    calibration_audit = {
        "scope": "CONDITIONAL_PHASE_B_EXECUTION_SCENE_CALIBRATION",
        "fresh_phase1_diagnostic_is_closure_gate": False,
        "fresh_semantic_witness_substituted": False,
        "frozen_phase1_input_integrity": {
            "root": str(PRIMARY_FROZEN.relative_to(ROOT)),
            "expected_sha256": {
                name: EXPECTED_PRIMARY_SHA256[name] for name in phase1_files
            },
            "actual_sha256": {name: primary_hashes[name] for name in phase1_files},
            "passed": phase1_integrity,
        },
        "frozen_phase2_input_integrity": {
            "root": str(PRIMARY_FROZEN.relative_to(ROOT)),
            "expected_sha256": {
                name: EXPECTED_PRIMARY_SHA256[name] for name in phase2_files
            },
            "actual_sha256": {name: primary_hashes[name] for name in phase2_files},
            "passed": phase2_integrity,
        },
        "gates": {
            "same_physical_object_inventory": (
                set(frozen_registry["instance_index"])
                == set(fresh_registry["instance_index"])
            ),
            "same_b1_container_membership": frozen_b1_tokens == fresh_b1_tokens,
            "same_b1_object_count": len(frozen_b1_tokens) == len(fresh_b1_tokens) == 2,
            "same_relevant_b1_families": sorted(map(physical_family, [
                row for row in authoritative_registry["objects"].values()
                if row.get("source_region") == "B1"
            ])) == sorted(map(physical_family, [
                row for row in fresh_registry["objects"].values()
                if row.get("source_region") == "B1"
            ])),
            "same_container_region_identity": all(
                row["source_region"] == "B1" for row in b1_objects
            ),
            "same_frozen_task_requirements": task_integrity,
            "frozen_phase1_witness_retained": phase1_integrity,
            "frozen_phase2_plan_retained": phase2_integrity,
            "only_approved_within_b1_pose_calibration_changed": (
                sorted(original_b1) == sorted(corrected_b1)
                and original_b1 == list(reversed(corrected_b1))
            ),
            "runtime_inventory_backend_free": (
                not inventory["planner_received_backend_names"]
            ),
            "evaluation_instance_token_excluded_from_runtime": (
                inventory["evaluation_instance_tokens_excluded"]
                and all("instance_token" not in row for row in inventory["objects"])
            ),
            "execution_resolution_one_to_one": resolution["one_to_one"],
            "execution_resolution_all_frozen_selected_ids_resolved": resolution["all_resolved"],
            "no_functional_substitution_during_execution": None,
        },
        "b1_objects": b1_objects,
        "approved_calibration_resolution": resolution.get(
            "approved_within_region_execution_calibration"
        ),
        "statement": (
            "The lane correction does not redefine the frozen functional "
            "witness. Fresh semantic evidence is diagnostic only and is not "
            "substituted into Phase-B symbolic input."
        ),
    }
    non_execution_gates = [
        value for key, value in calibration_audit["gates"].items()
        if key != "no_functional_substitution_during_execution"
    ]
    calibration_audit["pre_execution_passed"] = all(non_execution_gates)
    calibration_audit["passed"] = False
    write("b1_execution_scene_calibration_audit.json", calibration_audit)


if __name__ == "__main__":
    main()
