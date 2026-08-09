"""Frozen Phase-1 witness to pure symbolic living-room planning.

This module consumes only compact production artifacts.  It does not import
MuJoCo, perception, region allocation, or privileged evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .symbolic_planning_core import (
    Atom,
    SymbolicAction,
    SymbolicProblem,
    deterministic_astar,
    independent_replay,
)


SOURCE_PHASE1_COMMIT = "70baf5886427b9972e754a01ee45fbf0b74c151d"
PRODUCTION_INPUTS = (
    "functional_region_witness.json",
    "region_assignments.json",
    "task_requirements.json",
)
OBJECT_ID = re.compile(r"^object_[0-9]+$")
REGION_ID = re.compile(r"^region_[0-9]+$")


class LivingRoomCompilationError(RuntimeError):
    def __init__(self, reason: str, source_status: str, details: str = ""):
        super().__init__(details or reason)
        self.result = {
            "status": "REJECTED",
            "reason": reason,
            "source_phase1_status": source_status,
            "planner_invoked": False,
            "details": details or None,
        }


@dataclass(frozen=True)
class LivingRoomCompilation:
    problem: SymbolicProblem
    symbolic: dict[str, Any]
    source_manifest: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(variant_dir: Path) -> dict[str, Any]:
    files = {}
    for name in PRODUCTION_INPUTS:
        path = variant_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    witness = _load_json(variant_dir / "functional_region_witness.json")
    return {
        "variant": variant_dir.name,
        "source_phase1_commit": SOURCE_PHASE1_COMMIT,
        "source_phase1_status": witness.get("status"),
        "consumed_files": files,
        "source_production_only": True,
        "forbidden_oracle_artifacts_consumed": False,
    }


def _reject(reason: str, status: str, details: str) -> None:
    raise LivingRoomCompilationError(reason, status, details)


def compile_living_room_problem(variant_dir: str | Path) -> LivingRoomCompilation:
    """Compile the exact selected Phase-1 witness; never rerun allocation."""
    variant_path = Path(variant_dir)
    manifest = source_manifest(variant_path)
    witness = _load_json(variant_path / "functional_region_witness.json")
    assignments_record = _load_json(variant_path / "region_assignments.json")
    requirements = _load_json(variant_path / "task_requirements.json")
    status = str(witness.get("status", "UNKNOWN"))
    if status != "COMPLETE":
        _reject(
            "FUNCTIONAL_WITNESS_NOT_COMPLETE",
            status,
            "Phase 2 accepts only a COMPLETE production functional witness",
        )
    selected = witness.get("functional_requirements")
    assignments = assignments_record.get("assignments")
    if not isinstance(selected, list) or not isinstance(assignments, list):
        _reject("MALFORMED_WITNESS", status, "Missing assignment lists")
    if selected != assignments:
        _reject(
            "WITNESS_ASSIGNMENT_MISMATCH",
            status,
            "Witness selections differ from production region assignments",
        )
    if len(selected) != 3:
        _reject("INVALID_ASSIGNMENT_COUNT", status, "Expected exactly three slots")

    bindings: list[dict[str, str]] = []
    personal = []
    shared = []
    seen_payloads: set[str] = set()
    for assignment in selected:
        function_id = assignment.get("function_id")
        region_id = assignment.get("region_id")
        payload_ids = assignment.get("payload_ids")
        evidence = assignment.get("selected_compatibility_evidence", {})
        if not isinstance(region_id, str) or not REGION_ID.fullmatch(region_id):
            _reject("NON_GENERIC_REGION_ID", status, repr(region_id))
        if evidence.get("compatibility_status") != "TRUE":
            _reject("SELECTED_EDGE_NOT_TRUE", status, repr(assignment.get("slot_id")))
        required_relations = (
            requirements.get("function_groups", {})
            .get("personal_refreshment" if function_id == "PERSONAL_REFRESHMENT_REGION" else "shared_controls", {})
            .get("required_relations", [])
        )
        if any(evidence.get(relation) != "TRUE" for relation in required_relations):
            _reject("SELECTED_EDGE_NOT_TRUE", status, repr(assignment.get("slot_id")))
        if not isinstance(payload_ids, list) or len(payload_ids) != 2:
            _reject("INVALID_PAYLOAD_GROUP", status, repr(payload_ids))
        for object_id in payload_ids:
            if not isinstance(object_id, str) or not OBJECT_ID.fullmatch(object_id):
                _reject("NON_GENERIC_OBJECT_ID", status, repr(object_id))
            if object_id in seen_payloads:
                _reject("DUPLICATE_PAYLOAD_BINDING", status, object_id)
            seen_payloads.add(object_id)
            bindings.append(
                {
                    "object_id": object_id,
                    "region_id": region_id,
                    "function_id": function_id,
                    "slot_id": str(assignment.get("slot_id")),
                }
            )
        if function_id == "PERSONAL_REFRESHMENT_REGION":
            personal.append(assignment)
        elif function_id == "SHARED_CONTROLS_REGION":
            shared.append(assignment)
        else:
            _reject("UNKNOWN_FUNCTION", status, str(function_id))
    if len(seen_payloads) != 6:
        _reject("INCOMPLETE_PAYLOAD_COVERAGE", status, str(sorted(seen_payloads)))
    if len(personal) != 2 or len(shared) != 1:
        _reject("INVALID_FUNCTION_COVERAGE", status, "Expected 2 personal + 1 shared")
    if personal[0]["region_id"] == personal[1]["region_id"]:
        _reject("PERSONAL_REGIONS_NOT_DISTINCT", status, "Personal regions must differ")
    shared_payloads = [
        binding for binding in bindings
        if binding["function_id"] == "SHARED_CONTROLS_REGION"
    ]
    if len({item["region_id"] for item in shared_payloads}) != 1:
        _reject("SHARED_REGION_MISMATCH", status, "Controls must share a destination")

    objects = sorted(seen_payloads)
    regions = sorted({binding["region_id"] for binding in bindings})
    initial_atoms: set[Atom] = {("hand_empty",)}
    initial_atoms.update(("object", object_id) for object_id in objects)
    initial_atoms.update(("available", object_id) for object_id in objects)
    initial_atoms.update(("region", region_id) for region_id in regions)
    goals = frozenset(
        ("on", binding["object_id"], binding["region_id"])
        for binding in bindings
    )
    actions: list[SymbolicAction] = []
    for object_id in objects:
        actions.append(
            SymbolicAction(
                "pick",
                (object_id,),
                frozenset({("available", object_id), ("hand_empty",)}),
                frozenset(),
                frozenset({("holding", object_id)}),
                frozenset({("available", object_id), ("hand_empty",)}),
            )
        )
        for region_id in regions:
            actions.append(
                SymbolicAction(
                    "place",
                    (object_id, region_id),
                    frozenset({("holding", object_id), ("region", region_id)}),
                    frozenset(),
                    frozenset({("on", object_id, region_id), ("hand_empty",)}),
                    frozenset({("holding", object_id)}),
                )
            )
    actions.sort(key=lambda action: (action.name, action.arguments))
    problem = SymbolicProblem(frozenset(initial_atoms), goals, tuple(actions))
    symbolic = {
        "schema_version": 1,
        "task_id": witness.get("task_id"),
        "source_variant": variant_path.name,
        "source_phase1_status": status,
        "compilation_status": "SUCCESS",
        "planner_invoked": False,
        "objects": objects,
        "regions": regions,
        "witness_selected_bindings": sorted(
            bindings, key=lambda item: (item["object_id"], item["region_id"])
        ),
        "initial_atoms": [list(atom) for atom in sorted(initial_atoms)],
        "goal_atoms": [list(atom) for atom in sorted(goals)],
        "operator_vocabulary": ["PICK", "PLACE"],
        "initial_location_abstraction": "AVAILABLE_WITHOUT_FABRICATED_SOURCE_REGION",
        "allocation_policy": "USE_EXACT_PHASE1_WITNESS_SELECTION_NO_REALLOCATION",
    }
    return LivingRoomCompilation(problem, symbolic, manifest)


def action_json(index: int, action: SymbolicAction) -> dict[str, Any]:
    names = ("object",) if action.name == "pick" else ("object", "region")
    return {
        "step": index,
        "operator": action.name.upper(),
        "arguments": dict(zip(names, action.arguments)),
        "cost": action.cost,
    }


def render_domain_pddl() -> str:
    return """(define (domain living-room-placement)
  (:requirements :strips :typing)
  (:types object region)
  (:predicates (available ?o - object) (holding ?o - object)
               (hand-empty) (on ?o - object ?r - region))
  (:action pick
    :parameters (?o - object)
    :precondition (and (available ?o) (hand-empty))
    :effect (and (holding ?o) (not (available ?o)) (not (hand-empty))))
  (:action place
    :parameters (?o - object ?r - region)
    :precondition (holding ?o)
    :effect (and (on ?o ?r) (hand-empty) (not (holding ?o))))
)\n"""


def render_problem_pddl(compilation: LivingRoomCompilation) -> str:
    symbolic = compilation.symbolic
    objects = " ".join(symbolic["objects"])
    regions = " ".join(symbolic["regions"])
    initial = ["(hand-empty)"] + [f"(available {item})" for item in symbolic["objects"]]
    goals = [f"(on {atom[1]} {atom[2]})" for atom in symbolic["goal_atoms"]]
    lines = [
        "(define (problem living-room-witness-placement)",
        "  (:domain living-room-placement)",
        f"  (:objects {objects} - object {regions} - region)",
        "  (:init " + " ".join(initial) + ")",
        "  (:goal (and " + " ".join(goals) + "))",
        ")",
    ]
    return "\n".join(lines) + "\n"


def run_living_room_symbolic_pipeline(
    variant_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest(Path(variant_dir))
    _write_json(output / "phase1_source_manifest.json", manifest)
    try:
        compilation = compile_living_room_problem(variant_dir)
    except LivingRoomCompilationError as error:
        _write_json(output / "compilation_result.json", error.result)
        return error.result
    _write_json(output / "symbolic_problem.json", compilation.symbolic)
    _write_json(
        output / "symbolic_initial_state.json",
        {"atoms": compilation.symbolic["initial_atoms"]},
    )
    _write_json(
        output / "symbolic_goal.json",
        {"atoms": compilation.symbolic["goal_atoms"]},
    )
    _write_json(
        output / "compilation_result.json",
        {
            "status": "SUCCESS",
            "source_phase1_status": "COMPLETE",
            "planner_invoked": True,
            "payload_count": len(compilation.symbolic["objects"]),
            "goal_count": len(compilation.symbolic["goal_atoms"]),
        },
    )
    result = deterministic_astar(compilation.problem)
    replay = independent_replay(compilation.problem, result.plan)
    plan_records = [action_json(index, action) for index, action in enumerate(result.plan)]
    _write_json(
        output / "plan.json",
        {"actions": plan_records, "search_statistics": result.statistics},
    )
    (output / "plan.txt").write_text(
        "\n".join(f"{index:03d} {action.render()}" for index, action in enumerate(result.plan)) + "\n",
        encoding="utf-8",
    )
    _write_json(output / "replay_validation.json", replay)
    (output / "domain.pddl").write_text(render_domain_pddl(), encoding="utf-8")
    (output / "problem.pddl").write_text(render_problem_pddl(compilation), encoding="utf-8")
    return {
        "status": "SUCCESS",
        "source_phase1_status": "COMPLETE",
        "planner_invoked": True,
        "goal_status": replay["goal_status"],
        "plan_length": len(result.plan),
        "plan_cost": result.statistics["plan_cost"],
        "search_statistics": result.statistics,
        "bindings": compilation.symbolic["witness_selected_bindings"],
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
