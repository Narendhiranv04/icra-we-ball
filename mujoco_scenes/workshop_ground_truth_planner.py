"""Deterministic Workshop assignment and typed GT plan generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .workshop_scene import WORKSHOP_REGIONS, WORKSHOP_VARIANTS_CONFIG


TARGET_JOINT = "workshop_frame_joint"
MANUAL_DRIVER = "workshop_long_phillips_driver"
POWER_DRIVER = "workshop_power_driver"
COMPATIBLE_DRIVERS = (MANUAL_DRIVER, POWER_DRIVER)
COMPATIBLE_SCREW = "workshop_medium_phillips_screw"
ACTION_VOCABULARY = Path(__file__).resolve().parent / "configs" / "workshop_action_vocabulary.yaml"


@dataclass(frozen=True)
class WorkshopAssignment:
    variant_id: str
    intended_outcome: str
    is_feasible: bool
    driver: str | None = None
    fastener: str | None = None
    work_surface: str | None = None
    parts_container: str | None = None
    target_joint: str = TARGET_JOINT
    rejection_reason: str | None = None
    assignment_source: str = "GROUND_TRUTH_ORACLE"
    source_ids: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_action_vocabulary() -> dict[str, Any]:
    payload = yaml.safe_load(ACTION_VOCABULARY.read_text(encoding="utf-8"))
    if not payload or set(payload.get("operators", {})) != {
        "OPEN", "PICK", "PLACE", "SCREW",
    }:
        raise ValueError("Workshop action vocabulary is missing required operators")
    return payload


def load_variant_specs() -> dict[str, Any]:
    return yaml.safe_load(WORKSHOP_VARIANTS_CONFIG.read_text(encoding="utf-8"))["variants"]


def solve_gt_assignment(variant_id: str) -> WorkshopAssignment:
    spec = load_variant_specs()[variant_id]
    feasible = spec["intended_outcome"] == "FEASIBLE"
    if feasible:
        first_driver = None
        screw_present = False
        for region in WORKSHOP_REGIONS:
            for object_name in spec["storage_contents"][region]:
                if first_driver is None and object_name in COMPATIBLE_DRIVERS:
                    first_driver = object_name
                screw_present = screw_present or object_name == COMPATIBLE_SCREW
        if first_driver is None or not screw_present:
            raise ValueError(f"Feasible variant {variant_id} lacks its fixed repair pair")
        witness = dict(spec["expected_solution"])
        if witness["driver"] != first_driver or witness["fastener"] != COMPATIBLE_SCREW:
            raise ValueError(f"{variant_id} violates first-observed-driver selection")
        return WorkshopAssignment(variant_id, "FEASIBLE", True, **witness)
    return WorkshopAssignment(
        variant_id, "INFEASIBLE", False,
        rejection_reason=spec["rejection_reason"],
    )


def _source_of(spec: dict[str, Any], object_name: str) -> str:
    for region, objects in spec["storage_contents"].items():
        if object_name in objects:
            return region
    raise ValueError(f"Object {object_name} has no declared Workshop storage source")


def generate_gt_plan(assignment: WorkshopAssignment) -> list[dict[str, Any]]:
    vocabulary = load_action_vocabulary()["operators"]
    spec = load_variant_specs()[assignment.variant_id]
    plan: list[dict[str, Any]] = []

    def add(operator: str, *arguments: str, reason: str) -> None:
        index = len(plan) + 1
        schema = vocabulary[operator]
        if len(arguments) != len(schema["arguments"]):
            raise ValueError(f"{operator} expects {len(schema['arguments'])} arguments")
        plan.append({
            "action_index": index,
            "action_instance_id": f"wact_{index:03d}_{operator.lower()}",
            "operator": operator,
            "arguments": list(arguments),
            "parameter_names": list(schema["arguments"]),
            "preconditions": list(schema["preconditions"]),
            "effects": list(schema["effects"]),
            "reason": reason,
        })

    driver_source = _source_of(spec, assignment.driver) if assignment.driver else None
    fastener_source = _source_of(spec, assignment.fastener) if assignment.fastener else None
    for region in spec["expected_inspection_regions"]:
        add(
            "OPEN", region,
            reason="Open this storage region once for inspection; it remains open.",
        )

    # Infeasibility is the observation result after the final OPEN.  It is not
    # an additional domain action, so the public workshop vocabulary stays
    # generalizable across all object/container arguments.
    if not assignment.is_feasible:
        return plan

    assert assignment.driver and assignment.fastener and assignment.work_surface
    assert driver_source and fastener_source
    add(
        "PICK", assignment.fastener, fastener_source,
        reason="Acquire the discovered compatible screw directly from its open source.",
    )
    add(
        "PLACE", assignment.fastener, assignment.target_joint,
        reason="Place the screw tip-down directly in the frame-joint hole.",
    )
    add(
        "PICK", assignment.driver, driver_source,
        reason="Acquire the first compatible driver observed during inspection.",
    )
    add(
        "SCREW", assignment.driver, assignment.fastener, assignment.target_joint,
        reason=("Drive the inserted screw with the selected power driver."
                if assignment.driver == POWER_DRIVER else
                "Drive the inserted screw with the selected manual Phillips driver."),
    )
    add(
        "PLACE", assignment.driver, assignment.work_surface,
        reason="Return the used driver to the workbench after fastening.",
    )
    return plan
