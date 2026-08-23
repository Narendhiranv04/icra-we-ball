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
        "MOVE_TO", "OPEN_STORAGE", "INSPECT_STORAGE", "CLOSE_STORAGE", "PICK",
        "PLACE_ON_SURFACE", "INSERT_FASTENER",
        "DRIVE_FASTENER", "VERIFY_REPAIR", "TERMINATE_INFEASIBLE",
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
    inspection_regions = list(spec["expected_inspection_regions"])
    robot_at = "HOME"
    for region in inspection_regions:
        add("MOVE_TO", region, reason="Visit the next storage region in the fixed search order.")
        robot_at = region
        add("OPEN_STORAGE", region, reason="Expose the fixed object set in this region.")
        add("INSPECT_STORAGE", region, reason="Observe the screw, compatible drivers, or hammer distractor.")
        add("CLOSE_STORAGE", region, reason="Restore the inspected storage region before continuing search.")

    if not assignment.is_feasible:
        add("TERMINATE_INFEASIBLE", assignment.rejection_reason or "UNKNOWN",
            reason="All three regions were inspected and the required screw-driver pair is absent.")
        return plan

    assert assignment.driver and assignment.fastener and assignment.work_surface
    assert driver_source and fastener_source
    if robot_at != driver_source:
        add("MOVE_TO", driver_source, reason="Move to the first compatible driver observed during search.")
    add("OPEN_STORAGE", driver_source, reason="Make the selected driver accessible.")
    add("PICK", assignment.driver, driver_source, reason="Acquire the first compatible driver encountered.")
    add("MOVE_TO", assignment.work_surface, reason="Carry the selected driver to the fixed main workbench staging area.")
    add("PLACE_ON_SURFACE", assignment.driver, assignment.work_surface,
        reason="Stage the selected driver safely while retrieving the screw.")
    add("MOVE_TO", driver_source, reason="Return empty-handed to the driver's source.")
    add("CLOSE_STORAGE", driver_source, reason="Close the driver's source after retrieval.")

    if driver_source != fastener_source:
        add("MOVE_TO", fastener_source, reason="Move to the observed compatible screw.")
    add("OPEN_STORAGE", fastener_source, reason="Make the compatible screw accessible.")
    add("PICK", assignment.fastener, fastener_source, reason="Acquire the single compatible workbench screw.")
    if fastener_source in {"LEFT_DRAWER", "RIGHT_DRAWER"}:
        add("MOVE_TO", assignment.work_surface,
            reason="Carry the screw to a stable workbench staging pose before restoring the extracted drawer.")
        add("PLACE_ON_SURFACE", assignment.fastener, assignment.work_surface,
            reason="Stage the screw so the gripper is free to close its source drawer.")
        add("MOVE_TO", fastener_source,
            reason="Return empty-handed to the screw's source drawer.")
        add("CLOSE_STORAGE", fastener_source,
            reason="Close the extracted drawer before entering the repair workspace.")
        add("MOVE_TO", assignment.work_surface,
            reason="Return to the staged screw after the drawer is closed.")
        add("PICK", assignment.fastener, assignment.work_surface,
            reason="Reacquire the staged screw for insertion.")
    add("MOVE_TO", assignment.target_joint, reason="Carry the fastener to the loose frame joint.")
    add("INSERT_FASTENER", assignment.fastener, assignment.target_joint,
        reason="Place it perfectly tip-down with the screw head/recess on top.")
    if fastener_source == "TOOL_CABINET":
        add("MOVE_TO", fastener_source,
            reason="Return empty-handed after insertion to restore the cabinet.")
        add("CLOSE_STORAGE", fastener_source,
            reason="Close the cabinet after retrieving and inserting the screw.")
    add("MOVE_TO", assignment.work_surface, reason="Return to the selected staged driver.")
    add("PICK", assignment.driver, assignment.work_surface, reason="Acquire the staged driver for fastening.")
    add("MOVE_TO", assignment.target_joint, reason="Bring the compatible driver to the inserted fastener.")
    add("DRIVE_FASTENER", assignment.driver, assignment.fastener, assignment.target_joint,
        reason=("Hold the powered driver steady while its motor turns and advances the screw."
                if assignment.driver == POWER_DRIVER else
                "Rotate the manual driver in visible ratchet strokes while the screw advances."))
    add("MOVE_TO", assignment.work_surface, reason="Return the driver to the selected work surface.")
    add("PLACE_ON_SURFACE", assignment.driver, assignment.work_surface,
        reason="Leave the used driver safely on the main workbench.")
    add("MOVE_TO", assignment.target_joint, reason="Move to the repaired joint for terminal inspection.")
    add("VERIFY_REPAIR", assignment.target_joint, reason="Verify the intended repair and terminal organization.")
    return plan
