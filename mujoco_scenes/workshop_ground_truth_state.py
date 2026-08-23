"""Symbolic Workshop state and strict action preflight validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .workshop_scene import WORKSHOP_REGIONS


class WorkshopPreconditionError(RuntimeError):
    """Raised when a Workshop action violates its declared contract."""


@dataclass
class WorkshopWorldState:
    robot_at: str = "HOME"
    held_object: str | None = None
    object_locations: dict[str, str] = field(default_factory=dict)
    storage_open: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in WORKSHOP_REGIONS}
    )
    inspected_storage: set[str] = field(default_factory=set)
    staged_in_container: set[tuple[str, str]] = field(default_factory=set)
    inserted_fastener: tuple[str, str] | None = None
    repaired_joint: str | None = None
    verified_joint: str | None = None
    termination_reason: str | None = None

    def clone(self) -> "WorkshopWorldState":
        return WorkshopWorldState(
            robot_at=self.robot_at,
            held_object=self.held_object,
            object_locations=dict(self.object_locations),
            storage_open=dict(self.storage_open),
            inspected_storage=set(self.inspected_storage),
            staged_in_container=set(self.staged_in_container),
            inserted_fastener=self.inserted_fastener,
            repaired_joint=self.repaired_joint,
            verified_joint=self.verified_joint,
            termination_reason=self.termination_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_at": self.robot_at,
            "held_object": self.held_object,
            "object_locations": dict(sorted(self.object_locations.items())),
            "storage_open": dict(sorted(self.storage_open.items())),
            "inspected_storage": sorted(self.inspected_storage),
            "staged_in_container": sorted([list(item) for item in self.staged_in_container]),
            "inserted_fastener": list(self.inserted_fastener) if self.inserted_fastener else None,
            "repaired_joint": self.repaired_joint,
            "verified_joint": self.verified_joint,
            "termination_reason": self.termination_reason,
        }

    def check(self, action: dict[str, Any], assignment: Any) -> tuple[bool, str | None]:
        op = action["operator"]
        args = action.get("arguments", [])
        if op == "MOVE_TO":
            return True, None
        if op in {"OPEN_STORAGE", "INSPECT_STORAGE", "CLOSE_STORAGE"}:
            region = args[0]
            if region not in WORKSHOP_REGIONS:
                return False, f"unknown storage region {region}"
            if self.robot_at != region:
                return False, f"robot is at {self.robot_at}, not {region}"
            if op == "OPEN_STORAGE" and self.storage_open[region]:
                return False, f"{region} is already open"
            if op == "OPEN_STORAGE" and self.held_object is not None:
                return False, "hand must be empty to open storage"
            if op in {"INSPECT_STORAGE", "CLOSE_STORAGE"} and not self.storage_open[region]:
                return False, f"{region} is closed"
            if op == "CLOSE_STORAGE" and self.held_object is not None:
                return False, "hand must be empty to close storage"
            return True, None
        if op == "PICK":
            obj, source = args
            if self.robot_at != source:
                return False, f"robot is at {self.robot_at}, not {source}"
            if self.held_object is not None:
                return False, f"hand already holds {self.held_object}"
            if self.object_locations.get(obj) != source:
                return False, f"{obj} is not at {source}"
            if source in WORKSHOP_REGIONS and not self.storage_open[source]:
                return False, f"source storage {source} is closed"
            return True, None
        if op in {"PLACE_ON_SURFACE", "PLACE_IN_CONTAINER"}:
            obj, destination = args
            if self.robot_at != destination:
                return False, f"robot is at {self.robot_at}, not {destination}"
            if self.held_object != obj:
                return False, f"hand does not hold {obj}"
            expected = assignment.work_surface if op == "PLACE_ON_SURFACE" else assignment.parts_container
            if destination != expected:
                return False, f"{destination} is not assigned for {op}"
            return True, None
        if op == "INSERT_FASTENER":
            fastener, target = args
            if self.robot_at != target or self.held_object != fastener:
                return False, "robot must hold the fastener at the target joint"
            return True, None
        if op == "DRIVE_FASTENER":
            driver, fastener, target = args
            if self.robot_at != target or self.held_object != driver:
                return False, "robot must hold the assigned driver at the target joint"
            if self.inserted_fastener != (fastener, target):
                return False, "assigned fastener is not inserted"
            if (driver, fastener, target) != (assignment.driver, assignment.fastener, assignment.target_joint):
                return False, "repair tuple does not match the grounded assignment"
            return True, None
        if op == "VERIFY_REPAIR":
            target = args[0]
            if self.robot_at != target or self.repaired_joint != target:
                return False, "target joint is not repaired at the robot location"
            if self.held_object is not None:
                return False, "hand must be empty for terminal verification"
            return True, None
        if op == "TERMINATE_INFEASIBLE":
            if set(self.inspected_storage) != set(WORKSHOP_REGIONS):
                return False, "all storage must be inspected before infeasible termination"
            if self.held_object is not None:
                return False, "hand must be empty at infeasible termination"
            if assignment.is_feasible or args[0] != assignment.rejection_reason:
                return False, "termination does not match the grounded rejection"
            return True, None
        return False, f"unsupported operator {op}"

    def apply(self, action: dict[str, Any]) -> None:
        op, args = action["operator"], action.get("arguments", [])
        if op == "MOVE_TO":
            self.robot_at = args[0]
        elif op == "OPEN_STORAGE":
            self.storage_open[args[0]] = True
        elif op == "INSPECT_STORAGE":
            self.inspected_storage.add(args[0])
        elif op == "CLOSE_STORAGE":
            self.storage_open[args[0]] = False
        elif op == "PICK":
            self.held_object = args[0]
            self.object_locations[args[0]] = "GRIPPER"
        elif op == "PLACE_ON_SURFACE":
            self.held_object = None
            self.object_locations[args[0]] = args[1]
        elif op == "PLACE_IN_CONTAINER":
            self.held_object = None
            self.object_locations[args[0]] = args[1]
            self.staged_in_container.add((args[0], args[1]))
        elif op == "INSERT_FASTENER":
            self.held_object = None
            self.object_locations[args[0]] = args[1]
            self.inserted_fastener = (args[0], args[1])
        elif op == "DRIVE_FASTENER":
            self.repaired_joint = args[2]
        elif op == "VERIFY_REPAIR":
            self.verified_joint = args[0]
        elif op == "TERMINATE_INFEASIBLE":
            self.termination_reason = args[0]


def initial_workshop_state(storage_contents: dict[str, list[str]]) -> WorkshopWorldState:
    state = WorkshopWorldState()
    for region, objects in storage_contents.items():
        for object_name in objects:
            state.object_locations[object_name] = region
    return state


def symbolic_preflight(state: WorkshopWorldState, plan: list[dict[str, Any]], assignment: Any) -> dict[str, Any]:
    live = state.clone()
    trace = []
    for action in plan:
        valid, reason = live.check(action, assignment)
        trace.append({"action_instance_id": action["action_instance_id"], "valid": valid, "reason": reason})
        if not valid:
            return {"success": False, "failed_action": action, "failure_reason": reason, "trace": trace}
        live.apply(action)
    return {"success": True, "failed_action": None, "failure_reason": None, "trace": trace, "final_state": live.to_dict()}
