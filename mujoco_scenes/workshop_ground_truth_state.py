"""Symbolic Workshop state and strict action preflight validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .workshop_scene import WORKSHOP_REGIONS


class WorkshopPreconditionError(RuntimeError):
    """Raised when a Workshop action violates its declared contract."""


@dataclass
class WorkshopWorldState:
    held_object: str | None = None
    object_locations: dict[str, str] = field(default_factory=dict)
    storage_open: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in WORKSHOP_REGIONS}
    )
    inspected_storage: set[str] = field(default_factory=set)
    inserted_fastener: tuple[str, str] | None = None
    repaired_joint: str | None = None

    def clone(self) -> "WorkshopWorldState":
        return WorkshopWorldState(
            held_object=self.held_object,
            object_locations=dict(self.object_locations),
            storage_open=dict(self.storage_open),
            inspected_storage=set(self.inspected_storage),
            inserted_fastener=self.inserted_fastener,
            repaired_joint=self.repaired_joint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_object": self.held_object,
            "object_locations": dict(sorted(self.object_locations.items())),
            "storage_open": dict(sorted(self.storage_open.items())),
            "inspected_storage": sorted(self.inspected_storage),
            "inserted_fastener": list(self.inserted_fastener) if self.inserted_fastener else None,
            "repaired_joint": self.repaired_joint,
        }

    def check(self, action: dict[str, Any], assignment: Any) -> tuple[bool, str | None]:
        op = action["operator"]
        args = action.get("arguments", [])
        if op == "OPEN":
            region = args[0]
            if region not in WORKSHOP_REGIONS:
                return False, f"unknown storage region {region}"
            if self.storage_open[region]:
                return False, f"{region} is already open"
            if self.held_object is not None:
                return False, "hand must be empty to open storage"
            return True, None
        if op == "PICK":
            obj, source = args
            if self.held_object is not None:
                return False, f"hand already holds {self.held_object}"
            if self.object_locations.get(obj) != source:
                return False, f"{obj} is not at {source}"
            if source in WORKSHOP_REGIONS and not self.storage_open[source]:
                return False, f"source storage {source} is closed"
            return True, None
        if op == "PLACE":
            obj, destination = args
            if self.held_object != obj:
                return False, f"hand does not hold {obj}"
            if obj == assignment.fastener:
                if destination != assignment.target_joint:
                    return False, "the discovered screw must be placed directly at the frame joint"
            elif obj == assignment.driver:
                if destination != assignment.work_surface:
                    return False, f"{destination} is not the assigned driver return surface"
            else:
                return False, f"{obj} is not part of the grounded repair pair"
            return True, None
        if op == "SCREW":
            driver, fastener, target = args
            if self.held_object != driver:
                return False, "robot must hold the assigned driver"
            if self.inserted_fastener != (fastener, target):
                return False, "assigned fastener is not inserted"
            if (driver, fastener, target) != (assignment.driver, assignment.fastener, assignment.target_joint):
                return False, "repair tuple does not match the grounded assignment"
            return True, None
        return False, f"unsupported operator {op}"

    def apply(self, action: dict[str, Any]) -> None:
        op, args = action["operator"], action.get("arguments", [])
        if op == "OPEN":
            self.storage_open[args[0]] = True
            self.inspected_storage.add(args[0])
        elif op == "PICK":
            self.held_object = args[0]
            self.object_locations[args[0]] = "GRIPPER"
        elif op == "PLACE":
            self.held_object = None
            self.object_locations[args[0]] = args[1]
            if args[1] == "workshop_frame_joint":
                self.inserted_fastener = (args[0], args[1])
        elif op == "SCREW":
            self.repaired_joint = args[2]


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
