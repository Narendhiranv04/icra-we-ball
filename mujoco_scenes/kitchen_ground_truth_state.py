"""Ground-truth oracle world state tracking and symbolic preflight verification.

This module maintains the authoritative oracle world state for kitchen execution
without relying on learned perception or frozen-plan ledger restrictions.
It explicitly supports:
- Object reusability (e.g., PICK -> STIR -> PLACE -> PICK)
- Relocated cupboard/storage objects becoming accessible on the countertop
- Strict precondition and effect validation
- Symbolic preflight checking prior to physical execution
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


CONTAINER_NAMES = ("D1", "D2", "C2", "B1", "C1")


class StatePreconditionError(Exception):
    """Raised when an action's preconditions are violated in the oracle state."""

    def __init__(self, action_index: int, action_instance_id: str, operator: str, reason: str):
        super().__init__(
            f"Action #{action_index} ({action_instance_id}: {operator}) failed precondition: {reason}"
        )
        self.action_index = action_index
        self.action_instance_id = action_instance_id
        self.operator = operator
        self.reason = reason


@dataclass
class OracleWorldState:
    """Live symbolic representation of the kitchen world state."""

    held_object: str | None = None
    object_locations: dict[str, str] = field(default_factory=dict)
    container_open: dict[str, bool] = field(default_factory=lambda: {c: False for c in CONTAINER_NAMES})
    poured_relations: set[tuple[str, str]] = field(default_factory=set)
    stirred_relations: set[tuple[str, str]] = field(default_factory=set)
    served_objects: set[str] = field(default_factory=set)
    utensil_bowl_pairs: set[tuple[str, str]] = field(default_factory=set)

    def clone(self) -> OracleWorldState:
        return OracleWorldState(
            held_object=self.held_object,
            object_locations=dict(self.object_locations),
            container_open=dict(self.container_open),
            poured_relations=set(self.poured_relations),
            stirred_relations=set(self.stirred_relations),
            served_objects=set(self.served_objects),
            utensil_bowl_pairs=set(self.utensil_bowl_pairs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_object": self.held_object,
            "object_locations": dict(sorted(self.object_locations.items())),
            "container_open": dict(sorted(self.container_open.items())),
            "poured_relations": sorted([list(p) for p in self.poured_relations]),
            "stirred_relations": sorted([list(p) for p in self.stirred_relations]),
            "served_objects": sorted(list(self.served_objects)),
            "utensil_bowl_pairs": sorted([list(p) for p in self.utensil_bowl_pairs]),
        }

    def is_object_accessible(self, object_id: str) -> tuple[bool, str]:
        """Determine if an object can currently be accessed/manipulated."""
        if object_id not in self.object_locations:
            return False, f"Object '{object_id}' is not in known object locations"
        location = self.object_locations[object_id]
        if location == "GRIPPER":
            return False, f"Object '{object_id}' is already held in gripper"
        if location in CONTAINER_NAMES:
            if not self.container_open.get(location, False):
                return False, f"Object '{object_id}' is inside closed container '{location}'"
            return True, f"Object '{object_id}' is inside open container '{location}'"
        # On countertop, serving_area, or beside another object
        return True, f"Object '{object_id}' is accessible at '{location}'"

    def check_preconditions(self, action: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if an action is valid in this state without applying effects."""
        operator = str(action.get("operator", "")).upper()
        arguments = list(action.get("arguments", []))

        if operator == "OPEN":
            if not arguments:
                return False, "OPEN requires container argument"
            container = arguments[0]
            if container not in CONTAINER_NAMES:
                return False, f"Unknown container '{container}'"
            if self.container_open.get(container, False):
                return False, f"Container '{container}' is already open"
            return True, None

        if operator == "CLOSE":
            if not arguments:
                return False, "CLOSE requires container argument"
            container = arguments[0]
            if container not in CONTAINER_NAMES:
                return False, f"Unknown container '{container}'"
            if not self.container_open.get(container, False):
                return False, f"Container '{container}' is already closed"
            return True, None

        if operator == "PICK":
            if not arguments:
                return False, "PICK requires object argument"
            object_id = arguments[0]
            if self.held_object is not None:
                return False, f"Hand is not empty (already holding '{self.held_object}')"
            accessible, reason = self.is_object_accessible(object_id)
            if not accessible:
                return False, reason
            return True, None

        if operator == "PLACE":
            if not arguments:
                return False, "PLACE requires object and destination arguments"
            object_id = arguments[0]
            destination = arguments[1] if len(arguments) > 1 else "countertop"
            if self.held_object != object_id:
                return False, f"Cannot place '{object_id}' because hand holds '{self.held_object}'"
            return True, None

        if operator == "POUR":
            if len(arguments) < 2:
                return False, "POUR requires source and target arguments"
            source_id, target_id = arguments[0], arguments[1]
            if self.held_object != source_id:
                return False, f"Cannot pour from '{source_id}' because hand holds '{self.held_object}'"
            accessible, reason = self.is_object_accessible(target_id)
            if not accessible:
                return False, f"Target '{target_id}' inaccessible: {reason}"
            return True, None

        if operator == "STIR":
            if len(arguments) < 2:
                return False, "STIR requires tool and target arguments"
            tool_id, target_id = arguments[0], arguments[1]
            if self.held_object != tool_id:
                return False, f"Cannot stir with '{tool_id}' because hand holds '{self.held_object}'"
            accessible, reason = self.is_object_accessible(target_id)
            if not accessible:
                return False, f"Target '{target_id}' inaccessible: {reason}"
            return True, None

        if operator in {"SERVE_COFFEE", "SERVE_SOUP"}:
            if not arguments:
                return False, f"{operator} requires target vessel argument"
            target_id = arguments[0]
            accessible, reason = self.is_object_accessible(target_id)
            if not accessible and self.held_object != target_id:
                return False, f"Target '{target_id}' inaccessible: {reason}"
            return True, None

        if operator == "PLACE_SERVING_UTENSIL":
            if len(arguments) < 2:
                return False, "PLACE_SERVING_UTENSIL requires utensil and bowl arguments"
            utensil_id, bowl_id = arguments[0], arguments[1]
            if self.held_object != utensil_id:
                return False, f"Cannot place utensil '{utensil_id}' because hand holds '{self.held_object}'"
            return True, None

        # Unknown operator
        return False, f"Unsupported operator '{operator}'"

    def apply_action(self, action: dict[str, Any]) -> None:
        """Apply state effects for a validated action."""
        operator = str(action.get("operator", "")).upper()
        arguments = list(action.get("arguments", []))

        if operator == "OPEN":
            container = arguments[0]
            self.container_open[container] = True

        elif operator == "CLOSE":
            container = arguments[0]
            self.container_open[container] = False

        elif operator == "PICK":
            object_id = arguments[0]
            self.held_object = object_id
            self.object_locations[object_id] = "GRIPPER"

        elif operator == "PLACE":
            object_id = arguments[0]
            destination = arguments[1] if len(arguments) > 1 else "countertop"
            self.held_object = None
            self.object_locations[object_id] = destination
            if destination == "serving_area":
                self.served_objects.add(object_id)
            elif destination in self.object_locations:
                self.utensil_bowl_pairs.add((object_id, destination))

        elif operator == "POUR":
            source_id, target_id = arguments[0], arguments[1]
            self.poured_relations.add((source_id, target_id))

        elif operator == "STIR":
            tool_id, target_id = arguments[0], arguments[1]
            self.stirred_relations.add((tool_id, target_id))

        elif operator in {"SERVE_COFFEE", "SERVE_SOUP"}:
            target_id = arguments[0]
            self.held_object = None
            self.object_locations[target_id] = "serving_area"
            self.served_objects.add(target_id)

        elif operator == "PLACE_SERVING_UTENSIL":
            utensil_id, bowl_id = arguments[0], arguments[1]
            self.held_object = None
            self.object_locations[utensil_id] = f"beside_{bowl_id}"
            self.utensil_bowl_pairs.add((utensil_id, bowl_id))


def initialize_oracle_world_state(
    all_instances: Iterable[tuple[str, str, str | None]],
) -> OracleWorldState:
    """Build the initial OracleWorldState from known scene objects."""
    state = OracleWorldState()
    for instance_id, kind, region in all_instances:
        if region in CONTAINER_NAMES:
            state.object_locations[instance_id] = region
        else:
            state.object_locations[instance_id] = "countertop"
    return state


def run_symbolic_preflight(
    initial_state: OracleWorldState,
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute a purely symbolic simulation pass over the generated GT plan.

    Fails before physics if any action's preconditions are violated.
    """
    state = initial_state.clone()
    step_records = []
    success = True
    failure_reason = None
    failed_step_index = None

    for index, action in enumerate(plan):
        action_index = int(action.get("action_index", index))
        action_instance_id = str(action.get("action_instance_id", f"act_{index:03d}"))
        operator = str(action.get("operator", "")).upper()
        arguments = list(action.get("arguments", []))

        state_before = state.to_dict()
        valid, reason = state.check_preconditions(action)

        step_record = {
            "action_index": action_index,
            "action_instance_id": action_instance_id,
            "operator": operator,
            "arguments": arguments,
            "precondition_valid": valid,
            "precondition_failure_reason": reason,
            "state_before": state_before,
        }

        if not valid:
            success = False
            failure_reason = reason
            failed_step_index = action_index
            step_record["state_after"] = state_before
            step_records.append(step_record)
            break

        state.apply_action(action)
        step_record["state_after"] = state.to_dict()
        step_records.append(step_record)

    return {
        "preflight_status": "PASSED" if success else "FAILED",
        "success": success,
        "total_actions": len(plan),
        "validated_actions": len(step_records) if success else len(step_records) - 1,
        "failed_step_index": failed_step_index,
        "failure_reason": failure_reason,
        "final_symbolic_state": state.to_dict(),
        "steps": step_records,
    }
