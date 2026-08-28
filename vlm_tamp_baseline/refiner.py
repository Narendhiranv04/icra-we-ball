"""Deterministic symbolic-subgoal to shared-skill refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from baseline_common.catalog import load_catalog as load_action_catalog
from baseline_common.catalog import scene_actions
from baseline_common.models import Action, Observation, ValidationError, parse_plan

from .models import RefinementFailure, Subgoal


@dataclass(frozen=True)
class RefinementResult:
    actions: tuple[Action, ...] = ()
    failure: RefinementFailure | None = None

    @property
    def success(self) -> bool:
        return self.failure is None


def held_object(observation: Observation) -> str | None:
    value = observation.robot.get("holding")
    if value is None:
        value = observation.robot.get("held_object")
    if value is None:
        value = observation.robot.get("held_object_id")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    return None


def subgoal_satisfied(subgoal: Subgoal, observation: Observation) -> bool:
    arguments = subgoal.arguments
    predicate = subgoal.predicate
    if predicate == "HOLDING":
        return held_object(observation) == arguments["object_id"]
    if predicate == "INSPECTED":
        return any(
            region.region_id == arguments["region_id"] and region.inspected
            for region in observation.regions
        )
    facts_by_id = {item.entity_id: item.facts for item in observation.entities}
    if predicate == "PLACED":
        facts = facts_by_id.get(arguments["object_id"], {})
        location = facts.get(
            "region_id", facts.get("location", facts.get("source_region"))
        )
        return location == arguments["region_id"]
    target_id = arguments.get("target_id")
    facts = facts_by_id.get(target_id, {}) if target_id else {}
    fact_keys = {
        "POURED": ("poured_from", "source_id"),
        "STIRRED": ("stirred_with", "tool_id"),
        "CLEANED": ("cleaned_with", "tool_id"),
        "REMOVED": ("removed_objects", "object_id"),
        "INSERTED": ("inserted_fasteners", "fastener_id"),
        "FASTENED": ("fastened_with", "tool_id"),
    }
    if predicate not in fact_keys:
        return False
    fact_name, argument_name = fact_keys[predicate]
    value = facts.get(fact_name)
    expected = arguments[argument_name]
    return value == expected or (
        isinstance(value, (list, tuple, set)) and expected in value
    )


class CatalogSubgoalRefiner:
    """Generate a discrete skill skeleton; the shared motion layer refines it."""

    def __init__(self, action_catalog: Mapping[str, Any] | None = None):
        self.action_catalog = (
            load_action_catalog() if action_catalog is None else action_catalog
        )

    def refine(
        self, subgoal: Subgoal, observation: Observation
    ) -> RefinementResult:
        if subgoal_satisfied(subgoal, observation):
            return RefinementResult()
        visible = observation.object_ids
        arguments = subgoal.arguments
        required_objects = [
            value
            for name, value in arguments.items()
            if name.endswith("_id") and name != "region_id"
        ]
        hidden = [item for item in required_objects if item not in visible]
        destination = arguments.get("region_id")
        if (
            destination
            and destination not in observation.region_ids
            and destination not in visible
        ):
            hidden.append(destination)
        if hidden:
            return self._failure(
                subgoal,
                "ungrounded_object",
                "Required object is not currently visible: " + ", ".join(hidden),
            )
        held = held_object(observation)
        actions: list[Action] = []

        def acquire(object_id: str) -> RefinementFailure | None:
            nonlocal held
            if held == object_id:
                return None
            if held is not None:
                return RefinementFailure(
                    "hand_not_empty",
                    f"Robot is holding {held}; cannot acquire {object_id}.",
                    subgoal,
                )
            actions.append(Action("PICK", {"object_id": object_id}))
            held = object_id
            return None

        predicate = subgoal.predicate
        if predicate == "INSPECTED":
            if held is not None:
                return self._failure(
                    subgoal,
                    "hand_not_empty",
                    f"Robot is holding {held}; place it before inspection.",
                )
            actions.append(Action("INSPECT", {"region_id": arguments["region_id"]}))
        elif predicate == "HOLDING":
            failure = acquire(arguments["object_id"])
            if failure:
                return RefinementResult(failure=failure)
        elif predicate == "PLACED":
            failure = acquire(arguments["object_id"])
            if failure:
                return RefinementResult(failure=failure)
            actions.append(Action("PLACE", dict(arguments)))
        elif predicate == "POURED":
            failure = acquire(arguments["source_id"])
            if failure:
                return RefinementResult(failure=failure)
            actions.append(Action("POUR", dict(arguments)))
        elif predicate == "STIRRED":
            failure = acquire(arguments["tool_id"])
            if failure:
                return RefinementResult(failure=failure)
            actions.append(Action("STIR", dict(arguments)))
        elif predicate == "CLEANED":
            failure = acquire(arguments["tool_id"])
            if failure:
                return RefinementResult(failure=failure)
            actions.append(Action("CLEAN", dict(arguments)))
        elif predicate == "REMOVED":
            actions.append(Action("REMOVE", dict(arguments)))
        elif predicate == "INSERTED":
            failure = acquire(arguments["fastener_id"])
            if failure:
                return RefinementResult(failure=failure)
            actions.append(Action("INSERT", dict(arguments)))
        elif predicate == "FASTENED":
            failure = acquire(arguments["tool_id"])
            if failure:
                return RefinementResult(failure=failure)
            actions.append(Action("FASTEN", dict(arguments)))
        else:
            return self._failure(
                subgoal, "unsupported_subgoal", f"No refiner for {predicate}."
            )

        available = scene_actions(self.action_catalog, observation.scene)
        try:
            validated = parse_plan(
                {
                    "status": "PLAN",
                    "actions": [item.as_dict() for item in actions],
                },
                available,
                observation,
                max_actions=max(1, len(actions)),
            )
        except ValidationError as error:
            return self._failure(subgoal, "symbolic_refinement_failed", str(error))
        return RefinementResult(validated.actions)

    @staticmethod
    def _failure(
        subgoal: Subgoal, code: str, message: str
    ) -> RefinementResult:
        return RefinementResult(
            failure=RefinementFailure(code, message, subgoal)
        )
