"""Small domain compiler and OWL-TAMP relaxed grounding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

from .models import Action, ValidationError


@dataclass(frozen=True)
class Operator:
    name: str
    argument_kinds: tuple[str, ...]


DOMAINS: dict[str, tuple[Operator, ...]] = {
    "kitchen": (
        Operator("PICK", ("object",)),
        Operator("PLACE", ("object", "destination")),
        Operator("POUR", ("object", "object")),
        Operator("STIR", ("object", "object")),
        Operator("PLACE_SERVING_UTENSIL", ("object", "object")),
        Operator("OPEN", ("inspectable_region",)),
    ),
    "living_room": (
        Operator("PICK", ("object",)),
        Operator("PLACE", ("object", "region")),
    ),
    "workshop": (
        Operator("INSPECT", ("inspectable_region",)),
        Operator("PICK", ("object",)),
        Operator("PLACE", ("object", "destination")),
        Operator("INSERT", ("object", "object")),
        Operator("FASTEN", ("object", "object", "object")),
    ),
}


def relaxed_ground(
    scene: str,
    object_ids: Iterable[str],
    region_ids: Iterable[str],
    inspectable_regions: Iterable[str] = (),
) -> tuple[Action, ...]:
    """Ground reachable operator shapes with optimistic continuous values.

    The paper's relaxed reachability uses placeholders for continuous values.
    This planning-only adaptation omits those placeholders from the model-facing
    action signature and adds them during continuous refinement.
    """
    if scene not in DOMAINS:
        raise ValueError(f"Unsupported OWL-TAMP scene {scene!r}")
    pools = {
        "object": tuple(sorted(set(object_ids))),
        "region": tuple(sorted(set(region_ids))),
        "destination": tuple(sorted(set(object_ids) | set(region_ids))),
        "inspectable_region": tuple(sorted(set(inspectable_regions))),
    }
    result = []
    for operator in DOMAINS[scene]:
        for arguments in product(*(pools[kind] for kind in operator.argument_kinds)):
            if len(arguments) == 2 and arguments[0] == arguments[1]:
                continue
            result.append(Action(operator.name, tuple(arguments)))
    return tuple(result)


def validate_sketch(
    scene: str,
    actions: Iterable[Action],
    grounded_actions: Iterable[Action],
) -> tuple[Action, ...]:
    allowed = set(grounded_actions)
    arities = {operator.name: len(operator.argument_kinds) for operator in DOMAINS[scene]}
    result = []
    for action in actions:
        if action.operator not in arities:
            raise ValidationError(f"unknown {scene} operator {action.operator!r}")
        if len(action.arguments) != arities[action.operator]:
            raise ValidationError(f"{action.operator} has the wrong arity")
        if action not in allowed:
            raise ValidationError(
                f"action is not in the relaxed grounded set: {action.as_dict()}"
            )
        result.append(action)
    return tuple(result)


def executed_encoding(actions: Iterable[Action]) -> tuple[dict[str, object], ...]:
    """Return the paper's Executed(i) subsequence compilation."""
    rows = []
    for index, action in enumerate(actions):
        rows.append(
            {
                "action": action.as_dict(),
                "extra_precondition": "Executed(0)" if index == 0 else f"Executed({index})",
                "extra_effect": f"Executed({index + 1})",
            }
        )
    return tuple(rows)
