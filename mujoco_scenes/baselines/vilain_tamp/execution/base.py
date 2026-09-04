"""Syntactic projection from baseline PDDL actions to controller requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts import ExecutionProjection, SymbolicAction
from ..identity import EntityBinding


class ProjectionError(ValueError):
    """A symbolic action cannot be safely projected to a controller request."""


@dataclass(frozen=True)
class _ProjectionRule:
    controller_operator: str
    arity: int
    controller_argument_indices: tuple[int, ...]


_RULES: dict[str, dict[str, _ProjectionRule]] = {
    "kitchen": {
        "open-storage": _ProjectionRule("OPEN", 1, (0,)),
        "pick-from": _ProjectionRule("PICK", 2, (0,)),
        "place-on": _ProjectionRule("PLACE", 2, (0, 1)),
        "pour": _ProjectionRule("POUR", 3, (0, 1)),
        "stir": _ProjectionRule("STIR", 2, (0, 1)),
        "place-in": _ProjectionRule("PLACE", 2, (0, 1)),
    },
    "living_room": {
        "pick-from": _ProjectionRule("PICK", 2, (0,)),
        "place-on": _ProjectionRule("PLACE", 2, (0, 1)),
    },
    "workshop": {
        "open-storage": _ProjectionRule("OPEN", 1, (0,)),
        "pick-from": _ProjectionRule("PICK", 2, (0, 1)),
        "insert": _ProjectionRule("PLACE", 2, (0, 1)),
        "drive": _ProjectionRule("SCREW", 3, (0, 1, 2)),
        "place-on": _ProjectionRule("PLACE", 2, (0, 1)),
    },
}


def project_action(
    domain: str,
    action: SymbolicAction,
    bindings: Mapping[str, EntityBinding],
    *,
    fixed_bindings: Mapping[str, EntityBinding] | None = None,
    external_method_artifacts: Mapping[str, Any] | None = None,
) -> ExecutionProjection:
    """Apply only the fixed Section-13 syntactic projection rules."""
    if external_method_artifacts is not None:
        raise ProjectionError("external method artifacts are not valid projection input")
    if not isinstance(action, SymbolicAction):
        raise ProjectionError("projection requires a SymbolicAction contract")
    if not all(isinstance(item, EntityBinding) for item in bindings.values()):
        raise ProjectionError("projection bindings must use EntityBinding contracts")
    if fixed_bindings is not None and not all(
        isinstance(item, EntityBinding) for item in fixed_bindings.values()
    ):
        raise ProjectionError("fixed bindings must use EntityBinding contracts")
    if any(key != item.object_id for key, item in bindings.items()):
        raise ProjectionError("projection binding keys must match their symbolic object IDs")
    if fixed_bindings is not None and any(
        key != item.object_id for key, item in fixed_bindings.items()
    ):
        raise ProjectionError("fixed binding keys must match their symbolic object IDs")

    domain_key = domain.strip().lower().replace("-", "_")
    operator = action.operator.strip().lower().replace("_", "-")
    domain_rules = _RULES.get(domain_key)
    if domain_rules is None:
        raise ProjectionError(f"unsupported baseline domain {domain!r}")
    rule = domain_rules.get(operator)
    if rule is None:
        raise ProjectionError(
            f"UNSUPPORTED_CONTROLLER_ACTION: {operator!r} in {domain_key!r}"
        )
    if len(action.arguments) != rule.arity:
        raise ProjectionError(
            f"{operator!r} expects {rule.arity} arguments, got {len(action.arguments)}"
        )

    available = dict(fixed_bindings or {})
    overlap = set(available).intersection(bindings)
    if overlap:
        raise ProjectionError(f"duplicate movable/fixed binding IDs: {sorted(overlap)!r}")
    available.update(bindings)
    selected = []
    for index in rule.controller_argument_indices:
        symbolic_id = action.arguments[index]
        binding = available.get(symbolic_id)
        if binding is None:
            raise ProjectionError(f"UNRESOLVED_ENTITY: {symbolic_id!r}")
        selected.append(binding)

    evidence = tuple(
        sorted(
            {
                artifact
                for binding in selected
                for artifact in binding.evidence_artifacts
            }
        )
    )
    methods = sorted({binding.binding_method for binding in selected})
    return ExecutionProjection(
        action_instance_id=action.action_instance_id,
        pddl_operator=operator,
        pddl_arguments=action.arguments,
        controller_operator=rule.controller_operator,
        controller_arguments=tuple(binding.entity_name for binding in selected),
        resolved_entities=tuple(binding.entity_name for binding in selected),
        binding_method="+".join(methods),
        binding_confidence=min(binding.confidence for binding in selected),
        binding_evidence_artifacts=evidence,
        skill_parameters={},
    )


def project_plan(
    domain: str,
    actions: Sequence[SymbolicAction],
    bindings: Mapping[str, EntityBinding],
    *,
    fixed_bindings: Mapping[str, EntityBinding] | None = None,
    external_method_artifacts: Mapping[str, Any] | None = None,
) -> tuple[ExecutionProjection, ...]:
    if external_method_artifacts is not None:
        raise ProjectionError("external method artifacts are not valid projection input")
    return tuple(
        project_action(
            domain,
            action,
            bindings,
            fixed_bindings=fixed_bindings,
        )
        for action in actions
    )


def required_binding_ids(
    domain: str,
    actions: Sequence[SymbolicAction],
) -> tuple[str, ...]:
    """Return symbolic IDs that must resolve to controller-facing entities."""
    domain_key = domain.strip().lower().replace("-", "_")
    domain_rules = _RULES.get(domain_key)
    if domain_rules is None:
        raise ProjectionError(f"unsupported baseline domain {domain!r}")
    required: set[str] = set()
    for action in actions:
        if not isinstance(action, SymbolicAction):
            raise ProjectionError("projection requires SymbolicAction contracts")
        operator = action.operator.strip().lower().replace("_", "-")
        rule = domain_rules.get(operator)
        if rule is None:
            raise ProjectionError(
                f"UNSUPPORTED_CONTROLLER_ACTION: {operator!r} in {domain_key!r}"
            )
        if len(action.arguments) != rule.arity:
            raise ProjectionError(
                f"{operator!r} expects {rule.arity} arguments, "
                f"got {len(action.arguments)}"
            )
        required.update(
            action.arguments[index] for index in rule.controller_argument_indices
        )
    return tuple(sorted(required))
