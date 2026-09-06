"""Variant-aware, non-oracular symbolic contracts for live ViLaIn runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pathlib import Path
import yaml

from .domains.registry import DomainDefinition
from .execution.base import controller_capabilities


SExpression = str | list["SExpression"]


@dataclass(frozen=True)
class OperatorContract:
    name: str
    parameter_names: tuple[str, ...]
    parameter_types: tuple[str, ...]
    precondition: str
    effect: str
    controller_primitive: str
    controller_argument_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_names": list(self.parameter_names),
            "parameter_types": list(self.parameter_types),
            "precondition": self.precondition,
            "effect": self.effect,
            "controller_primitive": self.controller_primitive,
            "controller_argument_indices": list(self.controller_argument_indices),
        }


@dataclass(frozen=True)
class VariantActionContract:
    domain: str
    variant: str
    type_hierarchy: Mapping[str, str | None]
    predicate_signatures: Mapping[str, tuple[str, ...]]
    operators: Mapping[str, OperatorContract]
    structural_inventory: Mapping[str, str] = field(default_factory=dict)
    closed_world_initial_state: bool = True
    geometry_owned_by_refinement: bool = True

    def __post_init__(self) -> None:
        if not self.variant.strip():
            raise ValueError("variant must not be empty")
        if set(self.operators) == set():
            raise ValueError("variant action contract must expose operators")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "variant": self.variant,
            "type_hierarchy": dict(self.type_hierarchy),
            "predicate_signatures": {
                key: list(value) for key, value in self.predicate_signatures.items()
            },
            "operators": {
                key: value.to_dict() for key, value in self.operators.items()
            },
            "structural_inventory": dict(self.structural_inventory),
            "closed_world_initial_state": self.closed_world_initial_state,
            "geometry_owned_by_refinement": self.geometry_owned_by_refinement,
        }


@dataclass(frozen=True)
class LiteralValidationError:
    code: str
    section: str
    literal: str
    predicate: str | None = None
    argument_index: int | None = None
    expected: str | None = None
    actual: str | None = None
    object_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "code": self.code,
                "section": self.section,
                "literal": self.literal,
                "predicate": self.predicate,
                "argument_index": self.argument_index,
                "expected": self.expected,
                "actual": self.actual,
                "object": self.object_id,
                "detail": self.detail,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class GroundedFact:
    fact_id: str
    predicate: str
    arguments: tuple[str, ...]
    literal: str
    goal_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "predicate": self.predicate,
            "arguments": list(self.arguments),
            "literal": self.literal,
            "goal_eligible": self.goal_eligible,
        }


class SymbolicContractViolation(ValueError):
    def __init__(self, errors: Sequence[LiteralValidationError]) -> None:
        self.errors = tuple(errors)
        rendered = "; ".join(
            f"{error.code}: {error.detail or error.literal}" for error in self.errors
        )
        super().__init__(rendered)


def build_variant_action_contract(
    domain: DomainDefinition,
    variant: str,
    *,
    structural_inventory: Mapping[str, str] | None = None,
) -> VariantActionContract:
    """Build from immutable PDDL and the execution projection vocabulary.

    Structural inventory may contain only neutral entity IDs and PDDL types.
    Expected solutions, roles, plans, and benchmark answers are deliberately
    not accepted by this interface.
    """
    inventory = dict(structural_inventory or {})
    for object_id, object_type in inventory.items():
        if object_type not in domain.type_hierarchy:
            raise ValueError(f"structural entity {object_id!r} has unknown type")
    capabilities = controller_capabilities(domain.key)
    action_forms = _action_forms(domain.text)
    if set(capabilities) != set(domain.action_signatures):
        raise ValueError("PDDL and controller operator vocabularies differ")
    operators: dict[str, OperatorContract] = {}
    for name, parameter_types in domain.action_signatures.items():
        form = action_forms.get(name)
        if form is None:
            raise ValueError(f"fixed domain is missing action {name!r}")
        primitive, argument_indices = capabilities[name]
        operators[name] = OperatorContract(
            name=name,
            parameter_names=_action_parameter_names(form),
            parameter_types=parameter_types,
            precondition=_action_field(form, ":precondition"),
            effect=_action_field(form, ":effect"),
            controller_primitive=primitive,
            controller_argument_indices=argument_indices,
        )
    return VariantActionContract(
        domain=domain.key,
        variant=variant,
        type_hierarchy=MappingProxyType(dict(domain.type_hierarchy)),
        predicate_signatures=MappingProxyType(dict(domain.predicate_signatures)),
        operators=MappingProxyType(operators),
        structural_inventory=MappingProxyType(inventory),
    )


def load_variant_action_contract(
    domain: DomainDefinition,
    variant: str,
    *,
    config_root: str | Path,
) -> VariantActionContract:
    """Select neutral structural capabilities without loading variant answers."""
    root = Path(config_root)
    config_names = {
        "kitchen": "kitchen_feasibility_variants.yaml",
        "living_room": "living_room_variants.yaml",
        "workshop": "workshop_variants.yaml",
    }
    path = root / config_names[domain.key]
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"variant configuration must be a mapping: {path}")
    variants = loaded.get("variants")
    from mujoco_scenes.final_paper_variant_labels import resolve_variant_name

    internal_variant = resolve_variant_name(domain.key, variant)
    if not isinstance(variants, Mapping) or internal_variant not in variants:
        raise ValueError(f"unknown {domain.key} variant {variant!r}")
    row = variants[internal_variant]
    if not isinstance(row, Mapping):
        raise ValueError(f"variant {variant!r} must be a mapping")

    inventory: dict[str, str]
    if domain.key == "kitchen":
        storages = loaded.get("inspection_order", ())
        inventory = {str(item).lower(): "storage" for item in storages}
        inventory.update(
            {
                "countertop": "surface",
                "serving_area": "surface",
                # Public task-level symbolic constants, not hidden scene state.
                "coffee": "content",
                "soup": "content",
            }
        )
    elif domain.key == "living_room":
        regions = loaded.get("regions", {})
        if not isinstance(regions, Mapping):
            raise ValueError("living-room regions must be a mapping")
        inventory = {
            str(region_id).lower(): "support"
            for region_id in regions
        }
        inventory["staging"] = "location"
    else:
        storages = loaded.get("storage_regions", ())
        inventory = {str(item).lower(): "storage" for item in storages}
        target = str(loaded.get("fixed_insertion_target", "")).strip().lower()
        if target:
            inventory[target] = "target"
            # The workbench is also a legal support for returning a tool.  A
            # typed alias keeps the immutable PDDL type hierarchy honest while
            # both IDs resolve to the same neutral physical fixture.
            inventory[f"{target}_surface"] = "surface"
    return build_variant_action_contract(
        domain, internal_variant, structural_inventory=inventory
    )


def object_type_table(
    object_types: Mapping[str, str],
    contract: VariantActionContract,
) -> Mapping[str, tuple[str, ...]]:
    combined = dict(contract.structural_inventory)
    combined.update({str(key).lower(): str(value).lower() for key, value in object_types.items()})
    rows: dict[str, list[str]] = {name: [] for name in contract.type_hierarchy}
    for object_id, object_type in sorted(combined.items()):
        if object_type not in contract.type_hierarchy:
            raise ValueError(f"object {object_id!r} has unknown type {object_type!r}")
        rows[object_type].append(object_id)
    return MappingProxyType(
        {key: tuple(value) for key, value in rows.items() if value}
    )


def enumerate_grounded_facts(
    object_types: Mapping[str, str],
    contract: VariantActionContract,
    *,
    maximum_facts: int = 5000,
) -> tuple[GroundedFact, ...]:
    """Enumerate the stable, type-safe propositional universe for one run."""
    if maximum_facts <= 0:
        raise ValueError("maximum_facts must be positive")
    normalized = {
        str(object_id).strip().lower(): str(object_type).strip().lower()
        for object_id, object_type in object_types.items()
    }
    if len(normalized) != len(object_types) or any(not key for key in normalized):
        raise ValueError("object inventory contains duplicate or empty IDs")
    for object_id, object_type in normalized.items():
        if object_type not in contract.type_hierarchy:
            raise ValueError(f"object {object_id!r} has unknown type {object_type!r}")

    goal_predicates = _positive_effect_predicates(contract)
    rows: list[tuple[str, tuple[str, ...], bool]] = []
    for predicate in sorted(contract.predicate_signatures):
        signature = contract.predicate_signatures[predicate]
        compatible = [
            tuple(
                object_id
                for object_id, object_type in sorted(normalized.items())
                if _is_subtype(object_type, expected, contract.type_hierarchy)
            )
            for expected in signature
        ]
        if any(not values for values in compatible):
            continue
        combinations = product(*compatible) if compatible else ((),)
        for arguments in combinations:
            # A single physical/symbolic entity cannot fill two distinct
            # argument roles in these single-inheritance domains.
            if len(arguments) != len(set(arguments)):
                continue
            rows.append((predicate, tuple(arguments), predicate in goal_predicates))
            if len(rows) > maximum_facts:
                raise ValueError(
                    f"grounded fact universe exceeds safe limit {maximum_facts}"
                )
    return tuple(
        GroundedFact(
            fact_id=f"f{index:04d}",
            predicate=predicate,
            arguments=arguments,
            literal=(
                f"({predicate} {' '.join(arguments)})"
                if arguments
                else f"({predicate})"
            ),
            goal_eligible=goal_eligible,
        )
        for index, (predicate, arguments, goal_eligible) in enumerate(rows, start=1)
    )


def grouped_fact_payload(
    facts: Sequence[GroundedFact], *, goal_only: bool = False
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        if goal_only and not fact.goal_eligible:
            continue
        grouped.setdefault(fact.predicate, []).append(
            {"fact_id": fact.fact_id, "literal": fact.literal}
        )
    return MappingProxyType(
        {key: tuple(value) for key, value in sorted(grouped.items())}
    )


def missing_manipulable_types(
    object_types: Mapping[str, str], contract: VariantActionContract
) -> tuple[str, ...]:
    """Return absent leaf-like movable parameter types required by operators.

    This is a capability diagnostic only: it reports types and never invents
    entities or consults variant answers.
    """
    present = tuple(object_types.values())
    required = {
        parameter_type
        for operator in contract.operators.values()
        for parameter_type in operator.parameter_types
        if _is_subtype(parameter_type, "movable", contract.type_hierarchy)
        and parameter_type != "movable"
    }
    return tuple(
        sorted(
            required_type
            for required_type in required
            if not any(
                _is_subtype(actual, required_type, contract.type_hierarchy)
                for actual in present
            )
        )
    )


def relaxed_goal_diagnostics(
    initial_facts: Sequence[GroundedFact],
    goal_facts: Sequence[GroundedFact],
    object_types: Mapping[str, str],
    contract: VariantActionContract,
) -> tuple[Mapping[str, Any], ...]:
    """Compute positive relaxed reachability from fixed grounded operators."""
    reachable = {fact.literal for fact in initial_facts}
    typed_objects = tuple(sorted(object_types.items()))
    grounded: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []
    for operator in contract.operators.values():
        compatible = [
            tuple(
                object_id
                for object_id, actual_type in typed_objects
                if _is_subtype(actual_type, expected, contract.type_hierarchy)
            )
            for expected in operator.parameter_types
        ]
        if any(not values for values in compatible):
            continue
        for arguments in product(*compatible):
            if len(arguments) != len(set(arguments)):
                continue
            bindings = dict(zip(operator.parameter_names, arguments))
            positive_preconditions = tuple(
                _ground_atom(atom, bindings)
                for atom in _positive_atoms(_parse_single(operator.precondition))
            )
            positive_effects = tuple(
                _ground_atom(atom, bindings)
                for atom in _positive_atoms(_parse_single(operator.effect))
            )
            grounded.append((positive_preconditions, positive_effects, operator.name))

    changed = True
    while changed:
        changed = False
        for preconditions, effects, _ in grounded:
            if set(preconditions).issubset(reachable):
                before = len(reachable)
                reachable.update(effects)
                changed = changed or len(reachable) != before

    diagnostics: list[Mapping[str, Any]] = []
    for goal in goal_facts:
        if goal.literal in reachable:
            continue
        achievers = sorted(
            {
                operator
                for _, effects, operator in grounded
                if goal.literal in effects
            }
        )
        diagnostics.append(
            {
                "code": "RELAXED_GOAL_UNREACHABLE",
                "goal_fact_id": goal.fact_id,
                "goal_literal": goal.literal,
                "potential_ground_achievers": achievers,
                "detail": (
                    "no grounded operator can add this fact"
                    if not achievers
                    else "all grounded achievers have unreachable positive preconditions"
                ),
            }
        )
    return tuple(diagnostics)


def initial_state_diagnostics(
    initial_facts: Sequence[GroundedFact],
) -> tuple[Mapping[str, Any], ...]:
    """Reject generic physical contradictions in an FM-selected initial state."""
    by_predicate: dict[str, list[GroundedFact]] = {}
    for fact in initial_facts:
        by_predicate.setdefault(fact.predicate, []).append(fact)
    diagnostics: list[Mapping[str, Any]] = []
    holdings = by_predicate.get("holding", [])
    if holdings and by_predicate.get("handempty"):
        diagnostics.append(
            {
                "code": "HAND_STATE_CONTRADICTION",
                "fact_ids": [by_predicate["handempty"][0].fact_id]
                + [fact.fact_id for fact in holdings],
                "detail": "handempty and holding facts cannot both be true",
            }
        )
    if len(holdings) > 1:
        diagnostics.append(
            {
                "code": "MULTIPLE_HELD_OBJECTS",
                "fact_ids": [fact.fact_id for fact in holdings],
                "detail": "the single gripper cannot hold multiple objects",
            }
        )
    locations: dict[str, list[GroundedFact]] = {}
    for fact in by_predicate.get("at", []):
        locations.setdefault(fact.arguments[0], []).append(fact)
    held_objects = {fact.arguments[0] for fact in holdings}
    for object_id, facts in sorted(locations.items()):
        if len(facts) > 1:
            diagnostics.append(
                {
                    "code": "MULTIPLE_OBJECT_LOCATIONS",
                    "object_id": object_id,
                    "fact_ids": [fact.fact_id for fact in facts],
                    "detail": "one object cannot be at multiple locations",
                }
            )
        if object_id in held_objects:
            diagnostics.append(
                {
                    "code": "HELD_AND_AT_LOCATION",
                    "object_id": object_id,
                    "fact_ids": [fact.fact_id for fact in facts]
                    + [fact.fact_id for fact in holdings if fact.arguments[0] == object_id],
                    "detail": "a held object cannot simultaneously remain at a location",
                }
            )
    return tuple(diagnostics)


def _positive_atoms(expression: SExpression, *, negated: bool = False) -> tuple[list[SExpression], ...]:
    if not isinstance(expression, list) or not expression:
        return ()
    head = expression[0]
    if head == "and":
        return tuple(
            atom
            for child in expression[1:]
            for atom in _positive_atoms(child, negated=negated)
        )
    if head == "not":
        return ()
    if isinstance(head, str) and not negated:
        return (expression,)
    return ()


def _ground_atom(atom: Sequence[SExpression], bindings: Mapping[str, str]) -> str:
    values = [bindings.get(str(value), str(value)) for value in atom]
    return "(" + " ".join(values) + ")"


def _positive_effect_predicates(contract: VariantActionContract) -> set[str]:
    result: set[str] = set()
    for operator in contract.operators.values():
        expression = _parse_single(operator.effect)

        def visit(value: SExpression, negated: bool = False) -> None:
            if not isinstance(value, list) or not value:
                return
            head = value[0]
            if head == "and":
                for child in value[1:]:
                    visit(child, negated)
            elif head == "not":
                if len(value) == 2:
                    visit(value[1], True)
            elif isinstance(head, str) and not negated:
                result.add(head)

        visit(expression)
    return result


def validate_initial_fragment(
    fragment: str,
    object_types: Mapping[str, str],
    contract: VariantActionContract,
) -> None:
    form = _parse_single(fragment)
    if not form or form[0] != ":init":
        raise SymbolicContractViolation(
            (LiteralValidationError("INVALID_SECTION", "init", fragment, detail="expected :init"),)
        )
    errors: list[LiteralValidationError] = []
    for expression in form[1:]:
        if isinstance(expression, list) and expression and expression[0] == "not":
            errors.append(
                LiteralValidationError(
                    "NEGATIVE_INIT_LITERAL_NOT_ALLOWED",
                    "init",
                    _render(expression),
                    detail="false initial facts must be omitted under closed-world semantics",
                )
            )
            continue
        errors.extend(_validate_atom(expression, "init", object_types, contract))
    if errors:
        raise SymbolicContractViolation(errors)


def validate_goal_fragment(
    fragment: str,
    object_types: Mapping[str, str],
    contract: VariantActionContract,
) -> None:
    form = _parse_single(fragment)
    if len(form) != 2 or form[0] != ":goal":
        raise SymbolicContractViolation(
            (LiteralValidationError("INVALID_SECTION", "goal", fragment, detail="expected one :goal expression"),)
        )
    errors: list[LiteralValidationError] = []
    _validate_goal_expression(form[1], object_types, contract, errors)
    if errors:
        raise SymbolicContractViolation(errors)


def _validate_goal_expression(
    expression: SExpression,
    object_types: Mapping[str, str],
    contract: VariantActionContract,
    errors: list[LiteralValidationError],
) -> None:
    if isinstance(expression, list) and expression:
        if expression[0] in {"and", "or"}:
            for child in expression[1:]:
                _validate_goal_expression(child, object_types, contract, errors)
            return
        if expression[0] == "not":
            if len(expression) != 2:
                errors.append(LiteralValidationError("INVALID_LITERAL_SYNTAX", "goal", _render(expression)))
            else:
                errors.extend(_validate_atom(expression[1], "goal", object_types, contract))
            return
    errors.extend(_validate_atom(expression, "goal", object_types, contract))


def _validate_atom(
    expression: SExpression,
    section: str,
    object_types: Mapping[str, str],
    contract: VariantActionContract,
) -> list[LiteralValidationError]:
    literal = _render(expression)
    if not isinstance(expression, list) or not expression or not isinstance(expression[0], str):
        return [LiteralValidationError("INVALID_LITERAL_SYNTAX", section, literal)]
    predicate = expression[0]
    signature = contract.predicate_signatures.get(predicate)
    if signature is None:
        return [
            LiteralValidationError(
                "UNKNOWN_PREDICATE", section, literal, predicate=predicate,
                detail=f"unknown predicate {predicate!r}",
            )
        ]
    arguments = expression[1:]
    if len(arguments) != len(signature):
        return [
            LiteralValidationError(
                "ARITY_MISMATCH",
                section,
                literal,
                predicate=predicate,
                expected=str(len(signature)),
                actual=str(len(arguments)),
            )
        ]
    errors: list[LiteralValidationError] = []
    # Inventory entries are legal candidates, not implicit PDDL constants.
    # Every literal argument must still be explicitly declared by the module.
    available_types = dict(object_types)
    for index, (argument, expected_type) in enumerate(zip(arguments, signature)):
        if not isinstance(argument, str):
            errors.append(LiteralValidationError("INVALID_LITERAL_SYNTAX", section, literal, predicate=predicate))
            continue
        actual_type = available_types.get(argument)
        if actual_type is None:
            errors.append(
                LiteralValidationError(
                    "UNKNOWN_OBJECT", section, literal, predicate=predicate,
                    argument_index=index, object_id=argument,
                )
            )
        elif not _is_subtype(actual_type, expected_type, contract.type_hierarchy):
            errors.append(
                LiteralValidationError(
                    "INVALID_ARGUMENT_TYPE", section, literal, predicate=predicate,
                    argument_index=index, expected=expected_type, actual=actual_type,
                    object_id=argument,
                )
            )
    return errors


def _is_subtype(actual: str, expected: str, hierarchy: Mapping[str, str | None]) -> bool:
    if expected == "object":
        return True
    current: str | None = actual
    visited: set[str] = set()
    while current is not None and current not in visited:
        if current == expected:
            return True
        visited.add(current)
        current = hierarchy.get(current)
    return False


def _parse_single(text: str) -> list[SExpression]:
    tokens = re.findall(r"[()]|[^\s()]+", re.sub(r";[^\n]*", "", text.lower()))
    position = 0

    def parse() -> SExpression:
        nonlocal position
        if position >= len(tokens):
            raise SymbolicContractViolation((LiteralValidationError("INVALID_LITERAL_SYNTAX", "unknown", text),))
        token = tokens[position]
        position += 1
        if token != "(":
            return token
        result: list[SExpression] = []
        while position < len(tokens) and tokens[position] != ")":
            result.append(parse())
        if position >= len(tokens):
            raise SymbolicContractViolation((LiteralValidationError("INVALID_LITERAL_SYNTAX", "unknown", text),))
        position += 1
        return result

    value = parse()
    if position != len(tokens) or not isinstance(value, list):
        raise SymbolicContractViolation((LiteralValidationError("INVALID_LITERAL_SYNTAX", "unknown", text),))
    return value


def _render(value: SExpression) -> str:
    if isinstance(value, str):
        return value
    return "(" + " ".join(_render(item) for item in value) + ")"


def _action_forms(text: str) -> dict[str, str]:
    forms: dict[str, str] = {}
    for match in re.finditer(r"\(\s*:action\s+([^\s()]+)", text, flags=re.I):
        start = match.start()
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    forms[match.group(1).lower()] = text[start:index + 1]
                    break
    return forms


def _action_field(form: str, field: str) -> str:
    match = re.search(re.escape(field), form, flags=re.I)
    if match is None:
        raise ValueError(f"action is missing {field}")
    open_index = form.find("(", match.end())
    if open_index < 0:
        raise ValueError(f"action has malformed {field}")
    depth = 0
    for index in range(open_index, len(form)):
        if form[index] == "(":
            depth += 1
        elif form[index] == ")":
            depth -= 1
            if depth == 0:
                return " ".join(form[open_index:index + 1].lower().split())
    raise ValueError(f"action has unbalanced {field}")


def _action_parameter_names(form: str) -> tuple[str, ...]:
    match = re.search(r":parameters\s*\(([^)]*)\)", form, flags=re.I | re.S)
    if match is None:
        raise ValueError("action is missing :parameters")
    return tuple(re.findall(r"\?[a-z0-9_-]+", match.group(1).lower()))
