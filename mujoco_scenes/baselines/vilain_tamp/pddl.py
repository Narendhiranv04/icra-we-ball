"""Internal structural checks for fixed domains and generated PDDL problems."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import time
from typing import Iterable, Sequence

from .contracts import PDDLValidationResult, ValidationStage
from .domains.registry import DomainDefinition


SExpression = str | list["SExpression"]


class PDDLStructureError(ValueError):
    """Raised when PDDL cannot be interpreted as one balanced document."""


@dataclass(frozen=True)
class DomainSchema:
    name: str
    type_hierarchy: dict[str, str | None]
    predicate_signatures: dict[str, tuple[str, ...]]
    action_signatures: dict[str, tuple[str, ...]]


def parse_domain_schema(text: str) -> DomainSchema:
    document = _parse_document(text)
    _expect_head(document, "define")
    declaration = _find_child(document, "domain")
    if len(declaration) != 2 or not isinstance(declaration[1], str):
        raise PDDLStructureError("domain declaration must contain exactly one name")

    types_section = _find_child(document, ":types")
    type_hierarchy = {
        name: None if parent == "object" else parent
        for name, parent in _typed_symbols(types_section[1:], default_type="object")
    }
    if len(type_hierarchy) != len(_typed_symbols(types_section[1:], default_type="object")):
        raise PDDLStructureError("domain contains duplicate type declarations")
    for type_name, parent in type_hierarchy.items():
        if parent is not None and parent not in type_hierarchy:
            raise PDDLStructureError(f"type {type_name!r} has unknown parent {parent!r}")
        if parent == type_name:
            raise PDDLStructureError(f"type {type_name!r} cannot inherit from itself")

    predicates_section = _find_child(document, ":predicates")
    predicate_signatures: dict[str, tuple[str, ...]] = {}
    for declaration_node in predicates_section[1:]:
        declaration_list = _as_list(declaration_node, "predicate declaration")
        if not declaration_list or not isinstance(declaration_list[0], str):
            raise PDDLStructureError("predicate declaration is empty")
        name = declaration_list[0]
        if name in predicate_signatures:
            raise PDDLStructureError(f"duplicate predicate {name!r}")
        predicate_signatures[name] = tuple(
            value_type
            for _, value_type in _typed_symbols(
                declaration_list[1:], default_type="object", require_variables=True
            )
        )

    action_signatures: dict[str, tuple[str, ...]] = {}
    for node in document[1:]:
        if not isinstance(node, list) or not node or node[0] != ":action":
            continue
        if len(node) < 4 or not isinstance(node[1], str):
            raise PDDLStructureError("action declaration is incomplete")
        name = node[1]
        if name in action_signatures:
            raise PDDLStructureError(f"duplicate action {name!r}")
        try:
            parameter_index = node.index(":parameters")
        except ValueError as error:
            raise PDDLStructureError(f"action {name!r} has no parameters") from error
        if parameter_index + 1 >= len(node):
            raise PDDLStructureError(f"action {name!r} has no parameter list")
        parameters = _as_list(node[parameter_index + 1], "action parameter list")
        action_signatures[name] = tuple(
            value_type
            for _, value_type in _typed_symbols(
                parameters, default_type="object", require_variables=True
            )
        )

    return DomainSchema(
        name=declaration[1],
        type_hierarchy=type_hierarchy,
        predicate_signatures=predicate_signatures,
        action_signatures=action_signatures,
    )


def domain_definition_diagnostics(definition: DomainDefinition) -> tuple[str, ...]:
    diagnostics: list[str] = []
    actual_hash = hashlib.sha256(definition.text.encode("utf-8")).hexdigest()
    if actual_hash != definition.sha256:
        diagnostics.append("domain text does not match its registered SHA-256")
    try:
        schema = parse_domain_schema(definition.text)
    except PDDLStructureError as error:
        return (str(error),)
    if schema.name != definition.name:
        diagnostics.append(
            f"domain name mismatch: PDDL has {schema.name!r}, registry has {definition.name!r}"
        )
    if schema.type_hierarchy != dict(definition.type_hierarchy):
        diagnostics.append("type hierarchy differs between PDDL and knowledge file")
    if schema.predicate_signatures != dict(definition.predicate_signatures):
        diagnostics.append("predicate signatures differ between PDDL and knowledge file")
    if schema.action_signatures != dict(definition.action_signatures):
        diagnostics.append("action signatures differ between PDDL and knowledge file")
    return tuple(diagnostics)


def domain_is_unchanged(expected_sha256: str, current_text: str) -> bool:
    return hashlib.sha256(current_text.encode("utf-8")).hexdigest() == expected_sha256


def validate_problem(
    problem_text: str,
    definition: DomainDefinition,
    *,
    expected_domain_sha256: str | None = None,
) -> PDDLValidationResult:
    started = time.perf_counter()
    diagnostics = list(domain_definition_diagnostics(definition))
    if expected_domain_sha256 is not None and not domain_is_unchanged(
        expected_domain_sha256, definition.text
    ):
        diagnostics.append("immutable domain hash mismatch")
    try:
        if diagnostics:
            raise PDDLStructureError("fixed domain definition is inconsistent")
        schema = parse_domain_schema(definition.text)
        problem = _parse_document(problem_text)
        _validate_problem_document(problem, schema)
    except (PDDLStructureError, KeyError) as error:
        diagnostics.append(str(error))
    return PDDLValidationResult(
        valid=not diagnostics,
        stage=ValidationStage.INTERNAL,
        diagnostics=tuple(diagnostics),
        exit_code=None,
        stdout_artifact=None,
        stderr_artifact=None,
        elapsed_seconds=time.perf_counter() - started,
    )


def _validate_problem_document(problem: list[SExpression], schema: DomainSchema) -> None:
    _expect_head(problem, "define")
    declaration = _find_child(problem, "problem")
    if len(declaration) != 2 or not isinstance(declaration[1], str):
        raise PDDLStructureError("problem declaration must contain exactly one name")
    domain_declaration = _find_child(problem, ":domain")
    if len(domain_declaration) != 2 or domain_declaration[1] != schema.name:
        raise PDDLStructureError(
            f"problem domain must be {schema.name!r}"
        )

    objects_section = _find_child(problem, ":objects")
    typed_objects = _typed_symbols(objects_section[1:], default_type="object")
    object_types: dict[str, str] = {}
    for object_name, object_type in typed_objects:
        if object_name.startswith("?"):
            raise PDDLStructureError("problem objects must not be variables")
        if object_name in object_types:
            raise PDDLStructureError(f"duplicate object {object_name!r}")
        if object_type != "object" and object_type not in schema.type_hierarchy:
            raise PDDLStructureError(f"object {object_name!r} has unknown type {object_type!r}")
        object_types[object_name] = object_type

    init_section = _find_child(problem, ":init")
    for atom in init_section[1:]:
        _validate_atom(atom, schema, object_types)

    goal_section = _find_child(problem, ":goal")
    if len(goal_section) != 2:
        raise PDDLStructureError("goal section must contain exactly one expression")
    goal_atoms = list(_goal_atoms(goal_section[1]))
    if not goal_atoms:
        raise PDDLStructureError("goal must contain at least one atom")
    for atom in goal_atoms:
        _validate_atom(atom, schema, object_types)


def _validate_atom(
    atom: SExpression,
    schema: DomainSchema,
    object_types: dict[str, str],
) -> None:
    values = _as_list(atom, "atom")
    if not values or not isinstance(values[0], str):
        raise PDDLStructureError("atom must start with a predicate")
    predicate = values[0]
    if predicate not in schema.predicate_signatures:
        raise PDDLStructureError(f"unknown predicate {predicate!r}")
    arguments = values[1:]
    expected_types = schema.predicate_signatures[predicate]
    if len(arguments) != len(expected_types):
        raise PDDLStructureError(
            f"predicate {predicate!r} expects {len(expected_types)} arguments, got {len(arguments)}"
        )
    for argument, expected_type in zip(arguments, expected_types):
        if not isinstance(argument, str):
            raise PDDLStructureError(f"predicate {predicate!r} has a non-symbol argument")
        if argument not in object_types:
            raise PDDLStructureError(f"predicate {predicate!r} uses undeclared object {argument!r}")
        if not _is_subtype(object_types[argument], expected_type, schema.type_hierarchy):
            raise PDDLStructureError(
                f"object {argument!r} of type {object_types[argument]!r} is not a {expected_type!r}"
            )


def _goal_atoms(expression: SExpression) -> Iterable[SExpression]:
    values = _as_list(expression, "goal expression")
    if not values:
        return
    head = values[0]
    if head in {"and", "or"}:
        for child in values[1:]:
            yield from _goal_atoms(child)
        return
    if head == "not":
        if len(values) != 2:
            raise PDDLStructureError("not goal must contain exactly one atom")
        yield values[1]
        return
    yield values


def _is_subtype(actual: str, expected: str, hierarchy: dict[str, str | None]) -> bool:
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


def _parse_document(text: str) -> list[SExpression]:
    tokens = re.findall(r"[()]|[^\s()]+", re.sub(r";[^\n]*", "", text.lower()))
    if not tokens:
        raise PDDLStructureError("PDDL document is empty")
    position = 0

    def parse_expression() -> SExpression:
        nonlocal position
        if position >= len(tokens):
            raise PDDLStructureError("unexpected end of PDDL document")
        token = tokens[position]
        position += 1
        if token == "(":
            result: list[SExpression] = []
            while position < len(tokens) and tokens[position] != ")":
                result.append(parse_expression())
            if position >= len(tokens):
                raise PDDLStructureError("unbalanced opening parenthesis")
            position += 1
            return result
        if token == ")":
            raise PDDLStructureError("unexpected closing parenthesis")
        return token

    document = parse_expression()
    if position != len(tokens):
        raise PDDLStructureError("PDDL must contain exactly one top-level form")
    return _as_list(document, "PDDL document")


def _typed_symbols(
    values: Sequence[SExpression],
    *,
    default_type: str,
    require_variables: bool = False,
) -> list[tuple[str, str]]:
    if not all(isinstance(value, str) for value in values):
        raise PDDLStructureError("typed symbol list must contain only symbols")
    symbols = [str(value) for value in values]
    result: list[tuple[str, str]] = []
    pending: list[str] = []
    index = 0
    while index < len(symbols):
        token = symbols[index]
        if token == "-":
            if not pending or index + 1 >= len(symbols):
                raise PDDLStructureError("malformed typed symbol list")
            value_type = symbols[index + 1]
            if value_type == "-":
                raise PDDLStructureError("malformed type name")
            result.extend((name, value_type) for name in pending)
            pending = []
            index += 2
            continue
        pending.append(token)
        index += 1
    result.extend((name, default_type) for name in pending)
    names = [name for name, _ in result]
    if len(names) != len(set(names)):
        raise PDDLStructureError("typed symbol list contains duplicates")
    if require_variables and any(not name.startswith("?") for name in names):
        raise PDDLStructureError("parameter names must be variables")
    return result


def _find_child(document: Sequence[SExpression], head: str) -> list[SExpression]:
    matches = [
        node
        for node in document[1:]
        if isinstance(node, list) and node and node[0] == head
    ]
    if len(matches) != 1:
        raise PDDLStructureError(f"expected exactly one {head} section")
    return matches[0]


def _expect_head(document: Sequence[SExpression], expected: str) -> None:
    if not document or document[0] != expected:
        raise PDDLStructureError(f"document must start with {expected!r}")


def _as_list(value: SExpression, label: str) -> list[SExpression]:
    if not isinstance(value, list):
        raise PDDLStructureError(f"{label} must be a list")
    return value
