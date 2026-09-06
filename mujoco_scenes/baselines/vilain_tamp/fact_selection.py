"""Strict fact-ID selection and deterministic PDDL compilation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from .domains.registry import DomainDefinition
from .symbolic_contract import GroundedFact


@dataclass(frozen=True)
class FactSelectionError(ValueError):
    code: str
    field: str
    detail: str
    invalid_ids: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "detail": self.detail,
            "invalid_ids": list(self.invalid_ids),
        }


def parse_fact_selection(
    raw_text: str,
    *,
    field: str,
    candidates: Sequence[GroundedFact],
) -> tuple[GroundedFact, ...]:
    stripped = raw_text.strip()
    if stripped.startswith("(") or ":init" in stripped.lower() or ":goal" in stripped.lower():
        raise FactSelectionError(
            "RAW_PDDL_NOT_ALLOWED", field, "model must return JSON fact IDs only"
        )
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```" and lines[0].strip().lower() in {"```", "```json"}:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise FactSelectionError("MALFORMED_SELECTION_JSON", field, str(error)) from error
    if not isinstance(loaded, Mapping) or set(loaded) != {field}:
        raise FactSelectionError(
            "INVALID_SELECTION_SCHEMA", field, f"expected exactly one {field!r} field"
        )
    selected = loaded[field]
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise FactSelectionError(
            "INVALID_SELECTION_SCHEMA", field, "selected fact IDs must be a string list"
        )
    if len(selected) != len(set(selected)):
        raise FactSelectionError(
            "DUPLICATE_FACT_ID", field, "selected fact IDs must be unique"
        )
    by_id = {item.fact_id: item for item in candidates}
    invalid = tuple(sorted(set(selected).difference(by_id)))
    if invalid:
        raise FactSelectionError(
            "UNKNOWN_FACT_ID", field, "selection contains unknown fact IDs", invalid
        )
    return tuple(by_id[item] for item in selected)


def parse_corrective_selection(
    raw_text: str,
    *,
    candidates: Sequence[GroundedFact],
) -> tuple[tuple[GroundedFact, ...], tuple[GroundedFact, ...]]:
    stripped = raw_text.strip()
    if stripped.startswith("(") or ":init" in stripped.lower() or ":goal" in stripped.lower():
        raise FactSelectionError(
            "RAW_PDDL_NOT_ALLOWED", "correction", "CP must return JSON fact IDs only"
        )
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise FactSelectionError("MALFORMED_SELECTION_JSON", "correction", str(error)) from error
    required = {"true_fact_ids", "goal_fact_ids"}
    if not isinstance(loaded, Mapping) or set(loaded) != required:
        raise FactSelectionError(
            "INVALID_SELECTION_SCHEMA", "correction", "expected init and goal ID fields"
        )
    initial = parse_fact_selection(
        json.dumps({"true_fact_ids": loaded["true_fact_ids"]}),
        field="true_fact_ids",
        candidates=candidates,
    )
    goals = parse_fact_selection(
        json.dumps({"goal_fact_ids": loaded["goal_fact_ids"]}),
        field="goal_fact_ids",
        candidates=tuple(fact for fact in candidates if fact.goal_eligible),
    )
    if not goals:
        raise FactSelectionError(
            "EMPTY_GOAL_SELECTION", "goal_fact_ids", "goal selection must not be empty"
        )
    return initial, goals


def compile_objects_fragment(object_types: Mapping[str, str]) -> str:
    grouped: dict[str, list[str]] = {}
    for object_id, object_type in sorted(object_types.items()):
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", object_id):
            raise ValueError(f"invalid PDDL object ID {object_id!r}")
        grouped.setdefault(object_type, []).append(object_id)
    lines = ["(:objects"]
    for object_type in sorted(grouped):
        lines.append(f"  {' '.join(grouped[object_type])} - {object_type}")
    lines.append(")")
    return "\n".join(lines)


def compile_init_fragment(facts: Sequence[GroundedFact]) -> str:
    selected = _stable_unique(facts)
    return "(:init\n" + "".join(f"  {fact.literal}\n" for fact in selected) + ")"


def compile_goal_fragment(facts: Sequence[GroundedFact]) -> str:
    selected = _stable_unique(facts)
    if not selected:
        raise ValueError("goal fact selection must not be empty")
    if len(selected) == 1:
        return f"(:goal {selected[0].literal})"
    return "(:goal (and\n" + "".join(
        f"  {fact.literal}\n" for fact in selected
    ) + "))"


def compile_problem(
    *,
    domain: DomainDefinition,
    problem_name: str,
    object_types: Mapping[str, str],
    initial_facts: Sequence[GroundedFact],
    goal_facts: Sequence[GroundedFact],
) -> tuple[str, str, str, str]:
    objects = compile_objects_fragment(object_types)
    init = compile_init_fragment(initial_facts)
    goal = compile_goal_fragment(goal_facts)
    text = (
        f"(define (problem {problem_name})\n"
        f"  (:domain {domain.name})\n"
        f"  {_indent(objects)}\n"
        f"  {_indent(init)}\n"
        f"  {_indent(goal)}\n"
        ")\n"
    )
    return text, objects, init, goal


def _stable_unique(facts: Sequence[GroundedFact]) -> tuple[GroundedFact, ...]:
    by_id = {fact.fact_id: fact for fact in facts}
    if len(by_id) != len(facts):
        raise ValueError("selected facts must be unique")
    return tuple(by_id[key] for key in sorted(by_id))


def _indent(form: str) -> str:
    return "\n  ".join(line.rstrip() for line in form.splitlines())
