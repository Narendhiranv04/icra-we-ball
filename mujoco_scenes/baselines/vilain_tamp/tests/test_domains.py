from __future__ import annotations

from dataclasses import replace
import re

import pytest

from mujoco_scenes.baselines.vilain_tamp.domains import available_domains, load_domain
from mujoco_scenes.baselines.vilain_tamp.pddl import (
    domain_definition_diagnostics,
    domain_is_unchanged,
    parse_domain_schema,
    validate_problem,
)


VALID_PROBLEMS = {
    "kitchen": """
        (define (problem kitchen-synthetic)
          (:domain vilain-kitchen)
          (:objects
            mug_1 - vessel
            source_1 - source
            utensil_1 - utensil
            coffee - content
            counter - surface
          )
          (:init
            (handempty)
            (accessible counter)
            (at source_1 counter)
            (can-dispense source_1 coffee)
            (can-stir utensil_1 mug_1)
          )
          (:goal (and (contains mug_1 coffee) (stirred mug_1)))
        )
    """,
    "living_room": """
        (define (problem living-room-synthetic)
          (:domain vilain-living-room)
          (:objects
            cup_1 - cup
            table_1 - support
            staging - location
            seat_1 - seat
          )
          (:init
            (handempty)
            (present staging)
            (accessible staging)
            (present table_1)
            (accessible table_1)
            (personal-to table_1 seat_1)
            (at cup_1 staging)
          )
          (:goal (supports table_1 cup_1))
        )
    """,
    "workshop": """
        (define (problem workshop-synthetic)
          (:domain vilain-workshop)
          (:objects
            driver_1 - driver
            fastener_1 - fastener
            drawer_1 - storage
            bench - surface
            repair_target - target
          )
          (:init
            (handempty)
            (accessible drawer_1)
            (accessible bench)
            (at driver_1 drawer_1)
            (at fastener_1 drawer_1)
            (driver-compatible driver_1 fastener_1)
            (fits fastener_1 repair_target)
            (can-reach driver_1 repair_target)
          )
          (:goal (fastened fastener_1 repair_target))
        )
    """,
}


@pytest.mark.parametrize("domain_key", available_domains())
def test_fixed_domain_matches_declared_knowledge(domain_key: str) -> None:
    definition = load_domain(domain_key)
    schema = parse_domain_schema(definition.text)
    assert schema.name == definition.name
    assert schema.type_hierarchy == dict(definition.type_hierarchy)
    assert schema.predicate_signatures == dict(definition.predicate_signatures)
    assert schema.action_signatures == dict(definition.action_signatures)
    assert domain_definition_diagnostics(definition) == ()
    assert definition.descriptions["fixed_knowledge"]


@pytest.mark.parametrize("domain_key", available_domains())
def test_synthetic_problem_is_structurally_valid(domain_key: str) -> None:
    definition = load_domain(domain_key)
    result = validate_problem(
        VALID_PROBLEMS[domain_key],
        definition,
        expected_domain_sha256=definition.sha256,
    )
    assert result.valid, result.diagnostics
    assert result.diagnostics == ()


def test_problem_rejects_unknown_predicate() -> None:
    definition = load_domain("kitchen")
    invalid = VALID_PROBLEMS["kitchen"].replace(
        "(contains mug_1 coffee)", "(invented-predicate missing_object)"
    )
    result = validate_problem(invalid, definition)
    assert not result.valid
    assert "unknown predicate" in " ".join(result.diagnostics)


def test_problem_rejects_undeclared_object() -> None:
    definition = load_domain("kitchen")
    invalid = VALID_PROBLEMS["kitchen"].replace(
        "(contains mug_1 coffee)", "(contains missing_vessel coffee)"
    )
    result = validate_problem(invalid, definition)
    assert not result.valid
    assert "uses undeclared object 'missing_vessel'" in " ".join(result.diagnostics)


def test_problem_rejects_wrong_object_type() -> None:
    definition = load_domain("workshop")
    invalid = VALID_PROBLEMS["workshop"].replace(
        "(fastened fastener_1 repair_target)",
        "(fastened driver_1 repair_target)",
    )
    result = validate_problem(invalid, definition)
    assert not result.valid
    assert "is not a 'fastener'" in " ".join(result.diagnostics)


@pytest.mark.parametrize(
    "replacement, expected_diagnostic",
    [
        ("(:domain wrong-domain)", "problem domain must be"),
        ("(:goal (and))", "goal must contain at least one atom"),
    ],
)
def test_problem_rejects_wrong_domain_and_empty_goal(
    replacement: str, expected_diagnostic: str
) -> None:
    definition = load_domain("living_room")
    original = "(:domain vilain-living-room)" if replacement.startswith("(:domain") else "(:goal (supports table_1 cup_1))"
    invalid = VALID_PROBLEMS["living_room"].replace(original, replacement)
    result = validate_problem(invalid, definition)
    assert not result.valid
    assert expected_diagnostic in " ".join(result.diagnostics)


def test_domain_hash_detects_mutation() -> None:
    definition = load_domain("kitchen")
    changed_text = definition.text + "; changed\n"
    assert domain_is_unchanged(definition.sha256, definition.text)
    assert not domain_is_unchanged(definition.sha256, changed_text)
    changed = replace(definition, text=changed_text)
    result = validate_problem(
        VALID_PROBLEMS["kitchen"],
        changed,
        expected_domain_sha256=definition.sha256,
    )
    assert not result.valid
    assert "domain text does not match" in " ".join(result.diagnostics)


def test_fixed_files_contain_no_variant_answers() -> None:
    forbidden_exact = {
        "ab3_medium_deep_mug",
        "a2_drink_left",
        "workshop_long_phillips_driver",
        "expected_solution",
        "canonical_assignment",
        "intended_outcome",
    }
    for domain_key in available_domains():
        definition = load_domain(domain_key)
        combined = (
            definition.text
            + definition.knowledge_path.read_text(encoding="utf-8")
        ).lower()
        assert not any(identifier in combined for identifier in forbidden_exact)
        assert re.search(r"\b[fi]\d+[_-]", combined, re.IGNORECASE) is None
