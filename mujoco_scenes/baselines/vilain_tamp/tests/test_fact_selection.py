from __future__ import annotations

import pytest

from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.fact_selection import (
    FactSelectionError,
    compile_problem,
    parse_fact_selection,
)
from mujoco_scenes.baselines.vilain_tamp.pddl import validate_problem
from mujoco_scenes.baselines.vilain_tamp.symbolic_contract import (
    build_variant_action_contract,
    enumerate_grounded_facts,
)


def _universe(domain_key: str, objects: dict[str, str]):
    domain = load_domain(domain_key)
    contract = build_variant_action_contract(domain, "fixture")
    return domain, enumerate_grounded_facts(objects, contract)


@pytest.mark.parametrize(
    "domain_key, objects",
    [
        ("kitchen", {"mug": "vessel", "counter": "surface", "coffee": "content"}),
        ("living_room", {"cup": "cup", "table": "support"}),
        ("workshop", {"driver": "driver", "screw": "fastener", "hole": "target"}),
    ],
)
def test_grounded_universe_is_deterministic_typed_and_variable_free(
    domain_key: str, objects: dict[str, str]
) -> None:
    _, first = _universe(domain_key, objects)
    _, second = _universe(domain_key, dict(reversed(tuple(objects.items()))))
    assert first == second
    assert [fact.fact_id for fact in first] == [
        f"f{index:04d}" for index in range(1, len(first) + 1)
    ]
    assert all("?" not in fact.literal for fact in first)
    assert all(len(fact.arguments) == len(set(fact.arguments)) for fact in first)


def test_subtypes_are_grounded_but_incompatible_arguments_are_absent() -> None:
    _, facts = _universe(
        "kitchen", {"mug": "vessel", "counter": "surface", "spoon": "utensil"}
    )
    literals = {fact.literal for fact in facts}
    assert "(at mug counter)" in literals
    assert "(at spoon counter)" in literals
    assert "(at counter counter)" not in literals
    assert "(stirred spoon)" not in literals


def test_fact_selection_accepts_only_known_json_ids() -> None:
    _, facts = _universe("living_room", {"cup": "cup", "table": "support"})
    selected = parse_fact_selection(
        f'{{"true_fact_ids":["{facts[0].fact_id}"]}}',
        field="true_fact_ids",
        candidates=facts,
    )
    assert selected == (facts[0],)
    with pytest.raises(FactSelectionError, match="UNKNOWN_FACT_ID"):
        parse_fact_selection(
            '{"true_fact_ids":["unknown"]}',
            field="true_fact_ids",
            candidates=facts,
        )
    with pytest.raises(FactSelectionError, match="RAW_PDDL_NOT_ALLOWED"):
        parse_fact_selection(
            "(:init (handempty))", field="true_fact_ids", candidates=facts
        )


def test_deterministic_compiler_emits_positive_valid_problem() -> None:
    objects = {"mug": "vessel", "counter": "surface", "coffee": "content"}
    domain, facts = _universe("kitchen", objects)
    by_literal = {fact.literal: fact for fact in facts}
    problem, objects_form, init_form, goal_form = compile_problem(
        domain=domain,
        problem_name="selection-fixture",
        object_types=objects,
        initial_facts=(
            by_literal["(handempty)"],
            by_literal["(accessible counter)"],
            by_literal["(at mug counter)"],
        ),
        goal_facts=(by_literal["(contains mug coffee)"],),
    )
    assert "(not " not in init_form
    assert objects_form == "(:objects\n  coffee - content\n  counter - surface\n  mug - vessel\n)"
    assert goal_form == "(:goal (contains mug coffee))"
    assert validate_problem(problem, domain).valid


def test_empty_goal_and_duplicate_selection_fail_closed() -> None:
    domain, facts = _universe("living_room", {"cup": "cup", "table": "support"})
    with pytest.raises(ValueError, match="must not be empty"):
        compile_problem(
            domain=domain,
            problem_name="bad",
            object_types={"cup": "cup", "table": "support"},
            initial_facts=(),
            goal_facts=(),
        )
    with pytest.raises(FactSelectionError, match="DUPLICATE_FACT_ID"):
        parse_fact_selection(
            f'{{"goal_fact_ids":["{facts[0].fact_id}","{facts[0].fact_id}"]}}',
            field="goal_fact_ids",
            candidates=facts,
        )
