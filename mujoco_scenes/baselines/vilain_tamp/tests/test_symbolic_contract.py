from __future__ import annotations

from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.symbolic_contract import (
    SymbolicContractViolation,
    build_variant_action_contract,
    enumerate_grounded_facts,
    initial_state_diagnostics,
    load_variant_action_contract,
    missing_manipulable_types,
    object_type_table,
    relaxed_goal_diagnostics,
    validate_goal_fragment,
    validate_initial_fragment,
)


CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _kitchen():
    return build_variant_action_contract(load_domain("kitchen"), "fixture")


def _code(error: pytest.ExceptionInfo[SymbolicContractViolation]) -> str:
    return error.value.errors[0].code


def test_contract_extracts_fixed_signatures_actions_and_controller_capabilities() -> None:
    contract = _kitchen()

    assert contract.predicate_signatures["at"] == ("movable", "location")
    assert contract.operators["pick-from"].parameter_types == ("movable", "location")
    assert contract.operators["pick-from"].parameter_names == ("?object", "?location")
    assert "(accessible ?location)" in contract.operators["pick-from"].precondition
    assert contract.operators["pick-from"].controller_primitive == "PICK"
    assert contract.operators["pour"].controller_primitive == "POUR"


def test_object_type_table_groups_exact_ids_without_changing_types() -> None:
    contract = build_variant_action_contract(
        load_domain("kitchen"),
        "fixture",
        structural_inventory={"countertop": "surface"},
    )
    table = object_type_table({"mug_1": "vessel", "spoon_1": "utensil"}, contract)
    assert dict(table) == {
        "vessel": ("mug_1",),
        "utensil": ("spoon_1",),
        "surface": ("countertop",),
    }


@pytest.mark.parametrize(
    "fragment, expected_code",
    [
        ("(:init (not (open drawer)))", "NEGATIVE_INIT_LITERAL_NOT_ALLOWED"),
        ("(:init (invented mug))", "UNKNOWN_PREDICATE"),
        ("(:init (at mug))", "ARITY_MISMATCH"),
        ("(:init (at counter counter))", "INVALID_ARGUMENT_TYPE"),
        ("(:init (at missing counter))", "UNKNOWN_OBJECT"),
    ],
)
def test_initial_literals_fail_closed_with_structured_codes(
    fragment: str, expected_code: str
) -> None:
    contract = _kitchen()
    objects = {"mug": "vessel", "counter": "surface", "drawer": "storage"}
    with pytest.raises(SymbolicContractViolation) as raised:
        validate_initial_fragment(fragment, objects, contract)
    assert _code(raised) == expected_code


def test_positive_initial_and_typed_goal_are_accepted() -> None:
    contract = _kitchen()
    objects = {"mug": "vessel", "counter": "surface", "coffee": "content"}
    validate_initial_fragment(
        "(:init (handempty) (accessible counter) (at mug counter))",
        objects,
        contract,
    )
    validate_goal_fragment("(:goal (contains mug coffee))", objects, contract)


def test_goal_unknown_object_and_argument_type_are_rejected() -> None:
    contract = _kitchen()
    with pytest.raises(SymbolicContractViolation) as missing:
        validate_goal_fragment("(:goal (stirred missing))", {}, contract)
    assert _code(missing) == "UNKNOWN_OBJECT"
    with pytest.raises(SymbolicContractViolation) as wrong_type:
        validate_goal_fragment(
            "(:goal (stirred spoon))", {"spoon": "utensil"}, contract
        )
    assert _code(wrong_type) == "INVALID_ARGUMENT_TYPE"


@pytest.mark.parametrize(
    "domain_key, variant, expected_operator",
    [
        ("kitchen", "K1", "pour"),
        ("living_room", "L1", "place-on"),
        ("workshop", "W1", "drive"),
    ],
)
def test_domain_and_variant_contract_selection(
    domain_key: str, variant: str, expected_operator: str
) -> None:
    contract = load_variant_action_contract(
        load_domain(domain_key), variant, config_root=CONFIG_ROOT
    )
    assert expected_operator in contract.operators
    assert contract.structural_inventory


def test_physical_pick_variants_keep_one_symbolic_operator() -> None:
    first = load_variant_action_contract(
        load_domain("kitchen"), "K1", config_root=CONFIG_ROOT
    )
    drawer = load_variant_action_contract(
        load_domain("kitchen"), "K5", config_root=CONFIG_ROOT
    )
    assert tuple(first.operators) == tuple(drawer.operators)
    assert "pick-from" in first.operators
    assert "open-storage" in first.operators


def test_contract_exposes_no_solution_sequence_or_geometry() -> None:
    payload = load_variant_action_contract(
        load_domain("workshop"), "W1", config_root=CONFIG_ROOT
    ).to_dict()
    rendered_keys = repr(payload).lower()
    for forbidden in (
        "expected_solution",
        "expected_plan",
        "action_sequence",
        "trajectory",
        "grasp_pose",
        "ik_solution",
    ):
        assert forbidden not in rendered_keys
    assert payload["geometry_owned_by_refinement"] is True


def test_storage_accessibility_is_symbolic_not_variant_action_naming() -> None:
    contract = load_variant_action_contract(
        load_domain("workshop"), "W1", config_root=CONFIG_ROOT
    )
    assert "(accessible ?location)" in contract.operators["pick-from"].precondition
    assert "(accessible ?storage)" in contract.operators["open-storage"].effect
    assert all("variant" not in name for name in contract.operators)


def test_missing_capability_types_reports_type_without_inventing_entity() -> None:
    contract = _kitchen()
    missing = missing_manipulable_types(
        {"mug": "vessel", "spoon": "utensil"}, contract
    )
    assert missing == ("source",)


def test_relaxed_reachability_explains_unachievable_goal() -> None:
    contract = _kitchen()
    objects = {
        "mug": "vessel",
        "coffee": "content",
        "counter": "surface",
    }
    facts = enumerate_grounded_facts(objects, contract)
    by_literal = {fact.literal: fact for fact in facts}
    diagnostics = relaxed_goal_diagnostics(
        (by_literal["(handempty)"], by_literal["(accessible counter)"]),
        (by_literal["(contains mug coffee)"],),
        objects,
        contract,
    )
    assert diagnostics[0]["code"] == "RELAXED_GOAL_UNREACHABLE"
    assert diagnostics[0]["potential_ground_achievers"] == []


def test_relaxed_reachability_propagates_through_pick_and_pour() -> None:
    contract = _kitchen()
    objects = {
        "mug": "vessel",
        "pot": "source",
        "coffee": "content",
        "counter": "surface",
    }
    facts = enumerate_grounded_facts(objects, contract)
    by_literal = {fact.literal: fact for fact in facts}
    initial = tuple(
        by_literal[literal]
        for literal in (
            "(handempty)",
            "(accessible counter)",
            "(at pot counter)",
            "(can-dispense pot coffee)",
        )
    )
    assert not relaxed_goal_diagnostics(
        initial,
        (by_literal["(contains mug coffee)"],),
        objects,
        contract,
    )


def test_initial_state_diagnostics_rejects_hand_and_location_contradictions() -> None:
    contract = _kitchen()
    objects = {"mug": "vessel", "counter": "surface", "shelf": "surface"}
    facts = enumerate_grounded_facts(objects, contract)
    by_literal = {fact.literal: fact for fact in facts}
    diagnostics = initial_state_diagnostics(
        tuple(
            by_literal[literal]
            for literal in (
                "(handempty)",
                "(holding mug)",
                "(at mug counter)",
                "(at mug shelf)",
            )
        )
    )
    assert {item["code"] for item in diagnostics} == {
        "HAND_STATE_CONTRADICTION",
        "MULTIPLE_OBJECT_LOCATIONS",
        "HELD_AND_AT_LOCATION",
    }
