from dataclasses import replace

import pytest

from mujoco_scenes.symbolic_planning_core import (
    NoSymbolicPlan,
    SymbolicAction,
    SymbolicProblem,
    apply_action,
    deterministic_astar,
    independent_replay,
    is_applicable,
)


def _problem():
    pick = SymbolicAction(
        "pick", ("object_1",),
        frozenset({("available", "object_1"), ("hand_empty",)}),
        frozenset(), frozenset({("holding", "object_1")}),
        frozenset({("available", "object_1"), ("hand_empty",)}),
    )
    place = SymbolicAction(
        "place", ("object_1", "region_1"),
        frozenset({("holding", "object_1")}), frozenset(),
        frozenset({("on", "object_1", "region_1"), ("hand_empty",)}),
        frozenset({("holding", "object_1")}),
    )
    problem = SymbolicProblem(
        frozenset({("available", "object_1"), ("hand_empty",)}),
        frozenset({("on", "object_1", "region_1")}),
        (pick, place),
    )
    return problem, pick, place


def test_pick_requires_available_and_hand_empty():
    problem, pick, _ = _problem()
    assert is_applicable(problem.initial_atoms, pick)
    assert not is_applicable(frozenset({("available", "object_1")}), pick)
    assert not is_applicable(frozenset({("hand_empty",)}), pick)


def test_cannot_pick_second_while_holding_and_place_restores_hand():
    problem, pick, place = _problem()
    held = apply_action(problem.initial_atoms, pick)
    assert not is_applicable(held, pick)
    final = apply_action(held, place)
    assert ("on", "object_1", "region_1") in final
    assert ("hand_empty",) in final
    assert ("holding", "object_1") not in final


def test_place_requires_holding_specified_object():
    problem, _, place = _problem()
    assert not is_applicable(problem.initial_atoms, place)


def test_deterministic_astar_and_successor_order():
    problem, _, _ = _problem()
    first = deterministic_astar(problem)
    second = deterministic_astar(problem)
    assert [action.render() for action in first.plan] == [action.render() for action in second.plan]
    assert first.statistics["plan_cost"] == 2
    assert independent_replay(problem, first.plan)["goal_status"] == "GOAL_SATISFIED"


def test_impossible_problem_has_no_plan():
    problem, _, _ = _problem()
    with pytest.raises(NoSymbolicPlan):
        deterministic_astar(replace(problem, actions=tuple()))


def test_independent_replay_rejects_malformed_arity_and_corruption(monkeypatch):
    problem, pick, _ = _problem()
    malformed = replace(pick, arguments=("object_1", "extra"))
    result = independent_replay(problem, [malformed])
    assert result["status"] == "INVALID"
    plan = deterministic_astar(problem).plan
    monkeypatch.setattr(
        "mujoco_scenes.symbolic_planning_core.apply_action",
        lambda *_: (_ for _ in ()).throw(AssertionError("planner apply called")),
    )
    assert independent_replay(problem, plan)["status"] == "VALID"
