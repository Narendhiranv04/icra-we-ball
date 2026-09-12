"""The plan-fidelity comparison must not report a difference that is not one.

Two defects in the comparison made correct plans look wrong:

  * the serving destination is called `serving_area` by the ground-truth
    executor, the oracle world state and the expected-action catalogue, and
    `dining_table` by the compiled pipeline and by the goal evaluator, which
    checks `at(cup, dining_table)`.  One place, two names, and the disagreement
    is between the benchmark's own artifacts.
  * the unordered check matched greedily.  A one-argument step like PICK matches
    many candidates, so committing to the first could bind a body to the wrong
    instance and dead-end on a later multi-argument step.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "_cmp", REPO / "scripts" / "compare_plans_to_gt_actions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _a(op, *args):
    return {"operator": op, "arguments": list(args)}


def test_the_serving_destination_alias_is_applied():
    m = _tool()
    assert m.DESTINATION_ALIASES == {"dining_table": "serving_area"}
    assert m.arguments(_a("PLACE", "object_0001", "dining_table")) == [
        "object_0001", "serving_area"]


def test_unordered_matching_backtracks_out_of_a_greedy_dead_end():
    """PICK binds first, and a greedy matcher then strands the two-argument step."""
    m = _tool()
    expected = [_a("PICK", "kettle"), _a("PICK", "jar"),
                _a("POUR", "jar", "cup")]
    produced = [_a("PICK", "object_0001"), _a("PICK", "object_0002"),
                _a("POUR", "object_0002", "object_0003")]
    ok, detail = m.compare_unordered(expected, produced)
    assert ok, detail


def test_one_instance_may_not_stand_for_two_bodies():
    m = _tool()
    expected = [_a("PICK", "kettle"), _a("PICK", "jar")]
    produced = [_a("PICK", "object_0001"), _a("PICK", "object_0001")]
    ok, _ = m.compare_unordered(expected, produced)
    assert not ok


def test_a_genuinely_different_action_still_differs():
    m = _tool()
    expected = [_a("POUR", "jar", "cup")]
    produced = [_a("STIR", "object_0001", "object_0002")]
    ok, _ = m.compare_unordered(expected, produced)
    assert not ok


def test_a_missing_step_still_differs():
    m = _tool()
    expected = [_a("PICK", "kettle"), _a("POUR", "kettle", "cup")]
    produced = [_a("PICK", "object_0001")]
    ok, _ = m.compare_unordered(expected, produced)
    assert not ok
