"""Shared access is a claim about every seat, and is verified as one.

A support deliberately reachable from both seats is the one requirement in the
living room that cannot be checked one seat at a time.  If reaching a single
seat could satisfy it, a two-person task would be reported done against one
person's position -- the same failure as anchoring both personal placements to
one seat, arriving through the physical verifier instead of the compiler.

These tests exercise the geometric verifier directly, with synthetic seats and
distances.  No scene, reference graph or benchmark variant appears here.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.region_ablation2 import (
    _tri_and,
    evaluate_control_accessibility,
)


def _seat(x: float) -> dict:
    return {"centroid_world_m": [x, 0.0, 0.5],
            "semantic_role": {"status": "TRUE"},
            "provenance": {"evidence_path": "synthetic"}}


def _region(x: float) -> dict:
    return {"centroid_world_m": [x, 0.0, 0.5]}


REACH = 1.0


def test_both_seats_within_reach_is_accessible():
    verdict = evaluate_control_accessibility(
        _region(0.0), [_seat(-0.5), _seat(0.5)], maximum_distance_m=REACH)
    assert verdict["status"] == "TRUE"
    assert len(verdict["seat_relations"]) == 2, "one check per seat, always"


def test_one_seat_within_reach_is_not_accessible_from_both():
    """The regression this guards: a one-seat success passing as a two-seat one."""
    verdict = evaluate_control_accessibility(
        _region(0.0), [_seat(-0.2), _seat(5.0)], maximum_distance_m=REACH)
    assert verdict["status"] == "FALSE"
    statuses = [row["status"] for row in verdict["seat_relations"]]
    assert sorted(statuses) == ["FALSE", "TRUE"], "the near seat alone must not carry it"


def test_every_seat_is_checked_however_many_there_are():
    seats = [_seat(-0.4), _seat(0.4), _seat(0.0)]
    verdict = evaluate_control_accessibility(_region(0.0), seats, maximum_distance_m=REACH)
    assert len(verdict["seat_relations"]) == len(seats)
    assert verdict["status"] == "TRUE"
    far = evaluate_control_accessibility(
        _region(0.0), seats + [_seat(9.0)], maximum_distance_m=REACH)
    assert far["status"] == "FALSE", "adding an unreachable seat must break it"


def test_the_reported_margin_is_the_worst_seat_not_the_best():
    verdict = evaluate_control_accessibility(
        _region(0.0), [_seat(-0.1), _seat(0.9)], maximum_distance_m=REACH)
    margins = [row["signed_margin_m"] for row in verdict["seat_relations"]]
    assert verdict["signed_margin_m"] == min(margins)


def test_an_unknown_seat_is_not_an_accessible_one():
    """UNKNOWN must never be read as TRUE by the conjunction."""
    assert _tri_and("TRUE", "UNKNOWN") == "UNKNOWN"
    assert _tri_and("TRUE", "TRUE") == "TRUE"
    assert _tri_and("TRUE", "FALSE") == "FALSE"
    assert _tri_and("UNKNOWN", "FALSE") == "FALSE"


def test_the_method_says_what_it_checked():
    verdict = evaluate_control_accessibility(
        _region(0.0), [_seat(-0.5), _seat(0.5)], maximum_distance_m=REACH)
    assert verdict["method"] == "all_observed_seats_accessible_v1"
