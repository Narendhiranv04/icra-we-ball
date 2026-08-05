from __future__ import annotations

import inspect

import mujoco
import pytest

from mujoco_scenes.living_room_region_scene import (
    L2_ABLATION3_SCENES,
    L2LivingRoomRegionScene,
    build_l2_region_xml,
)
from mujoco_scenes.region_ablation3 import (
    DEFAULT_TASK_CONFIG,
    TargetSpecificMatcher,
    hall_diagnostics,
    load_ablation3_task,
)


TASK = load_ablation3_task(DEFAULT_TASK_CONFIG)


def _row(region, target, rank, status="TRUE", distance=0.7):
    return {
        "region_id": region,
        "seating_target_id": target,
        "candidate_rank": rank,
        "compatibility_status": status,
        "general_suitability_status": "TRUE",
        "near_seat_margin_m": 1.15 - distance,
        "fit_margin_m": 0.25,
    }


def _matcher(rows, targets=("target_a", "target_b")):
    return TargetSpecificMatcher(
        rows,
        target_ids=list(targets),
        required_target_count=len(targets),
    )


@pytest.mark.parametrize("scene_name", L2_ABLATION3_SCENES)
def test_ablation3_scene_compiles_without_robot(scene_name):
    model = mujoco.MjModel.from_xml_string(
        build_l2_region_xml(scene_name, "none")
    )
    assert model.ncam == 5
    assert sum(
        model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
        for index in range(model.njnt)
    ) == 2


@pytest.mark.parametrize("scene_name", L2_ABLATION3_SCENES)
def test_ablation3_scene_composes_with_google_robot(scene_name):
    scene = L2LivingRoomRegionScene(scene_name, robot="google")
    assert scene.has_robot
    assert scene.model.ncam >= 5


def test_ablation3_legacy_api_does_not_leak_payload_names():
    scene = L2LivingRoomRegionScene(L2_ABLATION3_SCENES[0], robot="none")
    assert scene.get_visible_object_instances() == []


def test_task_derives_two_targets_from_manual_function_contract():
    group = TASK["function_groups"]["personal_drinks"]
    assert group["required_target_count"] == 2
    assert group["usage_policy"] == "DEDICATED_REGION_PER_TARGET"
    assert group["target_assignment_policy"] == "TARGET_SPECIFIC"


def test_primary_count_false_positive_and_global_cardinality_one():
    rows = [
        _row("region_1", "target_a", 1),
        _row("region_1", "target_b", 1, "FALSE", 2.3),
        _row("region_2", "target_a", 2),
        _row("region_2", "target_b", 2, "FALSE", 1.8),
    ]
    matcher = _matcher(rows)
    count = matcher.count_result()
    global_result = matcher.global_result()
    assert count["status"] == "COMPLETE"
    assert count["decision_used_near_seat"] is False
    assert count["counted_region_count"] == 2
    assert count["maximum_matching_cardinality"] == 1
    assert global_result["status"] == "EXHAUSTED"
    assert global_result["maximum_matching_cardinality"] == 1
    assert len(global_result["uncovered_target_ids"]) == 1


def test_matching_trap_greedy_fails_and_global_reassigns():
    rows = [
        _row("flexible", "target_a", 1),
        _row("flexible", "target_b", 1),
        _row("a_only", "target_a", 2),
        _row("a_only", "target_b", 2, "FALSE", 2.0),
    ]
    matcher = _matcher(rows)
    greedy = matcher.greedy_result()
    global_result = matcher.global_result()
    assert greedy["status"] == "EXHAUSTED"
    assert greedy["region_target_assignments"] == [
        {
            "region_id": "flexible",
            "seating_target_id": "target_a",
            "candidate_rank": 1,
            "near_seat_margin_m": pytest.approx(0.45),
            "fit_margin_m": 0.25,
        }
    ]
    assert global_result["status"] == "COMPLETE"
    assert {
        (item["region_id"], item["seating_target_id"])
        for item in global_result["region_target_assignments"]
    } == {("a_only", "target_a"), ("flexible", "target_b")}


def test_balanced_scene_completes_greedy_and_global():
    rows = [
        _row("left", "target_a", 1),
        _row("left", "target_b", 1, "FALSE", 2.0),
        _row("right", "target_a", 2, "FALSE", 2.0),
        _row("right", "target_b", 2),
    ]
    matcher = _matcher(rows)
    assert matcher.greedy_result()["status"] == "COMPLETE"
    assert matcher.global_result()["status"] == "COMPLETE"


def test_unknown_compatibility_never_forms_an_edge():
    matcher = _matcher(
        [
            _row("region_1", "target_a", 1, "UNKNOWN"),
            _row("region_2", "target_b", 2),
        ]
    )
    assert matcher.global_result()["maximum_matching_cardinality"] == 1


def test_matching_is_general_for_three_targets():
    targets = ("t1", "t2", "t3")
    rows = [
        _row("r1", "t1", 2),
        _row("r2", "t1", 1),
        _row("r2", "t2", 1),
        _row("r3", "t2", 3),
        _row("r3", "t3", 3),
    ]
    result = _matcher(rows, targets).global_result()
    assert result["status"] == "COMPLETE"
    assert result["maximum_matching_cardinality"] == 3
    assert len(set(result["selected_region_ids"])) == 3


def test_tie_break_uses_rank_before_margin_and_ids():
    rows = [
        _row("rank_one", "target_a", 1, distance=1.0),
        _row("rank_two", "target_a", 2, distance=0.2),
        _row("other", "target_b", 3),
    ]
    result = _matcher(rows).global_result()
    selected = {
        item["seating_target_id"]: item["region_id"]
        for item in result["region_target_assignments"]
    }
    assert selected["target_a"] == "rank_one"


def test_hall_diagnostic_exposes_target_coverage_deficit():
    diagnostics = hall_diagnostics(
        ["a", "b"], {"a": {"r1", "r2"}, "b": set()}
    )
    joint = next(item for item in diagnostics if len(item["target_subset"]) == 2)
    assert joint["coverage_deficit"] == 0
    missing = next(item for item in diagnostics if item["target_subset"] == ["b"])
    assert missing["coverage_deficit"] == 1
    assert missing["hall_condition_satisfied"] is False


def test_matcher_has_no_scene_or_persistent_id_literals():
    source = inspect.getsource(TargetSpecificMatcher)
    assert "L2_living_room" not in source
    assert "region_000" not in source
    assert "seating_000" not in source
