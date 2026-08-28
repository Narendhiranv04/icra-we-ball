from __future__ import annotations

from types import SimpleNamespace

from mujoco_scenes.baseline_kitchen_runtime import (
    KitchenEffectLedger,
    KitchenGoalContract,
)

from vlm_tamp_baseline.kitchen_planning_runtime import (
    KitchenPlanningState,
    canonical_kitchen_actions,
    compare_kitchen_actions,
    normalize_kitchen_actions,
)
from vlm_tamp_baseline.planner import VLMTAMPPlannerConfig
from vlm_tamp_baseline.run_kitchen import _method_manifest, build_parser


def test_kitchen_planning_cli_exposes_private_gt_condition() -> None:
    arguments = build_parser().parse_args(
        [
            "--goal", "goal", "--output-dir", "run",
            "--planning-only", "--variant", "K1",
        ]
    )
    assert arguments.planning_only
    assert arguments.variant == "K1"
    assert arguments.max_model_calls is None
    assert arguments.seed == 0


def test_method_manifest_is_available_before_planning_result() -> None:
    arguments = build_parser().parse_args(
        [
            "--goal", "goal", "--output-dir", "run", "--planning-only",
            "--variant", "K1", "--camera-count", "3",
        ]
    )
    manifest = _method_manifest(
        arguments,
        VLMTAMPPlannerConfig(
            base_url="http://localhost/v1",
            model="test-model",
            enable_thinking=False,
        ),
        max_model_calls=1,
        inventory_object_count=9,
    )
    assert manifest["planning_rounds"] == 1
    assert manifest["raw_vlm_requests_per_round"] == 2
    assert manifest["camera_count"] == 3


def test_canonical_kitchen_actions_translates_ids_after_planning() -> None:
    history = (
        {
            "action": {"skill": "INSPECT", "arguments": {"region_id": "C1"}},
            "success": True,
        },
        {
            "action": {"skill": "PICK", "arguments": {"object_id": "object_1"}},
            "success": True,
        },
        {
            "action": {
                "skill": "PLACE",
                "arguments": {"object_id": "object_1", "region_id": "countertop"},
            },
            "success": True,
        },
    )
    assert canonical_kitchen_actions(history, {"object_1": "spoon_body"}) == [
        {"operator": "INSPECT", "arguments": ["C1"]},
        {"operator": "PICK", "arguments": ["spoon_body"]},
        {"operator": "PLACE", "arguments": ["spoon_body", "countertop"]},
    ]


def test_task_level_normalization_removes_execution_only_vocabulary() -> None:
    expected = [
        {"operator": "OPEN", "arguments": ["D1"]},
        {"operator": "CLOSE", "arguments": ["D1"]},
        {"operator": "PICK", "arguments": ["spoon"]},
        {"operator": "PLACE_SERVING_UTENSIL", "arguments": ["spoon", "bowl"]},
    ]
    assert normalize_kitchen_actions(expected) == [
        {"operator": "INSPECT", "arguments": ["D1"]},
        {"operator": "PICK", "arguments": ["spoon"]},
        {"operator": "PLACE", "arguments": ["spoon", "bowl"]},
    ]
    comparison = compare_kitchen_actions(
        normalize_kitchen_actions(expected), expected
    )
    assert comparison["shared_task_vocabulary"]["exact_sequence_match"]


def test_private_goal_verifier_checks_task_relations_not_gt_assignment() -> None:
    labels = {
        "cup": "cup", "mug": "mug", "bowl_1": "bowl", "bowl_2": "bowl",
        "water": "kettle", "coffee": "coffee_source",
        "stirrer": "spoon", "soup_tool_1": "spoon", "soup_tool_2": "spoon",
    }
    resolution = {
        "accepted": [
            {"generic_object_id": object_id, "semantic_label": label}
            for object_id, label in labels.items()
        ]
    }
    ledger = KitchenEffectLedger(KitchenGoalContract(("unused",), (), (), labels))
    runtime = SimpleNamespace(
        bundle=SimpleNamespace(resolution=resolution),
        ledger=ledger,
    )
    state = KitchenPlanningState(runtime)
    ledger.accept(
        (
            "poured(water,cup)", "poured(coffee,cup)", "stirred(stirrer,cup)",
            "placed(cup,serving_area)", "poured(water,mug)",
            "poured(coffee,mug)", "stirred(stirrer,mug)",
            "placed(mug,serving_area)", "placed(bowl_1,serving_area)",
            "placed(bowl_2,serving_area)", "placed(soup_tool_1,bowl_1)",
            "placed(soup_tool_2,bowl_2)",
        )
    )
    assert state.goal_verifier()
