from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import BenchmarkGoalEvaluation
from mujoco_scenes.baselines.vilain_tamp.corrective_planning import (
    CorrectivePlanningLoop,
)
from mujoco_scenes.baselines.vilain_tamp.evaluation import (
    EvaluationContractError,
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    evaluate_hidden_benchmark,
)
from mujoco_scenes.baselines.vilain_tamp.execution.kitchen import (
    KitchenEffectLedgerEntry,
)
from mujoco_scenes.baselines.vilain_tamp.interpreter import ViLaInInterpreter
from mujoco_scenes.baselines.vilain_tamp.planner import FastDownwardPlanner


def _on(support: str) -> dict[str, object]:
    return {
        "present": True,
        "support": support,
        "released": True,
        "stable": True,
        "inside_support_footprint": True,
        "support_contact": True,
        "floor_contact": False,
        "invalid_penetration": False,
    }


def _kitchen_context(*, feasible: bool = True) -> HiddenBenchmarkContext:
    return HiddenBenchmarkContext(
        domain="kitchen",
        variant="F0_ALL_VISIBLE" if feasible else "I4_MISSING_KETTLE",
        ground_truth_feasibility=feasible,
        requirements={
            "coffee_vessels": ["coffee_left", "coffee_right"],
            "soup_vessels": ["soup_left", "soup_right"],
            "water_sources": ["kettle"],
            "coffee_sources": ["coffee_jar"],
            "suitable_stirrers": ["stirrer"],
            "suitable_soup_utensils": ["spoon_left", "spoon_right"],
            "serving_support": "serving_area",
            "water_content": "water",
            "coffee_content": "coffee",
        },
        evidence_artifacts=("hidden/kitchen_variant.yaml",),
    )


def _kitchen_state(*, predicted_infeasible: bool = False) -> TerminalStateSnapshot:
    objects = {
        vessel: _on("serving_area")
        for vessel in ("coffee_left", "coffee_right", "soup_left", "soup_right")
    }
    objects.update(
        {
            "spoon_left": {"present": True, "contained_stably": True},
            "spoon_right": {"present": True, "contained_stably": True},
        }
    )
    return TerminalStateSnapshot(
        domain="kitchen",
        predicted_infeasible=predicted_infeasible,
        objects=objects,
        relations={
            "contained_in": {
                "soup_left": ["spoon_left"],
                "soup_right": ["spoon_right"],
            }
        },
    )


def _kitchen_ledger():
    rows = []
    index = 0
    for vessel in ("coffee_left", "coffee_right"):
        for source, content in (("kettle", "water"), ("coffee_jar", "coffee")):
            rows.append(
                KitchenEffectLedgerEntry(
                    action_index=index,
                    action_instance_id=f"action_{index}",
                    effect="POUR_COMPLETED",
                    symbolic_arguments=(source, vessel, content),
                    resolved_entities=(source, vessel),
                    controller_status="SUCCESS",
                )
            )
            index += 1
        rows.append(
            KitchenEffectLedgerEntry(
                action_index=index,
                action_instance_id=f"action_{index}",
                effect="STIR_COMPLETED",
                symbolic_arguments=("stirrer", vessel),
                resolved_entities=("stirrer", vessel),
                controller_status="SUCCESS",
            )
        )
        index += 1
    return tuple(rows)


def _living_context(*, feasible: bool = True) -> HiddenBenchmarkContext:
    return HiddenBenchmarkContext(
        domain="living_room",
        variant=("F0_ALL_OBJECTS_IN_STAGING" if feasible else "I0_NO_SHARED_TABLE"),
        ground_truth_feasibility=feasible,
        requirements={
            "left_payloads": ["left_cup", "left_saucer"],
            "right_payloads": ["right_cup", "right_saucer"],
            "remote": "remote",
            "left_support": "left_table",
            "right_support": "right_table",
            "shared_support": "coffee_table",
        },
    )


def _living_state(*, predicted_infeasible: bool = False) -> TerminalStateSnapshot:
    return TerminalStateSnapshot(
        domain="living_room",
        predicted_infeasible=predicted_infeasible,
        objects={
            "left_cup": _on("left_table"),
            "left_saucer": _on("left_table"),
            "right_cup": _on("right_table"),
            "right_saucer": _on("right_table"),
            "remote": _on("coffee_table"),
            "left_table": {"present": True},
            "right_table": {"present": True},
            "coffee_table": {"present": True},
        },
    )


def _workshop_context(*, feasible: bool = True) -> HiddenBenchmarkContext:
    return HiddenBenchmarkContext(
        domain="workshop",
        variant=("F0_MANUAL_FIRST_ONE_REGION" if feasible else "I0_NO_DRIVER"),
        ground_truth_feasibility=feasible,
        requirements={
            "compatible_drivers": ["manual_driver", "power_driver"],
            "compatible_fasteners": ["screw"],
            "target": "repair_joint",
            "workbench": "main_workbench",
            "inspection_order": ["left_drawer", "right_drawer", "tool_cabinet"],
            "storage_contents": (
                {
                    "left_drawer": ["manual_driver", "screw"],
                    "right_drawer": ["power_driver"],
                    "tool_cabinet": ["hammer"],
                }
                if feasible
                else {
                    "left_drawer": ["screw"],
                    "right_drawer": ["hammer"],
                    "tool_cabinet": [],
                }
            ),
            "minimum_insertion_depth_m": 0.012,
            "maximum_insertion_depth_m": 0.018,
            "radial_tolerance_m": 0.004,
            "orientation_tolerance_rad": 0.03,
        },
    )


def _workshop_state(*, predicted_infeasible: bool = False) -> TerminalStateSnapshot:
    return TerminalStateSnapshot(
        domain="workshop",
        predicted_infeasible=predicted_infeasible,
        objects={"manual_driver": _on("main_workbench")},
        relations={
            "insertion": {
                "fastener": "screw",
                "target": "repair_joint",
                "depth_m": 0.015,
                "radial_error_m": 0.002,
                "orientation_error_rad": 0.01,
                "head_above_tip": True,
            }
        },
        measurements={
            "used_driver": "manual_driver",
            "used_fastener": "screw",
            "used_target": "repair_joint",
            "joint_repaired": True,
        },
    )


@pytest.mark.parametrize(
    ("state", "ledger", "context"),
    [
        (_kitchen_state(), _kitchen_ledger(), _kitchen_context()),
        (_living_state(), (), _living_context()),
        (
            _workshop_state(),
            (
                {
                    "effect": "DRIVE_COMPLETED",
                    "symbolic_arguments": [
                        "manual_driver",
                        "screw",
                        "repair_joint",
                    ],
                },
            ),
            _workshop_context(),
        ),
    ],
)
def test_feasible_success_fixtures_pass_all_hidden_requirements(
    state, ledger, context
) -> None:
    result = evaluate_hidden_benchmark(state, ledger, context)

    assert result.actual_task_success
    assert result.benchmark_outcome_correct
    assert not result.predicted_infeasible
    assert result.requirement_checks
    assert all(check["passed"] for check in result.requirement_checks)


@pytest.mark.parametrize(
    ("state", "ledger", "context", "failed_check"),
    [
        (
            _kitchen_state(),
            (),
            _kitchen_context(),
            "coffee_ingredients_and_stirring_verified",
        ),
        (
            TerminalStateSnapshot(
                **{
                    **_living_state().to_dict(),
                    "objects": {
                        **_living_state().objects,
                        "remote": {
                            **_on("coffee_table"),
                            "floor_contact": True,
                        },
                    },
                }
            ),
            (),
            _living_context(),
            "remote_physically_on_shared_table",
        ),
        (
            TerminalStateSnapshot(
                **{
                    **_workshop_state().to_dict(),
                    "measurements": {
                        **_workshop_state().measurements,
                        "used_driver": "power_driver",
                    },
                    "objects": {"power_driver": _on("main_workbench")},
                }
            ),
            (
                {
                    "effect": "DRIVE_COMPLETED",
                    "symbolic_arguments": [
                        "power_driver",
                        "screw",
                        "repair_joint",
                    ],
                },
            ),
            _workshop_context(),
            "first_compatible_driver_used",
        ),
    ],
)
def test_feasible_failures_are_independent_of_generated_goal(
    state, ledger, context, failed_check
) -> None:
    result = evaluate_hidden_benchmark(state, ledger, context)

    assert not result.actual_task_success
    assert not result.benchmark_outcome_correct
    by_name = {check["name"]: check for check in result.requirement_checks}
    assert by_name[failed_check]["passed"] is False


@pytest.mark.parametrize(
    ("state", "context"),
    [
        (
            TerminalStateSnapshot(domain="kitchen", predicted_infeasible=True),
            _kitchen_context(feasible=False),
        ),
        (
            TerminalStateSnapshot(
                domain="living_room",
                predicted_infeasible=True,
                objects={
                    "left_table": {"present": True},
                    "right_table": {"present": True},
                    "coffee_table": {"present": False},
                },
            ),
            _living_context(feasible=False),
        ),
        (
            TerminalStateSnapshot(domain="workshop", predicted_infeasible=True),
            _workshop_context(feasible=False),
        ),
    ],
)
def test_infeasible_fixtures_score_correct_recognition(state, context) -> None:
    result = evaluate_hidden_benchmark(state, (), context)

    assert not result.ground_truth_feasibility
    assert not result.actual_task_success
    assert result.predicted_infeasible
    assert result.correct_infeasibility_recognition
    assert result.benchmark_outcome_correct


def test_no_plan_on_feasible_variant_is_not_success() -> None:
    result = evaluate_hidden_benchmark(
        _living_state(predicted_infeasible=True), (), _living_context()
    )

    assert result.actual_task_success
    assert not result.benchmark_outcome_correct
    assert not result.correct_infeasibility_recognition


def test_evaluator_contract_has_no_generated_goal_input() -> None:
    parameters = inspect.signature(evaluate_hidden_benchmark).parameters
    assert tuple(parameters) == (
        "terminal_state",
        "effect_ledger",
        "hidden_context",
    )
    assert all("goal" not in name for name in parameters)


def test_evaluator_output_is_not_an_input_type_for_planning_or_cp() -> None:
    result = evaluate_hidden_benchmark(_living_state(), (), _living_context())
    assert isinstance(result, BenchmarkGoalEvaluation)

    for method in (
        ViLaInInterpreter.interpret,
        FastDownwardPlanner.plan,
        CorrectivePlanningLoop.run,
    ):
        hints = get_type_hints(method)
        accepted = tuple(hint for name, hint in hints.items() if name != "return")
        assert BenchmarkGoalEvaluation not in accepted
        assert all(
            "BenchmarkGoalEvaluation" not in str(hint) for hint in accepted
        )


def test_rejects_cross_domain_and_malformed_ledger_inputs() -> None:
    with pytest.raises(EvaluationContractError, match="domains differ"):
        evaluate_hidden_benchmark(
            _kitchen_state(), (), _living_context()
        )
    with pytest.raises(EvaluationContractError, match="symbolic arguments"):
        evaluate_hidden_benchmark(
            _kitchen_state(),
            ({"effect": "POUR_COMPLETED", "symbolic_arguments": None},),
            _kitchen_context(),
        )
