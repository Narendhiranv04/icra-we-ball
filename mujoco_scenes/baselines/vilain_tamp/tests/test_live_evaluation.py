from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import GeneratedPDDLProblem, ProblemSource
from mujoco_scenes.baselines.vilain_tamp.evaluation import (
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    evaluate_hidden_benchmark,
)
from mujoco_scenes.baselines.vilain_tamp.live_evaluation import (
    FileHiddenContextProvider,
    LiveEvaluationError,
    PhysicalGeneratedGoalEvaluator,
)
from mujoco_scenes.baselines.vilain_tamp.runner import BaselineRunner


def problem(*goal_atoms: str) -> GeneratedPDDLProblem:
    return GeneratedPDDLProblem(
        attempt_index=0,
        source=ProblemSource.INITIAL,
        domain_name="vilain-kitchen",
        domain_sha256="a" * 64,
        problem_text="(define (problem test))",
        declared_objects=("cup", "table"),
        initial_atoms=(),
        goal_atoms=goal_atoms,
        raw_response_artifact="interpreter/raw.json",
        problem_sha256="b" * 64,
    )


def physical_object(support: str) -> dict[str, object]:
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


def test_generated_goal_uses_physical_state_and_verified_effects() -> None:
    terminal = TerminalStateSnapshot(
        domain="kitchen",
        predicted_infeasible=False,
        objects={
            "cup": physical_object("table"),
            "spoon": {"released": True, "stable": True},
        },
        relations={
            "contained_in": {"bowl": ["spoon"]},
            "articulation": {"cabinet": {"open": True}},
        },
        held_objects=(),
    )
    ledger = (
        {"effect": "POUR_COMPLETED", "symbolic_arguments": ["jar", "cup", "coffee"]},
        {"effect": "STIR_COMPLETED", "symbolic_arguments": ["spoon", "cup"]},
    )
    evaluation = PhysicalGeneratedGoalEvaluator().evaluate(
        problem=problem(
            "(at cup table)",
            "(contains cup coffee)",
            "(stirred cup)",
            "(inside spoon bowl)",
            "(open cabinet)",
            "(handempty)",
        ),
        terminal_state=terminal,
        effect_ledger=ledger,
    )
    assert evaluation["status"] == "SATISFIED"
    assert all(check["passed"] for check in evaluation["goal_checks"])


def test_failed_skill_effect_cannot_satisfy_generated_goal() -> None:
    terminal = TerminalStateSnapshot("kitchen", False)
    evaluation = PhysicalGeneratedGoalEvaluator().evaluate(
        problem=problem("(contains cup coffee)"),
        terminal_state=terminal,
        effect_ledger=(),
    )
    assert evaluation["status"] == "NOT_SATISFIED"


def test_generated_goal_and_hidden_benchmark_outcomes_remain_distinct() -> None:
    terminal = TerminalStateSnapshot(
        domain="living_room",
        predicted_infeasible=False,
        objects={
            "cup": physical_object("generated_table"),
            "expected_table": {"present": True},
            "generated_table": {"present": True},
        },
        relations={},
        held_objects=(),
    )
    generated = PhysicalGeneratedGoalEvaluator().evaluate(
        problem=problem("(at cup generated_table)"),
        terminal_state=terminal,
        effect_ledger=(),
    )
    hidden = HiddenBenchmarkContext(
        domain="living_room",
        variant="synthetic",
        ground_truth_feasibility=True,
        requirements={
            "left_payloads": ["cup"],
            "right_payloads": ["cup"],
            "remote": "cup",
            "left_support": "expected_table",
            "right_support": "expected_table",
            "shared_support": "expected_table",
        },
    )
    benchmark = evaluate_hidden_benchmark(terminal, (), hidden)
    assert generated["satisfied"] is True
    assert benchmark.actual_task_success is False
    assert benchmark.benchmark_outcome_correct is False


def test_hidden_context_provider_loads_only_requested_final_context(tmp_path: Path) -> None:
    context_path = tmp_path / "workshop" / "v1.json"
    context_path.parent.mkdir()
    context_path.write_text(
        json.dumps(
            {
                "domain": "workshop",
                "variant": "v1",
                "ground_truth_feasibility": True,
                "requirements": {"target": "joint"},
                "evidence_artifacts": ["private/variant.json"],
            }
        ),
        encoding="utf-8",
    )
    provider = FileHiddenContextProvider(tmp_path)
    context = provider.load("workshop", "v1")
    assert context.requirements == {"target": "joint"}
    assert context.evidence_artifacts == ("private/variant.json",)
    with pytest.raises(LiveEvaluationError, match="unsafe"):
        provider.load("workshop", "../v1")


def test_hidden_context_is_loaded_only_after_execution_and_generated_goal_evaluation() -> None:
    source = inspect.getsource(BaselineRunner.run)
    execution_index = source.index("components.execution.execute")
    generated_index = source.index("components.generated_goal_evaluator.evaluate")
    hidden_index = source.index("components.hidden_context.load")
    benchmark_index = source.index("evaluate_hidden_benchmark")
    assert execution_index < generated_index < hidden_index < benchmark_index
    prefix = source[:hidden_index]
    assert "hidden_context.load" not in prefix


def test_generated_goal_evaluator_fails_closed_on_nonphysical_predicate() -> None:
    with pytest.raises(LiveEvaluationError, match="unsupported"):
        PhysicalGeneratedGoalEvaluator().evaluate(
            problem=problem("(driver-compatible driver fastener)"),
            terminal_state=TerminalStateSnapshot("workshop", False),
            effect_ledger=(),
        )
