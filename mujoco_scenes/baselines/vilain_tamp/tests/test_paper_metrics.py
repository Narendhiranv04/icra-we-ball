"""Unit tests for ViLaIn-TAMP paper-readiness audit and metrics computation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import pytest

from mujoco_scenes.baselines.vilain_tamp.paper_metrics import (
    wilson_interval,
    continuous_stats,
    compute_confusion_matrix,
    compute_group_aggregates,
    compute_stage_funnel,
    extract_run_attempts,
    audit_run,
    run_full_paper_analysis,
)


def test_wilson_interval_bounds() -> None:
    # 0 successes
    low, high = wilson_interval(0, 100)
    assert low == 0.0
    assert 0.0 < high < 0.05
    
    # 100% successes
    low, high = wilson_interval(100, 100)
    assert 0.95 < low < 1.0
    assert high == 1.0
    
    # 50% successes
    low, high = wilson_interval(50, 100)
    assert 0.40 < low < 0.50
    assert 0.50 < high < 0.60
    assert math.isclose((low + high) / 2.0, 0.5, abs_tol=0.01)
    
    # Empty total
    low, high = wilson_interval(0, 0)
    assert (low, high) == (0.0, 0.0)


def test_continuous_stats_calculation() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = continuous_stats(data)
    assert stats["count"] == 5
    assert math.isclose(stats["mean"], 3.0)
    assert math.isclose(stats["median"], 3.0)
    assert math.isclose(stats["q25"], 2.0)
    assert math.isclose(stats["q75"], 4.0)
    assert math.isclose(stats["iqr"], 2.0)
    
    empty_stats = continuous_stats([])
    assert empty_stats["count"] == 0
    assert empty_stats["mean"] is None


def test_feasibility_confusion_matrix_standard_formulas() -> None:
    # 10 actual infeasible, 20 actual feasible
    # Infeasible = Positive class, Feasible = Negative class
    # 8 TP (infeasible predicted infeasible)
    # 2 FN (infeasible predicted feasible)
    # 5 FP (feasible predicted infeasible)
    # 15 TN (feasible predicted feasible)
    rows = []
    for _ in range(8):
        rows.append({"ground_truth_feasible": False, "raw_predicted_infeasible": True})
    for _ in range(2):
        rows.append({"ground_truth_feasible": False, "raw_predicted_infeasible": False})
    for _ in range(5):
        rows.append({"ground_truth_feasible": True, "raw_predicted_infeasible": True})
    for _ in range(15):
        rows.append({"ground_truth_feasible": True, "raw_predicted_infeasible": False})
    for _ in range(3):  # Unresolved rows (missing prediction)
        rows.append({"ground_truth_feasible": True, "raw_predicted_infeasible": None})

    cm = compute_confusion_matrix(rows)
    assert cm["evaluated_runs"] == 30
    assert cm["total_scheduled_runs"] == 33
    assert math.isclose(cm["decision_coverage"], 30 / 33)
    assert cm["true_positives"] == 8
    assert cm["false_positives"] == 5
    assert cm["true_negatives"] == 15
    assert cm["false_negatives"] == 2
    
    # Accuracy = (8 + 15) / 30 = 23 / 30 = 0.7667
    assert math.isclose(cm["accuracy"], 23 / 30)
    # Precision = 8 / (8 + 5) = 8 / 13 = 0.6154
    assert math.isclose(cm["precision"], 8 / 13)
    # Recall (Infeasible Recall) = 8 / (8 + 2) = 0.8
    assert math.isclose(cm["recall"], 0.8)
    # Specificity (Feasible Recall) = 15 / (15 + 5) = 0.75
    assert math.isclose(cm["specificity"], 0.75)
    # Balanced accuracy = (0.8 + 0.75) / 2 = 0.775
    assert math.isclose(cm["balanced_accuracy"], 0.775)
    # F1 = 2*8 / (2*8 + 5 + 2) = 16 / 23 = 0.6957
    assert math.isclose(cm["f1_score"], 16 / 23)
    # False Infeasible Rate = 5 / 20 = 0.25
    assert math.isclose(cm["false_infeasible_rate"], 0.25)
    # False Feasible Rate = 2 / 10 = 0.2
    assert math.isclose(cm["false_feasible_rate"], 0.2)


def test_action_sequence_detection_independent_of_refinement(tmp_path: Path) -> None:
    # Run directory where symbolic plan exists, but refinement fails
    run_dir = tmp_path / "run_test"
    art_dir = run_dir / "artifacts"
    att_dir = art_dir / "attempts" / "00"
    (att_dir / "planner").mkdir(parents=True)
    
    plan_content = {
        "actions": [
            {"action_index": 0, "action_instance_id": "act1", "arguments": ["a", "b"], "operator": "pick"},
            {"action_index": 1, "action_instance_id": "act2", "arguments": ["a", "c"], "operator": "place"},
        ],
        "attempt_index": 0,
        "plan_cost": 2.0,
        "plan_sha256": "abcdef123456",
    }
    (att_dir / "planner" / "symbolic_plan.json").write_text(json.dumps(plan_content), encoding="utf-8")
    (att_dir / "planner" / "plan_validation.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
    
    outcome_content = {
        "attempt_index": 0,
        "failure": {"kind": "REFINEMENT", "summary": "collision detected"},
        "success": False,
    }
    (att_dir / "attempt_outcome.json").write_text(json.dumps(outcome_content), encoding="utf-8")
    
    attempts = extract_run_attempts(art_dir, "run_test")
    assert len(attempts) == 1
    att = attempts[0]
    # Action sequence was detected and VAL-valid even though refinement failed
    assert att["plan_found"] is True
    assert att["nonempty_plan"] is True
    assert att["val_valid"] is True
    assert att["plan_length"] == 2
    assert att["identity_success"] is True
    assert att["refinement_success"] is False
    assert att["attempt_success"] is False


def test_empty_plan_handling(tmp_path: Path) -> None:
    # Run directory where empty plan is generated (cost 0, 0 actions)
    run_dir = tmp_path / "run_empty"
    art_dir = run_dir / "artifacts"
    att_dir = art_dir / "attempts" / "00"
    (att_dir / "planner").mkdir(parents=True)
    
    plan_content = {
        "actions": [],
        "attempt_index": 0,
        "plan_cost": 0.0,
        "plan_sha256": "empty123",
    }
    (att_dir / "planner" / "symbolic_plan.json").write_text(json.dumps(plan_content), encoding="utf-8")
    (att_dir / "planner" / "plan_validation.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
    
    outcome_content = {
        "attempt_index": 0,
        "success": True,
    }
    (att_dir / "attempt_outcome.json").write_text(json.dumps(outcome_content), encoding="utf-8")
    
    attempts = extract_run_attempts(art_dir, "run_empty")
    assert len(attempts) == 1
    att = attempts[0]
    assert att["plan_found"] is True
    assert att["nonempty_plan"] is False
    assert att["val_valid"] is True
    assert att["plan_length"] == 0
    assert att["identity_success"] is True
    assert att["refinement_success"] is True
    assert att["attempt_success"] is True


def test_multiple_cp_attempt_handling(tmp_path: Path) -> None:
    # Run directory with multiple CP attempts: attempt 0 fails NO_PLAN, attempt 1 succeeds
    run_dir = tmp_path / "run_cp"
    art_dir = run_dir / "artifacts"
    
    att0 = art_dir / "attempts" / "00"
    (att0 / "planner").mkdir(parents=True)
    (att0 / "attempt_outcome.json").write_text(
        json.dumps({"attempt_index": 0, "failure": {"kind": "NO_PLAN"}, "success": False}),
        encoding="utf-8",
    )
    
    att1 = art_dir / "attempts" / "01"
    (att1 / "planner").mkdir(parents=True)
    (att1 / "planner" / "symbolic_plan.json").write_text(
        json.dumps({"actions": [{"operator": "open", "arguments": ["d1"]}], "attempt_index": 1, "plan_cost": 1.0}),
        encoding="utf-8",
    )
    (att1 / "planner" / "plan_validation.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
    (att1 / "attempt_outcome.json").write_text(
        json.dumps({"attempt_index": 1, "failure": {"kind": "ENTITY_RESOLUTION"}, "success": False}),
        encoding="utf-8",
    )
    
    attempts = extract_run_attempts(art_dir, "run_cp")
    assert len(attempts) == 2
    assert attempts[0]["plan_found"] is False
    assert attempts[1]["plan_found"] is True
    assert attempts[1]["plan_length"] == 1
    assert attempts[1]["identity_success"] is False


def test_requirement_and_goal_atom_coverage() -> None:
    # Test partial-progress metrics
    rows = [
        {
            "benchmark_requirements_passed": 3,
            "benchmark_requirements_total": 6,
            "generated_goal_evaluated": True,
            "generated_goal_satisfied": False,
            "generated_goal_atoms_passed": 2,
            "generated_goal_atoms_total": 4,
            "symbolic_plan_found": True,
            "symbolic_plan_nonempty": True,
            "val_plan_valid": True,
            "identity_success": True,
            "refinement_success": False,
            "execution_attempted": False,
            "execution_success": False,
            "actual_task_success": False,
            "ground_truth_feasible": True,
        },
        {
            "benchmark_requirements_passed": 4,
            "benchmark_requirements_total": 8,
            "generated_goal_evaluated": True,
            "generated_goal_satisfied": False,
            "generated_goal_atoms_passed": 0,
            "generated_goal_atoms_total": 2,
            "symbolic_plan_found": True,
            "symbolic_plan_nonempty": False,
            "val_plan_valid": True,
            "identity_success": True,
            "refinement_success": True,
            "execution_attempted": True,
            "execution_success": True,
            "actual_task_success": False,
            "ground_truth_feasible": False,
        },
    ]
    
    aggr = compute_group_aggregates(rows, group_label="test")
    # Total reqs passed = 3 + 4 = 7, Total reqs = 6 + 8 = 14 -> 7/14 = 0.5
    assert math.isclose(aggr["benchmark_requirement_coverage_micro"], 0.5)
    # Total atoms passed = 2 + 0 = 2, Total atoms = 4 + 2 = 6 -> 2/6 = 0.3333
    assert math.isclose(aggr["generated_goal_atom_coverage_micro"], 2 / 6)
    assert aggr["generated_goal_satisfied_count"] == 0
    assert aggr["generated_goal_evaluated_count"] == 2
    assert math.isclose(aggr["generated_goal_evaluation_coverage"], 1.0)


def test_immutable_results_root_preservation() -> None:
    # Verify that running paper analysis leaves input files unmodified
    source_root = Path("/home/naren/ViLaIn-TAMP-results/stage24-4532495")
    if not source_root.exists():
        pytest.skip("Frozen stage24 results not available")
        
    mtimes_before = {p: p.stat().st_mtime for p in source_root.rglob("*.json")}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_full_paper_analysis(source_root, Path(tmpdir))
        
    mtimes_after = {p: p.stat().st_mtime for p in source_root.rglob("*.json")}
    assert mtimes_before == mtimes_after, "Analysis script modified raw experimental artifacts!"
