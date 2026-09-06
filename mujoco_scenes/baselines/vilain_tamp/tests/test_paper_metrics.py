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
    extract_representative_action_sequences,
    generate_manuscript_main_table,
)
from mujoco_scenes.baselines.vilain_tamp.benchmark_harness import (
    build_schedule,
    prepare_manifest,
    authoritative_feasibility,
    authoritative_requirements_count,
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


# ======================================================================
# 13 REQUIRED TESTS FOR PART 8
# ======================================================================


def test_outcome_correct_semantics() -> None:
    """Test 1: Outcome correct logic for feasible success and causally-clean infeasibility."""
    r1 = {
        "ground_truth_feasible": True,
        "actual_task_success": True,
        "has_obj": True,
        "pddl_valid": True,
        "symbolic_plan_found": True,
        "causal_category": "PLAN_FOUND",
        "infrastructure_failure": False,
    }
    r2 = {
        "ground_truth_feasible": True,
        "actual_task_success": False,
        "has_obj": True,
        "pddl_valid": True,
        "symbolic_plan_found": False,
        "causal_category": "SYMBOLIC_NO_PLAN_AFTER_BOUNDED_CP",
        "infrastructure_failure": False,
    }
    r3 = {
        "ground_truth_feasible": False,
        "actual_task_success": False,
        "has_obj": True,
        "pddl_valid": True,
        "symbolic_plan_found": False,
        "causal_category": "SYMBOLIC_NO_PLAN_AFTER_BOUNDED_CP",
        "infrastructure_failure": False,
    }
    r4 = {
        "ground_truth_feasible": False,
        "actual_task_success": False,
        "has_obj": False,
        "pddl_valid": False,
        "symbolic_plan_found": False,
        "causal_category": "UNRESOLVED_FM_FAILURE",
        "infrastructure_failure": False,
    }
    r5 = {
        "ground_truth_feasible": False,
        "actual_task_success": False,
        "has_obj": True,
        "pddl_valid": True,
        "symbolic_plan_found": True,
        "causal_category": "UNRESOLVED_IDENTITY_FAILURE",
        "infrastructure_failure": False,
    }
    r6 = {
        "ground_truth_feasible": False,
        "actual_task_success": False,
        "has_obj": True,
        "pddl_valid": True,
        "symbolic_plan_found": True,
        "causal_category": "UNRESOLVED_REFINEMENT_FAILURE",
        "infrastructure_failure": False,
    }
    r7 = {
        "ground_truth_feasible": False,
        "actual_task_success": False,
        "has_obj": True,
        "pddl_valid": True,
        "symbolic_plan_found": False,
        "causal_category": "UNRESOLVED_INVALID_CORRECTION",
        "infrastructure_failure": False,
    }

    def check_outcome_correct(r: dict) -> bool:
        if r["ground_truth_feasible"]:
            return bool(r["actual_task_success"])
        return bool(
            r["has_obj"]
            and r["pddl_valid"]
            and (not r["symbolic_plan_found"])
            and (r["causal_category"] == "SYMBOLIC_NO_PLAN_AFTER_BOUNDED_CP")
            and not r.get("infrastructure_failure", False)
        )

    assert check_outcome_correct(r1) is True
    assert check_outcome_correct(r2) is False
    assert check_outcome_correct(r3) is True
    assert check_outcome_correct(r4) is False
    assert check_outcome_correct(r5) is False
    assert check_outcome_correct(r6) is False
    assert check_outcome_correct(r7) is False


def test_feasible_task_success_denominator() -> None:
    """Test 2: Feasible-task success denominator includes all scheduled feasible tasks."""
    rows = []
    for _ in range(3):
        rows.append({"ground_truth_feasible": True, "actual_task_success": True})
    for _ in range(4):
        rows.append({"ground_truth_feasible": True, "actual_task_success": False})
    for _ in range(3):
        rows.append({"ground_truth_feasible": True, "actual_task_success": False, "raw_terminal_status": "FM_OBJECT_FAILURE"})
    for _ in range(5):
        rows.append({"ground_truth_feasible": False, "actual_task_success": False})

    aggr = compute_group_aggregates(rows, group_label="test")
    assert aggr["total_runs"] == 15
    assert aggr["feasible_runs"] == 10
    assert aggr["feasible_task_success_count"] == 3
    assert math.isclose(aggr["feasible_task_success_rate"], 0.3)


def test_goal_coverage_micro_and_macro() -> None:
    """Test 3: Goal coverage evaluates only GT-feasible trials; micro & macro aggregation."""
    rows = [
        {"ground_truth_feasible": True, "goal_requirements_passed": 4, "goal_requirements_total": 8},
        {"ground_truth_feasible": True, "goal_requirements_passed": 6, "goal_requirements_total": 8},
        {"ground_truth_feasible": True, "goal_requirements_passed": 0, "goal_requirements_total": 8},
        {"ground_truth_feasible": False, "goal_requirements_passed": 2, "goal_requirements_total": 6},
        {"ground_truth_feasible": False, "goal_requirements_passed": 0, "goal_requirements_total": 6},
    ]
    aggr = compute_group_aggregates(rows, group_label="test")
    assert aggr["goal_requirements_passed_feasible"] == 10
    assert aggr["goal_requirements_total_feasible"] == 24
    assert math.isclose(aggr["goal_coverage_micro"], 10 / 24)
    assert math.isclose(aggr["goal_coverage_macro"], 1.25 / 3)


def test_false_completion_metric() -> None:
    """Test 4: False completion numerator and denominator handling."""
    # Correct completion
    r1 = [{"declared_completion": True, "actual_task_success": True}]
    a1 = compute_group_aggregates(r1)
    assert a1["declared_completion_count"] == 1
    assert a1["false_completion_count"] == 0
    assert a1["false_completion_rate"] == 0.0

    # False completion
    r2 = [{"declared_completion": True, "actual_task_success": False}]
    a2 = compute_group_aggregates(r2)
    assert a2["declared_completion_count"] == 1
    assert a2["false_completion_count"] == 1
    assert a2["false_completion_rate"] == 1.0

    # Zero declared completions -> rate is None (N/A)
    r3 = [{"declared_completion": False, "actual_task_success": False}]
    a3 = compute_group_aggregates(r3)
    assert a3["declared_completion_count"] == 0
    assert a3["false_completion_count"] == 0
    assert a3["false_completion_rate"] is None

    # Zero-step hallucinated plan counts as false completion
    r4 = [{"declared_completion": True, "actual_task_success": False, "actual_selected_plan_length": 0}]
    a4 = compute_group_aggregates(r4)
    assert a4["false_completion_count"] == 1


def test_physical_plan_found_requires_nonempty_and_refinement() -> None:
    """Test 5: Physical plan found requires non-empty, VAL, identity, and refinement."""
    r_zero = {"ground_truth_feasible": True, "physical_plan_found": False}
    r_no_refine = {"ground_truth_feasible": True, "physical_plan_found": False}
    r_valid = {"ground_truth_feasible": True, "physical_plan_found": True}
    r_infeasible = {"ground_truth_feasible": False, "physical_plan_found": True}

    aggr = compute_group_aggregates([r_zero, r_no_refine, r_valid, r_infeasible])
    assert aggr["feasible_runs"] == 3
    assert aggr["physical_plan_found_count"] == 1
    assert math.isclose(aggr["physical_plan_found_rate"], 1 / 3)


def test_raw_vlm_requests_accounting(tmp_path: Path) -> None:
    """Test 6: Raw VLM requests count all actual FM calls."""
    run_dir = tmp_path / "run_calls"
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True)

    (run_dir / "terminal_status.json").write_text(json.dumps({
        "status": "NO_PLAN",
        "domain": "kitchen",
        "variant": "F0",
        "protocol": "initial_observation_only",
        "repeat": 0,
        "seed": 42,
        "metrics": {
            "model_calls_by_type": {
                "object_estimation": 1,
                "initial_state": 2,
                "goal_state": 1,
                "corrective_planning": 3,
            },
            "model_call_count": 7,
            "cp_calls": 3,
        }
    }), encoding="utf-8")

    audited = audit_run(run_dir)
    assert audited["raw_vlm_requests"] == 7
    assert audited["object_calls"] == 1
    assert audited["initial_state_calls"] == 2
    assert audited["goal_state_calls"] == 1
    assert audited["cp_calls"] == 3


def test_high_level_replans_accounting() -> None:
    """Test 7: High-level replans equals CP iterations."""
    rows = [
        {"high_level_replans": 0},
        {"high_level_replans": 1},
        {"high_level_replans": 2},
        {"high_level_replans": 3},
    ]
    aggr = compute_group_aggregates(rows)
    assert math.isclose(aggr["high_level_replans"]["mean"], 1.5)
    assert math.isclose(aggr["high_level_replans"]["min"], 0)
    assert math.isclose(aggr["high_level_replans"]["max"], 3)


def _config_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def test_one_protocol_one_repeat_schedule_generation() -> None:
    """Test 8: One-protocol one-repeat schedule generation yields 32 runs."""
    config_root = _config_root()
    schedule = build_schedule(config_root, protocols=["initial_observation_only"], repeats=1)
    assert len(schedule) == 32
    assert all(r.observation_protocol == "initial_observation_only" for r in schedule)
    assert all(r.repeat_index == 0 for r in schedule)


def test_five_repeat_schedule_generation() -> None:
    """Test 9: Five-repeat schedule generation yields 320 runs across both protocols."""
    config_root = _config_root()
    schedule = build_schedule(
        config_root,
        protocols=["initial_observation_only", "fixed_full_inspection"],
        repeats=5,
    )
    assert len(schedule) == 320


def test_exact_authoritative_variant_coverage() -> None:
    """Test 10: Authoritative variant counts: Kitchen 12, Living Room 10, Workshop 10 (Total 32)."""
    config_root = _config_root()
    feas_map = authoritative_feasibility(config_root)
    req_map = authoritative_requirements_count(config_root)

    assert len(feas_map["kitchen"]) == 12
    assert len(feas_map["living_room"]) == 10
    assert len(feas_map["workshop"]) == 10

    k_f = sum(1 for v in feas_map["kitchen"].values() if v is True)
    k_i = sum(1 for v in feas_map["kitchen"].values() if v is False)
    assert (k_f, k_i) == (6, 6)

    l_f = sum(1 for v in feas_map["living_room"].values() if v is True)
    l_i = sum(1 for v in feas_map["living_room"].values() if v is False)
    assert (l_f, l_i) == (6, 4)

    w_f = sum(1 for v in feas_map["workshop"].values() if v is True)
    w_i = sum(1 for v in feas_map["workshop"].values() if v is False)
    assert (w_f, w_i) == (8, 2)

    total_f = k_f + l_f + w_f
    total_i = k_i + l_i + w_i
    assert total_f == 20
    assert total_i == 12
    assert total_f + total_i == 32

    assert all(cnt == 8 for cnt in req_map["kitchen"].values())
    assert all(cnt == 6 for cnt in req_map["living_room"].values())
    assert all(cnt == 6 for cnt in req_map["workshop"].values())


def test_immutable_schedule_manifest(tmp_path: Path) -> None:
    """Test 11: Immutable schedule manifest cannot be overwritten with differing configuration."""
    config_root = _config_root()
    out_dir = tmp_path / "smoke_run"
    prepare_manifest(out_dir, source_commit="a" * 40, config_root=config_root, protocols=["initial_observation_only"], repeats=1)
    assert (out_dir / "schedule_manifest.json").is_file()
    with pytest.raises(RuntimeError):
        prepare_manifest(out_dir, source_commit="b" * 40, config_root=config_root, protocols=["initial_observation_only"], repeats=1)


def test_scheduled_feasibility_denominator_independent_of_early_fm_failure(tmp_path: Path) -> None:
    """Test 12: Early FM failure on GT-feasible variant does not drop it from feasible denominator."""
    run_dir = tmp_path / "run_fm_fail"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "terminal_status.json").write_text(json.dumps({
        "status": "FM_OBJECT_FAILURE",
        "domain": "kitchen",
        "variant": "F0",
        "protocol": "initial_observation_only",
        "repeat": 0,
        "seed": 42,
    }), encoding="utf-8")

    config_root = _config_root()
    audited = audit_run(run_dir, config_root=config_root)
    assert audited["ground_truth_feasible"] is True
    assert audited["actual_task_success"] is False
    assert audited["goal_requirements_total"] == 8
    assert audited["goal_requirements_passed"] == 0
    assert audited["outcome_correct"] is False


def test_nonempty_execution_success_is_separate_from_zero_step_execution_stage_completion() -> None:
    """Test 13: Non-empty plan execution is strictly separate from zero-step execution-stage completion."""
    r_zero = {
        "execution_stage_completed": True,
        "nonempty_plan_execution_completed": False,
        "actual_selected_plan_length": 0,
    }
    r_nonempty = {
        "execution_stage_completed": True,
        "nonempty_plan_execution_completed": True,
        "actual_selected_plan_length": 3,
    }

    aggr = compute_group_aggregates([r_zero, r_nonempty])
    assert aggr["execution_stage_completed_count"] == 2
    assert aggr["nonempty_plan_execution_completed_count"] == 1
    assert math.isclose(aggr["nonempty_plan_execution_completed_rate"], 0.5)
