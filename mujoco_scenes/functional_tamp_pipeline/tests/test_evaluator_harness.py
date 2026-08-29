import json
import os
import tempfile
import pytest

from scripts.evaluate_functional_tamp_variants import (
    load_grounding_info,
    load_plan_validation,
    load_grounding_audit,
)


def test_evaluator_helpers_feasible_with_audit(tmp_path):
    run_dir = tmp_path / "run_gt"
    run_dir.mkdir(parents=True)

    # 1. graph_grounding_result.json
    ggr_file = run_dir / "graph_grounding_result.json"
    ggr_file.write_text(json.dumps({
        "status": "COMPLETE",
        "complete": True,
        "satisfied": True
    }))

    # 2. action_plan.json
    act_seq = run_dir / "action_sequence"
    act_seq.mkdir()
    plan_file = act_seq / "action_plan.json"
    plan_file.write_text(json.dumps({
        "actions": [{"operator": "PICK"}],
        "validation": {
            "status": "VALID",
            "goal_status": "GOAL_SATISFIED"
        }
    }))

    # 3. plan_grounding_audit.json
    audit_file = run_dir / "plan_grounding_audit.json"
    audit_file.write_text(json.dumps({
        "grounding_complete": True,
        "all_assignment_nodes_observed": True,
        "all_required_relations_true": True,
        "plan_uses_only_grounded_task_objects": True,
        "preparation_accessibility_valid": True,
        "violations": []
    }))

    gr_status, gr_complete = load_grounding_info(str(run_dir))
    assert gr_status == "COMPLETE"
    assert gr_complete is True

    plan_replay_valid = load_plan_validation(str(run_dir))
    assert plan_replay_valid is True

    audit_valid, prep_access = load_grounding_audit(str(run_dir))
    assert audit_valid is True
    assert prep_access is True


def test_evaluator_helpers_infeasible_without_audit(tmp_path):
    run_dir = tmp_path / "run_gt"
    run_dir.mkdir(parents=True)

    # Infeasible graph_grounding_result.json
    ggr_file = run_dir / "graph_grounding_result.json"
    ggr_file.write_text(json.dumps({
        "status": "INFEASIBLE",
        "complete": False,
        "satisfied": False
    }))

    gr_status, gr_complete = load_grounding_info(str(run_dir))
    assert gr_status == "INFEASIBLE"
    assert gr_complete is False

    plan_replay_valid = load_plan_validation(str(run_dir))
    assert plan_replay_valid is None

    audit_valid, prep_access = load_grounding_audit(str(run_dir))
    assert audit_valid is None
    assert prep_access is None


def test_evaluator_helpers_audit_with_violations(tmp_path):
    run_dir = tmp_path / "run_gt"
    run_dir.mkdir(parents=True)

    audit_file = run_dir / "plan_grounding_audit.json"
    audit_file.write_text(json.dumps({
        "grounding_complete": True,
        "all_assignment_nodes_observed": True,
        "all_required_relations_true": True,
        "plan_uses_only_grounded_task_objects": True,
        "preparation_accessibility_valid": False,
        "violations": ["Target object not at home region"]
    }))

    audit_valid, prep_access = load_grounding_audit(str(run_dir))
    assert audit_valid is False
    assert prep_access is False
