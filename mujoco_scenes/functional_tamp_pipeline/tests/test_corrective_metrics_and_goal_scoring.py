from __future__ import annotations

import json
import shutil

from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
    compute_primary_metrics,
    full_task_coverage,
)


def test_goal_coverage_uses_feasible_denominator_only():
    records = [
        {"gt_feasible": True, "full_task_goal_coverage": 1.0},
        {"gt_feasible": True, "full_task_goal_coverage": 0.5},
        {"gt_feasible": False, "full_task_goal_coverage": 0.0},
    ]
    assert compute_primary_metrics(records)["goal_coverage"] == 75.0


def test_goal_scorer_invariant_to_operation_group_id(tmp_path):
    source = "benchmark_reports/raw_replay_final_v1/living_room/L1/vlm"
    original = tmp_path / "original"
    renamed = tmp_path / "renamed"
    shutil.copytree(source, original)
    shutil.copytree(source, renamed)
    grounding_path = renamed / "graph_grounding_result.json"
    grounding = json.loads(grounding_path.read_text())
    grounding["operation_bindings"] = {
        f"arbitrary_fm_id_{index}": value
        for index, value in enumerate(grounding["operation_bindings"].values())
    }
    grounding_path.write_text(json.dumps(grounding))
    assert full_task_coverage("living_room", original) == full_task_coverage("living_room", renamed)


def test_equivalent_v1_v2_plan_same_goal_coverage(tmp_path):
    # The scorer consumes independently replayed state and physical grounding,
    # never the raw schema version.
    source = "benchmark_reports/raw_replay_final_v1/living_room/L1/vlm"
    v1 = tmp_path / "v1"; v2 = tmp_path / "v2"
    shutil.copytree(source, v1); shutil.copytree(source, v2)
    spec_path = v2 / "functional_specification.json"
    spec = json.loads(spec_path.read_text())
    spec.setdefault("metadata", {})["is_v2_specification"] = True
    spec_path.write_text(json.dumps(spec))
    assert full_task_coverage("living_room", v1) == full_task_coverage("living_room", v2)


def test_invalid_candidate_plan_scores_zero_full_task(tmp_path):
    run_dir = tmp_path / "invalid"; run_dir.mkdir()
    (run_dir / "symbolic_problem.json").write_text(json.dumps({"initial_atoms": [], "actions": [], "goal_atoms": [["goal"]]}))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}}))
    assert full_task_coverage("kitchen", run_dir) == (4, 0, 0.0, False)


def test_missing_grounding_evidence_does_not_score_success(tmp_path):
    run_dir = tmp_path / "missing"; run_dir.mkdir()
    (run_dir / "action_plan.json").write_text(json.dumps({"validation": {"status": "VALID", "final_atoms": []}}))
    assert full_task_coverage("workshop", run_dir)[3] is False
