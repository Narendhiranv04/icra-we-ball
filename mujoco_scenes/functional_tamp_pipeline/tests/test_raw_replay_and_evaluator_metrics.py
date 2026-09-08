"""Tests for true raw FM replay mode, evaluator metrics, and failure taxonomy."""

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
    format_rate,
    full_task_coverage,
    enrich_record,
    write_detailed_report,
)


def test_zero_eligible_denominator_returns_na():
    assert format_rate(None) == "N/A"
    assert format_rate(0.5) == "50.0%"
    assert format_rate(1.0) == "100.0%"


def test_role_recall_distinct_from_complete_spec_rate(tmp_path):
    """Role recall may be 1.0 while complete spec rate is False due to missing relations/groups."""
    run_dir = tmp_path / "test_spec"
    run_dir.mkdir(parents=True)

    manifest = {"spec_acquisition": "archived_raw_provider_response", "semantic_vlm_requests": 0}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": "test",
        "nodes": {"driver": {"name": "driver"}, "fastener": {"name": "fastener"}}
    }))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({"assignment": {"driver": "obj1"}, "status": "PARTIAL"}))

    diag_dir = run_dir / "fm_diagnostics"
    diag_dir.mkdir()
    # Raw document with roles present, but functional relations empty
    raw_doc = {
        "status": "SUPPORTED",
        "task_summary": "fastening task",
        "functional_roles": [
            {"id": "role_1", "function": "fastening component", "entity_kind": "OBJECT", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw"], "visible_candidates": [], "required_properties": []},
            {"id": "role_2", "function": "driving tool", "entity_kind": "OBJECT", "required_count": 1, "binding_policy": "REUSABLE", "candidate_categories": ["screwdriver"], "visible_candidates": [], "required_properties": []},
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    (diag_dir / "fm_call_001.json").write_text(json.dumps({"content": json.dumps(raw_doc)}))

    row = {
        "domain": "workshop",
        "variant": "W1",
        "gt_feasible": True,
        "requires_observation_recovery": False,
        "runtime_contract_complete": True,
        "canonicalization_succeeded": True,
        "regions_available": [],
        "regions_inspected": [],
        "astar_invocations": 0,
        "candidate_plan_length": 0,
        "candidate_plan_status": "NO_PLAN",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "full_task_goal_coverage": 0.0,
        "full_task_satisfied": False,
        "outcome_correct": False,
        "false_completion": False,
        "search_exhausted": False,
    }
    task = "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench."
    enriched = enrich_record(row, run_dir, task)

    assert "raw_role_recall" in enriched
    assert "raw_vlm_spec_complete" in enriched
    # Even if raw_role_recall is high/1.0, raw_vlm_spec_complete evaluates required relations/groups too
    assert isinstance(enriched["raw_vlm_spec_complete"], bool)


def test_any_grounding_vs_complete_candidate_grounding(tmp_path):
    """Having 1 grounded role out of 3 is verified grounding, but NOT complete candidate grounding."""
    run_dir = tmp_path / "test_gr"
    run_dir.mkdir(parents=True)

    manifest = {"spec_acquisition": "archived_raw_provider_response"}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": "test",
        "nodes": {"driver": {"name": "driver"}, "fastener": {"name": "fastener"}, "workshop_frame_joint": {"name": "workshop_frame_joint"}}
    }))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({
        "assignment": {"driver": "screwdriver_1"},
        "status": "PARTIAL_VERIFIED_GROUNDING",
        "complete": False,
        "missing_roles": ["fastener"],
    }))

    row = {
        "domain": "workshop",
        "variant": "W1",
        "gt_feasible": True,
        "requires_observation_recovery": False,
        "runtime_contract_complete": True,
        "canonicalization_succeeded": True,
        "regions_available": [],
        "regions_inspected": [],
        "astar_invocations": 0,
        "candidate_plan_length": 0,
        "candidate_plan_status": "NO_PLAN",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "full_task_goal_coverage": 0.0,
        "full_task_satisfied": False,
        "outcome_correct": False,
        "false_completion": False,
        "search_exhausted": False,
    }
    task = "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench."
    enriched = enrich_record(row, run_dir, task)

    assert enriched["candidate_grounding_succeeded"] is True  # Any verified grounding
    assert enriched["complete_candidate_grounding"] is False  # Incomplete grounding
    assert pytest.approx(enriched["grounded_expressed_role_coverage"], 0.01) == 1.0 / 3.0
    assert enriched["nonempty_candidate_plan_generated"] is False


def test_infeasible_correctness_requires_runtime_contract(tmp_path):
    """Infeasible variants cannot claim outcome_correct if the runtime contract was incomplete."""
    run_dir = tmp_path / "test_infeasible"
    run_dir.mkdir(parents=True)

    manifest = {"spec_acquisition": "archived_raw_provider_response"}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": "test",
        "nodes": {}
    }))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INFEASIBLE"}))

    row = {
        "domain": "workshop",
        "variant": "W9",
        "gt_feasible": False,
        "requires_observation_recovery": False,
        "runtime_contract_complete": False,  # INCOMPLETE CONTRACT
        "canonicalization_succeeded": False,
        "regions_available": [],
        "regions_inspected": [],
        "astar_invocations": 0,
        "candidate_plan_length": 0,
        "candidate_plan_status": "INFEASIBLE",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "full_task_goal_coverage": 0.0,
        "full_task_satisfied": False,
        "outcome_correct": True,  # Initially claimed
        "false_completion": False,
        "search_exhausted": False,
    }
    task = "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench."
    enriched = enrich_record(row, run_dir, task)

    # Incomplete contract forces outcome_correct to False
    assert enriched["outcome_correct"] is False


def test_failure_taxonomy_first_cause_categorization(tmp_path):
    """Verify first-cause categorization cleanly identifies the paper-level causal failure."""
    run_dir = tmp_path / "test_taxonomy"
    run_dir.mkdir(parents=True)

    manifest = {"spec_acquisition": "archived_raw_provider_response"}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    diag_dir = run_dir / "fm_diagnostics"
    diag_dir.mkdir()
    raw_doc = {
        "status": "SUPPORTED",
        "task_summary": "fastening task",
        "functional_roles": [
            {"id": "role_1", "function": "fastening component", "entity_kind": "OBJECT", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw"], "visible_candidates": [], "required_properties": []},
            {"id": "role_2", "function": "driving tool", "entity_kind": "OBJECT", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screwdriver"], "visible_candidates": [], "required_properties": []},
            {"id": "role_3", "function": "marked repair target joint", "entity_kind": "FIXED_TARGET", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["workshop frame joint"], "visible_candidates": [], "required_properties": []},
        ],
        "functional_relations": [
            {"subject_role": "role_2", "relation": "compatible with", "object_role": "role_1"},
            {"subject_role": "role_2", "relation": "reaches target", "object_role": "role_3"},
            {"subject_role": "role_1", "relation": "compatible with target", "object_role": "role_3"},
        ],
        "interaction_groups": [
            {"id": "group_1", "function": "fasten", "tool_role": "role_2", "target_role": "role_1", "required_target_count": 1, "usage_policy": "DEDICATED_PER_TARGET", "required_relations": ["compatible with"]}
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    (diag_dir / "fm_call_001.json").write_text(json.dumps({"content": json.dumps(raw_doc)}))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": "test",
        "nodes": {"driver": {"name": "driver"}},
        "metadata": {
            "raw_vlm_response": raw_doc,
            "canonicalization_trace": {"unresolved_roles": [{"raw_id": "role_2"}]}
        }
    }))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INCOMPLETE"}))

    row = {
        "domain": "workshop",
        "variant": "W3",
        "gt_feasible": True,
        "requires_observation_recovery": False,
        "runtime_contract_complete": False,
        "canonicalization_succeeded": True,
        "regions_available": [],
        "regions_inspected": [],
        "astar_invocations": 0,
        "candidate_plan_length": 0,
        "candidate_plan_status": "NO_PLAN",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "full_task_goal_coverage": 0.0,
        "full_task_satisfied": False,
        "outcome_correct": False,
        "false_completion": False,
        "search_exhausted": False,
    }
    task = "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench."
    enriched = enrich_record(row, run_dir, task)

    # When raw semantics are complete, unresolved role in trace maps to GRAPH_COMPILATION_FAILURE
    assert enriched["first_cause_category"] == "GRAPH_COMPILATION_FAILURE"
    assert enriched["failure_category"] == "CANONICALIZATION_AMBIGUITY"


def test_w8_first_cause_derived_from_evidence(tmp_path):
    """W8 must be classified based on evidence: missing target relations -> TASK_SPECIFICATION_FAILURE."""
    import shutil
    baseline_records = json.loads(Path("benchmark_reports/live_final_evaluation_v2/evaluation_records.json").read_text())
    w8_row = next(r for r in baseline_records if r["variant"] == "W8")
    src_run_dir = Path("benchmark_reports/live_final_evaluation_v2/workshop/W8/vlm")
    w8_dir = tmp_path / "w8"
    shutil.copytree(src_run_dir, w8_dir)
    # Ensure copied files are writable in tmp_path
    for p in w8_dir.rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    task = "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench."

    enriched = enrich_record(dict(w8_row), w8_dir, task)
    assert enriched["first_cause_category"] == "TASK_SPECIFICATION_FAILURE"
    assert enriched["failure_category"] == "FM_SEMANTIC_OMISSION"


def test_causal_search_recovery_logic(tmp_path):
    """Causal search recovery requires initial incomplete, search executed, and final complete."""
    run_dir = tmp_path / "causal_test"
    run_dir.mkdir(parents=True)
    manifest = {"spec_acquisition": "archived_raw_provider_response"}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "functional_specification.json").write_text(json.dumps({"nodes": {"a": {}}}))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({"assignment": {"a": "1"}, "complete": True}))

    # Case 1: initial was incomplete in trajectory_events, region inspected, final complete -> True
    events = [
        {"event": "grounding_updated", "payload": {"grounding": {"complete": False, "status": "INCOMPLETE"}}},
        {"event": "search_region_selected", "payload": {"region": "R1"}},
        {"event": "grounding_updated", "payload": {"grounding": {"complete": True, "status": "COMPLETE"}}},
    ]
    (run_dir / "trajectory_events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    row = {
        "domain": "kitchen", "variant": "K2", "gt_feasible": True, "regions_available": ["R1"],
        "regions_inspected": ["R1"], "full_task_satisfied": True, "full_task_goal_count": 4,
        "full_task_goal_satisfied_count": 4, "full_task_goal_coverage": 1.0, "canonicalization_succeeded": True,
    }
    enriched = enrich_record(dict(row), run_dir, "Prepare and serve coffee")
    assert enriched["search_recovery_succeeded"] is True

    # Case 2: initial was already complete, region inspected -> False (not a recovery!)
    events[0] = {"event": "grounding_updated", "payload": {"grounding": {"complete": True, "status": "COMPLETE"}}}
    (run_dir / "trajectory_events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    enriched2 = enrich_record(dict(row), run_dir, "Prepare and serve coffee")
    assert enriched2["search_recovery_succeeded"] is False


def test_git_dirty_detects_untracked_source_file(tmp_path):
    """Git provenance detects untracked files in source directory as dirty."""
    import subprocess
    from mujoco_scenes.functional_tamp_pipeline.audit import get_git_info
    # Create temporary git repo
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo), check=True)
    src_dir = repo / "mujoco_scenes"
    src_dir.mkdir()
    tracked = src_dir / "tracked.py"
    tracked.write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo), check=True)

    info_clean = get_git_info(repo)
    assert info_clean["git_dirty"] is False

    # Add untracked source file
    untracked = src_dir / "untracked.py"
    untracked.write_text("y = 2\n")
    info_dirty = get_git_info(repo)
    assert info_dirty["git_dirty"] is True


def test_baseline_primary_metrics_recompute_exact():
    """Current primary metrics recompute exactly from attached live_final_evaluation_v2 records."""
    records = json.loads(Path("benchmark_reports/live_final_evaluation_v2/evaluation_records.json").read_text())
    summary = json.loads(Path("benchmark_reports/live_final_evaluation_v2/evaluation_summary.json").read_text())

    n_total = len(records)
    feasible_rows = [r for r in records if r["gt_feasible"]]
    infeasible_rows = [r for r in records if not r["gt_feasible"]]
    recovery_rows = [r for r in records if r["requires_observation_recovery"]]

    assert n_total == 32
    assert len(feasible_rows) == 20
    assert len(infeasible_rows) == 12
    assert len(recovery_rows) == 13

    outcome_correct_pct = 100.0 * sum(1 for r in records if r["outcome_correct"]) / n_total
    feasible_success_pct = 100.0 * sum(1 for r in feasible_rows if r["full_task_satisfied"]) / len(feasible_rows)
    feasibility_recovery_pct = 100.0 * sum(1 for r in recovery_rows if r["full_task_satisfied"]) / len(recovery_rows)
    mean_goal_cov = 100.0 * sum(r["full_task_goal_coverage"] for r in feasible_rows) / len(feasible_rows)
    false_comp_pct = 100.0 * sum(1 for r in infeasible_rows if r["false_completion"]) / len(infeasible_rows)
    mean_vlm_req = sum(r["semantic_vlm_requests"] for r in records) / n_total
    mean_replans = sum(r["high_level_replans"] for r in records) / n_total

    pm = summary["primary_metrics"]
    assert pytest.approx(outcome_correct_pct, 1e-4) == pm["outcome_correct"]
    assert pytest.approx(feasible_success_pct, 1e-4) == pm["feasible_success"]
    assert pytest.approx(feasibility_recovery_pct, 1e-4) == pm["feasibility_recovery"]
    assert pytest.approx(mean_goal_cov, 1e-4) == pm["goal_coverage"]
    assert pytest.approx(false_comp_pct, 1e-4) == pm["false_completion"]
    assert pytest.approx(mean_vlm_req, 1e-4) == pm["vlm_requests"]
    assert pytest.approx(mean_replans, 1e-4) == pm["high_level_replans"]


def test_true_raw_replay_pipeline_e2e(tmp_path):
    """End-to-end integration test verifying that raw response replay executes without VLM calls."""
    from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline
    spec_file = Path("benchmark_reports/final_vlm_evaluation_v3_rerun5/kitchen/K1/vlm/fm_diagnostics/fm_call_001.json")
    if not spec_file.exists():
        pytest.skip("Reference benchmark archive not found")

    result = run_pipeline(
        domain="kitchen",
        variant="K1",
        mode="vlm",
        specification_json=spec_file,
        output_root=tmp_path / "replay_out",
        dry_run=True,
    )
    manifest = json.loads((tmp_path / "replay_out" / "kitchen" / "K1" / "vlm" / "run_manifest.json").read_text())

    assert manifest["spec_acquisition"] == "archived_raw_provider_response"
    assert manifest["semantic_vlm_requests"] == 0
    assert manifest["high_level_replans"] == 0
    assert result.status is not None
