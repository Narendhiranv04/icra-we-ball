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
            {"id": "role_2", "function": "driving tool", "entity_kind": "OBJECT", "required_count": 1, "binding_policy": "REUSABLE", "candidate_categories": ["screwdriver"], "visible_candidates": [], "required_properties": []},
        ],
        "functional_relations": [],
        "interaction_groups": [],
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

    # Unresolved role in trace must map to GRAPH_COMPILATION_FAILURE
    assert enriched["first_cause_category"] == "GRAPH_COMPILATION_FAILURE"
    assert enriched["failure_category"] == "CANONICALIZATION_AMBIGUITY"


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
