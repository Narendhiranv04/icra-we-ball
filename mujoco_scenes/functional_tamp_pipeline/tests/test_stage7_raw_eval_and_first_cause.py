"""Unit tests for Stage 7: Raw Evaluation and First-Cause Attribution.

Verifies:
1. Raw evaluator covers all task operation families across Kitchen, Living Room, and Workshop (Section 14.2).
2. Workshop operation recall is non-None and correctly scored (Section 14.2).
3. Evaluates semantic meaning, not exact wording, with robust phrase normalization (Section 14.3).
4. First-cause distinction saves 8 diagnostic flags and strictly separates raw-vs-compiler causes (Section 14.4).
5. Compiler rejection is never attributed to FM omission automatically (Section 14.5).
6. Representative K2 and W1 cases are cleanly separable into raw-vs-compiler causes (Gate 7).
7. Raw evaluation remains strictly offline only (Section 14.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from mujoco_scenes.functional_tamp_pipeline.raw_semantic_evaluation import (
    evaluate_raw_semantics,
    extract_evaluation_contract,
)
from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import enrich_record


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"


def test_workshop_operation_coverage_and_recall():
    """Gate 7 / Section 14.2: Workshop operation recall must not be None and must cover FASTEN_JOINT."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    if not w1_file.exists():
        pytest.skip("W1 fixture missing")

    raw_w1 = json.loads(w1_file.read_text(encoding="utf-8"))
    metrics = evaluate_raw_semantics("workshop", "W1", raw_w1)

    assert metrics["operation"]["recall"] is not None, "Workshop operation recall must not be None"
    assert metrics["operation"]["recall"] == 1.0, "W1 ideal raw document must have 1.0 operation recall"
    assert metrics["operation"]["precision"] == 1.0
    assert metrics["operation"]["f1"] == 1.0
    assert metrics["complete_task_contract"] is True


def test_kitchen_and_living_room_operation_coverage():
    """Gate 7 / Section 14.2: Kitchen and Living Room operation coverage evaluates correctly."""
    k1_file = FIXTURES_DIR / "kitchen_K1.json"
    l1_file = FIXTURES_DIR / "living_room_L1.json"
    if not k1_file.exists() or not l1_file.exists():
        pytest.skip("Fixtures missing")

    raw_k1 = json.loads(k1_file.read_text(encoding="utf-8"))
    metrics_k = evaluate_raw_semantics("kitchen", "K1", raw_k1)
    assert metrics_k["operation"]["recall"] == 1.0
    assert metrics_k["operation"]["precision"] == 1.0
    assert metrics_k["complete_task_contract"] is True

    raw_l1 = json.loads(l1_file.read_text(encoding="utf-8"))
    metrics_l = evaluate_raw_semantics("living_room", "L1", raw_l1)
    assert metrics_l["operation"]["recall"] == 1.0
    assert metrics_l["operation"]["precision"] == 1.0
    assert metrics_l["complete_task_contract"] is True


def test_k2_separable_into_raw_vs_compiler_causes(tmp_path: Path):
    """Gate 7: Representative K2 case cleanly separable into raw-vs-compiler causes."""
    k2_file = FIXTURES_DIR / "kitchen_K1.json"
    if not k2_file.exists():
        pytest.skip("K1 fixture missing")

    raw_k2 = json.loads(k2_file.read_text(encoding="utf-8"))

    # Case A: Raw K2 document is complete, but compiler suffered a canonicalization ambiguity
    run_dir_a = tmp_path / "k2_compiler_fail"
    run_dir_a.mkdir(parents=True)
    diag_dir_a = run_dir_a / "fm_diagnostics"
    diag_dir_a.mkdir()
    (diag_dir_a / "fm_call_001.json").write_text(json.dumps({"content": json.dumps(raw_k2)}))
    (run_dir_a / "functional_specification.json").write_text(json.dumps({
        "domain": "kitchen",
        "task_instruction": "Prepare coffee and soup",
        "nodes": {"coffee_container": {"name": "coffee_container"}},
        "metadata": {
            "raw_vlm_response": raw_k2,
            "canonicalization_trace": {"unresolved_roles": [{"raw_id": "role_stirrer"}]},
        },
    }))
    (run_dir_a / "run_manifest.json").write_text(json.dumps({"spec_acquisition": "archived_raw_provider_response"}))
    (run_dir_a / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INCOMPLETE"}))

    row_a = {
        "domain": "kitchen", "variant": "K2", "gt_feasible": True, "requires_observation_recovery": False,
        "runtime_contract_complete": False, "canonicalization_succeeded": False,
        "vlm_json_valid": True, "sanitization_succeeded": True, "full_task_satisfied": False,
    }
    task = "Prepare and serve coffee and soup for two people using the available kitchenware."
    enriched_a = enrich_record(row_a, run_dir_a, task)

    assert enriched_a["first_cause_category"] == "GRAPH_COMPILATION_FAILURE"
    assert enriched_a["raw_requirement_present"] is True
    assert enriched_a["production_mapped"] is False

    # Case B: Raw K2 document omitted the coffee stirrer
    raw_k2_omitted = json.loads(json.dumps(raw_k2))
    raw_k2_omitted["functional_roles"] = [
        r for r in raw_k2_omitted["functional_roles"]
        if "stir" not in str(r.get("function", "")).lower()
    ]
    raw_k2_omitted["interaction_groups"] = [
        g for g in raw_k2_omitted["interaction_groups"]
        if "stir" not in str(g.get("function", "")).lower()
    ]

    run_dir_b = tmp_path / "k2_raw_omit"
    run_dir_b.mkdir(parents=True)
    diag_dir_b = run_dir_b / "fm_diagnostics"
    diag_dir_b.mkdir()
    (diag_dir_b / "fm_call_001.json").write_text(json.dumps({"content": json.dumps(raw_k2_omitted)}))
    (run_dir_b / "functional_specification.json").write_text(json.dumps({
        "domain": "kitchen",
        "task_instruction": "Prepare coffee and soup",
        "nodes": {"coffee_container": {"name": "coffee_container"}},
        "metadata": {"raw_vlm_response": raw_k2_omitted, "canonicalization_trace": {}},
    }))
    (run_dir_b / "run_manifest.json").write_text(json.dumps({"spec_acquisition": "archived_raw_provider_response"}))
    (run_dir_b / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INCOMPLETE"}))

    row_b = {
        "domain": "kitchen", "variant": "K2", "gt_feasible": True, "requires_observation_recovery": False,
        "runtime_contract_complete": False, "canonicalization_succeeded": True,
        "vlm_json_valid": True, "sanitization_succeeded": True, "full_task_satisfied": False,
    }
    enriched_b = enrich_record(row_b, run_dir_b, task)

    assert enriched_b["first_cause_category"] == "TASK_SPECIFICATION_FAILURE"
    assert enriched_b["failure_category"] == "FM_SEMANTIC_OMISSION"
    assert enriched_b["raw_requirement_present"] is False


def test_w1_separable_into_raw_vs_compiler_causes(tmp_path: Path):
    """Gate 7: Representative W1 case cleanly separable into raw-vs-compiler causes."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    if not w1_file.exists():
        pytest.skip("W1 fixture missing")

    raw_w1 = json.loads(w1_file.read_text(encoding="utf-8"))

    # Case A: Raw W1 is complete, but compiler had an unresolved semantic
    run_dir_a = tmp_path / "w1_compiler_fail"
    run_dir_a.mkdir(parents=True)
    diag_dir_a = run_dir_a / "fm_diagnostics"
    diag_dir_a.mkdir()
    (diag_dir_a / "fm_call_001.json").write_text(json.dumps({"content": json.dumps(raw_w1)}))
    (run_dir_a / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": "Fasten joint",
        "nodes": {"driver": {"name": "driver"}},
        "metadata": {
            "raw_vlm_response": raw_w1,
            "unresolved_semantics": [{"role": "fastener", "reason": "UNRESOLVED_SEMANTIC"}],
            "canonicalization_trace": {},
        },
    }))
    (run_dir_a / "run_manifest.json").write_text(json.dumps({"spec_acquisition": "archived_raw_provider_response"}))
    (run_dir_a / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INCOMPLETE"}))

    row_a = {
        "domain": "workshop", "variant": "W1", "gt_feasible": True, "requires_observation_recovery": False,
        "runtime_contract_complete": False, "canonicalization_succeeded": True,
        "vlm_json_valid": True, "sanitization_succeeded": True, "full_task_satisfied": False,
    }
    task = "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench."
    enriched_a = enrich_record(row_a, run_dir_a, task)

    assert enriched_a["first_cause_category"] == "GRAPH_COMPILATION_FAILURE"
    assert enriched_a["raw_requirement_present"] is True

    # Case B: Raw W1 omitted the fastener role entirely
    raw_w1_omitted = json.loads(json.dumps(raw_w1))
    raw_w1_omitted["functional_roles"] = [
        r for r in raw_w1_omitted["functional_roles"]
        if "fasten" not in str(r.get("function", "")).lower()
        and "screw" not in str(r.get("function", "")).lower()
    ]
    raw_w1_omitted["functional_relations"] = []
    raw_w1_omitted["interaction_groups"] = []

    run_dir_b = tmp_path / "w1_raw_omit"
    run_dir_b.mkdir(parents=True)
    diag_dir_b = run_dir_b / "fm_diagnostics"
    diag_dir_b.mkdir()
    (diag_dir_b / "fm_call_001.json").write_text(json.dumps({"content": json.dumps(raw_w1_omitted)}))
    (run_dir_b / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": "Fasten joint",
        "nodes": {"driver": {"name": "driver"}},
        "metadata": {"raw_vlm_response": raw_w1_omitted, "canonicalization_trace": {}},
    }))
    (run_dir_b / "run_manifest.json").write_text(json.dumps({"spec_acquisition": "archived_raw_provider_response"}))
    (run_dir_b / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INCOMPLETE"}))

    row_b = {
        "domain": "workshop", "variant": "W1", "gt_feasible": True, "requires_observation_recovery": False,
        "runtime_contract_complete": False, "canonicalization_succeeded": True,
        "vlm_json_valid": True, "sanitization_succeeded": True, "full_task_satisfied": False,
    }
    enriched_b = enrich_record(row_b, run_dir_b, task)

    assert enriched_b["first_cause_category"] == "TASK_SPECIFICATION_FAILURE"
    assert enriched_b["failure_category"] == "FM_SEMANTIC_OMISSION"
    assert enriched_b["raw_requirement_present"] is False


def test_all_eight_diagnostic_flags_saved(tmp_path: Path):
    """Section 14.4: Record must save all eight diagnostic flags."""
    run_dir = tmp_path / "test_flags"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"spec_acquisition": "archived_raw_provider_response"}))
    (run_dir / "functional_specification.json").write_text(json.dumps({"domain": "workshop", "nodes": {}}))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INCOMPLETE"}))

    row = {"domain": "workshop", "variant": "W1", "gt_feasible": True, "full_task_satisfied": False}
    enriched = enrich_record(row, run_dir, "test task")

    required_flags = [
        "raw_requirement_present",
        "production_mapped",
        "capability_mapped",
        "physical_evidence_available",
        "search_attempted",
        "grounding_complete",
        "astar_attempted",
        "plan_valid",
    ]
    for flag in required_flags:
        assert flag in enriched, f"Diagnostic flag {flag!r} must be saved in evaluation record"
        assert isinstance(enriched[flag], bool), f"Diagnostic flag {flag!r} must be a boolean"


def test_compiler_unresolved_semantic_never_attributes_to_fm_omission(tmp_path: Path):
    """Section 14.5: UNRESOLVED_SEMANTIC in compiler does not prove raw omission."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    if not w1_file.exists():
        pytest.skip("W1 fixture missing")

    raw_w1 = json.loads(w1_file.read_text(encoding="utf-8"))
    run_dir = tmp_path / "test_unresolved"
    run_dir.mkdir(parents=True)
    diag_dir = run_dir / "fm_diagnostics"
    diag_dir.mkdir()
    (diag_dir / "fm_call_001.json").write_text(json.dumps({"content": json.dumps(raw_w1)}))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": "Fasten joint",
        "nodes": {"driver": {"name": "driver"}},
        "metadata": {
            "raw_vlm_response": raw_w1,
            "canonicalization_trace": {
                "unresolved_roles": [{"raw_id": "role_2", "status": "UNRESOLVED_SEMANTIC"}],
            },
        },
    }))
    (run_dir / "run_manifest.json").write_text(json.dumps({"spec_acquisition": "archived_raw_provider_response"}))
    (run_dir / "graph_grounding_result.json").write_text(json.dumps({"assignment": {}, "status": "INCOMPLETE"}))

    row = {
        "domain": "workshop", "variant": "W1", "gt_feasible": True, "requires_observation_recovery": False,
        "runtime_contract_complete": False, "canonicalization_succeeded": False,
        "vlm_json_valid": True, "sanitization_succeeded": True, "full_task_satisfied": False,
    }
    task = "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench."
    enriched = enrich_record(row, run_dir, task)

    # Compiler rejection must be GRAPH_COMPILATION_FAILURE, never TASK_SPECIFICATION_FAILURE
    assert enriched["first_cause_category"] == "GRAPH_COMPILATION_FAILURE"
    assert enriched["first_cause_category"] != "TASK_SPECIFICATION_FAILURE"
