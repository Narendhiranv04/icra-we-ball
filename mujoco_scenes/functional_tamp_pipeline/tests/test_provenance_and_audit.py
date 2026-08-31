"""Unit tests for Phase 3 provenance fingerprinting, resume semantics, and prompt leakage audit (P3-A.1)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.audit import (
    FORBIDDEN_CANONICAL_REGION_TOKENS,
    FORBIDDEN_CHECKER_STRINGS,
    FORBIDDEN_ORACLE_STRINGS,
    audit_prompt_leakage,
    compute_prompt_and_schema_hash,
    compute_provenance_fingerprint,
    get_git_info,
)
from scripts.run_phase36b2_matrix import clean_case_directory


def test_provenance_fingerprint_stability():
    """Verify that identical configuration produces identical SHA-256 fingerprint."""
    fp1 = compute_provenance_fingerprint(
        domain="kitchen",
        variant="K1",
        model="qwen35-9b",
        base_url="http://127.0.0.1:18000/v1",
        task_instruction="Prepare coffee and soup",
        search_order="auto",
    )
    fp2 = compute_provenance_fingerprint(
        domain="kitchen",
        variant="K1",
        model="qwen35-9b",
        base_url="http://127.0.0.1:18000/v1",
        task_instruction="Prepare coffee and soup",
        search_order="auto",
    )
    assert fp1["fingerprint_sha256"] == fp2["fingerprint_sha256"]
    assert len(fp1["fingerprint_sha256"]) == 64
    assert fp1["domain"] == "kitchen"
    assert fp1["variant"] == "K1"
    assert fp1["model_identifier"] == "qwen35-9b"


def test_provenance_fingerprint_sensitivity():
    """Verify that differing models, tasks, domains, or search orders produce different fingerprints."""
    base_fp = compute_provenance_fingerprint(
        domain="kitchen",
        variant="K1",
        model="qwen35-9b",
        task_instruction="Prepare coffee and soup",
    )

    diff_model_fp = compute_provenance_fingerprint(
        domain="kitchen",
        variant="K1",
        model="qwen35-27b",
        task_instruction="Prepare coffee and soup",
    )
    assert base_fp["fingerprint_sha256"] != diff_model_fp["fingerprint_sha256"]

    diff_task_fp = compute_provenance_fingerprint(
        domain="kitchen",
        variant="K1",
        model="qwen35-9b",
        task_instruction="Different task instruction",
    )
    assert base_fp["fingerprint_sha256"] != diff_task_fp["fingerprint_sha256"]

    diff_domain_fp = compute_provenance_fingerprint(
        domain="workshop",
        variant="W1",
        model="qwen35-9b",
        task_instruction="Prepare coffee and soup",
    )
    assert base_fp["fingerprint_sha256"] != diff_domain_fp["fingerprint_sha256"]


def test_audit_prompt_leakage_clean_request():
    """Verify that a clean semantic request passes the prompt leakage audit."""
    clean_payload = {
        "sanitized_request": {
            "messages": [
                {"role": "system", "content": "You are a vision-language functional specification generator."},
                {"role": "user", "content": "Find the screw and driver, and drive the screw into the hole."},
            ],
            "response_format": {
                "json_schema": {"name": "workshop_functional_graph"}
            },
        }
    }
    result = audit_prompt_leakage(clean_payload)
    assert result["audited"] is True
    assert result["zero_leakage"] is True
    assert result["oracle_symbols_in_prompt"] is False
    assert result["checker_predicates_in_prompt"] is False
    assert result["canonical_regions_in_prompt"] is False
    assert len(result["forbidden_checkers_found"]) == 0
    assert len(result["forbidden_regions_found"]) == 0
    assert len(result["forbidden_oracle_symbols_found"]) == 0


def test_audit_prompt_leakage_ignores_model_response_content():
    """Regression test (Issue 1): Model response containing forbidden tokens is NOT flagged as prompt leakage."""
    diagnostic_with_model_response = {
        "call_kind": "completion",
        "sanitized_request": {
            "messages": [
                {"role": "system", "content": "You are a functional planner."},
                {"role": "user", "content": "Stir the coffee cup with a spoon."},
            ]
        },
        # The model independently output INSERTABLE_IN and D1 in its response content
        "content": json.dumps({
            "functional_roles": [{"name": "stirrer", "unary": "ELONGATED_OBJECT"}],
            "relations": [{"relation": "INSERTABLE_IN", "region": "D1"}],
        }),
        "raw_response": {"choices": [{"message": {"content": "INSERTABLE_IN D1"}}]},
    }
    result = audit_prompt_leakage(diagnostic_with_model_response)
    assert result["audited"] is True
    assert result["zero_leakage"] is True
    assert len(result["forbidden_checkers_found"]) == 0
    assert len(result["forbidden_regions_found"]) == 0


def test_audit_prompt_leakage_catches_forbidden_content_in_request():
    """Regression test (Issue 1): Forbidden tokens inside outgoing request ARE caught."""
    leaky_request = {
        "sanitized_request": {
            "messages": [
                {"role": "system", "content": "Ensure you verify OPEN_CAVITY and search in D1."},
                {"role": "user", "content": "Prepare coffee."},
            ]
        },
        "content": "clean content",
    }
    result = audit_prompt_leakage(leaky_request)
    assert result["audited"] is True
    assert result["zero_leakage"] is False
    assert "OPEN_CAVITY" in result["forbidden_checkers_found"]
    assert "D1" in result["forbidden_regions_found"]


def test_audit_prompt_leakage_response_only_does_not_fabricate_pass():
    """Verify that auditing an object with zero request data does not falsely claim zero_leakage=True."""
    response_only = {
        "content": "some model output",
        "choices": [{"message": {"content": "some output"}}],
    }
    result = audit_prompt_leakage(response_only)
    assert result["audited"] is False
    assert result["zero_leakage"] is None
    assert result["audit_status"] == "SKIPPED_NO_REQUEST_PAYLOAD"


def test_audit_prompt_leakage_forbidden_oracle_symbols():
    """Verify that leaking ground truth oracle symbol names in request is caught."""
    for oracle in ("GTSpecProvider", "KitchenGroundTruth", "LivingRoomRegionOracle"):
        leaky_payload = {
            "sanitized_request": {
                "prompt": f"Loaded context from {oracle}",
            }
        }
        result = audit_prompt_leakage(leaky_payload)
        assert result["zero_leakage"] is False
        assert result["oracle_symbols_in_prompt"] is True
        assert oracle in result["forbidden_oracle_symbols_found"]


def test_stale_resume_fingerprint_matching(tmp_path: Path):
    """Verify resume logic correctly distinguishes matching from mismatched fingerprints."""
    domain, variant = "kitchen", "K1"
    curr_fp = compute_provenance_fingerprint(
        domain=domain,
        variant=variant,
        model="qwen35-9b",
        task_instruction="Prepare coffee and soup",
    )

    # Saved manifest with matching fingerprint
    matching_manifest = {
        "case_id": f"{domain}_{variant}",
        "provenance_fingerprint": curr_fp,
        "live_terminal_status": "ACTION_SEQUENCE_READY",
        "runtime_seconds": 12.34,
    }
    match_file = tmp_path / "case_manifest_match.json"
    match_file.write_text(json.dumps(matching_manifest), encoding="utf-8")

    loaded_match = json.loads(match_file.read_text(encoding="utf-8"))
    assert (
        loaded_match.get("provenance_fingerprint", {}).get("fingerprint_sha256")
        == curr_fp["fingerprint_sha256"]
    )

    # Saved manifest with outdated/mismatched fingerprint (e.g. different model)
    old_fp = compute_provenance_fingerprint(
        domain=domain,
        variant=variant,
        model="older-model-7b",
        task_instruction="Prepare coffee and soup",
    )
    stale_manifest = {
        "case_id": f"{domain}_{variant}",
        "provenance_fingerprint": old_fp,
        "live_terminal_status": "INFEASIBLE",
        "runtime_seconds": 9.99,
    }
    stale_file = tmp_path / "case_manifest_stale.json"
    stale_file.write_text(json.dumps(stale_manifest), encoding="utf-8")

    loaded_stale = json.loads(stale_file.read_text(encoding="utf-8"))
    assert (
        loaded_stale.get("provenance_fingerprint", {}).get("fingerprint_sha256")
        != curr_fp["fingerprint_sha256"]
    )


def test_clean_case_directory_removes_stale_artifacts(tmp_path: Path):
    """Test (Issue 3): clean_case_directory wipes all stale files before a fresh run."""
    case_dir = tmp_path / "kitchen_K1"
    case_dir.mkdir(parents=True)
    (case_dir / "case_manifest.json").write_text("old manifest")
    fm_dir = case_dir / "fm_diagnostics"
    fm_dir.mkdir()
    (fm_dir / "fm_call_001.json").write_text("old diag 1")
    (fm_dir / "fm_call_002.json").write_text("old diag 2")
    (case_dir / "combined.log").write_text("old log")

    assert (fm_dir / "fm_call_001.json").is_file()
    assert (case_dir / "case_manifest.json").is_file()

    clean_case_directory(case_dir)

    assert case_dir.exists()
    assert list(case_dir.iterdir()) == []
    assert not (case_dir / "case_manifest.json").exists()
    assert not (case_dir / "fm_diagnostics").exists()
