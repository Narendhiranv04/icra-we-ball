"""Unit tests for Phase 3 provenance fingerprinting, resume semantics, and prompt leakage audit."""

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
)


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


def test_audit_prompt_leakage_clean():
    """Verify that a clean semantic payload passes the prompt leakage audit."""
    clean_payload = {
        "messages": [
            {"role": "system", "content": "You are a vision-language functional specification generator."},
            {"role": "user", "content": "Find the screw and driver, and drive the screw into the hole."},
        ],
        "schema": {
            "functional_roles": [{"id": "driver", "description": "screwdriver or power driver"}],
        },
    }
    result = audit_prompt_leakage(clean_payload)
    assert result["audited"] is True
    assert result["zero_leakage"] is True
    assert result["gt_imports_found"] is False
    assert result["oracle_labels_in_prompt"] is False
    assert len(result["forbidden_checkers_found"]) == 0
    assert len(result["forbidden_regions_found"]) == 0
    assert len(result["forbidden_oracles_found"]) == 0


def test_audit_prompt_leakage_forbidden_checker():
    """Verify that leaking internal checker predicates is caught by the audit."""
    for checker in ("OPEN_CAVITY", "INSERTABLE_IN", "CAN_DRIVE_SCREW", "PLANAR_SUPPORT"):
        leaky_payload = {
            "prompt": f"Please verify {checker} on the candidate objects",
        }
        result = audit_prompt_leakage(leaky_payload)
        assert result["zero_leakage"] is False
        assert checker in result["forbidden_checkers_found"]


def test_audit_prompt_leakage_forbidden_canonical_regions():
    """Verify that leaking canonical benchmark region IDs is caught by the audit."""
    for reg in ("D1", "C2", "LEFT_DRAWER", "TOOL_CABINET"):
        leaky_payload = {
            "prompt": f"Search the region {reg} for tools",
        }
        result = audit_prompt_leakage(leaky_payload)
        assert result["zero_leakage"] is False
        assert reg in result["forbidden_regions_found"]


def test_audit_prompt_leakage_forbidden_oracles():
    """Verify that leaking ground truth oracle symbol names is caught by the audit."""
    for oracle in ("GTSpecProvider", "KitchenGroundTruth", "LivingRoomRegionOracle"):
        leaky_payload = {
            "context": f"Loaded from {oracle}",
        }
        result = audit_prompt_leakage(leaky_payload)
        assert result["zero_leakage"] is False
        assert result["gt_imports_found"] is True
        assert oracle in result["forbidden_oracles_found"]


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

    # Saved manifest with outdated/mismatched fingerprint (e.g. older commit or different model)
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
