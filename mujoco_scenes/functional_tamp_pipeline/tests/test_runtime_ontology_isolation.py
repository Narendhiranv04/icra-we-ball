"""Phase 1: Runtime Semantic Ontology Isolation and Information Firewall Tests.

Verifies:
1. Runtime functional semantic ontology loads all canonical domain categories.
2. Runtime ontology contains no benchmark answer graphs, task goals, or expected plans.
3. Role semantic ontology source code does not touch reference task configs.
4. Online VLM spec compilation never invokes or touches GTSpecProvider.
5. W1 and W2 regression fixtures canonicalize correctly under runtime-only ontology.
6. Manifest records the runtime ontology SHA-256 hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml

from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (
    clear_cached_ontology,
    get_all_system_role_semantic_categories,
    get_runtime_detector_aliases,
    get_runtime_semantic_ontology_hash,
    get_runtime_semantic_ontology_path,
    get_system_role_semantic_categories,
    build_task_detector_vocabulary,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ideal_raw_vlm"
BASELINE_DIR = ROOT / "benchmark_reports" / "live_final_evaluation_v2"


def test_runtime_ontology_loads_all_domain_categories():
    """All domain canonical roles and categories load strictly from the isolated runtime ontology."""
    clear_cached_ontology()

    # Kitchen canonical roles
    k_roles = get_all_system_role_semantic_categories("kitchen")
    for req in ("coffee_container", "soup_container", "coffee_stirrer", "soup_eating_utensil", "coffee_source", "water_source"):
        assert req in k_roles
        cats = get_system_role_semantic_categories("kitchen", req)
        assert len(cats) >= 1
    assert "cup" in get_system_role_semantic_categories("kitchen", "coffee_container")
    assert "bowl" in get_system_role_semantic_categories("kitchen", "soup_container")

    # Living room canonical roles
    l_roles = get_all_system_role_semantic_categories("living_room")
    for req in ("PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION", "CUP_SAUCER_SET", "REMOTE", "SEATING_POSITION", "SEATING_PAIR"):
        assert req in l_roles
        cats = get_system_role_semantic_categories("living_room", req)
        assert len(cats) >= 1
    assert "side_table" in get_system_role_semantic_categories("living_room", "PERSONAL_CUP_SAUCER_REGION")

    # Workshop canonical roles
    w_roles = get_all_system_role_semantic_categories("workshop")
    for req in ("driver", "fastener", "repair_target"):
        assert req in w_roles
        cats = get_system_role_semantic_categories("workshop", req)
        assert len(cats) >= 1
    assert "screwdriver" in get_system_role_semantic_categories("workshop", "driver")
    assert "screw" in get_system_role_semantic_categories("workshop", "fastener")
    assert "repair_target" in get_system_role_semantic_categories("workshop", "repair_target")

    # Detector aliases
    aliases = get_runtime_detector_aliases()
    assert len(aliases) >= 20
    assert "screwdriver" in aliases
    assert "screw" in aliases
    assert "bowl" in aliases

    # Hash
    h = get_runtime_semantic_ontology_hash()
    assert isinstance(h, str) and len(h) == 64


def test_runtime_ontology_contains_no_benchmark_answer_graph():
    """The runtime ontology configuration must not contain benchmark task goals, plans, or oracle graphs."""
    ont_path = get_runtime_semantic_ontology_path()
    assert ont_path.is_file(), f"Runtime ontology file missing at {ont_path}"
    raw_text = ont_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)

    # Must only have approved root keys
    allowed_root_keys = {"schema_version", "ontology_version", "domains", "detector_aliases", "canonical_labels"}
    assert set(data.keys()).issubset(allowed_root_keys)

    forbidden_tokens = [
        "task_goals",
        "task_instruction",
        "task_summary",
        "required_relations",
        "expected_relations",
        "interaction_groups",
        "operation_groups",
        "expected_plans",
        "solution_plan",
        "feasibility",
        "variant_id",
        "hidden_locations",
        "ground_truth",
        "oracle",
        "reference_graph",
    ]
    for tok in forbidden_tokens:
        assert tok not in data, f"Forbidden benchmark token {tok!r} found at root of runtime ontology"
        assert tok not in raw_text, f"Forbidden token {tok!r} found in runtime ontology file content"

    # Must not contain benchmark variant IDs like K1, W1, L1
    for variant in ("K1", "K2", "K3", "L1", "L2", "L3", "W1", "W2", "W3", "F0"):
        assert f": {variant}" not in raw_text
        assert f"'{variant}'" not in raw_text
        assert f'"{variant}"' not in raw_text


def test_runtime_ontology_loader_source_scan():
    """Verify that role_semantic_ontology.py source code never touches reference task YAML files."""
    code_path = ROOT / "mujoco_scenes" / "functional_tamp_pipeline" / "role_semantic_ontology.py"
    code = code_path.read_text(encoding="utf-8")

    forbidden_configs = [
        "s1_integrated_kitchen_object_function.yaml",
        "l2_integrated_region_function_task.yaml",
        "workshop_phase1_fm_contract.yaml",
        "kitchen_feasibility_variants.yaml",
        "living_room_variants.yaml",
        "workshop_variants.yaml",
    ]
    for cfg in forbidden_configs:
        assert cfg not in code, f"role_semantic_ontology.py must not reference {cfg}"

    assert "runtime_functional_semantic_ontology.yaml" in code


def test_online_vlm_path_gt_provider_leakage_guard():
    """Online VLM specification compiling must succeed with GTSpecProvider completely blocked."""
    provider = VLMSpecProvider()

    # Poison GTSpecProvider
    def poison_provide(*args, **kwargs):
        raise AssertionError("LEAKAGE: GTSpecProvider.provide was called during online execution!")

    def poison_init(*args, **kwargs):
        raise AssertionError("LEAKAGE: GTSpecProvider was instantiated during online execution!")

    with patch.object(GTSpecProvider, "provide", side_effect=poison_provide):
        with patch.object(GTSpecProvider, "__init__", side_effect=poison_init):
            # 1. Workshop W1
            w1_raw = json.loads((FIXTURE_DIR / "workshop_W1.json").read_text(encoding="utf-8"))
            g_w1 = provider.provide("workshop", "Repair the loose frame joint", raw_document=w1_raw)
            assert g_w1 is not None
            assert "driver" in g_w1.nodes
            assert "fastener" in g_w1.nodes

            # 2. Kitchen K1
            k1_raw = json.loads((FIXTURE_DIR / "kitchen_K1.json").read_text(encoding="utf-8"))
            g_k1 = provider.provide("kitchen", "Prepare coffee and soup", raw_document=k1_raw)
            assert g_k1 is not None
            assert "coffee_container" in g_k1.nodes

            # 3. Living Room L1
            l1_raw = json.loads((FIXTURE_DIR / "living_room_L1.json").read_text(encoding="utf-8"))
            g_l1 = provider.provide("living_room", "Serve refreshments and remote", raw_document=l1_raw)
            assert g_l1 is not None


def test_w1_w2_regression_fixtures_canonicalize_under_runtime_only_ontology():
    """Verify W1 and W2 regression fixtures canonicalize cleanly under the isolated runtime ontology."""
    provider = VLMSpecProvider()

    # W1 fixture
    w1_raw = json.loads((FIXTURE_DIR / "workshop_W1.json").read_text(encoding="utf-8"))
    g_w1 = provider.provide("workshop", "Repair the loose frame joint", raw_document=w1_raw)
    assert g_w1.nodes["driver"].semantic_categories == get_system_role_semantic_categories("workshop", "driver")
    assert g_w1.nodes["fastener"].semantic_categories == get_system_role_semantic_categories("workshop", "fastener")
    assert g_w1.nodes["repair_target"].semantic_categories == get_system_role_semantic_categories("workshop", "repair_target")

    # W2 baseline raw response
    w2_call_path = BASELINE_DIR / "workshop" / "W2" / "vlm" / "fm_diagnostics" / "fm_call_001.json"
    assert w2_call_path.is_file()
    w2_call_data = json.loads(w2_call_path.read_text(encoding="utf-8"))
    w2_raw = json.loads(w2_call_data["content"])
    g_w2 = provider.provide("workshop", "Complete fastening at workbench", raw_document=w2_raw)
    assert "driver" in g_w2.nodes
    assert "fastener" in g_w2.nodes
    assert g_w2.nodes["driver"].semantic_categories == get_system_role_semantic_categories("workshop", "driver")
    assert g_w2.nodes["fastener"].semantic_categories == get_system_role_semantic_categories("workshop", "fastener")


def test_run_manifest_records_runtime_ontology_hash(tmp_path):
    """Verify that run_manifest.json includes runtime_semantic_ontology_hash."""
    from mujoco_scenes.functional_tamp_pipeline.run import _RunState, _write_run_manifest

    state = _RunState(
        domain="kitchen",
        variant="K1",
        internal_variant="K1",
        mode="vlm",
        run_dir=tmp_path,
        search_order="auto",
    )
    _write_run_manifest(state)

    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "runtime_semantic_ontology_hash" in manifest
    assert manifest["runtime_semantic_ontology_hash"] == get_runtime_semantic_ontology_hash()
    assert len(manifest["runtime_semantic_ontology_hash"]) == 64
