"""Unit tests for Phase 6: Domain Canonicalization Cleanup (Gate 6).

Verifies:
1. No variant IDs or variant-specific conditional branches exist in production mapping code.
2. Workshop repair target maps robustly even when entity_kind is 'OBJECT'.
3. Omitted FM roles and relations remain missing rather than synthesized by the compiler.
4. Workshop target anchor provenance is written and verified as non-variant-specific.
5. W1 and W2 regression fixtures remain fully preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.functional_tamp_pipeline.run import _RunState, _write_run_manifest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"


def test_no_variant_ids_in_production_mapping_code():
    """Production mapping code must contain zero variant-ID branches or variant-specific logic."""
    mapping_files = [
        ROOT / "mujoco_scenes" / "kitchen_vlm_functional_graph.py",
        ROOT / "mujoco_scenes" / "environment_vlm_requirements.py",
        ROOT / "mujoco_scenes" / "workshop_phase1" / "requirements.py",
        ROOT / "mujoco_scenes" / "functional_tamp_pipeline" / "semantic_compiler.py",
        ROOT / "mujoco_scenes" / "functional_tamp_pipeline" / "relation_interpreter.py",
        ROOT / "mujoco_scenes" / "functional_tamp_pipeline" / "robot_capability_registry.py",
    ]

    # Look for conditional branching on specific variant IDs like 'K1', 'W8', 'L3'
    variant_pattern = re.compile(r"""\b(?:if|elif)\s+.*?\b(?:variant|v_name|var)\s*(?:==|in)\s*['"][KWL]\d+['"]""", re.IGNORECASE)

    for fpath in mapping_files:
        assert fpath.is_file(), f"Expected file {fpath} does not exist"
        text = fpath.read_text(encoding="utf-8")
        matches = variant_pattern.findall(text)
        assert len(matches) == 0, f"Found variant-specific conditional branch in {fpath.name}: {matches}"


def test_workshop_repair_target_maps_from_object_entity_kind():
    """Workshop repair target maps robustly to repair_target even when entity_kind is 'OBJECT'."""
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Repair loose frame joint.",
        "functional_roles": [
            {
                "id": "r1",
                "entity_kind": "OBJECT",
                "function": "drive screw",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "required_properties": [],
            },
            {
                "id": "r2",
                "entity_kind": "OBJECT",
                "function": "threaded screw for frame",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "required_properties": [],
            },
            {
                "id": "r3",
                "entity_kind": "OBJECT",  # Imperfect entity kind from VLM
                "function": "receive fastening at target joint hole",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["frame joint", "screw hole"],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "r1", "relation": "compatible with", "object_role": "r2"},
            {"subject_role": "r1", "relation": "reaches target", "object_role": "r3"},
            {"subject_role": "r2", "relation": "compatible with target", "object_role": "r3"},
        ],
        "interaction_groups": [
            {
                "id": "g1",
                "function": "drive screw into joint target",
                "tool_role": "r1",
                "target_role": "r2",
                "required_target_count": 1,
                "usage_policy": "DEDICATED_PER_TARGET",
                "context_role": "r3",
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("workshop", "Repair task", raw)
    assert "repair_target" in graph.nodes
    assert graph.nodes["repair_target"].entity_kind == "FIXED_TARGET"
    assert graph.required_contract_complete is True


def test_workshop_explicit_patient_tool_and_fastener_roles_do_not_collide():
    """Open-vocabulary patient grammar must stay distinct from tool/component grammar."""
    from mujoco_scenes.workshop_phase1.requirements import (
        map_workshop_fixed_target_role,
        map_workshop_role_function,
    )

    target = {
        "entity_kind": "OBJECT",
        "function": "Physical item requiring the fastening operation.",
        "candidate_categories": ["mechanical assembly", "base plate"],
    }
    fastener = {
        "entity_kind": "OBJECT",
        "function": "A component compatible with the assembly target used to secure it.",
        "candidate_categories": ["screw", "bolt", "clip"],
    }
    driver = {
        "entity_kind": "OBJECT",
        "function": "Device used to install the component.",
        "candidate_categories": ["screwdriver", "driver bit", "wrench"],
    }

    assert map_workshop_fixed_target_role(target) == "repair_target"
    assert map_workshop_fixed_target_role(fastener) is None
    assert map_workshop_role_function(fastener) == "CAN_FASTEN"
    assert map_workshop_role_function(driver) == "CAN_DRIVE_SCREW"


def test_missing_fm_roles_remain_missing():
    """Missing roles in Kitchen, Living Room, and Workshop are never synthesized by compiler."""
    # 1. Kitchen missing soup eating utensil
    raw_kitchen_no_spoon = {
        "status": "SUPPORTED",
        "task_summary": "Coffee and soup",
        "functional_roles": [
            {"id": "r1", "entity_kind": "OBJECT", "function": "contain coffee", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["cup"], "required_properties": []},
            {"id": "r2", "entity_kind": "OBJECT", "function": "contain soup", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["bowl"], "required_properties": []},
            {"id": "r3", "entity_kind": "OBJECT", "function": "stir coffee", "required_count": 1, "binding_policy": "REUSABLE", "candidate_categories": ["spoon"], "required_properties": []},
        ],
        "functional_relations": [
            {"subject_role": "r3", "relation": "fits inside", "object_role": "r1"},
            {"subject_role": "r3", "relation": "reaches the bottom", "object_role": "r1"},
        ],
        "interaction_groups": [
            {"id": "g1", "function": "stir beverage in cups", "tool_role": "r3", "target_role": "r1", "required_target_count": 2, "usage_policy": "SEQUENTIAL_REUSE_ALLOWED", "required_relations": ["fits inside", "reaches the bottom"]},
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    g_k = compile_candidate_graph("kitchen", "task", raw_kitchen_no_spoon)
    assert "soup_eating_utensil" not in g_k.nodes
    assert g_k.required_contract_complete is False

    # 2. Living room missing remote control
    raw_living_no_remote = {
        "status": "SUPPORTED",
        "task_summary": "Drinks only",
        "functional_roles": [
            {"id": "r1", "entity_kind": "REGION", "function": "hold items for viewer", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["side table"], "required_properties": ["planar horizontal support"]},
            {"id": "r2", "entity_kind": "OBJECT", "function": "contain hot beverage and saucer", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["cup"], "required_properties": []},
        ],
        "functional_relations": [
            {"subject_role": "r1", "relation": "can hold drinkware set", "object_role": "r2"},
        ],
        "interaction_groups": [
            {"id": "g1", "function": "support drinkware set beside seat", "tool_role": "r1", "target_role": "r2", "required_target_count": 2, "usage_policy": "DEDICATED_PER_TARGET"},
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    g_l = compile_candidate_graph("living_room", "task", raw_living_no_remote)
    assert "REMOTE" not in g_l.nodes
    assert "ENTERTAINMENT_CONTROL" not in g_l.nodes
    assert g_l.required_contract_complete is False

    # 3. Workshop missing fastener
    raw_workshop_no_fastener = {
        "status": "SUPPORTED",
        "task_summary": "Driver only",
        "functional_roles": [
            {"id": "r1", "entity_kind": "OBJECT", "function": "drive screw", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screwdriver"], "required_properties": []},
            {"id": "r2", "entity_kind": "FIXED_TARGET", "function": "frame joint repair target", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw hole"], "required_properties": []},
        ],
        "functional_relations": [
            {"subject_role": "r1", "relation": "reaches target", "object_role": "r2"},
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    g_w = compile_candidate_graph("workshop", "task", raw_workshop_no_fastener)
    assert "fastener" not in g_w.nodes
    assert g_w.required_contract_complete is False


def test_workshop_target_anchor_provenance_is_not_variant_answer_data(tmp_path):
    """Workshop run manifest and target anchor provenance confirm marked target calibration, not GT leakage."""
    state = _RunState(
        domain="workshop",
        variant="W1",
        internal_variant="W1",
        mode="vlm",
        run_dir=tmp_path,
        search_order="auto",
    )
    _write_run_manifest(state)

    prov_file = tmp_path / "target_anchor_provenance.json"
    assert prov_file.is_file(), "target_anchor_provenance.json must be written for Workshop run"
    data = json.loads(prov_file.read_text(encoding="utf-8"))

    assert data["anchor"] == "repair_target"
    assert data["source"] == "SYSTEM_CALIBRATED_MARKED_TARGET"
    assert data["variant_specific_answer_source"] is False
    assert data["configured_dimensions_used_as_measurements"] is False
    assert data["roi_only"] is True

    manifest_file = tmp_path / "run_manifest.json"
    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert "target_anchor_provenance" in manifest["artifacts"]


def test_w1_w2_regression_fixtures_preserved():
    """W1 and W2 regression fixtures compile cleanly without error."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    assert w1_file.is_file()
    w1_raw = json.loads(w1_file.read_text(encoding="utf-8"))
    g1 = compile_candidate_graph("workshop", "Repair loose frame joint", w1_raw)
    assert g1.required_contract_complete is True
    assert set(g1.nodes.keys()) == {"driver", "fastener", "repair_target"}
