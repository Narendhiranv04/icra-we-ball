"""Pass P3-I.1: Semantic Authority and Deterministic Convergence Verification.

Verifies:
1. One shared system-owned role semantic acceptance authority across all domains.
2. Candidate category mutation leaves role.semantic_categories unchanged (B1).
3. Candidate category reordering leaves role.semantic_categories and phi* unchanged (B2).
4. Kitchen task-scoped YOLO vocabulary construction excluding global labels (C1-C4).
5. Unresolved FM detector terms do not manufacture grounding acceptance (C3).
6. Workshop reviewed ontology alias preservation (D).
7. Truthful provenance and version tracking (E).
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import yaml

import pytest

from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph
from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (
    PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION,
    build_task_detector_vocabulary,
    get_all_system_role_semantic_categories,
    get_system_role_semantic_categories,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode, ObservedSceneGraph
from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"


@pytest.fixture(scope="session", autouse=True)
def ensure_dummy_image(tmp_path_factory):
    p = Path("/tmp/dummy_obs.png")
    p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
    return p


class MockFMAdapter:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = deepcopy(document)
        self.last_raw_response = deepcopy(document)
        self.last_raw_requirement_response = deepcopy(document)
        self.last_raw_kitchen_graph_response = deepcopy(document)
        self.last_validated_kitchen_graph_response = deepcopy(document)
        self.raw_decomposition = deepcopy(document)
        self.raw_vlm_response = deepcopy(document)
        self.validated_vlm_specification = deepcopy(document)

    def generate_task_requirements(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return deepcopy(self.document)

    def generate_kitchen_functional_graph(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return deepcopy(self.document)


def load_fixture(domain: str) -> dict[str, Any]:
    file_map = {
        "kitchen": "kitchen_K1.json",
        "living_room": "living_room_L1.json",
        "workshop": "workshop_W1.json",
    }
    return json.loads((FIXTURES_DIR / file_map[domain]).read_text(encoding="utf-8"))


# ==============================================================================
# 1. SHARED SYSTEM SEMANTIC AUTHORITY & NO PRIVATE VLM TABLES
# ==============================================================================

def test_system_role_semantic_ontology_authority():
    """Verify system-owned role semantic ontology is the single authority consumed by GT and VLM."""
    import mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider as vlm_mod

    # 1. Verify private VLM tables are deleted
    assert not hasattr(vlm_mod, "KITCHEN_ROLE_CANONICAL_CATEGORIES")
    assert not hasattr(vlm_mod, "WORKSHOP_ROLE_CANONICAL_CATEGORIES")

    # 2. Verify all domains have reviewed categories
    for domain in ("workshop", "kitchen", "living_room"):
        cats = get_all_system_role_semantic_categories(domain)
        assert len(cats) > 0

    # 3. Verify Workshop GT and VLM role categories match system ontology exactly
    gt = GTSpecProvider()
    vlm = VLMSpecProvider()

    w_inst = "Fasten the frame joint on the workpiece using a compatible screw and a driver from the workshop storage."
    gt_w = gt.provide("workshop", w_inst)
    w_fixture = load_fixture("workshop")
    adapter_w = MockFMAdapter(w_fixture)
    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_w),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_w),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_w):
        vlm_w = vlm.provide("workshop", w_inst, observation_images=[Path("/tmp/dummy_obs.png")])

    for role_id in ("driver", "fastener", "repair_target"):
        expected_cats = get_system_role_semantic_categories("workshop", role_id)
        assert gt_w.nodes[role_id].semantic_categories == expected_cats
        assert vlm_w.nodes[role_id].semantic_categories == expected_cats

    # 4. Verify Kitchen GT and VLM role categories match system ontology
    k_inst = "Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil. Search the closed kitchen storage for anything still required."
    gt_k = gt.provide("kitchen", k_inst)
    k_fixture = load_fixture("kitchen")
    adapter_k = MockFMAdapter(k_fixture)
    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_k),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_k),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_k):
        vlm_k = vlm.provide("kitchen", k_inst, observation_images=[Path("/tmp/dummy_obs.png")])

    for role_id in ("coffee_container", "soup_container", "coffee_stirrer", "soup_eating_utensil", "coffee_source", "water_source"):
        expected_cats = get_system_role_semantic_categories("kitchen", role_id)
        assert gt_k.nodes[role_id].semantic_categories == expected_cats
        assert vlm_k.nodes[role_id].semantic_categories == expected_cats


# ==============================================================================
# 2. B1 — CANDIDATE CATEGORY MUTATION TEST
# ==============================================================================

def test_b1_candidate_category_mutation_workshop():
    """Workshop B1: Mutating candidate categories does NOT change role semantic categories or canonical role ID."""
    vlm = VLMSpecProvider()
    inst = "Fasten the frame joint on the workpiece using a compatible screw and a driver from the workshop storage."

    fixture_a = load_fixture("workshop")
    fixture_b = deepcopy(fixture_a)

    # In fixture B, mutate driver and fastener candidate_categories
    for req in fixture_b["functional_roles"]:
        if req.get("id") == "role_1":
            req["candidate_categories"] = ["cordless torque tool", "mystery driver phrase"]
        elif req.get("id") == "role_2":
            req["candidate_categories"] = ["novel threaded fastener", "micro bolt"]

    adapter_a = MockFMAdapter(fixture_a)
    adapter_b = MockFMAdapter(fixture_b)

    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_a),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_a),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_a):
        gf_a = vlm.provide("workshop", inst, observation_images=[Path("/tmp/dummy_obs.png")])

    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_b),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_b),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_b):
        gf_b = vlm.provide("workshop", inst, observation_images=[Path("/tmp/dummy_obs.png")])

    # Assert canonical role IDs identical
    assert set(gf_a.nodes.keys()) == set(gf_b.nodes.keys())

    # Assert role.semantic_categories IDENTICAL (authoritative system semantics)
    for role_id in ("driver", "fastener", "repair_target"):
        assert gf_a.nodes[role_id].semantic_categories == gf_b.nodes[role_id].semantic_categories
        assert gf_a.nodes[role_id].verification_mode == gf_b.nodes[role_id].verification_mode

    # Assert functional relations and operation groups identical
    assert len(gf_a.relations) == len(gf_b.relations)
    assert len(gf_a.operation_groups) == len(gf_b.operation_groups)

    # But detector vocabulary prompts differ as intended
    assert gf_a.detector_vocabulary != gf_b.detector_vocabulary
    assert "cordless torque tool" in gf_b.detector_vocabulary


def test_b1_candidate_category_mutation_kitchen():
    """Kitchen B1: Mutating candidate categories does NOT change role semantic categories."""
    vlm = VLMSpecProvider()
    inst = "Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil. Search the closed kitchen storage for anything still required."

    fixture_a = load_fixture("kitchen")
    fixture_b = deepcopy(fixture_a)

    # Mutate candidate categories in fixture B
    for req in fixture_b["functional_roles"]:
        if req.get("id") == "role_1":  # coffee_container
            req["candidate_categories"] = ["fancy ceramic mug", "thermal drink holder"]

    adapter_a = MockFMAdapter(fixture_a)
    adapter_b = MockFMAdapter(fixture_b)

    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_a),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_a),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_a):
        gf_a = vlm.provide("kitchen", inst, observation_images=[Path("/tmp/dummy_obs.png")])

    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_b),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_b),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_b):
        gf_b = vlm.provide("kitchen", inst, observation_images=[Path("/tmp/dummy_obs.png")])

    assert gf_a.nodes["coffee_container"].semantic_categories == gf_b.nodes["coffee_container"].semantic_categories
    assert gf_a.nodes["coffee_container"].semantic_categories == ("cup", "mug")


# ==============================================================================
# 3. B2 — CANDIDATE CATEGORY ORDER INVARIANCE TEST
# ==============================================================================

def test_b2_candidate_category_order_invariance_workshop():
    """Workshop B2: Reordering raw candidate categories does NOT change role semantic categories or phi*."""
    vlm = VLMSpecProvider()
    inst = "Fasten the frame joint on the workpiece using a compatible screw and a driver from the workshop storage."

    fixture_a = load_fixture("workshop")
    fixture_b = deepcopy(fixture_a)

    # Reverse candidate categories order
    for req in fixture_b["functional_roles"]:
        if "candidate_categories" in req:
            req["candidate_categories"] = list(reversed(req["candidate_categories"]))

    adapter_a = MockFMAdapter(fixture_a)
    adapter_b = MockFMAdapter(fixture_b)

    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_a),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_a),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_a):
        gf_a = vlm.provide("workshop", inst, observation_images=[Path("/tmp/dummy_obs.png")])

    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_b),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_b),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_b):
        gf_b = vlm.provide("workshop", inst, observation_images=[Path("/tmp/dummy_obs.png")])

    # Semantic categories are identical
    for role_id in gf_a.nodes:
        assert gf_a.nodes[role_id].semantic_categories == gf_b.nodes[role_id].semantic_categories

    # Create a synthetic fixed G_O to test grounding invariance
    go = ObservedSceneGraph()
    go.add_node(ObservedNode(
        instance_id="d1",
        canonical_category="screwdriver",
        geometry={"tool_length_m": 0.15, "bit_type": "phillips"},
    ))
    go.add_node(ObservedNode(
        instance_id="f1",
        canonical_category="screw",
        geometry={"thread_length_m": 0.03, "head_type": "phillips", "outer_diameter_m": 0.004},
    ))
    go.add_node(ObservedNode(
        instance_id="repair_target",
        entity_kind="FIXED_TARGET",
        canonical_category="repair_target",
        geometry={"hole_depth_m": 0.035, "hole_diameter_m": 0.005},
    ))
    from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedRelation
    go.add_relation(ObservedRelation(subject_id="d1", predicate="COMPATIBLE_WITH", object_id="f1", status="TRUE"))
    go.add_relation(ObservedRelation(subject_id="d1", predicate="REACHES_TARGET", object_id="repair_target", status="TRUE"))
    go.add_relation(ObservedRelation(subject_id="f1", predicate="COMPATIBLE_WITH_TARGET", object_id="repair_target", status="TRUE"))

    res_a = ground_graph(gf_a, go, {"search_exhausted": False})
    res_b = ground_graph(gf_b, go, {"search_exhausted": False})

    assert res_a.complete is True
    assert res_b.complete is True
    assert res_a.assignment == res_b.assignment


# ==============================================================================
# 4. C1-C4 — TASK-SCORED DETECTOR VOCABULARY IN KITCHEN
# ==============================================================================

def test_c4_kitchen_task_scoped_detector_vocabulary():
    """Kitchen C4: Task detector vocabulary contains relevant concepts, excludes unrelated global labels."""
    root = Path(__file__).resolve().parents[3]
    base_vocab_path = root / "mujoco_scenes" / "configs" / "semantic_vocabulary.yaml"
    base_vocab = yaml.safe_load(base_vocab_path.read_text(encoding="utf-8"))

    system_cats = {"cup", "mug", "bowl", "spoon", "coffee_source", "kettle"}
    raw_candidates = ["cup", "coffee cup", "bowl", "soup bowl", "spoon", "stirrer", "coffee jar", "kettle"]

    task_vocab = build_task_detector_vocabulary(
        system_role_categories=system_cats,
        raw_vlm_candidate_categories=raw_candidates,
        base_semantic_ontology=base_vocab,
    )

    # 1. Relevant concepts MUST be present
    assert "cup" in task_vocab
    assert "mug" in task_vocab["cup"]
    assert "bowl" in task_vocab
    assert "spoon" in task_vocab
    assert "coffee_source" in task_vocab
    assert "kettle" in task_vocab

    # 2. Unrelated global concepts MUST be ABSENT
    unrelated_labels = ["remote_control", "book", "coaster", "game_controller", "duster", "spatula", "fork", "pen", "marker"]
    for unk in unrelated_labels:
        assert unk not in task_vocab, f"Unrelated label {unk!r} leaked into task vocabulary"
        for aliases in task_vocab.values():
            assert unk not in aliases


def test_c3_unresolved_fm_detector_terms_policy():
    """Kitchen C3: Unresolved detector prompt is included in detector vocab but does NOT widen semantic acceptance."""
    root = Path(__file__).resolve().parents[3]
    base_vocab_path = root / "mujoco_scenes" / "configs" / "semantic_vocabulary.yaml"
    base_vocab = yaml.safe_load(base_vocab_path.read_text(encoding="utf-8"))

    system_cats = {"cup", "mug"}
    raw_candidates = ["ceramic drinking vessel"]  # novel unmapped term

    task_vocab = build_task_detector_vocabulary(
        system_role_categories=system_cats,
        raw_vlm_candidate_categories=raw_candidates,
        base_semantic_ontology=base_vocab,
    )

    # Prompt is retained in detector vocabulary
    assert "ceramic drinking vessel" in task_vocab

    # But system role semantic categories remain strict
    accepted_cats = get_system_role_semantic_categories("kitchen", "coffee_container")
    assert "ceramic drinking vessel" not in accepted_cats


# ==============================================================================
# 5. D — WORKSHOP ALIAS PRESERVATION
# ==============================================================================

def test_d_workshop_ontology_alias_preservation():
    """Workshop D: Reviewed aliases like Phillips screw -> screw remain authoritative and are not overwritten."""
    w_fixture = load_fixture("workshop")
    adapter_w = MockFMAdapter(w_fixture)
    provider = FMRequirementProvider(fm_adapter=adapter_w)
    provider.get_requirements("Fasten...", observation_images=[Path("/tmp/dummy_obs.png")])
    detector_map = provider.get_detector_label_to_canonical_map()
    alias_map = provider.get_alias_to_canonical_map()

    # Phillips screw must map to canonical "screw"
    assert detector_map.get("phillips screw") == "screw"
    assert alias_map.get("phillips screw") == "screw"

    # Phillips screwdriver must map to canonical "screwdriver"
    assert detector_map.get("phillips screwdriver") == "screwdriver"
    assert alias_map.get("phillips screwdriver") == "screwdriver"


# ==============================================================================
# 6. E — PROVENANCE AND METADATA TRACE
# ==============================================================================

def test_provenance_and_metadata():
    """Provenance E: Verify semantic ontology version and explicit trace keys in G_F metadata."""
    vlm = VLMSpecProvider()
    gt = GTSpecProvider()

    w_inst = "Fasten the frame joint on the workpiece using a compatible screw and a driver from the workshop storage."
    k_inst = "Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil. Search the closed kitchen storage for anything still required."

    gf_gt = gt.provide("workshop", w_inst)
    assert gf_gt.metadata["role_semantic_ontology_version"] == PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION
    assert gf_gt.metadata["semantic_acceptance_source"] == "SYSTEM_ROLE_SEMANTIC_ONTOLOGY"

    w_fixture = load_fixture("workshop")
    adapter_w = MockFMAdapter(w_fixture)
    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter_w),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter_w),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter_w):
        gf_vlm_w = vlm.provide("workshop", w_inst, observation_images=[Path("/tmp/dummy_obs.png")])

    assert gf_vlm_w.metadata["role_semantic_ontology_version"] == PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION
    assert gf_vlm_w.metadata["semantic_acceptance_source"] == "SYSTEM_ROLE_SEMANTIC_ONTOLOGY"
    assert gf_vlm_w.metadata["detector_vocabulary_source"] == "VLM_CANDIDATES_PLUS_RELEVANT_SYSTEM_ALIASES"
    assert gf_vlm_w.metadata["candidate_categories_used_for_role_identity"] is False
    assert gf_vlm_w.metadata["candidate_categories_used_for_grounding_acceptance"] is False
    assert gf_vlm_w.metadata["candidate_categories_used_for_detector_vocabulary"] is True
