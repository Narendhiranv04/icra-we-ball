"""Master Pass Stage B (P3-I/P3-I.1): Full Ideal Fixture Pipeline Convergence.

Verifies end-to-end execution of K1, L1, and W1 across 3 deterministic runs each:
1. Production pipeline execution using ideal raw VLM fixtures.
2. 100% COMPLETE status (ACTION_SEQUENCE_READY).
3. Multi-run determinism (identical structural G_F, search contract, final G_O, phi*, and plan hashes).
4. Plan grounding audit and execution validity (F10).
5. Replay validation using the saved specification JSON with 0 FM calls (F14).
6. Exact one-shot provider calls (F1).
7. Same-G_O GT control validation (F11).
8. Frozen search contract integrity and run manifest validation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    SearchRegionContractError,
    UnmappedFunctionalConceptError,
)
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import evaluate_gf_against_reference
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.audit import audit_plan_grounding
from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph
from mujoco_scenes.functional_tamp_pipeline.search_contract import (
    PHASE3_SEARCH_REGION_POLICY_VERSION,
    SearchRegionContract,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.workshop_phase1.fm_adapter import FMCallMetrics

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"

EXPECTED_FIXTURE_SHA256 = {
    "kitchen": "8aa50952216fd01270b95a4a5fa22f7206cf648bd81fa1054f6b64229cafadfe",
    "living_room": "a72d86cf1e054f6d7a9533be6e04b5ad6eeb5400190d6f52abb75447d29c30a4",
    "workshop": "42e8c9215ec7f5f946c050ef65c978c4edb9c8efcf524c0fc6095d6fb01eba72",
}

TASK_INSTRUCTIONS = {
    "kitchen": "Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil. Search the closed kitchen storage for anything still required.",
    "living_room": "Organize the living room so each seating position has an individual drinkware support and both seats share an accessible remote control support.",
    "workshop": "Fasten the frame joint on the workpiece using a compatible screw and a driver from the workshop storage.",
}


class MockFMAdapter:
    """Offline test adapter serving ideal raw VLM documents."""

    def __init__(self, document: dict[str, Any]) -> None:
        self.document = deepcopy(document)
        self.last_raw_response = deepcopy(document)
        self.last_raw_requirement_response = deepcopy(document)
        self.last_raw_inspection_response: dict[str, Any] | None = None
        self.last_observation_images: list[str] = []
        self.last_raw_kitchen_graph_response = deepcopy(document)
        self.last_validated_kitchen_graph_response = deepcopy(document)
        self.raw_decomposition = deepcopy(document)
        self.raw_vlm_response = deepcopy(document)
        self.validated_vlm_specification = deepcopy(document)
        self.metrics = FMCallMetrics(requirement_calls=0, search_prior_calls=0, total_calls=0)
        self.call_count: int = 0

    def generate_task_requirements(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        return deepcopy(self.document)

    def generate_kitchen_functional_graph(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        return deepcopy(self.document)

    def generate_inspection_priors(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.metrics.search_prior_calls += 1
        self.metrics.total_calls += 1
        return {
            "initial_requirements_satisfied": True,
            "decision_reason": "Offline mock",
            "inspectable_regions": [],
            "inspection_order": [],
        }


def compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_dict_hash(d: Any) -> str:
    canonical_json = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def structural_gf_projection(gf: FunctionalRequirementGraph) -> dict[str, Any]:
    """Extract architecture-authoritative fields only for deterministic hashing."""
    nodes_proj = {}
    for name, node in sorted(gf.nodes.items()):
        nodes_proj[name] = {
            "name": node.name,
            "entity_kind": node.entity_kind,
            "count": node.count,
            "min_count": node.min_count,
            "max_count": node.max_count,
            "preference": node.preference,
            "binding_policy": node.binding_policy,
            "semantic_categories": sorted(node.semantic_categories),
            "unary_predicates": sorted(node.unary_predicates),
            "numeric_constraints": sorted([
                {
                    "property_name": nc.property_name,
                    "operator": nc.operator,
                    "threshold": nc.threshold,
                    "unit": nc.unit,
                }
                for nc in node.numeric_constraints
            ], key=lambda x: x["property_name"]),
            "verification_mode": node.verification_mode,
        }

    rels_proj = sorted([
        {
            "subject_role": rel.subject_role,
            "predicate": rel.predicate,
            "object_role": rel.object_role,
            "expected": rel.expected,
        }
        for rel in gf.relations
    ], key=lambda x: (x["subject_role"], x["predicate"], x["object_role"]))

    ops_proj = sorted([
        {
            "id": op.id,
            "function": op.function,
            "tool_role": op.tool_role,
            "target_role": op.target_role,
            "required_target_count": op.required_target_count,
            "usage_policy": op.usage_policy,
            "required_relations": sorted(op.required_relations),
            "context_role": op.context_role,
            "context_relations": sorted(op.context_relations),
            "distinct_within_group": op.distinct_within_group,
            "same_tool_must_cover_all_targets": op.same_tool_must_cover_all_targets,
            "selection_preference": op.selection_preference,
        }
        for op in gf.operation_groups
    ], key=lambda x: x["id"])

    return {
        "domain": gf.domain,
        "nodes": nodes_proj,
        "relations": rels_proj,
        "operation_groups": ops_proj,
        "cross_group_reuse_allowed": gf.cross_group_reuse_allowed,
    }


def load_ideal_fixture(domain: str) -> tuple[dict[str, Any], str]:
    filename_map = {
        "kitchen": "kitchen_K1.json",
        "living_room": "living_room_L1.json",
        "workshop": "workshop_W1.json",
    }
    fixture_path = FIXTURES_DIR / filename_map[domain]
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"
    sha = compute_file_sha256(fixture_path)
    assert sha == EXPECTED_FIXTURE_SHA256[domain], f"Fixture hash mismatch for {domain}: got {sha}"
    return json.loads(fixture_path.read_text(encoding="utf-8")), sha


def build_go_from_dict(d: dict[str, Any]) -> ObservedSceneGraph:
    return ObservedSceneGraph.from_dict(d)


# ==============================================================================
# 1. KITCHEN K1 IDEAL FULL PIPELINE CONVERGENCE
# ==============================================================================

def test_p3i_kitchen_k1_ideal_convergence(tmp_path: Path):
    """K1 Full Pipeline Convergence: 3 deterministic runs on ideal fixture with complete evidence closure."""
    fixture_data, fixture_sha = load_ideal_fixture("kitchen")
    runs_results: list[PipelineResult] = []
    gf_hashes: list[str] = []
    contract_hashes: list[str] = []
    go_hashes: list[str] = []
    phi_hashes: list[str] = []
    plan_hashes: list[str] = []

    for run_idx in range(3):
        out_dir = tmp_path / f"kitchen_run_{run_idx}"
        adapter = MockFMAdapter(fixture_data)

        with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter),              patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter),              patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter):
            res = run_pipeline(
                domain="kitchen",
                variant="K1",
                mode="vlm",
                search_order="auto",
                output_root=out_dir,
            )
            runs_results.append(res)

            # Verification F1: Exact One-Shot Call Metrics
            assert adapter.metrics.requirement_calls == 1
            assert adapter.metrics.search_prior_calls == 0
            assert adapter.metrics.total_calls == 1

            # Verification 1: Terminal Status
            assert res.status == "ACTION_SEQUENCE_READY", f"Run {run_idx} failed with status {res.status}"
            assert res.assignment is not None and len(res.assignment) == 6
            assert res.plan is not None and len(res.plan) > 0

            # Verification 2: Manifest & Contract validation
            run_dir = out_dir / "kitchen" / "K1" / "vlm"
            manifest_file = run_dir / "run_manifest.json"
            assert manifest_file.exists()
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert manifest["terminal_status"] == "ACTION_SEQUENCE_READY"
            assert manifest["search_policy_version"] == PHASE3_SEARCH_REGION_POLICY_VERSION
            assert manifest["search_contract"]["source"] == "VLM_PROVIDER_RANKED_SYSTEM_COMPLETED"
            assert manifest["search_contract"]["domain"] == "kitchen"

            # Compute and record hashes (F2, F6, F7, F8, F9)
            spec_file = run_dir / "functional_specification.json"
            assert spec_file.exists()
            spec_dict = json.loads(spec_file.read_text(encoding="utf-8"))
            spec_obj = FunctionalRequirementGraph.from_dict(spec_dict)
            gf_proj = structural_gf_projection(spec_obj)
            gf_h = compute_dict_hash(gf_proj)
            gf_hashes.append(gf_h)

            contract_h = compute_dict_hash(manifest["search_contract"])
            contract_hashes.append(contract_h)

            go_file = run_dir / "observed_scene_graph.json"
            assert go_file.exists()
            go_data = json.loads(go_file.read_text(encoding="utf-8"))
            go_h = compute_dict_hash(go_data)
            go_hashes.append(go_h)

            phi_h = compute_dict_hash(res.assignment)
            phi_hashes.append(phi_h)

            plan_h = compute_dict_hash(list(res.plan))
            plan_hashes.append(plan_h)

            # Verification F10: Audit Plan Grounding Artifact
            plan_audit_file = run_dir / "plan_grounding_audit.json"
            assert plan_audit_file.exists()
            plan_audit_data = json.loads(plan_audit_file.read_text(encoding="utf-8"))
            assert plan_audit_data.get("grounding_complete") is True
            assert plan_audit_data.get("all_assignment_nodes_observed") is True
            assert plan_audit_data.get("all_required_relations_true") is True
            assert plan_audit_data.get("plan_uses_only_grounded_task_objects") is True
            assert plan_audit_data.get("preparation_accessibility_valid") is True
            assert plan_audit_data.get("plan_replay_valid") is True
            assert len(plan_audit_data.get("violations", [])) == 0

            # Verification F11: Same-G_O GT Control
            gt_provider = GTSpecProvider()
            gt_gf = gt_provider.provide("kitchen", TASK_INSTRUCTIONS["kitchen"])
            go_obj = build_go_from_dict(go_data)
            gt_ground_res = ground_graph(gt_gf, go_obj, {"search_exhausted": False})
            assert gt_ground_res.complete is True
            assert gt_ground_res.status == "COMPLETE"
            for role_name in ("coffee_container", "soup_container", "coffee_stirrer", "soup_eating_utensil", "coffee_source", "water_source"):
                assert gt_ground_res.assignment.get(role_name) == res.assignment.get(role_name)

            # Verification F14: Replay Validation with 0 FM calls
            spec_file = run_dir / "functional_specification.json"
            assert spec_file.exists()
            replay_adapter = MockFMAdapter(fixture_data)
            replay_out = tmp_path / f"kitchen_replay_{run_idx}"
            with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=replay_adapter),                  patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=replay_adapter),                  patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=replay_adapter):
                replay_res = run_pipeline(
                    domain="kitchen",
                    variant="K1",
                    mode="vlm",
                    specification_json=spec_file,
                    output_root=replay_out,
                )
            assert replay_adapter.metrics.total_calls == 0
            assert replay_res.status == "ACTION_SEQUENCE_READY"
            assert replay_res.assignment == res.assignment
            assert replay_res.plan == res.plan

    # Determinism Assertions (F13)
    assert len(set(gf_hashes)) == 1, "K1 G_F structural hash must be identical across runs"
    assert len(set(contract_hashes)) == 1, "K1 Search contract hash must be identical across runs"
    assert len(set(go_hashes)) == 1, "K1 Final G_O hash must be identical across runs"
    assert len(set(phi_hashes)) == 1, "K1 phi* hash must be identical across runs"
    assert len(set(plan_hashes)) == 1, "K1 Plan hash must be identical across runs"


# ==============================================================================
# 2. LIVING ROOM L1 IDEAL FULL PIPELINE CONVERGENCE
# ==============================================================================

def test_p3i_living_room_l1_ideal_convergence(tmp_path: Path):
    """L1 Full Pipeline Convergence: 3 deterministic runs on ideal fixture with complete evidence closure."""
    fixture_data, fixture_sha = load_ideal_fixture("living_room")
    runs_results: list[PipelineResult] = []
    gf_hashes: list[str] = []
    contract_hashes: list[str] = []
    go_hashes: list[str] = []
    phi_hashes: list[str] = []
    plan_hashes: list[str] = []

    for run_idx in range(3):
        out_dir = tmp_path / f"living_room_run_{run_idx}"
        adapter = MockFMAdapter(fixture_data)

        with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter),              patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter),              patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter):
            res = run_pipeline(
                domain="living_room",
                variant="L1",
                mode="vlm",
                search_order="auto",
                output_root=out_dir,
            )
            runs_results.append(res)

            # Verification F1: Exact One-Shot Call Metrics
            assert adapter.metrics.requirement_calls == 1
            assert adapter.metrics.search_prior_calls == 0
            assert adapter.metrics.total_calls == 1

            # Verification 1: Terminal Status
            assert res.status == "ACTION_SEQUENCE_READY", f"Run {run_idx} failed with status {res.status}"
            assert res.assignment is not None and len(res.assignment) == 6
            assert res.plan is not None and len(res.plan) > 0

            # Verification 2: Manifest & Contract validation
            run_dir = out_dir / "living_room" / "L1" / "vlm"
            manifest_file = run_dir / "run_manifest.json"
            assert manifest_file.exists()
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert manifest["terminal_status"] == "ACTION_SEQUENCE_READY"
            assert manifest["search_policy_version"] == PHASE3_SEARCH_REGION_POLICY_VERSION
            assert manifest["search_contract"]["source"] == "SYSTEM_DECLARED_NO_SEARCH"
            assert manifest["search_contract"]["no_search_required"] is True

            # Compute and record hashes (F2, F6, F7, F8, F9)
            spec_file = run_dir / "functional_specification.json"
            assert spec_file.exists()
            spec_dict = json.loads(spec_file.read_text(encoding="utf-8"))
            spec_obj = FunctionalRequirementGraph.from_dict(spec_dict)
            gf_proj = structural_gf_projection(spec_obj)
            gf_h = compute_dict_hash(gf_proj)
            gf_hashes.append(gf_h)

            contract_h = compute_dict_hash(manifest["search_contract"])
            contract_hashes.append(contract_h)

            go_file = run_dir / "observed_scene_graph.json"
            assert go_file.exists()
            go_data = json.loads(go_file.read_text(encoding="utf-8"))
            go_h = compute_dict_hash(go_data)
            go_hashes.append(go_h)

            phi_h = compute_dict_hash(res.assignment)
            phi_hashes.append(phi_h)

            plan_h = compute_dict_hash(list(res.plan))
            plan_hashes.append(plan_h)

            # Verification F10: Audit Plan Grounding Artifact
            plan_audit_file = run_dir / "plan_grounding_audit.json"
            assert plan_audit_file.exists()
            plan_audit_data = json.loads(plan_audit_file.read_text(encoding="utf-8"))
            assert plan_audit_data.get("grounding_complete") is True
            assert plan_audit_data.get("all_assignment_nodes_observed") is True
            assert plan_audit_data.get("all_required_relations_true") is True
            assert plan_audit_data.get("plan_uses_only_grounded_task_objects") is True
            assert plan_audit_data.get("preparation_accessibility_valid") is True
            assert plan_audit_data.get("plan_replay_valid") is True
            assert len(plan_audit_data.get("violations", [])) == 0

            # Verification F11: Same-G_O GT Control
            gt_provider = GTSpecProvider()
            gt_gf = gt_provider.provide("living_room", TASK_INSTRUCTIONS["living_room"])
            go_obj = build_go_from_dict(go_data)
            gt_ground_res = ground_graph(gt_gf, go_obj, {"search_exhausted": False})
            assert gt_ground_res.complete is True
            assert gt_ground_res.status == "COMPLETE"
            for role_name in gt_gf.nodes:
                assert gt_ground_res.assignment.get(role_name) == res.assignment.get(role_name)

            # Verification F14: Replay Validation with 0 FM calls
            spec_file = run_dir / "functional_specification.json"
            assert spec_file.exists()
            replay_adapter = MockFMAdapter(fixture_data)
            replay_out = tmp_path / f"living_room_replay_{run_idx}"
            with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=replay_adapter),                  patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=replay_adapter),                  patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=replay_adapter):
                replay_res = run_pipeline(
                    domain="living_room",
                    variant="L1",
                    mode="vlm",
                    specification_json=spec_file,
                    output_root=replay_out,
                )
            assert replay_adapter.metrics.total_calls == 0
            assert replay_res.status == "ACTION_SEQUENCE_READY"
            assert replay_res.assignment == res.assignment
            assert replay_res.plan == res.plan

    # Determinism Assertions (F13)
    assert len(set(gf_hashes)) == 1, "L1 G_F structural hash must be identical across runs"
    assert len(set(contract_hashes)) == 1, "L1 Search contract hash must be identical across runs"
    assert len(set(go_hashes)) == 1, "L1 Final G_O hash must be identical across runs"
    assert len(set(phi_hashes)) == 1, "L1 phi* hash must be identical across runs"
    assert len(set(plan_hashes)) == 1, "L1 Plan hash must be identical across runs"


# ==============================================================================
# 3. WORKSHOP W1 IDEAL FULL PIPELINE CONVERGENCE
# ==============================================================================

def test_p3i_workshop_w1_ideal_convergence(tmp_path: Path):
    """W1 Full Pipeline Convergence: 3 deterministic runs on ideal fixture with complete evidence closure."""
    fixture_data, fixture_sha = load_ideal_fixture("workshop")
    runs_results: list[PipelineResult] = []
    gf_hashes: list[str] = []
    contract_hashes: list[str] = []
    go_hashes: list[str] = []
    phi_hashes: list[str] = []
    plan_hashes: list[str] = []

    for run_idx in range(3):
        out_dir = tmp_path / f"workshop_run_{run_idx}"
        adapter = MockFMAdapter(fixture_data)

        with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter),              patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter),              patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter):
            res = run_pipeline(
                domain="workshop",
                variant="W1",
                mode="vlm",
                search_order="auto",
                output_root=out_dir,
            )
            runs_results.append(res)

            # Verification F1: Exact One-Shot Call Metrics
            assert adapter.metrics.requirement_calls == 1
            assert adapter.metrics.search_prior_calls == 0
            assert adapter.metrics.total_calls == 1

            # Verification 1: Terminal Status
            assert res.status == "ACTION_SEQUENCE_READY", f"Run {run_idx} failed with status {res.status}"
            assert res.assignment is not None and len(res.assignment) >= 2
            assert res.plan is not None and len(res.plan) > 0

            # Verification 2: Manifest & Contract validation
            run_dir = out_dir / "workshop" / "W1" / "vlm"
            manifest_file = run_dir / "run_manifest.json"
            assert manifest_file.exists()
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert manifest["terminal_status"] == "ACTION_SEQUENCE_READY"
            assert manifest["search_policy_version"] == PHASE3_SEARCH_REGION_POLICY_VERSION
            assert manifest["search_contract"]["source"] == "VLM_PROVIDER_RANKED_SYSTEM_COMPLETED"
            assert manifest["search_contract"]["domain"] == "workshop"

            # Compute and record hashes (F2, F6, F7, F8, F9)
            spec_file = run_dir / "functional_specification.json"
            assert spec_file.exists()
            spec_dict = json.loads(spec_file.read_text(encoding="utf-8"))
            spec_obj = FunctionalRequirementGraph.from_dict(spec_dict)
            gf_proj = structural_gf_projection(spec_obj)
            gf_h = compute_dict_hash(gf_proj)
            gf_hashes.append(gf_h)

            contract_h = compute_dict_hash(manifest["search_contract"])
            contract_hashes.append(contract_h)

            go_file = run_dir / "observed_scene_graph.json"
            assert go_file.exists()
            go_data = json.loads(go_file.read_text(encoding="utf-8"))
            go_h = compute_dict_hash(go_data)
            go_hashes.append(go_h)

            phi_h = compute_dict_hash(res.assignment)
            phi_hashes.append(phi_h)

            plan_h = compute_dict_hash(list(res.plan))
            plan_hashes.append(plan_h)

            # Verification F10: Audit Plan Grounding Artifact
            plan_audit_file = run_dir / "plan_grounding_audit.json"
            assert plan_audit_file.exists()
            plan_audit_data = json.loads(plan_audit_file.read_text(encoding="utf-8"))
            assert plan_audit_data.get("grounding_complete") is True
            assert plan_audit_data.get("all_assignment_nodes_observed") is True
            assert plan_audit_data.get("all_required_relations_true") is True
            assert plan_audit_data.get("plan_uses_only_grounded_task_objects") is True
            assert plan_audit_data.get("preparation_accessibility_valid") is True
            assert plan_audit_data.get("plan_replay_valid") is True
            assert len(plan_audit_data.get("violations", [])) == 0

            # Verification F11: Same-G_O GT Control
            gt_provider = GTSpecProvider()
            gt_gf = gt_provider.provide("workshop", TASK_INSTRUCTIONS["workshop"])
            go_obj = build_go_from_dict(go_data)
            gt_ground_res = ground_graph(gt_gf, go_obj, {"search_exhausted": False})
            assert gt_ground_res.complete is True
            assert gt_ground_res.status == "COMPLETE"
            for role_name in ("driver", "fastener", "repair_target"):
                assert gt_ground_res.assignment.get(role_name) == res.assignment.get(role_name)

            # Verification F14: Replay Validation with 0 FM calls
            spec_file = run_dir / "functional_specification.json"
            assert spec_file.exists()
            replay_adapter = MockFMAdapter(fixture_data)
            replay_out = tmp_path / f"workshop_replay_{run_idx}"
            with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=replay_adapter),                  patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=replay_adapter),                  patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=replay_adapter):
                replay_res = run_pipeline(
                    domain="workshop",
                    variant="W1",
                    mode="vlm",
                    specification_json=spec_file,
                    output_root=replay_out,
                )
            assert replay_adapter.metrics.total_calls == 0
            assert replay_res.status == "ACTION_SEQUENCE_READY"
            assert replay_res.assignment == res.assignment
            assert replay_res.plan == res.plan

    # Determinism Assertions (F13)
    assert len(set(gf_hashes)) == 1, "W1 G_F structural hash must be identical across runs"
    assert len(set(contract_hashes)) == 1, "W1 Search contract hash must be identical across runs"
    assert len(set(go_hashes)) == 1, "W1 Final G_O hash must be identical across runs"
    assert len(set(phi_hashes)) == 1, "W1 phi* hash must be identical across runs"
    assert len(set(plan_hashes)) == 1, "W1 Plan hash must be identical across runs"


# ==============================================================================
# 4. F3 & F4 — GT VS IDEAL VLM STRUCTURAL CONTROL & EVALUATOR METRICS
# ==============================================================================

@pytest.mark.parametrize("domain,variant", [
    ("kitchen", "K1"),
    ("living_room", "L1"),
    ("workshop", "W1"),
])
def test_p3i_gt_vs_ideal_vlm_structural_control(domain: str, variant: str):
    """F3 & F4: Compare structural properties and reference metrics for all 3 ideal fixtures."""
    fixture_data, _ = load_ideal_fixture(domain)
    adapter = MockFMAdapter(fixture_data)
    vlm = VLMSpecProvider()
    gt = GTSpecProvider()
    inst = TASK_INSTRUCTIONS[domain]

    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter),          patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter),          patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter):
        vlm_gf = vlm.provide(domain, inst, observation_images=[Path("/tmp/dummy_obs.png")])

    gt_gf = gt.provide(domain, inst)

    eval_res = evaluate_gf_against_reference(vlm_gf, gt_gf)
    assert eval_res.role_identity_recall == 1.0
    assert eval_res.role_identity_precision == 1.0
    assert eval_res.role_exact_recall == 1.0
    assert eval_res.role_exact_precision == 1.0
    assert eval_res.relation_recall == 1.0
    assert eval_res.relation_precision == 1.0
    assert eval_res.operation_group_identity_recall == 1.0
    assert eval_res.operation_group_identity_precision == 1.0
    assert eval_res.operation_group_exact_recall == 1.0
    assert eval_res.operation_group_exact_precision == 1.0
    assert eval_res.reference_complete is True
    assert eval_res.exact_structural_match is True
    assert compute_dict_hash(structural_gf_projection(vlm_gf)) == compute_dict_hash(structural_gf_projection(gt_gf))
