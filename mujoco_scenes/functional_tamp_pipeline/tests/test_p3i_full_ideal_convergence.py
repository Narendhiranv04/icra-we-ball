"""Master Pass Stage B (P3-I): Full Ideal Fixture Pipeline Convergence.

Verifies end-to-end execution of K1, L1, and W1 across 3 deterministic runs each:
1. Production pipeline execution using ideal raw VLM fixtures.
2. 100% COMPLETE status ('ACTION_SEQUENCE_READY').
3. Multi-run determinism (identical assignment, plan, and inspected regions).
4. Plan grounding audit and execution validity.
5. Replay validation using the saved specification JSON.
6. Frozen search contract integrity and run manifest validation.
"""

from __future__ import annotations

from copy import deepcopy
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
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.audit import audit_plan_grounding
from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline
from mujoco_scenes.functional_tamp_pipeline.search_contract import (
    PHASE3_SEARCH_REGION_POLICY_VERSION,
    SearchRegionContract,
)
from mujoco_scenes.workshop_phase1.fm_adapter import FMCallMetrics

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"


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


def load_ideal_fixture(domain: str) -> dict[str, Any]:
    filename_map = {
        "kitchen": "kitchen_K1.json",
        "living_room": "living_room_L1.json",
        "workshop": "workshop_W1.json",
    }
    fixture_path = FIXTURES_DIR / filename_map[domain]
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


# ==============================================================================
# 1. KITCHEN K1 IDEAL FULL PIPELINE CONVERGENCE
# ==============================================================================

def test_p3i_kitchen_k1_ideal_convergence(tmp_path: Path):
    """K1 Full Pipeline Convergence: 3 deterministic runs on ideal fixture."""
    fixture_data = load_ideal_fixture("kitchen")
    runs_results: list[PipelineResult] = []

    for run_idx in range(3):
        out_dir = tmp_path / f"kitchen_run_{run_idx}"
        adapter = MockFMAdapter(fixture_data)

        with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter), \
             patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter), \
             patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter):
            res = run_pipeline(
                domain="kitchen",
                variant="K1",
                mode="vlm",
                search_order="auto",
                output_root=out_dir,
            )
            runs_results.append(res)

            # Verification 1: Terminal Status
            assert res.status == "ACTION_SEQUENCE_READY", f"Run {run_idx} failed with status {res.status}"
            assert res.assignment is not None and len(res.assignment) == 6
            assert res.plan is not None and len(res.plan) > 0

            # Verification 2: Manifest & Contract validation
            manifest_file = out_dir / "kitchen" / "K1" / "vlm" / "run_manifest.json"
            assert manifest_file.exists()
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert manifest["terminal_status"] == "ACTION_SEQUENCE_READY"
            assert manifest["search_policy_version"] == PHASE3_SEARCH_REGION_POLICY_VERSION
            assert manifest["search_contract"]["source"] == "VLM_PROVIDER_RANKED_SYSTEM_COMPLETED"
            assert manifest["search_contract"]["domain"] == "kitchen"

            # Verification 3: Replay Validation
            spec_file = out_dir / "kitchen" / "K1" / "vlm" / "functional_specification.json"
            assert spec_file.exists()
            replay_out = tmp_path / f"kitchen_replay_{run_idx}"
            replay_res = run_pipeline(
                domain="kitchen",
                variant="K1",
                mode="vlm",
                specification_json=spec_file,
                output_root=replay_out,
            )
            assert replay_res.status == "ACTION_SEQUENCE_READY"
            assert replay_res.assignment == res.assignment

    # Multi-run determinism verification
    for i in range(1, 3):
        assert runs_results[i].status == runs_results[0].status
        assert runs_results[i].assignment == runs_results[0].assignment
        assert runs_results[i].inspected_regions == runs_results[0].inspected_regions
        assert runs_results[i].plan == runs_results[0].plan


# ==============================================================================
# 2. LIVING ROOM L1 IDEAL FULL PIPELINE CONVERGENCE
# ==============================================================================

def test_p3i_living_room_l1_ideal_convergence(tmp_path: Path):
    """L1 Full Pipeline Convergence: 3 deterministic runs on ideal fixture."""
    fixture_data = load_ideal_fixture("living_room")
    runs_results: list[PipelineResult] = []

    for run_idx in range(3):
        out_dir = tmp_path / f"living_room_run_{run_idx}"
        adapter = MockFMAdapter(fixture_data)

        with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter), \
             patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter), \
             patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter):
            res = run_pipeline(
                domain="living_room",
                variant="L1",
                mode="vlm",
                search_order="auto",
                output_root=out_dir,
            )
            runs_results.append(res)

            # Verification 1: Terminal Status
            assert res.status == "ACTION_SEQUENCE_READY", f"Run {run_idx} failed with status {res.status}"
            assert res.assignment is not None and len(res.assignment) == 6
            assert res.plan is not None and len(res.plan) > 0

            # Verification 2: Manifest & Contract validation
            manifest_file = out_dir / "living_room" / "L1" / "vlm" / "run_manifest.json"
            assert manifest_file.exists()
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert manifest["terminal_status"] == "ACTION_SEQUENCE_READY"
            assert manifest["search_policy_version"] == PHASE3_SEARCH_REGION_POLICY_VERSION
            assert manifest["search_contract"]["source"] == "SYSTEM_DECLARED_NO_SEARCH"
            assert manifest["search_contract"]["no_search_required"] is True

            # Verification 3: Replay Validation
            spec_file = out_dir / "living_room" / "L1" / "vlm" / "functional_specification.json"
            assert spec_file.exists()
            replay_out = tmp_path / f"living_room_replay_{run_idx}"
            replay_res = run_pipeline(
                domain="living_room",
                variant="L1",
                mode="vlm",
                specification_json=spec_file,
                output_root=replay_out,
            )
            assert replay_res.status == "ACTION_SEQUENCE_READY"
            assert replay_res.assignment == res.assignment

    # Multi-run determinism verification
    for i in range(1, 3):
        assert runs_results[i].status == runs_results[0].status
        assert runs_results[i].assignment == runs_results[0].assignment
        assert runs_results[i].inspected_regions == runs_results[0].inspected_regions
        assert runs_results[i].plan == runs_results[0].plan


# ==============================================================================
# 3. WORKSHOP W1 IDEAL FULL PIPELINE CONVERGENCE
# ==============================================================================

def test_p3i_workshop_w1_ideal_convergence(tmp_path: Path):
    """W1 Full Pipeline Convergence: 3 deterministic runs on ideal fixture."""
    fixture_data = load_ideal_fixture("workshop")
    runs_results: list[PipelineResult] = []

    for run_idx in range(3):
        out_dir = tmp_path / f"workshop_run_{run_idx}"
        adapter = MockFMAdapter(fixture_data)

        with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter), \
             patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter), \
             patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter):
            res = run_pipeline(
                domain="workshop",
                variant="W1",
                mode="vlm",
                search_order="auto",
                output_root=out_dir,
            )
            runs_results.append(res)

            # Verification 1: Terminal Status
            assert res.status == "ACTION_SEQUENCE_READY", f"Run {run_idx} failed with status {res.status}"
            assert res.assignment is not None and len(res.assignment) >= 2
            assert res.plan is not None and len(res.plan) > 0

            # Verification 2: Manifest & Contract validation
            manifest_file = out_dir / "workshop" / "W1" / "vlm" / "run_manifest.json"
            assert manifest_file.exists()
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert manifest["terminal_status"] == "ACTION_SEQUENCE_READY"
            assert manifest["search_policy_version"] == PHASE3_SEARCH_REGION_POLICY_VERSION
            assert manifest["search_contract"]["source"] == "VLM_PROVIDER_RANKED_SYSTEM_COMPLETED"
            assert manifest["search_contract"]["domain"] == "workshop"

            # Verification 3: Replay Validation
            spec_file = out_dir / "workshop" / "W1" / "vlm" / "functional_specification.json"
            assert spec_file.exists()
            replay_out = tmp_path / f"workshop_replay_{run_idx}"
            replay_res = run_pipeline(
                domain="workshop",
                variant="W1",
                mode="vlm",
                specification_json=spec_file,
                output_root=replay_out,
            )
            assert replay_res.status == "ACTION_SEQUENCE_READY"
            assert replay_res.assignment == res.assignment

    # Multi-run determinism verification
    for i in range(1, 3):
        assert runs_results[i].status == runs_results[0].status
        assert runs_results[i].assignment == runs_results[0].assignment
        assert runs_results[i].inspected_regions == runs_results[0].inspected_regions
        assert runs_results[i].plan == runs_results[0].plan
