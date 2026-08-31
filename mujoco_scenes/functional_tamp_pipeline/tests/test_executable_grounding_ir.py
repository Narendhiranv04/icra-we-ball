"""Unit and regression tests for Pass 3.6A.8 Executable Grounding IR and VLM interfaces."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import validate_runtime_gf
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider
from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
from mujoco_scenes.kitchen_vlm_functional_graph import compile_vlm_functional_graph
from mujoco_scenes.workshop_phase1.fm_adapter import SYSTEM_PROMPT


FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tmp" / "phase36b1r_live_audit_20260831_140518"


def _load_fixture(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    content = data.get("content", data)
    if isinstance(content, str):
        content = json.loads(content)
    return content


def test_kitchen_k2_live_audit_fixture() -> None:
    k2_path = FIXTURE_DIR / "kitchen_K2" / "live" / "fm_diagnostics" / "fm_call_001.json"
    if not k2_path.exists():
        pytest.skip(f"Fixture {k2_path} not found")
    raw_doc = _load_fixture(k2_path)

    class MockAdapter:
        def generate_kitchen_functional_graph(self, task_instruction, observation_images):
            self.last_raw_kitchen_graph_response = raw_doc
            self.last_validated_kitchen_graph_response = raw_doc
            return raw_doc

    graph = VLMSpecProvider._kitchen("Prepare coffee and soup", [], adapter=MockAdapter())
    assert isinstance(graph, FunctionalRequirementGraph)
    assert graph.domain == "kitchen"
    assert "coffee_container" in graph.nodes
    assert "soup_container" in graph.nodes
    assert "coffee_stirrer" in graph.nodes
    assert "soup_eating_utensil" in graph.nodes
    assert graph.nodes["coffee_container"].entity_kind == "OBJECT"
    assert graph.nodes["soup_container"].entity_kind == "OBJECT"
    assert len(graph.operation_groups) >= 1
    validate_runtime_gf(graph)


def test_living_room_l1_live_audit_fixture() -> None:
    l1_path = FIXTURE_DIR / "living_L1" / "live" / "fm_diagnostics" / "fm_call_001.json"
    if not l1_path.exists():
        pytest.skip(f"Fixture {l1_path} not found")
    raw_doc = _load_fixture(l1_path)

    prov = EnvironmentVLMRequirementProvider("living_room")
    prov.generate_canonical("Set up cup, saucer, and remote", raw_document=raw_doc)
    graph = VLMSpecProvider._living_room("Set up cup, saucer, and remote", [], provider=prov)

    assert isinstance(graph, FunctionalRequirementGraph)
    assert graph.domain == "living_room"
    # HARD GATE invariant: OBJECT cup must never become a REGION
    assert graph.nodes["CUP_SAUCER_SET"].entity_kind == "OBJECT"
    assert graph.nodes["REMOTE"].entity_kind == "OBJECT"
    assert graph.nodes["PERSONAL_CUP_SAUCER_REGION"].entity_kind == "REGION"
    assert graph.nodes["SHARED_REMOTE_REGION"].entity_kind == "REGION"
    assert len(graph.operation_groups) == 3
    validate_runtime_gf(graph)


def test_workshop_w1_live_audit_fixture() -> None:
    w1_path = FIXTURE_DIR / "workshop_W1" / "live" / "fm_diagnostics" / "fm_call_001.json"
    if not w1_path.exists():
        pytest.skip(f"Fixture {w1_path} not found")
    raw_doc = _load_fixture(w1_path)

    prov = FMRequirementProvider()
    prov._ensure_generated("Find screw and driver", raw_document=raw_doc)
    graph = VLMSpecProvider._workshop("Find screw and driver", [], provider=prov)

    assert isinstance(graph, FunctionalRequirementGraph)
    assert graph.domain == "workshop"
    assert "fastener" in graph.nodes
    assert "driver" in graph.nodes
    assert graph.nodes["fastener"].entity_kind == "OBJECT"
    assert graph.nodes["driver"].entity_kind == "OBJECT"
    assert "workbench_surface" in graph.nodes
    assert graph.nodes["workbench_surface"].entity_kind == "REGION"
    assert "repair_target" in graph.nodes
    assert graph.nodes["repair_target"].entity_kind == "FIXED_TARGET"
    assert len(graph.operation_groups) == 1
    assert graph.candidate_regions == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
    validate_runtime_gf(graph)


def test_static_runtime_isolation() -> None:
    """Verify live VLM runtime modules do not import GT providers or oracle evaluators."""
    import sys
    import mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider
    import mujoco_scenes.environment_vlm_requirements
    import mujoco_scenes.kitchen_vlm_functional_graph
    import mujoco_scenes.workshop_phase1.requirements
    import mujoco_scenes.workshop_phase1.fm_adapter

    vlm_modules = [
        "mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider",
        "mujoco_scenes.environment_vlm_requirements",
        "mujoco_scenes.kitchen_vlm_functional_graph",
        "mujoco_scenes.workshop_phase1.requirements",
        "mujoco_scenes.workshop_phase1.fm_adapter",
    ]

    for mod_name in vlm_modules:
        mod = sys.modules.get(mod_name)
        assert mod is not None, f"Module {mod_name} not imported"
        source_path = getattr(mod, "__file__", "")
        if source_path:
            text = Path(source_path).read_text(encoding="utf-8")
            assert "GTSpecProvider" not in text, f"{mod_name} imports GTSpecProvider"
            assert "gf_reference_evaluator" not in text, f"{mod_name} imports gf_reference_evaluator"
            assert "eval_gt_graph" not in text, f"{mod_name} references eval_gt_graph"


def test_robot_verifier_contract_in_prompts() -> None:
    """Verify generic robot verifier capabilities are present and do not leak internal predicate names."""
    assert "Robot Verifier Capabilities" in SYSTEM_PROMPT
    assert "open/deep cavity" in SYSTEM_PROMPT
    assert "elongated" in SYSTEM_PROMPT
    assert "planar support" in SYSTEM_PROMPT

    # Internal predicate names must NOT be exposed in system prompts
    assert "OPEN_CAVITY" not in SYSTEM_PROMPT
    assert "REACHES_BOTTOM" not in SYSTEM_PROMPT
    assert "COMPATIBLE_WITH_TARGET" not in SYSTEM_PROMPT
