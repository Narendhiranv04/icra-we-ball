"""Unit tests for Phase 3.0/3.0.1 contract: search order, specification replay, manifest, artifact discovery, and safety guards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.run import (
    _collect_artifacts,
    _compute_file_sha256,
    _get_git_provenance,
    _load_or_acquire_specification,
    _write_run_manifest,
    run_pipeline,
)
from mujoco_scenes.functional_tamp_pipeline.search_order import (
    FIXED_SEARCH_ORDERS,
    resolve_search_order,
)


def _make_dummy_spec(
    domain: str = "kitchen",
    candidate_regions: tuple[str, ...] = ("D1", "D2", "C2", "B1", "C1"),
    region_ranking: tuple[str, ...] = ("D1", "D2", "C2", "B1", "C1"),
    source: str = "GT_FUNCTIONAL_SPEC_ONLY",
) -> FunctionalRequirementGraph:
    nodes = {
        "tool": FunctionalRole(
            name="tool",
            entity_kind="OBJECT",
            count=1,
            semantic_categories=("spoon",),
        )
    }
    return FunctionalRequirementGraph(
        domain=domain,
        task_instruction="dummy task",
        nodes=nodes,
        candidate_regions=candidate_regions,
        region_ranking=region_ranking,
        source=source,
    )


# 1. Provider search order returns specification.region_ranking unchanged
def test_search_order_provider_returns_unchanged():
    spec = _make_dummy_spec(
        domain="kitchen",
        candidate_regions=("C2", "D1", "D2", "B1", "C1"),
        region_ranking=("C2", "D1", "D2", "B1", "C1"),
    )
    order = resolve_search_order(spec, "kitchen", source="provider")
    assert order == ("C2", "D1", "D2", "B1", "C1")


# 2. Kitchen fixed order resolves expected canonical order
def test_search_order_kitchen_fixed():
    spec = _make_dummy_spec(
        domain="kitchen",
        candidate_regions=("C2", "D1", "D2", "B1", "C1"),
        region_ranking=("C2", "D1", "D2", "B1", "C1"),
    )
    order = resolve_search_order(spec, "kitchen", source="fixed")
    assert order == ("D1", "D2", "C2", "B1", "C1")


# 3. Workshop fixed order resolves expected canonical order
def test_search_order_workshop_fixed():
    spec = _make_dummy_spec(
        domain="workshop",
        candidate_regions=("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER"),
        region_ranking=("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER"),
    )
    order = resolve_search_order(spec, "workshop", source="fixed")
    assert order == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")


# 4. Duplicate region order rejected
def test_search_order_duplicate_region_rejected():
    nodes = {"tool": FunctionalRole(name="tool", count=1, semantic_categories=("spoon",))}
    spec = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="dummy",
        nodes=nodes,
        candidate_regions=("D1", "D2"),
        region_ranking=("D1", "D1"),
    )
    with pytest.raises(ValueError, match="duplicate regions"):
        resolve_search_order(spec, "kitchen", source="provider")


# 5. Missing candidate region rejected
def test_search_order_missing_candidate_rejected():
    spec = _make_dummy_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2", "C2", "B1", "C1", "EXTRA"),
        region_ranking=("D1", "D2", "C2", "B1", "C1", "EXTRA"),
    )
    with pytest.raises(ValueError, match="missing candidate regions"):
        resolve_search_order(spec, "kitchen", source="fixed")


# 6. Unknown extra region rejected
def test_search_order_unknown_extra_region_rejected():
    spec = _make_dummy_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2"),
        region_ranking=("D1", "D2"),
    )
    with pytest.raises(ValueError, match="extra unknown regions"):
        resolve_search_order(spec, "kitchen", source="fixed")


# 7. Living Room resolves search order as N/A / empty
def test_search_order_living_room_empty():
    spec = _make_dummy_spec(
        domain="living_room",
        candidate_regions=(),
        region_ranking=(),
    )
    assert resolve_search_order(spec, "living_room", source="provider") == ()
    assert resolve_search_order(spec, "living_room", source="fixed") == ()


# 8. Specification replay: valid saved G_F loads successfully
def test_spec_replay_valid_loads(tmp_path: Path):
    spec = _make_dummy_spec(domain="kitchen", source="GT_FUNCTIONAL_SPEC_ONLY")
    spec_file = tmp_path / "saved_spec.json"
    spec_file.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    loaded, acquisition, path_str = _load_or_acquire_specification(
        domain="kitchen",
        mode="gt",
        task="dummy",
        images=[],
        specification_json=spec_file,
    )
    assert acquisition == "replayed_provider_output"
    assert path_str == str(spec_file.resolve())
    assert loaded.domain == "kitchen"
    assert loaded.source == "GT_FUNCTIONAL_SPEC_ONLY"


# 9. Specification replay: malformed JSON rejected
def test_spec_replay_malformed_json_rejected(tmp_path: Path):
    spec_file = tmp_path / "bad.json"
    spec_file.write_text("NOT_JSON_DATA", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        _load_or_acquire_specification(
            domain="kitchen",
            mode="gt",
            task="dummy",
            images=[],
            specification_json=spec_file,
        )


# 10. Specification replay: domain mismatch rejected
def test_spec_replay_domain_mismatch_rejected(tmp_path: Path):
    spec = _make_dummy_spec(domain="workshop", candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"), region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"))
    spec_file = tmp_path / "workshop_spec.json"
    spec_file.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match requested domain"):
        _load_or_acquire_specification(
            domain="kitchen",
            mode="gt",
            task="dummy",
            images=[],
            specification_json=spec_file,
        )


# 11. Specification replay: source/mode mismatch rejected
def test_spec_replay_mode_mismatch_rejected(tmp_path: Path):
    spec = _make_dummy_spec(domain="kitchen", source="VLM_FUNCTIONAL_SPEC")
    spec_file = tmp_path / "vlm_spec.json"
    spec_file.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="is not compatible with requested mode 'gt'"):
        _load_or_acquire_specification(
            domain="kitchen",
            mode="gt",
            task="dummy",
            images=[],
            specification_json=spec_file,
        )


# 12. Specification replay: provider is not called when replay file is supplied
def test_spec_replay_bypasses_provider(tmp_path: Path):
    spec = _make_dummy_spec(domain="kitchen", source="GT_FUNCTIONAL_SPEC_ONLY")
    spec_file = tmp_path / "saved_spec.json"
    spec_file.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory:
        loaded, acquisition, _ = _load_or_acquire_specification(
            domain="kitchen",
            mode="gt",
            task="dummy",
            images=[],
            specification_json=spec_file,
        )
        mock_prov_factory.assert_not_called()
        assert loaded.domain == "kitchen"
        assert acquisition == "replayed_provider_output"


# 13. Specification SHA-256: deterministic for identical bytes
def test_specification_sha256_deterministic(tmp_path: Path):
    file1 = tmp_path / "spec1.json"
    file2 = tmp_path / "spec2.json"
    content = b'{"hello": "world"}'
    file1.write_bytes(content)
    file2.write_bytes(content)

    h1 = _compute_file_sha256(file1)
    h2 = _compute_file_sha256(file2)
    assert h1 == h2
    assert len(h1) == 64


# 14. Manifest: successful mocked run produces required provenance fields
def test_manifest_success_mocked(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")
    dummy_result = PipelineResult(
        domain="kitchen",
        variant="K1",
        mode="gt",
        status="ACTION_SEQUENCE_READY",
        inspected_regions=(),
        assignment={"tool": "object_0001"},
        plan=({"action_index": 1, "operator": "PICK", "arguments": ["object_0001"]},),
    )

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", return_value=dummy_result):
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        res = run_pipeline(
            domain="kitchen",
            variant="K1",
            mode="gt",
            output_root=tmp_path,
        )
        assert res.status == "ACTION_SEQUENCE_READY"

    manifest_path = tmp_path / "kitchen" / "K1" / "gt" / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["domain"] == "kitchen"
    assert manifest["variant"] == "K1"
    assert manifest["spec_mode"] == "gt"
    assert manifest["spec_acquisition"] == "live_provider"
    assert manifest["search_order_source_requested"] == "provider"
    assert manifest["search_order_source_effective"] == "provider"
    assert manifest["execution_state"] == "planning_only"
    assert manifest["visualization_requested"] is False
    assert manifest["terminal_status"] == "ACTION_SEQUENCE_READY"
    assert isinstance(manifest["specification_sha256"], str)
    assert isinstance(manifest["pipeline_runtime_seconds"], float)
    assert isinstance(manifest["started_at_utc"], str)
    assert isinstance(manifest["finished_at_utc"], str)
    assert "result" in manifest["artifacts"]


# 15. Manifest: mocked exception still produces failure manifest
def test_manifest_exception_mocked(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", side_effect=RuntimeError("Simulated perception crash")):
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        with pytest.raises(RuntimeError, match="Simulated perception crash"):
            run_pipeline(
                domain="kitchen",
                variant="K1",
                mode="gt",
                output_root=tmp_path,
            )

    manifest_path = tmp_path / "kitchen" / "K1" / "gt" / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["terminal_status"] == "PIPELINE_EXCEPTION"


# 16. Git dirty field test through mocking
def test_git_provenance_mocked():
    with patch("subprocess.check_output") as mock_subproc:
        mock_subproc.side_effect = [
            "abcdef1234567890abcdef1234567890abcdef12\n",  # git rev-parse HEAD
            " M some_file.py\n",                            # git status --porcelain
        ]
        commit, dirty = _get_git_provenance()
        assert commit == "abcdef1234567890abcdef1234567890abcdef12"
        assert dirty is True

    with patch("subprocess.check_output") as mock_subproc:
        mock_subproc.side_effect = [
            "abcdef1234567890abcdef1234567890abcdef12\n",
            "",
        ]
        commit, dirty = _get_git_provenance()
        assert commit == "abcdef1234567890abcdef1234567890abcdef12"
        assert dirty is False


# 17. Default provider path: existing provider call remains used when --specification-json is absent
def test_default_provider_call_when_spec_json_absent(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory:
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        spec, acq, path_input = _load_or_acquire_specification(
            domain="kitchen",
            mode="gt",
            task="dummy",
            images=[],
            specification_json=None,
        )
        mock_prov_factory.assert_called_once_with("gt")
        mock_provider.provide.assert_called_once_with("kitchen", "dummy", [])
        assert acq == "live_provider"
        assert path_input is None


# 18. Fixed runtime guard: requesting fixed before Pass 3.1 must fail fast
def test_fixed_search_runtime_guard(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"):
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        with pytest.raises(RuntimeError, match="fixed search order is resolved but not executable until Phase 3.1 wiring"):
            run_pipeline(
                domain="kitchen",
                variant="K1",
                mode="gt",
                search_order="fixed",
                output_root=tmp_path,
            )


# 19. Pass 3.0.1: Kitchen nested action plan discovery
def test_artifact_discovery_kitchen_nested(tmp_path: Path):
    run_dir = tmp_path / "kitchen_run"
    plan_dir = run_dir / "action_sequence"
    plan_dir.mkdir(parents=True)
    (plan_dir / "action_plan.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan_grounding_audit.json").write_text("{}", encoding="utf-8")

    artifacts = _collect_artifacts(run_dir)
    assert artifacts["action_plan"] == "action_sequence/action_plan.json"
    assert artifacts["final_plan"] == "action_sequence/action_plan.json"
    assert artifacts["plan_grounding_audit"] == "plan_grounding_audit.json"


# 20. Pass 3.0.1: Living Room plan and replay discovery
def test_artifact_discovery_living_room(tmp_path: Path):
    run_dir = tmp_path / "lr_run"
    plan_dir = run_dir / "action_sequence"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_text("{}", encoding="utf-8")
    (plan_dir / "replay_validation.json").write_text("{}", encoding="utf-8")

    artifacts = _collect_artifacts(run_dir)
    assert artifacts["plan"] == "action_sequence/plan.json"
    assert artifacts["final_plan"] == "action_sequence/plan.json"
    assert artifacts["replay_validation"] == "action_sequence/replay_validation.json"


# 21. Pass 3.0.1: Workshop root action plan discovery (BUG FIX)
def test_artifact_discovery_workshop_root(tmp_path: Path):
    run_dir = tmp_path / "workshop_run"
    run_dir.mkdir(parents=True)
    (run_dir / "action_plan.json").write_text("{}", encoding="utf-8")
    (run_dir / "satisfaction.json").write_text("{}", encoding="utf-8")
    (run_dir / "detection_diagnostics.json").write_text("{}", encoding="utf-8")

    artifacts = _collect_artifacts(run_dir)
    assert artifacts["action_plan"] == "action_plan.json"
    assert artifacts["final_plan"] == "action_plan.json"
    assert artifacts["satisfaction"] == "satisfaction.json"
    assert artifacts["detection_diagnostics"] == "detection_diagnostics.json"


# 22. Pass 3.0.1: Centralized finalization called exactly once on success
def test_manifest_finalization_called_once_on_success(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")
    dummy_result = PipelineResult(
        domain="kitchen",
        variant="K1",
        mode="gt",
        status="ACTION_SEQUENCE_READY",
        assignment={"tool": "object_0001"},
        plan=({"action_index": 1, "operator": "PICK", "arguments": ["object_0001"]},),
    )

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", return_value=dummy_result), \
         patch("mujoco_scenes.functional_tamp_pipeline.run._write_run_manifest", wraps=_write_run_manifest) as mock_manifest_writer:
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        res = run_pipeline(
            domain="kitchen",
            variant="K1",
            mode="gt",
            output_root=tmp_path,
        )
        assert res.status == "ACTION_SEQUENCE_READY"
        assert mock_manifest_writer.call_count == 1


# 23. Pass 3.0.1: Centralized finalization called exactly once on infeasible
def test_manifest_finalization_called_once_on_infeasible(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")
    dummy_result = PipelineResult(
        domain="kitchen",
        variant="K7",
        mode="gt",
        status="INFEASIBLE",
        failure_reason="NO_TOOL_OBSERVED",
    )

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", return_value=dummy_result), \
         patch("mujoco_scenes.functional_tamp_pipeline.run._write_run_manifest", wraps=_write_run_manifest) as mock_manifest_writer:
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        res = run_pipeline(
            domain="kitchen",
            variant="K7",
            mode="gt",
            output_root=tmp_path,
        )
        assert res.status == "INFEASIBLE"
        assert mock_manifest_writer.call_count == 1


# 24. Pass 3.0.1: Centralized finalization called exactly once on exception
def test_manifest_finalization_called_once_on_exception(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", side_effect=ValueError("Perception crashed")), \
         patch("mujoco_scenes.functional_tamp_pipeline.run._write_run_manifest", wraps=_write_run_manifest) as mock_manifest_writer:
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        with pytest.raises(ValueError, match="Perception crashed"):
            run_pipeline(
                domain="kitchen",
                variant="K1",
                mode="gt",
                output_root=tmp_path,
            )
        assert mock_manifest_writer.call_count == 1
