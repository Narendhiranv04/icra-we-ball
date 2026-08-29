"""Unit tests for Phase 3 contract: search order, spec replay, manifest, artifact discovery, and failure isolation."""

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
    _safe_write_run_manifest,
    _write_run_manifest,
    run_pipeline,
)
from mujoco_scenes.functional_tamp_pipeline.search_order import (
    FIXED_SEARCH_ORDERS,
    ORACLE_SEARCH_ORDERS,
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
    order, effective_source, effective_seed = resolve_search_order(spec, "kitchen", source="provider")
    assert order == ("C2", "D1", "D2", "B1", "C1")
    assert effective_source == "provider"
    assert effective_seed is None


# 2. Kitchen oracle orders resolve expected canonical orders
def test_search_order_kitchen_oracle():
    spec = _make_dummy_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2", "C2", "B1", "C1"),
        region_ranking=("C2", "D1", "D2", "B1", "C1"),
    )
    order, source, seed = resolve_search_order(spec, "kitchen", source="oracle", mode="gt", variant="K2")
    assert order == ("C2", "D1", "D2", "B1", "C1")
    assert source == "oracle"
    assert seed is None

    order_k3, _, _ = resolve_search_order(spec, "kitchen", source="oracle", mode="gt", variant="K3")
    assert order_k3 == ("B1", "D1", "D2", "C2", "C1")


# 3. Workshop oracle orders resolve expected canonical orders
def test_search_order_workshop_oracle():
    spec = _make_dummy_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        region_ranking=("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER"),
    )
    order, source, seed = resolve_search_order(spec, "workshop", source="oracle", mode="gt", variant="W1")
    assert order == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
    assert source == "oracle"
    assert seed is None

    order_w5, _, _ = resolve_search_order(spec, "workshop", source="oracle", mode="gt", variant="W5")
    assert order_w5 == ("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER")


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
        resolve_search_order(spec, "kitchen", source="oracle", mode="gt", variant="K1")


# 6. Unknown extra region rejected
def test_search_order_unknown_extra_region_rejected():
    spec = _make_dummy_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2"),
        region_ranking=("D1", "D2"),
    )
    with pytest.raises(ValueError, match="extra unknown regions"):
        resolve_search_order(spec, "kitchen", source="oracle", mode="gt", variant="K1")


# 7. Living Room resolves search order as N/A / empty
def test_search_order_living_room_empty():
    spec = _make_dummy_spec(
        domain="living_room",
        candidate_regions=(),
        region_ranking=(),
    )
    assert resolve_search_order(spec, "living_room", source="auto") == ((), "not_applicable", None)
    assert resolve_search_order(spec, "living_room", source="provider") == ((), "not_applicable", None)
    with pytest.raises(ValueError, match="not applicable for living_room"):
        resolve_search_order(spec, "living_room", source="oracle")
    with pytest.raises(ValueError, match="not applicable for living_room"):
        resolve_search_order(spec, "living_room", source="random", seed=0)


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
    assert manifest["search_order_source_requested"] == "auto"
    assert manifest["search_order_source_effective"] == "oracle"
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


# 18. Auto search mode defaults
def test_search_order_auto_resolution():
    spec = _make_dummy_spec(domain="kitchen")
    _, eff_gt, _ = resolve_search_order(spec, "kitchen", "auto", mode="gt", variant="K1")
    assert eff_gt == "oracle"

    _, eff_vlm, _ = resolve_search_order(spec, "kitchen", "auto", mode="vlm", variant="K1")
    assert eff_vlm == "provider"


# 19. VLM + Oracle rejected
def test_vlm_oracle_rejected():
    spec = _make_dummy_spec(domain="kitchen")
    with pytest.raises(ValueError, match="oracle search is privileged and only valid with GT mode"):
        resolve_search_order(spec, "kitchen", "oracle", mode="vlm", variant="K1")


# 20. Random search reproducibility and validation
def test_random_search_properties():
    spec = _make_dummy_spec(domain="kitchen")
    order1, eff1, seed1 = resolve_search_order(spec, "kitchen", "random", mode="vlm", seed=42)
    order2, eff2, seed2 = resolve_search_order(spec, "kitchen", "random", mode="vlm", seed=42)
    assert order1 == order2
    assert eff1 == "random"
    assert seed1 == 42
    assert set(order1) == set(spec.candidate_regions)

    # Missing seed rejected
    with pytest.raises(ValueError, match="random search requires --search-seed"):
        resolve_search_order(spec, "kitchen", "random", mode="vlm", seed=None)

    # Negative seed rejected
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        resolve_search_order(spec, "kitchen", "random", mode="vlm", seed=-5)

    # Seed with oracle/provider rejected
    with pytest.raises(ValueError, match="--search-seed is only valid for random"):
        resolve_search_order(spec, "kitchen", "provider", mode="vlm", seed=0)


# 21. Kitchen nested action plan discovery
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


# 22. Living Room plan and replay discovery
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


# 23. Workshop root action plan discovery
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


# 24. Manifest failure on success is non-intrusive
def test_manifest_failure_on_success_is_non_intrusive(tmp_path: Path, capsys: pytest.CaptureFixture):
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
         patch("mujoco_scenes.functional_tamp_pipeline.run._write_run_manifest", side_effect=OSError("disk full")):
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
        assert res is dummy_result

    captured = capsys.readouterr()
    assert "RUN MANIFEST WRITE FAILED: OSError: disk full" in captured.err


# 25. Manifest failure on infeasible is non-intrusive
def test_manifest_failure_on_infeasible_is_non_intrusive(tmp_path: Path, capsys: pytest.CaptureFixture):
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
         patch("mujoco_scenes.functional_tamp_pipeline.run._write_run_manifest", side_effect=PermissionError("permission denied")):
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
        assert res is dummy_result

    captured = capsys.readouterr()
    assert "RUN MANIFEST WRITE FAILED: PermissionError: permission denied" in captured.err


# 26. Manifest failure on pipeline exception preserves original exception
def test_manifest_failure_on_pipeline_exception_preserves_original_exception(tmp_path: Path, capsys: pytest.CaptureFixture):
    dummy_spec = _make_dummy_spec(domain="kitchen")

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", side_effect=RuntimeError("original pipeline failure")), \
         patch("mujoco_scenes.functional_tamp_pipeline.run._write_run_manifest", side_effect=OSError("manifest failure")):
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        with pytest.raises(RuntimeError) as exc_info:
            run_pipeline(
                domain="kitchen",
                variant="K1",
                mode="gt",
                output_root=tmp_path,
            )
        assert str(exc_info.value) == "original pipeline failure"
        assert type(exc_info.value) is RuntimeError

    captured = capsys.readouterr()
    assert "RUN MANIFEST WRITE FAILED: OSError: manifest failure" in captured.err
