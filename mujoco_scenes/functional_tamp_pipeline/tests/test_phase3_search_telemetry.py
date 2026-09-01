"""Unit tests for Phase 3.1 & 3.1.1: search order regimes and telemetry events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    GraphGroundingResult,
    PipelineResult,
    SatisfactionResult,
    SearchRegionContract,
    freeze_search_region_contract,
)
from mujoco_scenes.functional_tamp_pipeline.run import (
    run_pipeline,
)
from mujoco_scenes.functional_tamp_pipeline.search import (
    SearchDomain,
    search_until_satisfied,
)
from mujoco_scenes.functional_tamp_pipeline.search_order import (
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


# 1. Oracle coverage test: all K1-K12 and W1-W10 covered
def test_oracle_orders_complete_coverage():
    kitchen_candidates = {"D1", "D2", "C2", "B1", "C1"}
    for idx in range(1, 13):
        k_label = f"K{idx}"
        assert k_label in ORACLE_SEARCH_ORDERS["kitchen"]
        order = ORACLE_SEARCH_ORDERS["kitchen"][k_label]
        assert set(order) == kitchen_candidates
        assert len(order) == len(kitchen_candidates)

    workshop_candidates = {"LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"}
    for idx in range(1, 11):
        w_label = f"W{idx}"
        assert w_label in ORACLE_SEARCH_ORDERS["workshop"]
        order = ORACLE_SEARCH_ORDERS["workshop"][w_label]
        assert set(order) == workshop_candidates
        assert len(order) == len(workshop_candidates)


# 2. Random search seeds produce multiple distinct permutations across seeds 0..9
def test_random_search_seeds_distribution():
    spec = _make_dummy_spec(domain="kitchen")
    permutations = set()
    for seed in range(10):
        order, eff_source, eff_seed = resolve_search_order(
            spec, "kitchen", "random", mode="vlm", seed=seed
        )
        assert eff_source == "random"
        assert eff_seed == seed
        assert set(order) == set(spec.candidate_regions)
        assert len(order) == len(spec.candidate_regions)
        permutations.add(order)
    # With 10 different seeds and 5! = 120 permutations, we should get multiple distinct permutations
    assert len(permutations) > 1


# 3. Workshop search with fake domain and explicit search order
class FakeWorkshopDomain:
    def __init__(self, satisfy_at_region: str | None = None):
        self.satisfy_at_region = satisfy_at_region
        self.observed_initial = False
        self.opened_regions: list[str] = []
        self.observed_after_open: list[str] = []
        self.call_history: list[str] = []
        self.graph = MagicMock()
        self.graph.to_dict.return_value = {"nodes": [], "edges": []}

    def observe_initial(self) -> None:
        self.observed_initial = True
        self.call_history.append("observe_initial")

    def evaluate_satisfaction(self, search_exhausted: bool = False) -> SatisfactionResult:
        self.call_history.append(f"evaluate_satisfaction(exhausted={search_exhausted})")
        satisfied = False
        if self.satisfy_at_region and self.satisfy_at_region in self.opened_regions:
            satisfied = True
        elif search_exhausted and not self.satisfy_at_region:
            return GraphGroundingResult(status="INFEASIBLE", complete=False)
        return GraphGroundingResult(
            status="ACTION_SEQUENCE_READY" if satisfied else "INCOMPLETE",
            complete=satisfied,
            assignment={"driver": "d1", "fastener": "f1"} if satisfied else None,
        )

    def open_region(self, region: str) -> dict[str, Any]:
        self.opened_regions.append(region)
        self.call_history.append(f"open_region({region})")
        return {"success": True}

    def observe_after_open(self, region: str) -> None:
        self.observed_after_open.append(region)
        self.call_history.append(f"observe_after_open({region})")


def test_workshop_search_explicit_order_and_early_stopping():
    spec = _make_dummy_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    )
    domain = FakeWorkshopDomain(satisfy_at_region="TOOL_CABINET")
    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

    custom_order = ("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER")
    contract = SearchRegionContract(domain="workshop", canonical_region_ids=custom_order, source="TEST_POLICY")
    result, inspected = search_until_satisfied(
        domain, spec, search_contract=contract, observer=observer
    )
    assert result.satisfied is True
    # Should stop immediately after TOOL_CABINET
    assert inspected == ("TOOL_CABINET",)
    assert domain.opened_regions == ["TOOL_CABINET"]

    # Verify event types received in order and live scene_graph presence
    event_names = [e[0] for e in events]
    assert event_names == [
        "observation_updated",  # initial
        "grounding_updated",    # initial evaluation
        "search_region_selected",  # TOOL_CABINET selected
        "search_region_opened",    # TOOL_CABINET opened
        "observation_updated",  # after_TOOL_CABINET
        "grounding_updated",    # after_TOOL_CABINET evaluation
    ]
    # Check that scene_graph is exposed in observation and grounding events
    assert events[0][1]["scene_graph"] == {"nodes": [], "edges": []}
    assert events[1][1]["scene_graph"] == {"nodes": [], "edges": []}


# 4. FIX #9: Observer does not alter computation (identically zero extra calls)
def test_observer_does_not_change_computation():
    spec = _make_dummy_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    )
    contract = freeze_search_region_contract(spec)
    domain_no_obs = FakeWorkshopDomain(satisfy_at_region="RIGHT_DRAWER")
    res_no_obs, insp_no_obs = search_until_satisfied(
        domain_no_obs, spec, search_contract=contract, observer=None
    )

    domain_with_obs = FakeWorkshopDomain(satisfy_at_region="RIGHT_DRAWER")
    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

    res_with_obs, insp_with_obs = search_until_satisfied(
        domain_with_obs, spec, search_contract=contract, observer=observer
    )

    # Identical call history and result
    assert domain_no_obs.call_history == domain_with_obs.call_history
    assert insp_no_obs == insp_with_obs
    assert res_no_obs.status == res_with_obs.status
    assert res_no_obs.assignment == res_with_obs.assignment


# 5. Kitchen sequential inspection explicit order and early stopping
def test_kitchen_sequential_inspection_order_and_early_stopping(tmp_path: Path):
    from mujoco_scenes.sequential_inspection import run_fixed_order_inspection

    class FakeScene:
        def get_region_observation_states(self):
            return {"C1": {}, "C2": {}, "D1": {}, "D2": {}, "B1": {}}

    class FakeAdapter:
        def __init__(self):
            self.inspected = []
        def inspect(self, region_id):
            self.inspected.append(region_id)

    class FakeSession:
        def __init__(self):
            self.events = []
            self.latest_witness = {"status": "INCOMPLETE"}
            self.next_stage = 1
            self.registry = {"objects": []}
        def append_event(self, event):
            self.events.append(event)

    observed_calls = []
    def fake_observe(stage_label, region_opened):
        observed_calls.append((stage_label, region_opened))
        cloud_mock = MagicMock(total_points=100)
        return cloud_mock, tmp_path / stage_label

    adapter = FakeAdapter()
    session = FakeSession()
    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

    def completion_predicate(s):
        return "C2" in adapter.inspected

    custom_order = ("C2", "D1", "D2", "B1", "C1")
    run_fixed_order_inspection(
        FakeScene(),
        session,
        custom_order,
        adapter=adapter,
        observe=fake_observe,
        stop_on_complete=True,
        completion_predicate=completion_predicate,
        observer=observer,
    )

    # Must stop immediately after inspecting C2
    assert adapter.inspected == ["C2"]

    event_names = [e[0] for e in events]
    assert event_names == [
        "observation_updated",
        "search_region_selected",
        "search_region_opened",
        "observation_updated",
    ]


# 6. Kitchen stage events sequence and central payload enrichment
def test_kitchen_stage_events_sequence_and_enrichment(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")
    dummy_result = PipelineResult(
        domain="kitchen",
        variant="K1",
        mode="gt",
        status="ACTION_SEQUENCE_READY",
        assignment={"tool": "object_0001"},
        plan=({"action_index": 1, "operator": "PICK", "arguments": ["object_0001"]},),
    )

    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

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
            search_order="auto",
            observer=observer,
            output_root=tmp_path,
        )
        assert res.status == "ACTION_SEQUENCE_READY"

    stages = [payload["stage"] for name, payload in events if name == "stage_changed"]
    assert stages == ["specification", "perception", "search_grounding", "planning", "complete"]


# 7. Central search event policy and seed enrichment
def test_central_search_event_enrichment(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="workshop", candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"), region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"), source="VLM_FUNCTIONAL_SPEC")
    dummy_result = PipelineResult(
        domain="workshop",
        variant="W1",
        mode="vlm",
        status="ACTION_SEQUENCE_READY",
        inspected_regions=("LEFT_DRAWER",),
        assignment={"driver": "d1", "fastener": "f1", "work_surface": "w1"},
        plan=({"action_index": 1, "operator": "PICK", "arguments": ["d1"]},),
    )

    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.workshop_scene.WorkshopScene"), \
         patch("mujoco_scenes.functional_tamp_pipeline.run._capture_workshop_vlm_inputs", return_value=[]), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.workshop.WorkshopDomainAdapter") as mock_adapter_cls, \
         patch("mujoco_scenes.functional_tamp_pipeline.run.search_until_satisfied") as mock_search, \
         patch("mujoco_scenes.functional_tamp_pipeline.run.plan_with_common_astar") as mock_plan:
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        mock_adapter_instance = MagicMock()
        mock_adapter_instance.graph.to_dict.return_value = {"nodes": [], "edges": []}
        mock_adapter_instance.controller.detection_diagnostics = []
        mock_adapter_cls.return_value = mock_adapter_instance

        mock_satisfaction = GraphGroundingResult(
            status="ACTION_SEQUENCE_READY",
            complete=True,
            assignment={"driver": "d1", "fastener": "f1", "work_surface": "w1"},
        )
        def fake_search(adapter, spec, *args, **kwargs):
            observer = kwargs.get("observer")
            if observer is not None:
                observer("search_region_selected", {"region": "LEFT_DRAWER", "index": 0, "total_regions": 3})
                observer("search_region_opened", {"region": "LEFT_DRAWER", "success": True, "exploratory": True})
            return mock_satisfaction, ("LEFT_DRAWER",)
        mock_search.side_effect = fake_search

        planned_mock = MagicMock()
        planned_mock.actions = ({"action_index": 1, "operator": "PICK", "arguments": ["d1"]},)
        planned_mock.search.statistics = {}
        planned_mock.validation = {}
        mock_plan.return_value = planned_mock

        res = run_pipeline(
            domain="workshop",
            variant="W1",
            mode="vlm",
            search_order="random",
            search_seed=42,
            observer=observer,
            output_root=tmp_path,
        )
        assert res.status == "ACTION_SEQUENCE_READY"

    selected_events = [p for n, p in events if n == "search_region_selected"]
    opened_events = [p for n, p in events if n == "search_region_opened"]
    assert len(selected_events) == 1
    assert selected_events[0]["search_order_source_effective"] == "random"
    assert selected_events[0]["search_seed_effective"] == 42
    assert len(opened_events) == 1
    assert opened_events[0]["search_order_source_effective"] == "random"
    assert opened_events[0]["search_seed_effective"] == 42


# 8. Living Room telemetry: zero search/open events
def test_living_room_telemetry_no_search_events(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="living_room", candidate_regions=(), region_ranking=())
    dummy_result = PipelineResult(
        domain="living_room",
        variant="L1",
        mode="gt",
        status="ACTION_SEQUENCE_READY",
        assignment={"shared_remote": "table_1"},
        plan=({"action_index": 1, "operator": "PLACE", "arguments": ["remote", "table_1"]},),
    )

    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.living_room.run_to_plan", return_value=dummy_result):
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        res = run_pipeline(
            domain="living_room",
            variant="L1",
            mode="gt",
            observer=observer,
            output_root=tmp_path,
        )
        assert res.status == "ACTION_SEQUENCE_READY"

    event_names = [e[0] for e in events]
    assert "run_started" in event_names
    assert "spec_ready" in event_names
    assert "stage_changed" in event_names
    assert "plan_ready" in event_names
    assert "run_finished" in event_names
    assert "search_region_selected" not in event_names
    assert "search_region_opened" not in event_names


# 9. Observer failure isolation
def test_observer_failure_does_not_affect_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture):
    dummy_spec = _make_dummy_spec(domain="kitchen")
    dummy_result = PipelineResult(
        domain="kitchen",
        variant="K1",
        mode="gt",
        status="ACTION_SEQUENCE_READY",
        assignment={"tool": "object_0001"},
        plan=({"action_index": 1, "operator": "PICK", "arguments": ["object_0001"]},),
    )

    def failing_observer(event: str, payload: dict[str, Any]):
        if event == "spec_ready":
            raise RuntimeError("Telemetry visualization crash")

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
            observer=failing_observer,
            output_root=tmp_path,
        )
        assert res.status == "ACTION_SEQUENCE_READY"

    captured = capsys.readouterr()
    assert "OBSERVER ERROR on spec_ready: RuntimeError: Telemetry visualization crash" in captured.err

    manifest_path = tmp_path / "kitchen" / "K1" / "gt" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["observer_errors"]) == 1
    assert manifest["observer_errors"][0]["event"] == "spec_ready"
    assert manifest["observer_errors"][0]["type"] == "RuntimeError"


# 10. Replay with random search does not call provider
def test_replay_with_random_search_bypasses_provider(tmp_path: Path):
    spec = _make_dummy_spec(domain="kitchen", source="VLM_FUNCTIONAL_SPEC")
    spec_file = tmp_path / "saved_spec.json"
    spec_file.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    dummy_result = PipelineResult(
        domain="kitchen",
        variant="K1",
        mode="vlm",
        status="ACTION_SEQUENCE_READY",
        assignment={"tool": "object_0001"},
        plan=(),
    )

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", return_value=dummy_result):
        res = run_pipeline(
            domain="kitchen",
            variant="K1",
            mode="vlm",
            search_order="random",
            search_seed=7,
            specification_json=spec_file,
            output_root=tmp_path,
        )
        assert res.status == "ACTION_SEQUENCE_READY"
        mock_prov_factory.assert_not_called()

    manifest_path = tmp_path / "kitchen" / "K1" / "vlm" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["spec_acquisition"] == "replayed_provider_output"
    assert manifest["search_order_source_requested"] == "random"
    assert manifest["search_order_source_effective"] == "random"
    assert manifest["search_seed_requested"] == 7
    assert manifest["search_seed_effective"] == 7


# 11. Manifest records visualization_requested flag correctly
def test_manifest_visualization_requested_flag(tmp_path: Path):
    dummy_spec = _make_dummy_spec(domain="kitchen")
    dummy_result = PipelineResult(
        domain="kitchen",
        variant="K1",
        mode="gt",
        status="ACTION_SEQUENCE_READY",
        assignment={"tool": "object_0001"},
        plan=(),
    )

    with patch("mujoco_scenes.functional_tamp_pipeline.run.provider_for_mode") as mock_prov_factory, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.scene_for_variant"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.kitchen.run_to_plan", return_value=dummy_result):
        mock_provider = MagicMock()
        mock_provider.provide.return_value = dummy_spec
        mock_prov_factory.return_value = mock_provider

        # Default run: visualization_requested = False
        run_pipeline(
            domain="kitchen",
            variant="K1",
            mode="gt",
            output_root=tmp_path / "default_run",
        )
        manifest_default = json.loads((tmp_path / "default_run" / "kitchen" / "K1" / "gt" / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest_default["visualization_requested"] is False

        # Visualized run: visualization_requested = True
        run_pipeline(
            domain="kitchen",
            variant="K1",
            mode="gt",
            visualize=True,
            output_root=tmp_path / "visualized_run",
        )
        manifest_viz = json.loads((tmp_path / "visualized_run" / "kitchen" / "K1" / "gt" / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest_viz["visualization_requested"] is True


# 12. Camera plumbing tests without MuJoCo
def test_kitchen_camera_plumbing_surfaces_raw_rgb_image(tmp_path: Path):
    from mujoco_scenes.sequential_inspection import run_fixed_order_inspection

    class FakeScene:
        def get_region_observation_states(self):
            return {"C2": {}}

    class FakeAdapter:
        def inspect(self, region_id):
            pass

    class FakeSession:
        def __init__(self):
            self.events = []
            self.latest_witness = {"status": "COMPLETE"}
            self.next_stage = 1
            self.registry = {"objects": []}
        def append_event(self, event):
            self.events.append(event)

    stage_dir = tmp_path / "stage_000"
    stage_dir.mkdir(parents=True, exist_ok=True)
    overview_img = stage_dir / "overview.png"
    overview_img.write_text("fake_overview_composite", encoding="utf-8")

    cam_dir = stage_dir / "cameras" / "CAM_A"
    cam_dir.mkdir(parents=True, exist_ok=True)
    raw_rgb_img = cam_dir / "rgb.png"
    raw_rgb_img.write_text("fake_raw_rgb", encoding="utf-8")

    def fake_observe(stage_label, region_opened):
        cloud_mock = MagicMock(total_points=100, cameras=("CAM_A", "CAM_B"))
        return cloud_mock, stage_dir

    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

    run_fixed_order_inspection(
        FakeScene(),
        FakeSession(),
        ("C2",),
        adapter=FakeAdapter(),
        observe=fake_observe,
        stop_on_complete=True,
        observer=observer,
    )

    obs_events = [p for e, p in events if e == "observation_updated"]
    assert len(obs_events) > 0
    assert obs_events[0]["frame_path"] == str(raw_rgb_img)
    assert obs_events[0]["frame_path"] != str(overview_img)

    # When no raw rgb exists, frame_path must be None (no fallback to overview.png)
    raw_rgb_img.unlink()
    events.clear()
    run_fixed_order_inspection(
        FakeScene(),
        FakeSession(),
        ("C2",),
        adapter=FakeAdapter(),
        observe=fake_observe,
        stop_on_complete=True,
        observer=observer,
    )
    obs_events = [p for e, p in events if e == "observation_updated"]
    assert len(obs_events) > 0
    assert obs_events[0]["frame_path"] is None


def test_workshop_camera_plumbing_copies_rgb_array():
    import numpy as np
    from mujoco_scenes.functional_tamp_pipeline.search import search_until_satisfied

    class FakeWorkshopDomain:
        def __init__(self):
            self.latest_frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        def observe_initial(self):
            pass
        def evaluate_satisfaction(self):
            return SatisfactionResult(complete=True, status="SATISFIED", assignment={"driver": "d1"})
        def open_region(self, region):
            return {"success": True}
        def observe_after_open(self, region):
            pass

    events = []
    def observer(event: str, payload: dict[str, Any]):
        events.append((event, payload))

    domain = FakeWorkshopDomain()
    spec = _make_dummy_spec(domain="workshop")
    contract = freeze_search_region_contract(spec)
    search_until_satisfied(domain, spec, search_contract=contract, observer=observer)

    obs_events = [p for e, p in events if e == "observation_updated"]
    assert len(obs_events) > 0
    assert obs_events[0]["frame_rgb"] is not None
    assert obs_events[0]["frame_rgb"].shape == (100, 100, 3)

