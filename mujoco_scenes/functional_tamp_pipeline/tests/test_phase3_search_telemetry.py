"""Unit tests for Phase 3.1: search order regimes and telemetry events."""

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

    def observe_initial(self) -> None:
        self.observed_initial = True

    def evaluate_satisfaction(self, search_exhausted: bool = False) -> SatisfactionResult:
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
        return {"success": True}

    def observe_after_open(self, region: str) -> None:
        self.observed_after_open.append(region)


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
    result, inspected = search_until_satisfied(
        domain, spec, search_order=custom_order, observer=observer
    )
    assert result.satisfied is True
    # Should stop immediately after TOOL_CABINET
    assert inspected == ("TOOL_CABINET",)
    assert domain.opened_regions == ["TOOL_CABINET"]

    # Verify event types received in order
    event_names = [e[0] for e in events]
    assert event_names == [
        "observation_updated",  # initial
        "grounding_updated",    # initial evaluation
        "search_region_selected",  # TOOL_CABINET selected
        "search_region_opened",    # TOOL_CABINET opened
        "observation_updated",  # after_TOOL_CABINET
        "grounding_updated",    # after_TOOL_CABINET evaluation
    ]


# 4. Kitchen sequential inspection explicit order and early stopping
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


# 5. Living Room telemetry: zero search/open events
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


# 5. Observer failure isolation
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


# 6. Replay with random search does not call provider
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
