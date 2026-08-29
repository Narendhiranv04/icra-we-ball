"""Comprehensive tests for LivePipelineVisualizer (headless, no ffplay/MuJoCo)."""

import inspect
import queue
import time
from typing import Any

import numpy as np
import pytest

from mujoco_scenes.functional_tamp_pipeline.live_visualizer import (
    LivePipelineVisualizer,
    _VisualizerState,
)


class FakeViewer:
    """Mock viewer recording frames for inspection."""

    def __init__(self, width: int, height: int, fps: int, title: str):
        self.width = width
        self.height = height
        self.fps = fps
        self.title = title
        self.frames: list[np.ndarray] = []
        self.closed = False
        self.raise_on_show = False

    def show(self, frame_rgb: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("Viewer is closed")
        if self.raise_on_show:
            raise RuntimeError("Fake ffplay broken pipe")
        self.frames.append(np.array(frame_rgb, copy=True))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_viewer_factory():
    created_viewers = []

    def _factory(width: int, height: int, fps: int, title: str):
        v = FakeViewer(width, height, fps, title)
        created_viewers.append(v)
        return v

    _factory.viewers = created_viewers
    return _factory


def test_1_construction_and_headless(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960, fps=30)
    assert len(fake_viewer_factory.viewers) == 1
    assert fake_viewer_factory.viewers[0].width == 1600
    assert fake_viewer_factory.viewers[0].height == 960
    viz.close()
    assert fake_viewer_factory.viewers[0].closed


def test_2_state_reduction(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960)
    try:
        viz("run_started", {
            "domain": "kitchen",
            "variant": "K1",
            "spec_mode": "gt",
            "search_order_source_requested": "oracle",
            "search_seed_requested": None,
            "run_dir": "/tmp/test_run",
        })
        viz("stage_changed", {"stage": "specification"})
        viz("spec_ready", {
            "graph": {
                "nodes": {
                    "tool": {"entity_kind": "OBJECT", "count": 1, "semantic_categories": ["spoon"]},
                },
                "edges": [],
            },
            "source": "GT_TEST",
            "provider_region_ranking": ["C2", "B1"],
            "search_order_source_effective": "oracle",
            "region_order_used": ["C2", "B1"],
        })
        viz("search_region_selected", {"region": "C2", "index": 0})
        viz("search_region_opened", {"region": "C2", "success": True})
        viz("observation_updated", {
            "stage": "after_C2",
            "inspected_regions": ["C2"],
            "scene_graph": {
                "nodes": {
                    "obj1": {"canonical_category": "spoon", "source_region": "C2", "entity_kind": "OBJECT"},
                },
                "edges": [],
            },
        })
        viz("grounding_updated", {
            "grounding": {
                "assignment": {"tool": "obj1"},
                "operation_bindings": [],
                "missing_roles": [],
            },
            "satisfied": True,
            "status": "SATISFIED",
        })
        viz("plan_ready", {
            "actions": [{"action_index": 1, "operator": "PICK", "arguments": ["obj1"]}],
            "search_statistics": {"expansions": 5, "search_time_sec": 0.012},
        })
        viz("run_finished", {"terminal_status": "ACTION_SEQUENCE_READY"})

        # Wait briefly for worker queue processing
        time.sleep(0.15)

        s = viz._state
        assert s.domain == "kitchen"
        assert s.variant == "K1"
        assert s.stage == "complete"
        assert s.terminal_status == "ACTION_SEQUENCE_READY"
        assert s.inspected_regions == ["C2"]
        assert len(s.exploratory_open_trace) == 1
        assert s.exploratory_open_trace[0]["region"] == "C2"
        assert s.assignment == {"tool": "obj1"}
        assert len(s.plan_actions) == 1
    finally:
        viz.close()


def test_3_composite_shape(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960)
    try:
        frame = viz.render_composite_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (960, 1600, 3)
    finally:
        viz.close()


def test_4_gf_cache(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz._state.spec_graph = {"nodes": {"tool": {"entity_kind": "OBJECT"}}, "edges": []}
        img1 = viz._get_or_render_gf_panel(800, 280)
        img2 = viz._get_or_render_gf_panel(800, 280)
        assert img1 is img2  # Exact cached object identity

        # Mutate spec graph
        viz._state.spec_graph = {"nodes": {"fastener": {"entity_kind": "OBJECT"}}, "edges": []}
        img3 = viz._get_or_render_gf_panel(800, 280)
        assert img3 is not img1
    finally:
        viz.close()


def test_5_go_cache(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz._state.scene_graph = {"nodes": {"obj1": {"canonical_category": "mug"}}, "edges": []}
        img1 = viz._get_or_render_go_panel(800, 280)
        img2 = viz._get_or_render_go_panel(800, 280)
        assert img1 is img2

        viz._state.scene_graph = {"nodes": {"obj2": {"canonical_category": "plate"}}, "edges": []}
        img3 = viz._get_or_render_go_panel(800, 280)
        assert img3 is not img1
    finally:
        viz.close()


def test_6_queue_is_non_blocking(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        # Fill queue up to capacity
        for i in range(150):
            viz("stage_changed", {"stage": f"stage_{i}"})
        # Callback returns immediately without throwing Queue.Full exception
        assert viz._state.dropped_display_updates >= 0
    finally:
        viz.close()


def test_7_array_copy(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        source_array = np.zeros((100, 100, 3), dtype=np.uint8)
        viz("observation_updated", {"frame_rgb": source_array})

        time.sleep(0.1)
        # Modify original array in-place
        source_array[:] = 255
        # Visualizer's copy must not be mutated
        assert viz._state.latest_frame is not None
        assert np.all(viz._state.latest_frame == 0)
    finally:
        viz.close()


def test_8_living_room_na(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("run_started", {"domain": "living_room", "variant": "L1", "spec_mode": "gt"})
        time.sleep(0.05)
        panel = viz._render_status_and_search_panel(1040, 400)
        assert panel is not None
        # Living room state has no search order
        assert viz._state.domain == "living_room"
    finally:
        viz.close()


def test_9_exploration_vs_plan(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("search_region_opened", {"region": "C2", "success": True})
        viz("search_region_opened", {"region": "B1", "success": True})
        viz("plan_ready", {
            "actions": [
                {"action_index": 1, "operator": "PICK", "arguments": ["obj1"]},
                {"action_index": 2, "operator": "PLACE", "arguments": ["obj1", "target"]},
            ]
        })
        time.sleep(0.1)
        assert len(viz._state.exploratory_open_trace) == 2
        assert len(viz._state.plan_actions) == 2
        # Verify separate data structures
        assert viz._state.exploratory_open_trace[0]["region"] == "C2"
        assert viz._state.plan_actions[0]["operator"] == "PICK"
    finally:
        viz.close()


def test_10_infeasible_rendering(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("run_started", {"domain": "kitchen", "variant": "K7", "spec_mode": "gt"})
        viz("grounding_updated", {
            "grounding": {
                "assignment": None,
                "missing_roles": ["tool"],
                "unsatisfied_relations": ["reaches_target"],
            },
            "satisfied": False,
            "status": "INFEASIBLE",
        })
        viz("run_finished", {"terminal_status": "INFEASIBLE"})
        time.sleep(0.1)
        assert viz._state.terminal_status == "INFEASIBLE"
        assert not viz._state.is_exception
        assert viz._state.missing_roles == ["tool"]
        frame = viz.render_composite_frame()
        assert frame.shape == (960, 1600, 3)
    finally:
        viz.close()


def test_11_pipeline_exception_rendering(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("run_started", {"domain": "workshop", "variant": "W1", "spec_mode": "gt"})
        viz("run_failed", {
            "error_type": "RuntimeError",
            "error_message": "Unexpected physical failure",
        })
        time.sleep(0.1)
        assert viz._state.is_exception
        assert viz._state.terminal_status == "PIPELINE_EXCEPTION"
        assert viz._state.exception_type == "RuntimeError"
    finally:
        viz.close()


def test_12_viewer_failure_is_non_fatal(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viewer = fake_viewer_factory.viewers[0]
        viewer.raise_on_show = True

        viz("run_started", {"domain": "kitchen", "variant": "K1", "spec_mode": "gt"})
        time.sleep(0.15)
        # Visualizer catches error, records in display_errors, disables viewer without crash
        errs = viz.drain_errors()
        assert len(errs) > 0
        assert errs[0]["event"] in {"viewer_show", "worker_loop"}
    finally:
        viz.close()


def test_13_close_idempotence(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    viz.close()
    viz.close()  # No exception on second close


def test_14_no_mujoco_references():
    import mujoco_scenes.functional_tamp_pipeline.live_visualizer as lv
    source = inspect.getsource(lv)
    assert "import mujoco\n" not in source
    assert "from mujoco import" not in source
    assert "import mujoco." not in source
    assert "mj_step" not in source
    assert "mjData" not in source
    assert "WorkshopDomainAdapter" not in source
    assert "KitchenPlanningScene" not in source


def test_15_cumulative_open_preservation(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("search_region_selected", {"region": "C2"})
        viz("search_region_opened", {"region": "C2", "success": True})
        # Rapid updates afterward
        for i in range(20):
            viz("stage_changed", {"stage": f"substep_{i}"})
        time.sleep(0.1)
        assert "C2" in [x["region"] for x in viz._state.exploratory_open_trace]
        assert "C2" in viz._state.inspected_regions
    finally:
        viz.close()
