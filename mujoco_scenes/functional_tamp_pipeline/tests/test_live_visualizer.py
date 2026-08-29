"""Comprehensive real-schema tests for LivePipelineVisualizer (headless, no ffplay/MuJoCo)."""

import inspect
from pathlib import Path
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest

from mujoco_scenes.functional_tamp_pipeline.live_visualizer import (
    LivePipelineVisualizer,
    _VisualizerState,
    _flatten_assignment_instance_ids,
    _format_assignment_binding,
    _format_gf_role,
    _format_gf_relation,
    _format_operation_group,
    _format_go_relation,
    _format_operation_binding,
    _format_unsatisfied_relation,
    _gf_relevant_predicates,
    _safe_copy_value,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    FunctionalRelation,
    OperationGroup,
    NumericConstraint,
    GraphGroundingResult,
    SatisfactionResult,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedSceneGraph,
    ObservedNode,
    ObservedRelation,
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
        self.pause_event = threading.Event()
        self.pause_event.set()

    def show(self, frame_rgb: np.ndarray) -> None:
        self.pause_event.wait()
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


def _make_canonical_test_gf() -> FunctionalRequirementGraph:
    """Create real canonical FunctionalRequirementGraph object."""
    role1 = FunctionalRole(
        name="driver",
        entity_kind="OBJECT",
        count=1,
        semantic_categories=("screwdriver", "drill"),
        unary_predicates=("TOOL_HEAD_MATCH",),
        numeric_constraints=(
            NumericConstraint(property_name="length", operator=">=", threshold=0.12, unit="m"),
        ),
        binding_policy="DISTINCT",
    )
    role2 = FunctionalRole(
        name="fastener",
        entity_kind="OBJECT",
        count=1,
        semantic_categories=("screw",),
        binding_policy="DISTINCT",
    )
    rel = FunctionalRelation(
        subject_role="driver",
        predicate="COMPATIBLE_WITH",
        object_role="fastener",
        expected=True,
    )
    op = OperationGroup(
        id="repair_group",
        function="DRIVE_FASTENER",
        tool_role="driver",
        target_role="fastener",
        required_target_count=1,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("COMPATIBLE_WITH",),
        context_role="fastener",
        context_relations=("NEAR_TARGET",),
    )
    return FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Fasten workpiece screw",
        nodes={"driver": role1, "fastener": role2},
        relations=(rel,),
        operation_groups=(op,),
        candidate_regions=("TOOL_CABINET", "DRAWER_LEFT"),
        region_ranking=("TOOL_CABINET", "DRAWER_LEFT"),
        source="GT_WORKSHOP_SPEC",
    )


def _make_canonical_test_go() -> ObservedSceneGraph:
    """Create real canonical ObservedSceneGraph object."""
    g_o = ObservedSceneGraph()
    n1 = ObservedNode(
        instance_id="driver_01",
        entity_kind="OBJECT",
        canonical_category="screwdriver",
        source_region="TOOL_CABINET",
    )
    n2 = ObservedNode(
        instance_id="fastener_01",
        entity_kind="OBJECT",
        canonical_category="screw",
        source_region="DRAWER_LEFT",
    )
    g_o.add_node(n1)
    g_o.add_node(n2)
    g_o.add_relation(ObservedRelation(
        subject_id="driver_01",
        predicate="COMPATIBLE_WITH",
        object_id="fastener_01",
        status="TRUE",
    ))
    g_o.add_relation(ObservedRelation(
        subject_id="driver_01",
        predicate="REACHES_TARGET",
        object_id="fastener_01",
        status="UNKNOWN",
    ))
    g_o.add_relation(ObservedRelation(
        subject_id="fastener_01",
        predicate="COLLIDES_WITH",
        object_id="driver_01",
        status="FALSE",
    ))
    return g_o


# 1. Construction and Headless
def test_1_construction_and_headless(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960, fps=30)
    assert len(fake_viewer_factory.viewers) == 1
    assert fake_viewer_factory.viewers[0].width == 1600
    assert fake_viewer_factory.viewers[0].height == 960
    viz.close()
    assert fake_viewer_factory.viewers[0].closed


# 2. Real-Schema G_F formatting and state reduction
def test_2_real_schema_gf(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960)
    try:
        gf = _make_canonical_test_gf()
        gf_dict = gf.to_dict()

        # Test pure helpers
        role_str = _format_gf_role(gf_dict["nodes"]["driver"])
        assert "driver" in role_str
        assert "DISTINCT" in role_str
        assert "screwdriver" in role_str

        rel_str = _format_gf_relation(gf_dict["relations"][0])
        assert rel_str == "driver --COMPATIBLE_WITH--> fastener"

        op_str = _format_operation_group(gf_dict["operation_groups"][0])
        assert "repair_group" in op_str
        assert "DRIVE_FASTENER" in op_str
        assert "driver -> fastener" in op_str
        assert "SEQUENTIAL_REUSE_ALLOWED" in op_str

        viz("spec_ready", {
            "graph": gf_dict,
            "source": gf.source,
            "provider_region_ranking": list(gf.region_ranking),
            "search_order_source_effective": "oracle",
            "region_order_used": list(gf.region_ranking),
        })

        time.sleep(0.1)
        with viz._state_lock:
            s = viz._state
            assert s.spec_graph == gf_dict
            assert s.spec_source == "GT_WORKSHOP_SPEC"
            assert s.resolved_region_order == ["TOOL_CABINET", "DRAWER_LEFT"]
    finally:
        viz.close()


# 3. Real-Schema G_O formatting and state reduction
def test_3_real_schema_go(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960)
    try:
        go = _make_canonical_test_go()
        go_dict = go.to_dict()

        # Test relation formatting across all canonical relations
        rel_map = {}
        for r in go_dict["relations"]:
            rel_str, status = _format_go_relation(r)
            rel_map[rel_str] = status

        assert rel_map["driver_01 --COMPATIBLE_WITH--> fastener_01"] == "TRUE"
        assert rel_map["driver_01 --REACHES_TARGET--> fastener_01"] == "UNKNOWN"
        assert rel_map["fastener_01 --COLLIDES_WITH--> driver_01"] == "FALSE"

        viz("observation_updated", {
            "stage": "after_TOOL_CABINET",
            "inspected_regions": ["TOOL_CABINET"],
            "scene_graph": go_dict,
        })

        time.sleep(0.1)
        with viz._state_lock:
            s = viz._state
            assert s.scene_graph == go_dict
            assert s.inspected_regions == ["TOOL_CABINET"]
    finally:
        viz.close()


# 4. Real-Schema Grounding & Operation Bindings
def test_4_real_schema_grounding(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960)
    try:
        grounding = GraphGroundingResult(
            status="COMPLETE",
            complete=True,
            assignment={"driver": "driver_01", "fastener": "fastener_01"},
            operation_bindings={
                "repair_group": [
                    {"tool": "driver_01", "target": "fastener_01"}
                ]
            },
        )
        grounding_dict = grounding.to_dict()

        # Test operation binding formatter
        binding_str = _format_operation_binding("repair_group", grounding_dict["operation_bindings"]["repair_group"][0])
        assert "repair_group" in binding_str
        assert "tool=driver_01" in binding_str
        assert "target=fastener_01" in binding_str

        viz("grounding_updated", {
            "grounding": grounding_dict,
            "satisfied": True,
            "status": "COMPLETE",
        })

        time.sleep(0.1)
        with viz._state_lock:
            s = viz._state
            assert isinstance(s.operation_bindings, dict)
            assert "repair_group" in s.operation_bindings
            assert s.assignment == {"driver": "driver_01", "fastener": "fastener_01"}
    finally:
        viz.close()


# 5. Composite shape from real models
def test_5_composite_shape_from_real_models(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960)
    try:
        gf = _make_canonical_test_gf()
        go = _make_canonical_test_go()
        grounding = GraphGroundingResult(
            status="COMPLETE",
            complete=True,
            assignment={"driver": "driver_01", "fastener": "fastener_01"},
            operation_bindings={"repair_group": [{"tool": "driver_01", "target": "fastener_01"}]},
        )
        viz("run_started", {
            "domain": "workshop", "variant": "W1", "spec_mode": "gt",
            "search_order_source_requested": "oracle", "exploration_actuation": "robot_physical",
        })
        viz("spec_ready", {"graph": gf.to_dict(), "source": gf.source, "region_order_used": ["TOOL_CABINET"]})
        viz("observation_updated", {"stage": "initial", "scene_graph": go.to_dict()})
        viz("grounding_updated", {"grounding": grounding.to_dict(), "satisfied": True, "status": "COMPLETE"})
        viz("plan_ready", {"actions": [{"action_index": 1, "operator": "FASTEN", "arguments": ["driver_01", "fastener_01"]}]})
        viz("run_finished", {"terminal_status": "ACTION_SEQUENCE_READY"})

        time.sleep(0.1)
        frame = viz.render_composite_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (960, 1600, 3)
    finally:
        viz.close()


# 6. Queue saturation test: OPEN event is preserved under queue drops
def test_6_queue_saturation_preserves_open_history(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, width=1600, height=960, fps=10)
    try:
        viewer = fake_viewer_factory.viewers[0]
        # Pause viewer to back up the render queue
        viewer.pause_event.clear()

        # Emit search events and many rapid stage updates to saturate queue
        viz("search_region_selected", {"region": "C2", "index": 0})
        viz("search_region_opened", {"region": "C2", "success": True, "exploratory": True})
        viz("observation_updated", {"stage": "after_C2", "inspected_regions": ["C2"]})
        for i in range(50):
            viz("stage_changed", {"stage": f"substep_{i}"})
        viz("plan_ready", {"actions": [{"action_index": 1, "operator": "PICK", "arguments": ["obj1"]}]})
        viz("run_finished", {"terminal_status": "ACTION_SEQUENCE_READY"})

        # Unpause viewer and wait for worker to render
        viewer.pause_event.set()
        time.sleep(0.2)

        with viz._state_lock:
            s = viz._state
            assert s.dropped_display_updates > 0
            assert "C2" in s.inspected_regions
            assert len(s.exploratory_open_trace) == 1
            assert s.exploratory_open_trace[0]["region"] == "C2"
            assert s.terminal_status == "ACTION_SEQUENCE_READY"
            assert len(s.plan_actions) == 1
    finally:
        viewer.pause_event.set()
        viz.close()


# 7. Nested snapshot defensive copying and unsupported object rejection
def test_7_nested_snapshot_copy_and_unsupported_rejection(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        nested_payload = {
            "scene_graph": {
                "nodes": {
                    "obj1": {
                        "semantic_labels": {"status": "SUPPORTED"}
                    }
                }
            }
        }
        viz("observation_updated", nested_payload)
        # Mutate source in place
        nested_payload["scene_graph"]["nodes"]["obj1"]["semantic_labels"]["status"] = "MUTATED"

        with viz._state_lock:
            stored_status = viz._state.scene_graph["nodes"]["obj1"]["semantic_labels"]["status"]
            assert stored_status == "SUPPORTED"

        # Unsupported live object rejection
        class ForbiddenSimulatorObject:
            pass

        with pytest.raises(TypeError, match="Unsupported live object in visualization payload"):
            _safe_copy_value({"live_obj": ForbiddenSimulatorObject()})
    finally:
        viz.close()


# 8. Random seed 0 display
def test_8_random_seed_zero_display(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("run_started", {
            "domain": "kitchen", "variant": "K1", "spec_mode": "vlm",
            "search_order_source_requested": "random", "search_seed_requested": 0,
        })
        viz("spec_ready", {
            "graph": {}, "source": "VLM", "search_order_source_effective": "random",
            "search_seed_effective": 0, "region_order_used": ["C2", "B1"],
        })
        time.sleep(0.05)
        s = viz._snapshot_state_for_render()
        panel = viz._render_status_and_search_panel(s, 1040, 400)
        assert panel is not None
        with viz._state_lock:
            assert viz._state.search_seed_effective == 0
    finally:
        viz.close()


# 9. Incomplete grounding display (missing roles, unsatisfied relations, unresolved constraints)
def test_9_incomplete_grounding_panel(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        grounding = GraphGroundingResult(
            status="INCOMPLETE",
            complete=False,
            assignment=None,
            missing_roles=("driver",),
            unsatisfied_relations=(
                {"predicate": "COMPATIBLE_WITH", "subject_role": "driver", "object_role": "fastener"},
            ),
            unresolved_constraints=("driver: length >= 0.12m",),
        )
        assert _format_unsatisfied_relation(grounding.unsatisfied_relations[0]) == "COMPATIBLE_WITH(driver, fastener)"

        viz("grounding_updated", {
            "grounding": grounding.to_dict(),
            "satisfied": False,
            "status": "INCOMPLETE",
        })
        time.sleep(0.05)
        s = viz._snapshot_state_for_render()
        panel = viz._render_assignment_panel(s, 800, 280)
        assert panel is not None
        with viz._state_lock:
            assert viz._state.missing_roles == ["driver"]
            assert len(viz._state.unsatisfied_relations) == 1
            assert viz._state.unresolved_constraints == ["driver: length >= 0.12m"]
    finally:
        viz.close()


# 10. Headless Workshop telemetry copy safety (mutating SAME source array)
def test_10_workshop_telemetry_enable_flag_mutates_same_source():
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import WorkshopDomainAdapter

    class FakeObs:
        def __init__(self):
            self.rgb = np.zeros((60, 80, 3), dtype=np.uint8)
            self.camera_id = "CAM_TEST"

    class FakeController:
        def __init__(self):
            self.proposal_backend = MagicMock()
            self.geometric_grounder = MagicMock()
            self.tracker = MagicMock(tracks={})
            self._yolo_aux_backend = None
            self.region_categories = set()
            self.graph = MagicMock()
            self.detection_diagnostics = []
            self.last_observation = None

        def _capture_and_process_stage(self, stage_idx, source_region_id):
            obs = FakeObs()
            self.last_observation = obs
            return [obs]

        def _evaluate_grounding_and_search(self, stage_idx, source_region_id):
            return MagicMock(status="INCOMPLETE"), None

    spec = _make_canonical_test_gf()

    with patch("mujoco_scenes.functional_tamp_pipeline.domains.workshop.WorkshopPhase1InspectionController") as mock_ctrl_cls, \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.workshop.FunctionalSatisfactionSearch"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.workshop.WorkshopExecutionDispatcher"), \
         patch("mujoco_scenes.functional_tamp_pipeline.domains.workshop.compile_workshop_requirements_from_graph"):
        mock_ctrl_cls.return_value = FakeController()

        # Telemetry Disabled: latest_frame_rgb remains None
        adapter_disabled = WorkshopDomainAdapter(
            "W1", spec, scene=MagicMock(), physical_open=False, telemetry_enabled=False
        )
        adapter_disabled._capture_and_evaluate("TOOL_CABINET")
        assert adapter_disabled.latest_frame_rgb is None

        # Telemetry Enabled: latest_frame_rgb is an independent copy
        adapter_enabled = WorkshopDomainAdapter(
            "W1", spec, scene=MagicMock(), physical_open=False, telemetry_enabled=True
        )
        adapter_enabled._capture_and_evaluate("TOOL_CABINET")
        assert adapter_enabled.latest_frame_rgb is not None
        assert adapter_enabled.latest_frame_rgb.shape == (60, 80, 3)

        # Mutate the EXACT SAME source observation returned in _capture_and_evaluate
        source = adapter_enabled.controller.last_observation.rgb
        source[:] = 255
        # Stored frame in adapter must still contain original zeros
        assert np.all(adapter_enabled.latest_frame_rgb == 0)


# 11. Headless zero-overhead test: observer=None never calls _extract_scene_graph_dict
def test_11_zero_overhead_when_observer_is_none():
    from mujoco_scenes.functional_tamp_pipeline.search import search_until_satisfied

    class FakeDomain:
        def __init__(self):
            self.graph = _make_canonical_test_go()
        def observe_initial(self): pass
        def evaluate_satisfaction(self, **kwargs):
            return SatisfactionResult(status="COMPLETE", complete=True, assignment={"driver": "d1"})
        def open_region(self, region): return {"success": True}
        def observe_after_open(self, region): pass

    domain = FakeDomain()
    spec = _make_canonical_test_gf()

    with patch("mujoco_scenes.functional_tamp_pipeline.search._extract_scene_graph_dict") as mock_extract:
        result, inspected = search_until_satisfied(domain, spec, observer=None)
        assert result.complete is True
        mock_extract.assert_not_called()


# 12. G_F and G_O panel caching and assignment invalidation
def test_12_panel_caching_and_assignment_invalidation(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        gf = _make_canonical_test_gf()
        go = _make_canonical_test_go()

        viz._state.spec_graph = gf.to_dict()
        s1 = viz._snapshot_state_for_render()
        img_gf1 = viz._get_or_render_gf_panel(s1, 800, 280)
        img_gf2 = viz._get_or_render_gf_panel(s1, 800, 280)
        assert img_gf1 is img_gf2

        viz._state.scene_graph = go.to_dict()
        viz._state.assignment = {"driver": "driver_01"}
        s2 = viz._snapshot_state_for_render()
        img_go1 = viz._get_or_render_go_panel(s2, 800, 280)
        img_go2 = viz._get_or_render_go_panel(s2, 800, 280)
        assert img_go1 is img_go2

        # Change assignment only: G_O cache must invalidate and produce a new image
        viz._state.assignment = {"driver": "driver_02"}
        s3 = viz._snapshot_state_for_render()
        img_go3 = viz._get_or_render_go_panel(s3, 800, 280)
        assert img_go3 is not img_go1
    finally:
        viz.close()


# 13. Viewer failure is non-fatal and records error
def test_13_viewer_failure_non_fatal(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viewer = fake_viewer_factory.viewers[0]
        viewer.raise_on_show = True

        viz("run_started", {"domain": "kitchen", "variant": "K1", "spec_mode": "gt"})
        time.sleep(0.15)
        errs = viz.drain_errors()
        assert len(errs) > 0
        assert errs[0]["event"] in {"viewer_show", "worker_loop"}
    finally:
        viz.close()


# 14. No MuJoCo C-binding references in live_visualizer.py
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


# 15. Exploration vs Plan separation
def test_15_exploration_vs_plan_separation(fake_viewer_factory):
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
        with viz._state_lock:
            assert len(viz._state.exploratory_open_trace) == 2
            assert len(viz._state.plan_actions) == 2
            assert viz._state.exploratory_open_trace[0]["region"] == "C2"
            assert viz._state.plan_actions[0]["operator"] == "PICK"
    finally:
        viz.close()


# 16. Multi-valued role assignments formatting and normalization
def test_16_multi_valued_assignment(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        assignment = {
            "cup_set": ["cup_01", "cup_02"],
            "shared_remote": "remote_01",
        }
        # Helper normalization test
        inst_ids = _flatten_assignment_instance_ids(assignment)
        assert inst_ids == {"cup_01", "cup_02", "remote_01"}

        # Binding formatting test
        list_str = _format_assignment_binding("cup_set", ["cup_01", "cup_02"])
        assert "cup_set" in list_str
        assert "[cup_01, cup_02]" in list_str

        scalar_str = _format_assignment_binding("shared_remote", "remote_01")
        assert "shared_remote" in scalar_str
        assert "remote_01" in scalar_str

        # Full grounding integration with real scene graph
        go = _make_canonical_test_go()
        grounding = GraphGroundingResult(
            status="COMPLETE",
            complete=True,
            assignment=assignment,
        )

        viz("observation_updated", {"stage": "initial", "scene_graph": go.to_dict()})
        viz("grounding_updated", {"grounding": grounding.to_dict(), "satisfied": True, "status": "COMPLETE"})

        time.sleep(0.05)
        # Render composite must succeed without TypeError on unhashable list
        frame = viz.render_composite_frame()
        assert frame.shape == (960, 1600, 3)
        assert frame.dtype == np.uint8
    finally:
        viz.close()


# 17. Guaranteed terminal render race test
def test_17_terminal_render_race_guarantee(fake_viewer_factory):
    # Set slow FPS (e.g. 5 FPS = 200ms cooldown)
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory, fps=5)
    try:
        viewer = fake_viewer_factory.viewers[0]
        # Emit initial event
        viz("run_started", {"domain": "kitchen", "variant": "K1", "spec_mode": "gt"})
        time.sleep(0.05)
        initial_frames = len(viewer.frames)
        assert initial_frames >= 1

        # Emit plan_ready and run_finished rapidly within cooldown
        viz("plan_ready", {"actions": [{"action_index": 1, "operator": "PICK", "arguments": ["cup_01"]}]})
        viz("run_finished", {"terminal_status": "ACTION_SEQUENCE_READY"})

        # Wait for terminal render to execute
        time.sleep(0.15)

        # Viewer MUST have received an additional frame containing terminal status
        assert len(viewer.frames) > initial_frames
        assert viz.last_rendered_terminal_status == "ACTION_SEQUENCE_READY"
    finally:
        viz.close()


# 18. Frame-path disk image cache test
def test_18_frame_path_image_cache(tmp_path: Path, fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        p1 = tmp_path / "frame1.png"
        img1 = Image.new("RGB", (64, 64), (100, 150, 200))
        img1.save(p1)

        p2 = tmp_path / "frame2.png"
        img2 = Image.new("RGB", (64, 64), (50, 100, 150))
        img2.save(p2)

        open_calls = []
        real_open = Image.open

        def tracked_open(fp, *args, **kwargs):
            open_calls.append(str(fp))
            return real_open(fp, *args, **kwargs)

        with patch("PIL.Image.open", side_effect=tracked_open):
            viz("observation_updated", {"stage": "initial", "frame_path": str(p1)})
            # Render composite multiple times
            for _ in range(5):
                viz.render_composite_frame()
            # Image.open for p1 should be called exactly once
            assert open_calls.count(str(p1)) == 1

            # Change path to p2
            viz("observation_updated", {"stage": "after_C2", "frame_path": str(p2)})
            for _ in range(5):
                viz.render_composite_frame()
            # Image.open for p2 should be called exactly once
            assert open_calls.count(str(p2)) == 1
    finally:
        viz.close()


# 19. NumPy scalar snapshot support
def test_19_numpy_scalar_snapshot_support():
    payload = {
        "score": np.float32(0.85),
        "count": np.int64(42),
        "valid": np.bool_(True),
        "nested": {"val": np.float64(3.14)},
    }
    copied = _safe_copy_value(payload)
    assert isinstance(copied["score"], float)
    assert isinstance(copied["count"], int)
    assert isinstance(copied["valid"], bool)
    assert isinstance(copied["nested"]["val"], float)
    assert copied["score"] == pytest.approx(0.85, 1e-5)
    assert copied["count"] == 42
    assert copied["valid"] is True

    # Arbitrary simulator objects still raise TypeError
    class DummySimObj:
        pass
    with pytest.raises(TypeError, match="Unsupported live object in visualization payload"):
        _safe_copy_value({"sim": DummySimObj()})


# 20. G_F relation relevance includes operation groups
def test_20_gf_relation_relevance_includes_operation_groups():
    gf_dict = {
        "relations": [
            {"subject_role": "driver", "predicate": "COMPATIBLE_WITH", "object_role": "fastener"}
        ],
        "operation_groups": [
            {
                "id": "group1",
                "function": "FASTEN",
                "tool_role": "driver",
                "target_role": "fastener",
                "required_relations": ["REACHES_TARGET"],
                "context_role": "fastener",
                "context_relations": ["NEAR_CONTEXT"],
            }
        ],
    }
    preds = _gf_relevant_predicates(gf_dict)
    assert preds == {"COMPATIBLE_WITH", "REACHES_TARGET", "NEAR_CONTEXT"}


# 21. Living Room telemetry search N/A
def test_21_living_room_telemetry_search_na(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("run_started", {"domain": "living_room", "variant": "L1", "spec_mode": "gt"})
        time.sleep(0.05)
        s = viz._snapshot_state_for_render()
        panel = viz._render_status_and_search_panel(s, 1040, 400)
        assert panel is not None
        assert s.domain == "living_room"
    finally:
        viz.close()


# 22. INFEASIBLE scientific outcome is not an exception
def test_22_infeasible_grounding_outcome(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        grounding = GraphGroundingResult(
            status="INFEASIBLE",
            complete=False,
            missing_roles=("driver",),
        )
        viz("run_started", {"domain": "workshop", "variant": "W1", "spec_mode": "gt"})
        viz("grounding_updated", {"grounding": grounding.to_dict(), "satisfied": False, "status": "INFEASIBLE"})
        viz("run_finished", {"terminal_status": "INFEASIBLE"})

        time.sleep(0.05)
        with viz._state_lock:
            s = viz._state
            assert s.terminal_status == "INFEASIBLE"
            assert s.is_exception is False
            assert len(s.plan_actions) == 0

        frame = viz.render_composite_frame()
        assert frame.shape == (960, 1600, 3)
    finally:
        viz.close()


# 23. run_failed marks PIPELINE_EXCEPTION
def test_23_run_failed_exception(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    try:
        viz("run_started", {"domain": "workshop", "variant": "W1", "spec_mode": "gt"})
        viz("run_failed", {
            "error_type": "RuntimeError",
            "error_message": "Unexpected simulator articulation failure",
        })
        time.sleep(0.05)
        with viz._state_lock:
            s = viz._state
            assert s.is_exception is True
            assert s.terminal_status == "PIPELINE_EXCEPTION"
            assert s.exception_type == "RuntimeError"
            assert "articulation failure" in (s.exception_message or "")
    finally:
        viz.close()


# 24. close() idempotence
def test_24_close_idempotence(fake_viewer_factory):
    viz = LivePipelineVisualizer(viewer_factory=fake_viewer_factory)
    viz.close()
    viz.close()  # Must not raise on second close


# 25. CLI failure does not indefinitely hold window
def test_25_cli_lifecycle_failure_does_not_hold():
    from mujoco_scenes.functional_tamp_pipeline.run import main

    test_args = [
        "run.py",
        "--domain", "kitchen",
        "--variant", "K1",
        "--mode", "gt",
        "--visualize",
    ]

    mock_visualizer = MagicMock()
    with patch("sys.argv", test_args), \
         patch("mujoco_scenes.functional_tamp_pipeline.live_visualizer.LivePipelineVisualizer", return_value=mock_visualizer), \
         patch("mujoco_scenes.functional_tamp_pipeline.run.run_pipeline", side_effect=RuntimeError("Early simulator failure")):
        exit_code = main()
        assert exit_code == 1
        # hold_until_closed must NOT be called on pipeline exception
        mock_visualizer.hold_until_closed.assert_not_called()
        # visualizer must be closed cleanly
        mock_visualizer.close.assert_called_once()
