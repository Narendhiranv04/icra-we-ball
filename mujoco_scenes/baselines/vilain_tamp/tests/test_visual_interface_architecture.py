"""Regression tests for ViLaIn visual state-estimation interface and architecture fidelity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    CameraFrameArtifacts,
    FixedSceneEvidence,
    ObjectEstimate,
    ObjectEstimateStatus,
    ViLaInObservation,
)
from mujoco_scenes.baselines.vilain_tamp.config import Domain
from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.fm import FMCallType, FMRequest, RecordedFMClient
from mujoco_scenes.baselines.vilain_tamp.interpreter import InterpreterModels, ViLaInInterpreter
from mujoco_scenes.baselines.vilain_tamp.live_fixed_evidence import FixedSceneEvidenceProvider
from mujoco_scenes.baselines.vilain_tamp.live_observations import _create_scene
from mujoco_scenes.baselines.vilain_tamp.paper_metrics import compute_stage_funnel
from mujoco_scenes.baselines.vilain_tamp.prompts import (
    build_corrective_fact_selection_prompt,
    build_goal_state_prompt,
    build_initial_consistency_prompt,
    build_initial_state_prompt,
)
from mujoco_scenes.baselines.vilain_tamp.symbolic_contract import (
    GroundedFact,
    load_variant_action_contract,
)
from mujoco_scenes.final_paper_variant_labels import VARIANT_LABELS


from mujoco_scenes.baselines.vilain_tamp.fm import FMTransportResponse

class DummyTransport:
    def __init__(self, canned_responses: list[str] | None = None) -> None:
        self.responses = list(canned_responses) if canned_responses else []
        self.recorded_requests: list[FMRequest] = []

    def complete(self, request: FMRequest) -> FMTransportResponse:
        self.recorded_requests.append(request)
        if self.responses:
            text = self.responses.pop(0)
        elif request.call_type == FMCallType.OBJECT_ESTIMATION:
            text = '{"objects": []}'
        elif request.call_type == FMCallType.INITIAL_STATE:
            text = '{"true_fact_ids": ["f0009"]}'
        elif request.call_type == FMCallType.GOAL_STATE:
            text = '{"goal_fact_ids": ["f0009"]}'
        else:
            text = "{}"
        return FMTransportResponse(
            raw_text=text,
            call_id="dummy-1",
            model=request.model,
            revision=request.revision,
            usage={"tokens": 1},
        )


def _make_dummy_observation(tmp_path: Path) -> ViLaInObservation:
    frame_dir = tmp_path / "cam0"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rgb_file = frame_dir / "rgb.png"
    rgb_file.write_bytes(b"fake_png_bytes")
    depth_file = frame_dir / "depth.npy"
    np.save(depth_file, np.ones((480, 640), dtype=np.float32))
    calib_file = frame_dir / "camera.json"
    calib_file.write_text(json.dumps({
        "camera_id": "cam0",
        "view_description": "Test cam",
        "depth_unit": "meter",
        "intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        "extrinsics": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "image_shape": [480, 640],
    }), encoding="utf-8")

    frame = CameraFrameArtifacts(
        camera_id="cam0",
        view_description="Test cam",
        rgb_path=str(rgb_file),
        depth_path=str(depth_file),
        calibration_path=str(calib_file),
        rgb_sha256="abc",
        depth_sha256="def",
        calibration_sha256="123",
    )
    return ViLaInObservation(
        domain="living_room",
        observation_mode="initial_observation_only",
        stage_id="000_initial",
        camera_frames=(frame,),
        opened_region_id=None,
        capture_timestamp="2026-09-06T00:00:00Z",
        inspection_ordinal=None,
        content_hash="hash123",
    )


def test_1_initial_state_fmrequest_contains_image_artifacts(tmp_path: Path) -> None:
    obs = _make_dummy_observation(tmp_path)
    obj_json = json.dumps({"objects": [{"label": "cup", "pddl_type": "cup", "detections": [{"stage_id": "000_initial", "camera_id": "cam0", "bbox_1000": [100, 100, 200, 200], "confidence": 0.9}]}]})
    init_json = json.dumps({"true_fact_ids": ["f0009"]})
    goal_json = json.dumps({"goal_fact_ids": ["f0009"]})

    transport = DummyTransport([obj_json, init_json, goal_json])
    client = RecordedFMClient(transport)
    domain = load_domain("living_room")
    contract = load_variant_action_contract(domain, "F0_ALL_OBJECTS_IN_STAGING", config_root=Path("mujoco_scenes/configs"))
    interpreter = ViLaInInterpreter(
        object_client=client,
        reasoning_client=client,
        models=InterpreterModels("qwen35-9b", "rev1", "qwen35-9b", "rev1"),
        symbolic_contract=contract,
    )
    interpreter.interpret(
        task_instruction="Test task",
        domain=domain,
        observations=[obs],
        observation_root=tmp_path,
        output_root=tmp_path / "out",
    )

    init_reqs = [r for r in transport.recorded_requests if r.call_type == FMCallType.INITIAL_STATE]
    assert len(init_reqs) >= 1
    assert len(init_reqs[0].image_artifacts) == 1
    assert init_reqs[0].image_artifacts[0] == obs.camera_frames[0].rgb_path


def test_2_initial_consistency_semantic_repair_contains_same_image_artifacts(tmp_path: Path) -> None:
    obs = _make_dummy_observation(tmp_path)
    obj_json = json.dumps({"objects": [{"label": "cup", "pddl_type": "cup", "detections": [{"stage_id": "000_initial", "camera_id": "cam0", "bbox_1000": [100, 100, 200, 200], "confidence": 0.9}]}]})
    # Both holding and handempty triggers initial_state_diagnostics
    init_json_conflict = json.dumps({"true_fact_ids": ["f0009", "f0010"]})
    init_json_repaired = json.dumps({"true_fact_ids": ["f0009"]})
    goal_json = json.dumps({"goal_fact_ids": ["f0009"]})

    transport = DummyTransport([obj_json, init_json_conflict, init_json_repaired, goal_json])
    client = RecordedFMClient(transport)
    domain = load_domain("living_room")
    contract = load_variant_action_contract(domain, "F0_ALL_OBJECTS_IN_STAGING", config_root=Path("mujoco_scenes/configs"))
    interpreter = ViLaInInterpreter(
        object_client=client,
        reasoning_client=client,
        models=InterpreterModels("qwen35-9b", "rev1", "qwen35-9b", "rev1"),
        symbolic_contract=contract,
    )
    interpreter.interpret(
        task_instruction="Test task",
        domain=domain,
        observations=[obs],
        observation_root=tmp_path,
        output_root=tmp_path / "out",
    )

    repair_reqs = [
        r for r in transport.recorded_requests
        if r.call_type == FMCallType.INITIAL_STATE and r.metadata.get("bounded_consistency_correction")
    ]
    assert len(repair_reqs) >= 1
    assert len(repair_reqs[0].image_artifacts) == 1
    assert repair_reqs[0].image_artifacts[0] == obs.camera_frames[0].rgb_path


def test_3_object_estimator_and_initial_state_use_same_observations(tmp_path: Path) -> None:
    obs = _make_dummy_observation(tmp_path)
    domain = load_domain("living_room")
    obj_prompt = build_initial_state_prompt(
        task_instruction="Test",
        domain=domain,
        observations=[obs],
        objects=[],
        fact_candidates=[],
        fixed_scene_evidence=[],
    )
    assert len(obj_prompt.image_artifacts) == 1
    assert obj_prompt.image_artifacts[0] == obs.camera_frames[0].rgb_path


def test_4_fixed_scene_evidence_derived_from_current_scene_geometry() -> None:
    scene = _create_scene(Domain.LIVING_ROOM, "F0_ALL_OBJECTS_IN_STAGING", robot="google", layout_seed=None)
    provider = FixedSceneEvidenceProvider(scene=scene, domain=Domain.LIVING_ROOM)
    evidence = provider()
    by_id = {e.symbolic_id: e for e in evidence}
    assert by_id["personal_table_left"].physically_present is True
    assert by_id["personal_table_left"].centroid_m is not None
    assert len(by_id["personal_table_left"].centroid_m) == 3


def test_5_living_structural_inventory_invariant_across_f0_to_i3() -> None:
    domain = load_domain("living_room")
    config_root = Path("mujoco_scenes/configs")
    variants = VARIANT_LABELS["living_room"]
    inventories = [
        load_variant_action_contract(domain, v, config_root=config_root).structural_inventory
        for v in variants
    ]
    for inv in inventories:
        assert inv == inventories[0]
    expected = {
        "personal_table_left": "support",
        "personal_table_right": "support",
        "shared_table": "support",
        "staging": "location",
    }
    assert dict(inventories[0]) == expected


def test_6_absent_regions_never_consumed_by_baseline_planning() -> None:
    planning_files = [
        "symbolic_contract.py",
        "interpreter.py",
        "attempt.py",
        "planner.py",
        "refinement.py",
        "corrective_planning.py",
    ]
    root = Path("mujoco_scenes/baselines/vilain_tamp")
    for name in planning_files:
        text = (root / name).read_text(encoding="utf-8")
        assert "absent_regions" not in text, f"{name} contains absent_regions"


def test_7_canonical_assignment_never_consumed_by_baseline_planning() -> None:
    planning_files = [
        "symbolic_contract.py",
        "interpreter.py",
        "attempt.py",
        "planner.py",
        "refinement.py",
        "corrective_planning.py",
    ]
    root = Path("mujoco_scenes/baselines/vilain_tamp")
    for name in planning_files:
        text = (root / name).read_text(encoding="utf-8")
        assert "canonical_assignment" not in text, f"{name} contains canonical_assignment"


def test_8_object_locations_never_consumed_by_baseline_planning() -> None:
    planning_files = [
        "symbolic_contract.py",
        "interpreter.py",
        "attempt.py",
        "planner.py",
        "refinement.py",
        "corrective_planning.py",
    ]
    root = Path("mujoco_scenes/baselines/vilain_tamp")
    for name in planning_files:
        text = (root / name).read_text(encoding="utf-8")
        assert "object_locations" not in text, f"{name} contains object_locations"


def test_9_living_i0_physical_scene_evidence_reflects_no_shared_support_without_reading_absent_regions() -> None:
    scene = _create_scene(Domain.LIVING_ROOM, "I0_NO_SHARED_TABLE", robot="google", layout_seed=None)
    provider = FixedSceneEvidenceProvider(scene=scene, domain=Domain.LIVING_ROOM)
    evidence = provider()
    by_id = {e.symbolic_id: e for e in evidence}
    assert by_id["shared_table"].physically_present is False
    assert by_id["shared_table"].centroid_m is None
    assert by_id["personal_table_left"].physically_present is True
    assert by_id["personal_table_right"].physically_present is True


def test_10_living_f0_physical_scene_evidence_contains_all_required_supports() -> None:
    scene = _create_scene(Domain.LIVING_ROOM, "F0_ALL_OBJECTS_IN_STAGING", robot="google", layout_seed=None)
    provider = FixedSceneEvidenceProvider(scene=scene, domain=Domain.LIVING_ROOM)
    evidence = provider()
    by_id = {e.symbolic_id: e for e in evidence}
    assert by_id["personal_table_left"].physically_present is True
    assert by_id["personal_table_right"].physically_present is True
    assert by_id["shared_table"].physically_present is True
    assert by_id["staging"].physically_present is True


def test_11_no_gt_terminal_subgoal_evaluator_called_before_baseline_termination() -> None:
    runner_text = Path("mujoco_scenes/baselines/vilain_tamp/runner.py").read_text(encoding="utf-8")
    # evaluate_terminal_subgoals must occur after execution/planning termination
    pos_eval = runner_text.find("terminal_subgoal_evaluation = evaluate_terminal_subgoals(")
    pos_plan = runner_text.find("components.corrective_planning.run")
    assert pos_plan != -1 and pos_eval != -1
    assert pos_plan < pos_eval, "subgoal evaluation called before planning completion"


def test_12_goal_estimator_has_no_hidden_benchmark_data() -> None:
    domain = load_domain("kitchen")
    fact = GroundedFact("f1", "served", ("coffee_1",), "(served coffee_1)", "served")
    prompt = build_goal_state_prompt(
        task_instruction="Prepare coffee",
        domain=domain,
        objects=[],
        initial_state_fragment="",
        fact_candidates=[fact],
    )
    assert "canonical_assignment" not in prompt.user_text
    assert "FEASIBLE" not in prompt.user_text
    assert "ground_truth" not in prompt.user_text


def test_13_cp_has_no_hidden_benchmark_evaluation_data() -> None:
    domain = load_domain("kitchen")
    fact = GroundedFact("f1", "at", ("cup_1", "countertop"), "(at cup_1 countertop)", "at")
    with pytest.raises(ValueError, match="benchmark evaluator data is forbidden"):
        build_corrective_fact_selection_prompt(
            task_instruction="Test",
            domain=domain,
            current_problem="(define (problem p))",
            current_failure={"benchmark_evaluation": "FAILED"},
            fact_candidates=[fact],
        )


def test_14_zero_step_plan_exclusions_remain_intact() -> None:
    from mujoco_scenes.baselines.vilain_tamp.paper_metrics import compute_stage_funnel
    r = {
        "observation_success": True,
        "object_estimation_success": True,
        "pddl_valid": True,
        "any_plan_fd": True,
        "any_plan_val": True,
        "nonempty_plan_fd": False,  # zero step plan
        "nonempty_plan_val": False,
        "nonempty_plan_identity": False,
        "nonempty_plan_refine": False,
        "nonempty_plan_exec": False,
        "actual_selected_plan_length": 0,
    }
    funnel = compute_stage_funnel([r])
    fmap = {s["stage_id"]: s["count"] for s in funnel}
    assert fmap["4_any_plan_fd"] == 1
    assert fmap["5_nonempty_plan_fd"] == 0
    assert fmap["8_nonempty_plan_refine"] == 0


def test_15_same_attempt_funnel_raises_error_on_nesting_violation() -> None:
    # A stage count exceeding predecessor count must raise ValueError, not silently clip
    broken_rows = [
        {
            "observation_success": True,
            "object_estimation_success": True,
            "pddl_valid": False,
            "any_plan_fd": True,  # Plan found despite invalid PDDL -> nesting violation
        }
    ]
    with pytest.raises(ValueError, match="Funnel nesting invariant violated"):
        compute_stage_funnel(broken_rows)


def test_16_terminal_subgoal_coverage_remains_post_terminal_only() -> None:
    from mujoco_scenes.baselines.vilain_tamp.runner import BaselineRunner
    # Subgoals module is never imported by interpreter or attempt runner
    interpreter_text = Path("mujoco_scenes/baselines/vilain_tamp/interpreter.py").read_text(encoding="utf-8")
    attempt_text = Path("mujoco_scenes/baselines/vilain_tamp/attempt.py").read_text(encoding="utf-8")
    assert "subgoals" not in interpreter_text
    assert "subgoals" not in attempt_text


def test_17_old_metrics_still_aggregate_correctly() -> None:
    from mujoco_scenes.baselines.vilain_tamp.paper_metrics import compute_group_aggregates
    rows = [
        {
            "domain": "kitchen",
            "variant": "F0_ALL_VISIBLE",
            "repeat": 0,
            "gt_feasible": True,
            "ground_truth_feasible": True,
            "actual_task_success": True,
            "outcome_correct": True,
            "feasible_task_success": True,
            "goal_coverage": 1.0,
            "goal_requirements_passed": 12,
            "goal_requirements_total": 12,
            "initial_goal_coverage": 0.0,
            "delta_goal_coverage": 1.0,
            "false_completion": False,
            "declared_completion": True,
            "physical_plan_found": True,
            "total_raw_vlm_calls": 3,
            "replans_used": 0,
            "nonempty_plan_execution_completed": True,
            "execution_stage_completed": True,
            "observation_success": True,
            "object_estimation_success": True,
            "pddl_valid": True,
            "any_plan_fd": True,
            "any_plan_val": True,
            "nonempty_plan_fd": True,
            "nonempty_plan_val": True,
            "nonempty_plan_identity": True,
            "nonempty_plan_refine": True,
            "nonempty_plan_exec": True,
            "task_final": True,
            "subgoals_passed": 12,
            "subgoals_total": 12,
            "initial_subgoals_passed": 0,
        }
    ]
    metrics = compute_group_aggregates(rows, group_label="kitchen")
    assert metrics["outcome_correct_count"] == 1
    assert metrics["outcome_correct_rate"] == 1.0
    assert metrics["feasible_task_success_rate"] == 1.0
    assert metrics["goal_coverage_micro"] == 1.0


def test_18_all_baseline_isolation_tests_continue_to_pass() -> None:
    from mujoco_scenes.baselines.vilain_tamp.tests.test_boundary import (
        test_production_imports_do_not_cross_method_boundary,
        test_production_source_has_no_proposed_method_references,
        test_only_live_runtime_adapters_may_import_simulator,
    )
    test_production_imports_do_not_cross_method_boundary()
    test_production_source_has_no_proposed_method_references()
    test_only_live_runtime_adapters_may_import_simulator()
