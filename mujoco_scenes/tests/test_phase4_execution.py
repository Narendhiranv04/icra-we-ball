import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest
import mujoco
from mujoco_scenes.generic_manipulation import environment_collision_is_allowed

from mujoco_scenes.phase4_execution import (
    ActionExecutionResult,
    ExecutionFailure,
    Phase4Executor,
    Phase4EntityMappingError,
    Phase4LiveViewer,
    Phase4ViewerClosed,
    ResolvedEntity,
    UpstreamPhase3Blocked,
    load_phase3_handoff,
    audit_strict_telemetry,
    classify_planner_failure,
    normalize_planner_failure_code,
)
from mujoco_scenes.phase4_kitchen import KitchenPhase4Adapter
from mujoco_scenes.phase4_workshop_entities import (
    WorkshopEntityResolutionError,
    resolve_workshop_entities,
    workshop_body_world_geometry_aabb_center,
)
from mujoco_scenes.phase4_workshop import (
    WorkshopPhase4Adapter,
    planner_failure_code,
    strict_workshop_place_block,
    resolve_workshop_entities_for_execution,
)
from mujoco_scenes.workshop_ground_truth_execution import (
    WorkshopExecutionDispatcher,
    strict_insertion_verified,
    strict_surface_place_verified,
    strict_grasp_attachment_verified,
    strict_pick_source_clearance_verified,
    reviewed_workshop_grasp_geometries,
    benchmark_open_verified,
    workshop_preclose_limit_m,
)
from mujoco_scenes.living_room_mobile_execution import (
    post_release_dynamics_modification_enabled,
)
from mujoco_scenes.phase4_living_room import (
    normalize_living_room_action_result,
    resolve_living_room_action_arguments,
    validate_living_room_plan_ids,
)
from mujoco_scenes import run_workshop_phase4_controller_development as workshop_harness
from mujoco_scenes.run_phase4_execution import build_parser, execute_phase3_run


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _handoff(tmp_path: Path, *, inspected_regions=(), action_count=1):
    actions = [
        {
            "action_index": index,
            "action_instance_id": f"fact_{index:03d}_pick",
            "operator": "PICK",
            "arguments": [f"object_{index:04d}"],
        }
        for index in range(1, action_count + 1)
    ]
    replay = {
        "status": "VALID",
        "goal_status": "GOAL_SATISFIED",
        "missing_goals": [],
        "final_atoms": [],
        "steps": [
            {
                "step": index,
                "operator": action["operator"],
                "arguments": action["arguments"],
                "status": "VALID",
                "failure": None,
            }
            for index, action in enumerate(actions)
        ],
        "validator": "independent_symbolic_replay_v1",
        "uses_planner_transition": False,
    }
    _write(
        tmp_path / "run_manifest.json",
        {
            "domain": "kitchen",
            "variant": "K1",
            "internal_variant": "F0_ALL_VISIBLE",
            "spec_provider_source": "GT_FUNCTIONAL_SPEC_ONLY",
            "specification_sha256": "abc",
            "execution_state": "planning_only",
            "terminal_status": "ACTION_SEQUENCE_READY",
            "artifacts": {
                "final_plan": "action_sequence/action_plan.json",
                "replay_validation": "action_sequence/replay_validation.json",
                "plan_grounding_audit": "plan_grounding_audit.json",
                "observed_graph": "observed_scene_graph.json",
                "result": "result.json",
            },
        },
    )
    _write(
        tmp_path / "result.json",
        {
            "status": "ACTION_SEQUENCE_READY",
            "plan": actions,
            "inspected_regions": list(inspected_regions),
        },
    )
    _write(
        tmp_path / "graph_grounding_result.json",
        {
            "complete": True,
            "assignment": {"tool": "object_0001"},
            "operation_bindings": {},
        },
    )
    _write(
        tmp_path / "action_sequence/action_plan.json",
        {"actions": actions, "validation": replay},
    )
    _write(
        tmp_path / "action_sequence/replay_validation.json",
        replay,
    )
    _write(
        tmp_path / "plan_grounding_audit.json",
        {
            "violations": [],
            "plan_replay_valid": True,
            "grounding_complete": True,
        },
    )
    _write(tmp_path / "observed_scene_graph.json", {"objects": {}})
    return load_phase3_handoff(tmp_path)


def test_load_phase3_handoff_preserves_exact_plan_contract(tmp_path):
    handoff = _handoff(tmp_path)
    assert handoff.source == "GT_FUNCTIONAL_SPEC_ONLY"
    assert handoff.assignment == {"tool": "object_0001"}
    assert handoff.actions[0]["arguments"] == ["object_0001"]
    assert set(handoff.artifact_sha256) == {
        "manifest", "result", "grounding", "plan", "replay_validation",
        "plan_grounding_audit",
        "observed_graph",
    }
    assert handoff.replay_validation_source == "EXPLICIT_REPLAY_ARTIFACT"


def test_living_room_explicit_replay_handoff_still_loads(tmp_path):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["domain"] = "living_room"
    manifest["variant"] = "L1"
    _write(tmp_path / "run_manifest.json", manifest)
    handoff = load_phase3_handoff(tmp_path)
    assert handoff.domain == "living_room"
    assert handoff.variant == "L1"
    assert handoff.replay_validation_source == "EXPLICIT_REPLAY_ARTIFACT"


def test_ready_handoff_requires_manifest_declared_result(tmp_path):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["artifacts"].pop("result")
    _write(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="result artifact"):
        load_phase3_handoff(tmp_path)


@pytest.mark.parametrize("domain", ["kitchen", "living_room", "workshop"])
def test_valid_embedded_replay_is_accepted_generically(tmp_path, domain):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["domain"] = domain
    manifest["artifacts"].pop("replay_validation")
    _write(tmp_path / "run_manifest.json", manifest)
    handoff = load_phase3_handoff(tmp_path)
    assert handoff.replay_validation_source == (
        "EMBEDDED_FINAL_PLAN_VALIDATION"
    )
    assert handoff.actions[0]["arguments"] == ["object_0001"]


def test_missing_explicit_and_embedded_replay_evidence_fails_closed(tmp_path):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["artifacts"].pop("replay_validation")
    _write(tmp_path / "run_manifest.json", manifest)
    plan = json.loads(
        (tmp_path / "action_sequence/action_plan.json").read_text()
    )
    plan.pop("validation")
    _write(tmp_path / "action_sequence/action_plan.json", plan)
    with pytest.raises(ValueError, match="evidence is missing"):
        load_phase3_handoff(tmp_path)


def test_non_valid_replay_status_is_rejected(tmp_path):
    _handoff(tmp_path)
    validation_path = tmp_path / "action_sequence/replay_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["status"] = "INVALID"
    _write(validation_path, validation)
    with pytest.raises(ValueError, match="did not validate"):
        load_phase3_handoff(tmp_path)


def test_replay_steps_must_equal_generated_action_sequence(tmp_path):
    _handoff(tmp_path)
    validation_path = tmp_path / "action_sequence/replay_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["steps"][0]["arguments"] = ["different_object"]
    _write(validation_path, validation)
    with pytest.raises(ValueError, match="differs from final action"):
        load_phase3_handoff(tmp_path)


def test_load_phase3_handoff_rejects_result_plan_mismatch(tmp_path):
    _handoff(tmp_path)
    _write(
        tmp_path / "result.json",
        {"status": "ACTION_SEQUENCE_READY", "plan": []},
    )
    with pytest.raises(ValueError, match="non-empty|differs"):
        load_phase3_handoff(tmp_path)


def test_load_phase3_handoff_classifies_current_upstream_incomplete(tmp_path):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["terminal_status"] = "INCOMPLETE"
    manifest["artifacts"].pop("final_plan")
    _write(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(UpstreamPhase3Blocked, match="CURRENT_UPSTREAM_PHASE3_BLOCKED"):
        load_phase3_handoff(tmp_path)


def test_upstream_failure_needs_only_manifest(tmp_path):
    _write(tmp_path / "run_manifest.json", {"terminal_status": "VLM_SPEC_FAILED"})
    with pytest.raises(UpstreamPhase3Blocked, match="VLM_SPEC_FAILED"):
        load_phase3_handoff(tmp_path)


def test_missing_replay_validation_rejects_handoff(tmp_path):
    _handoff(tmp_path)
    (tmp_path / "action_sequence/replay_validation.json").unlink()
    with pytest.raises(FileNotFoundError, match="replay_validation"):
        load_phase3_handoff(tmp_path)


def test_invalid_plan_grounding_audit_rejects_handoff(tmp_path):
    _handoff(tmp_path)
    _write(tmp_path / "plan_grounding_audit.json", {
        "violations": ["bad"],
        "plan_replay_valid": True,
        "grounding_complete": True,
    })
    with pytest.raises(ValueError, match="plan-grounding audit"):
        load_phase3_handoff(tmp_path)


class _Adapter:
    entity_resolution = {"all_resolved": True}

    def __init__(self, succeed=True, inspection_succeed=True):
        self.succeed = succeed
        self.inspection_succeed = inspection_succeed
        self.calls = []

    def execute_inspection_open(self, region):
        return {
            "region": region,
            "success": self.inspection_succeed,
            "failure": (
                ExecutionFailure.NONE.value
                if self.inspection_succeed
                else ExecutionFailure.CONTROLLER_FAILURE.value
            ),
            "direct_container_state_write_used": False,
        }

    def execute_action(self, action):
        self.calls.append(action)
        return ActionExecutionResult(
            action_index=action["action_index"],
            action_instance_id=action["action_instance_id"],
            operator=action["operator"],
            arguments=action["arguments"],
            success=self.succeed,
            failure=(
                ExecutionFailure.NONE.value
                if self.succeed
                else ExecutionFailure.CONTROLLER_FAILURE.value
            ),
            resolved_arguments=[],
            primitive="test",
            pre_check={"success": True},
            controller_result={"success": self.succeed},
            post_check={"success": self.succeed},
            wall_duration_s=0.0,
        )

    def final_verification(self):
        return {"performed": True, "success": True}


def test_executor_records_partial_smoke_without_claiming_full_verification(tmp_path):
    handoff = _handoff(tmp_path, action_count=2)
    result = Phase4Executor(handoff, _Adapter()).run(max_actions=1)
    assert not result["success"]
    assert not result["full_sequence_requested"]
    assert result["partial_smoke"]
    assert result["partial_smoke_success"]
    assert not result["final_verification"]["performed"]


def test_executor_stops_and_classifies_controller_failure(tmp_path):
    handoff = _handoff(tmp_path)
    result = Phase4Executor(handoff, _Adapter(succeed=False)).run()
    assert not result["success"]
    assert result["actions_completed"] == 0
    assert result["failure"] == ExecutionFailure.CONTROLLER_FAILURE.value


def test_phase4_cli_viewer_flag_defaults_off_and_parses_on():
    parser = build_parser()
    base = ["--domain", "kitchen", "--variant", "K1"]
    assert parser.parse_args(base).viewer is False
    assert parser.parse_args([*base, "--viewer"]).viewer is True
    assert inspect.signature(execute_phase3_run).parameters["viewer"].default is False


def test_passive_viewer_syncs_same_model_data_and_closes(monkeypatch):
    model, data = object(), object()

    class Handle:
        running = True
        sync_calls = 0
        closed = False

        def is_running(self):
            return self.running

        def sync(self):
            self.sync_calls += 1

        def close(self):
            self.closed = True

    handle = Handle()
    import mujoco.viewer
    launch = []
    monkeypatch.setattr(
        mujoco.viewer, "launch_passive",
        lambda received_model, received_data: (
            launch.append((received_model, received_data)) or handle
        ),
    )
    viewer = Phase4LiveViewer(model, data)
    viewer.sync()
    assert launch == [(model, data)]
    assert handle.sync_calls == 1
    handle.running = False
    with pytest.raises(Phase4ViewerClosed):
        viewer.sync()
    viewer.close()
    assert handle.closed


def test_executor_prints_one_based_task_progress_without_changing_result(
    tmp_path, capsys
):
    handoff = _handoff(tmp_path)
    result = Phase4Executor(handoff, _Adapter()).run()
    output = capsys.readouterr().out
    assert "[TASK 01/01] PICK(object_0001)" in output
    assert "[TASK 01/01] SUCCESS PICK(object_0001)" in output
    assert result["final_action_sequence"] == list(handoff.actions)


def test_executor_prints_inspections_separately_from_task_indices(tmp_path, capsys):
    Phase4Executor(
        _handoff(tmp_path, inspected_regions=("C2", "B1")), _Adapter()
    ).run()
    output = capsys.readouterr().out
    assert "[INSPECTION 01/02] OPEN(C2)" in output
    assert "[INSPECTION 02/02] OPEN(B1)" in output
    assert "[TASK 01/01] PICK(object_0001)" in output


def test_executor_prints_failed_action_status_and_failure_code(tmp_path, capsys):
    class FailedAdapter(_Adapter):
        def execute_action(self, action):
            result = super().execute_action(action)
            result.success = False
            result.failure = ExecutionFailure.CONTROLLER_FAILURE.value
            result.failure_code = "ACCESS_BLOCKED"
            result.controller_result = {
                "success": False, "status": "GRASP_FAILED"
            }
            return result

    result = Phase4Executor(_handoff(tmp_path), FailedAdapter()).run()
    output = capsys.readouterr().out
    assert "[TASK 01/01] FAILED PICK(object_0001)" in output
    assert "controller_status=GRASP_FAILED" in output
    assert "failure_code=ACCESS_BLOCKED" in output
    assert result["failure_code"] == "ACCESS_BLOCKED"


def test_executor_contains_unexpected_adapter_exception(tmp_path):
    class RaisingAdapter(_Adapter):
        def execute_action(self, action):
            raise ValueError("unexpected native failure")

    result = Phase4Executor(_handoff(tmp_path), RaisingAdapter()).run()
    row = result["action_results"][0]
    assert result["failure_stage"] == "TASK_ACTION"
    assert row["failure"] == ExecutionFailure.CONTROLLER_FAILURE.value
    assert row["failure_code"] == "EXECUTION_ERROR"
    assert row["controller_result"]["failure_type"] == "ValueError"


def test_failure_code_normalization_preserves_only_public_codes():
    assert normalize_planner_failure_code(
        "MOTION_INFEASIBLE", "ignored",
        infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
    ) == "MOTION_INFEASIBLE"
    assert normalize_planner_failure_code(
        "IK_INTERNAL_17", "IK_UNREACHABLE",
        infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
    ) == "MOTION_INFEASIBLE"
    assert normalize_planner_failure_code(
        None, "collision during approach",
        infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
    ) == "MOTION_INFEASIBLE"
    assert normalize_planner_failure_code(
        None, "bilateral grasp contact missing",
        infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
    ) == "ACCESS_BLOCKED"
    assert normalize_planner_failure_code(
        "PRIVATE_RETRY_CODE", "unknown internal controller issue",
        infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
    ) == "EXECUTION_ERROR"


def test_executor_replays_inspection_before_immutable_plan(tmp_path):
    handoff = _handoff(tmp_path, inspected_regions=("D1", "C2"))
    adapter = _Adapter()
    result = Phase4Executor(handoff, adapter).run()
    assert result["inspection_execution"]["regions"] == ["D1", "C2"]
    assert result["inspection_execution"]["actions_completed"] == 2
    assert adapter.calls == list(handoff.actions)
    assert result["execution_mode"] == "P4_BENCH"
    assert result["strict_execution"] is False
    assert result["direct_task_state_fallback_used"] is False


def test_executor_stops_before_plan_when_inspection_fails(tmp_path):
    handoff = _handoff(tmp_path, inspected_regions=("D1",))
    adapter = _Adapter(inspection_succeed=False)
    result = Phase4Executor(handoff, adapter).run()
    assert not result["success"]
    assert result["failure_stage"] == "INSPECTION_OPEN"
    assert result["failure"] == "INSPECTION_EXECUTION_FAILURE"
    assert adapter.calls == []


class _AssistedTelemetryAdapter(_Adapter):
    def execute_action(self, action):
        result = super().execute_action(action)
        result.controller_result["direct_payload_pose_write"] = True
        return result


def test_benchmark_executor_retains_strict_audit_without_gating_success(tmp_path):
    result = Phase4Executor(
        _handoff(tmp_path), _AssistedTelemetryAdapter()
    ).run()
    assert result["success"]
    assert result["failure"] == ExecutionFailure.NONE.value
    assert result["direct_payload_state_write_used"] is True
    assert result["direct_task_state_fallback_used"] is False
    assert result["strict_telemetry_verification"]["violations"][0]["flag"] == (
        "direct_payload_pose_write"
    )


def test_strict_audit_rejects_ungated_handle_constraint():
    audit = audit_strict_telemetry([{
        "physical_handle_grasp_constraint_used": True,
        "handle_grasp_constraint_contact_gated": False,
    }], [])
    assert not audit["verified"]
    assert audit["violations"][0]["flag"] == (
        "ungated_handle_grasp_constraint_used"
    )


class _FailedViolatingAdapter(_Adapter):
    def execute_action(self, action):
        result = super().execute_action(action)
        result.success = False
        result.failure = ExecutionFailure.CONTROLLER_FAILURE.value
        result.controller_result["staging_fixture_used"] = True
        return result


def test_failed_primitive_is_included_in_strict_telemetry_audit(tmp_path):
    result = Phase4Executor(
        _handoff(tmp_path), _FailedViolatingAdapter()
    ).run()
    assert result["strict_execution_violation_detected"] is True
    assert result["assisted_task_fixture_used"] is True
    assert any(
        row["flag"] == "staging_fixture_used"
        for row in result["strict_telemetry_verification"]["violations"]
    )


def test_nested_target_alignment_sets_violation_and_assisted_aggregate():
    audit = audit_strict_telemetry([], [{
        "controller_result": {
            "insertion": {"target_alignment_constraint_used": True}
        }
    }])
    assert audit["strict_execution_violation_detected"] is True
    assert audit["assisted_task_fixture_used"] is True


def test_workshop_controller_harness_reports_benchmark_and_strict_audit():
    source = inspect.getsource(workshop_harness.run_controller_sequence)
    assert "audit_strict_telemetry" in source
    assert 'payload["benchmark_execution_mode"] = True' in source


def test_geom_allowance_does_not_exempt_sibling_furniture_geometry():
    allowed_geoms = frozenset({11})
    assert environment_collision_is_allowed(5, 11, frozenset(), allowed_geoms)
    assert not environment_collision_is_allowed(5, 12, frozenset(), allowed_geoms)


def _workshop_observed(x=0.2):
    return {
        "objects": {
            "object_0007": {
                "instance_id": "object_0007",
                "source_region": "TOOL_CABINET",
                "canonical_category": "power_driver",
                "geometry": {"centroid_world_m": [x, 0.5, 0.8]},
            }
        }
    }


def _workshop_pick():
    return [{
        "operator": "PICK",
        "arguments": ["object_0007", "TOOL_CABINET"],
    }]


def test_workshop_resolves_current_canonical_instance_by_source_and_geometry():
    result = resolve_workshop_entities(
        ["object_0007"],
        _workshop_observed(),
        _workshop_pick(),
        [
            {
                "simulator_id": "workshop_manual_driver_long",
                "source_region": "TOOL_CABINET",
                "centroid_world_m": [0.201, 0.5, 0.8],
            },
            {
                "simulator_id": "workshop_power_driver",
                "source_region": "LEFT_DRAWER",
                "centroid_world_m": [0.2, 0.5, 0.8],
            },
        ],
    )
    row = result["objects"][0]
    assert row["planner_id"] == "object_0007"
    assert row["simulator_id"] == "workshop_manual_driver_long"
    assert row["evidence"]["semantic_evidence_used_for_selection"] is False


def test_workshop_ambiguous_geometry_mapping_fails_closed():
    with pytest.raises(WorkshopEntityResolutionError, match="Ambiguous"):
        resolve_workshop_entities(
            ["object_0007"],
            _workshop_observed(x=0.0),
            _workshop_pick(),
            [
                {
                    "simulator_id": "candidate_a",
                    "source_region": "TOOL_CABINET",
                    "centroid_world_m": [-0.004, 0.5, 0.8],
                },
                {
                    "simulator_id": "candidate_b",
                    "source_region": "TOOL_CABINET",
                    "centroid_world_m": [0.004, 0.5, 0.8],
                },
            ],
        )


def test_workshop_source_only_mapping_fails_closed():
    observed = _workshop_observed()
    del observed["objects"]["object_0007"]["geometry"]
    with pytest.raises(WorkshopEntityResolutionError, match="centroid evidence"):
        resolve_workshop_entities(
            ["object_0007"], observed, _workshop_pick(), [{
                "simulator_id": "only_candidate",
                "source_region": "TOOL_CABINET",
                "centroid_world_m": [0.2, 0.5, 0.8],
            }]
        )


def test_workshop_distant_centroid_mapping_fails_closed():
    with pytest.raises(WorkshopEntityResolutionError, match="absolute gate"):
        resolve_workshop_entities(
            ["object_0007"], _workshop_observed(), _workshop_pick(), [{
                "simulator_id": "distant_candidate",
                "source_region": "TOOL_CABINET",
                "centroid_world_m": [1.0, 0.5, 0.8],
            }]
        )


def test_workshop_candidate_centroid_uses_geometry_aabb_not_body_origin():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body name="payload" pos="0 0 0">
          <freejoint/><geom type="box" pos="1 0 0" size="0.2 0.1 0.1"/>
        </body></worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    centroid, definition = workshop_body_world_geometry_aabb_center(
        model, data, body_id
    )
    assert centroid == pytest.approx([1.0, 0.0, 0.0])
    assert centroid != pytest.approx(data.xpos[body_id].tolist())
    assert definition == "WORLD_GEOMETRY_AABB_CENTER"


def test_workshop_resolution_failure_has_phase4_mapping_type():
    with pytest.raises(Phase4EntityMappingError, match="absolute gate"):
        resolve_workshop_entities_for_execution(
            ["object_0007"], _workshop_observed(), _workshop_pick(), [{
                "simulator_id": "distant_candidate",
                "source_region": "TOOL_CABINET",
                "centroid_world_m": [1.0, 0.5, 0.8],
            }]
        )


def test_workshop_open_without_handle_contact_never_activates_weld():
    dispatcher = WorkshopExecutionDispatcher.__new__(
        WorkshopExecutionDispatcher
    )
    dispatcher.scene = SimpleNamespace(
        data=SimpleNamespace(eq_active=np.array([0], dtype=np.uint8))
    )
    result = dispatcher._activate_handle_grasp_constraint(0, 0, {
        "bilateral_handle_contact_confirmed": False,
        "bilateral_handle_contact_steps": 0,
        "handle_contact_geometry": "left_drawer_handle_col",
        "finger_handle_contacts": [],
    })
    assert result["status"] == "STRICT_PHYSICAL_HANDLE_GRASP_UNAVAILABLE"
    assert not result["physical_handle_grasp_constraint_used"]
    assert not result["storage_joint_intentionally_opened"]
    assert int(dispatcher.scene.data.eq_active[0]) == 0


def test_workshop_large_preclose_miss_is_never_a_tolerance_fix():
    for source, object_name in (
        ("TOOL_CABINET", "workshop_wooden_hammer"),
        ("LEFT_DRAWER", "workshop_medium_phillips_screw"),
        ("RIGHT_DRAWER", "workshop_power_driver"),
    ):
        assert workshop_preclose_limit_m(source, object_name) < 0.4


def test_workshop_pick_attachment_remains_bilateral_contact_gated():
    source = inspect.getsource(WorkshopExecutionDispatcher._activate_grasp)
    assert "require_bilateral" in source
    assert "attachment requested without bilateral" in source
    assert "eq_active" in source


def test_workshop_object_attachment_strict_translation_and_angle_boundaries():
    valid = dict(bilateral_contact=True, translation_snap_m=0.004, angle_snap_rad=0.02)
    assert strict_grasp_attachment_verified(**valid)
    assert not strict_grasp_attachment_verified(**(valid | {"translation_snap_m": 0.004001}))
    assert not strict_grasp_attachment_verified(**(valid | {"angle_snap_rad": 0.020001}))
    assert not strict_grasp_attachment_verified(**(valid | {"bilateral_contact": False}))


def test_workshop_reviewed_grasp_geometry_is_not_body_wide():
    assert reviewed_workshop_grasp_geometries(
        "workshop_long_phillips_driver", "TOOL_CABINET"
    ) == ("workshop_long_phillips_driver_col_handle",)
    assert reviewed_workshop_grasp_geometries(
        "workshop_medium_phillips_screw", "TOOL_CABINET"
    ) == ("workshop_medium_phillips_screw_col_shaft",)
    assert reviewed_workshop_grasp_geometries(
        "workshop_medium_phillips_screw", "LEFT_DRAWER"
    ) == ("workshop_medium_phillips_screw_col_head",)


def test_workshop_pick_requires_source_clearance_not_only_active_grasp():
    assert strict_pick_source_clearance_verified(
        source_clearance_m=0.005, displacement_m=0.08
    )
    assert not strict_pick_source_clearance_verified(
        source_clearance_m=-0.001, displacement_m=0.40
    )


def test_workshop_pick_uses_live_resolved_object_geometry():
    source = inspect.getsource(WorkshopExecutionDispatcher._object_grasp_position)
    assert "mjOBJ_GEOM" in source
    assert "geom_xpos" in source
    assert "data.xpos" not in source


def test_strict_insertion_source_has_no_legacy_target_fixture_calls():
    source = inspect.getsource(
        WorkshopExecutionDispatcher._strict_insert_fastener
    )
    assert "_activate_installed_fastener" not in source
    assert "workshop_alignment_weld" not in source
    assert "qpos[" not in source
    assert '"target_alignment_constraint_used": False' in source
    assert '"installed_fastener_constraint_used": False' in source


def test_strict_surface_release_skips_staging_constraint():
    source = inspect.getsource(WorkshopExecutionDispatcher.execute)
    assert "and not self.strict_physical_execution" in source
    assert 'result["staging_constraint_used"] = False' in source


def test_strict_surface_place_rejects_fall_or_retained_grasp():
    valid = dict(
        xy_margin_m=0.1, height_m=0.05, support_contact=True,
        grasp_inactive=True, held=False, linear_velocity_m_s=0.0,
        angular_velocity_rad_s=0.0,
    )
    assert strict_surface_place_verified(**valid)
    assert not strict_surface_place_verified(**(valid | {"support_contact": False}))
    assert not strict_surface_place_verified(**(valid | {"held": True}))
    assert not strict_surface_place_verified(**(valid | {"linear_velocity_m_s": 0.1}))


def test_strict_insertion_rejects_tip_axis_depth_and_contact_failures():
    valid = dict(
        lateral_error_m=0.001, axis_error_rad=0.01, depth_m=0.012,
        target_contact=True, held=False, linear_velocity_m_s=0.0,
        angular_velocity_rad_s=0.0,
    )
    assert strict_insertion_verified(**valid)
    assert not strict_insertion_verified(**(valid | {"lateral_error_m": 0.004}))
    assert not strict_insertion_verified(**(valid | {"axis_error_rad": 0.06}))
    assert not strict_insertion_verified(**(valid | {"depth_m": 0.004}))
    assert not strict_insertion_verified(**(valid | {"target_contact": False}))


def test_workshop_place_validation_preserves_exact_fastener_identity():
    insertion = strict_workshop_place_block({
        "operator": "PLACE",
        "arguments": ["object_0001", "workshop_frame_joint"],
    }, "object_0001")
    assert insertion is None
    wrong_object = strict_workshop_place_block({
        "operator": "PLACE",
        "arguments": ["object_0002", "workshop_frame_joint"],
    }, "object_0001")
    assert wrong_object["status"] == "INVALID_IMMUTABLE_WORKSHOP_PLAN"
    assert wrong_object["immutable_plan_precondition_mismatch"] is True
    surface = strict_workshop_place_block({
        "operator": "PLACE",
        "arguments": ["object_0002", "MAIN_WORKBENCH_ZONE"],
    }, "object_0001")
    assert surface is None


def test_benchmark_workshop_routes_place_and_screw_to_calibrated_dispatcher():
    init_source = inspect.getsource(WorkshopPhase4Adapter.__init__)
    execute_source = inspect.getsource(WorkshopPhase4Adapter.execute_action)
    assert "strict_physical_execution=False" in init_source
    assert "STRICT_PHYSICAL_SCREW_UNAVAILABLE" not in execute_source
    assert "strict_workshop_place_block(" not in execute_source


@pytest.mark.parametrize(("message", "expected"), (
    ("region is closed", "REGION_CLOSED"),
    ("IK_UNREACHABLE: no solution", "MOTION_INFEASIBLE"),
    ("GRASP_REJECTED: no contact", "ACCESS_BLOCKED"),
    ("wrong tool selected", "WRONG_TOOL"),
    ("unexpected controller fault", "EXECUTION_ERROR"),
))
def test_workshop_planner_failure_codes_hide_low_level_retry_details(
    message, expected
):
    assert planner_failure_code(message) == expected


def test_workshop_failure_code_propagates_alongside_infrastructure_failure():
    class State:
        def check(self, action, assignment):
            return True, None

        def to_dict(self):
            return {}

    class Dispatcher:
        def execute(self, action, state):
            raise RuntimeError("MOTION_INFEASIBLE: no collision-free IK candidate")

    adapter = WorkshopPhase4Adapter.__new__(WorkshopPhase4Adapter)
    adapter.by_id = {
        "object_0001": ResolvedEntity(
            "object_0001", "OBJECT", "workshop_power_driver"
        )
    }
    adapter.state = State()
    adapter.controller_state = object()
    adapter.assignment = object()
    adapter.dispatcher = Dispatcher()
    result = adapter.execute_action({
        "action_index": 1,
        "action_instance_id": "fact_001_pick",
        "operator": "PICK",
        "arguments": ["object_0001"],
    })
    assert result.failure == ExecutionFailure.CONTROLLER_FAILURE.value
    assert result.failure_code == "MOTION_INFEASIBLE"


def test_closed_region_precondition_uses_public_region_closed_code():
    class State:
        def check(self, action, assignment):
            return False, "source region LEFT_DRAWER is closed"

        def to_dict(self):
            return {}

    adapter = WorkshopPhase4Adapter.__new__(WorkshopPhase4Adapter)
    adapter.by_id = {
        "object_0001": ResolvedEntity(
            "object_0001", "OBJECT", "workshop_power_driver"
        )
    }
    adapter.state = State()
    adapter.assignment = object()
    result = adapter.execute_action({
        "action_index": 1,
        "action_instance_id": "fact_001_pick",
        "operator": "PICK",
        "arguments": ["object_0001"],
    })
    assert result.failure == ExecutionFailure.PRECONDITION_STATE_FAILURE.value
    assert result.failure_code == "REGION_CLOSED"


def test_living_pick_requires_held_postcondition_even_after_controller_success():
    action = {"operator": "PICK", "arguments": ["object_0001"]}
    result = normalize_living_room_action_result(
        action,
        {"result": "SUCCESS"},
        [],
        {"validation_status": "FALSE"},
    )
    assert not result["success"]
    assert result["failure"] == (
        ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
    )
    assert result["failure_code"] == "ACCESS_BLOCKED"


def test_kitchen_inspection_closes_interfering_region_and_preserves_history():
    class Dispatcher:
        def __init__(self):
            self.open_regions = set()
            self.closed = []

        def physically_open_containers(self):
            return set(self.open_regions)

        def open_container(self, region):
            self.open_regions.add(region)
            return {"success": True, "status": "OPENED"}

        def close_container(self, region):
            self.closed.append(region)
            self.open_regions.discard(region)
            return {"success": True, "status": "CLOSED"}

    adapter = KitchenPhase4Adapter.__new__(KitchenPhase4Adapter)
    adapter.dispatcher = Dispatcher()
    adapter.successful_inspection_history = []
    assert adapter.execute_inspection_open("C2")["success"]
    assert adapter.execute_inspection_open("B1")["success"]
    assert adapter.dispatcher.closed == ["C2"]
    assert adapter.successful_inspection_history == ["C2", "B1"]
    assert adapter.dispatcher.physically_open_containers() == {"B1"}
    adapter.expected_inspected_regions = ("C2", "B1")
    adapter.expected_actions = []
    adapter.successful_actions = []
    adapter.by_id = {}
    adapter._held_planner_id = lambda: None
    assert adapter.final_verification()["success"]


def test_kitchen_closed_pick_source_requires_prior_inspection():
    class Dispatcher:
        inventory_by_id = {
            "object_0001": {"source_context": {"source_container": "C2"}}
        }

        def physically_open_containers(self):
            return set()

    adapter = KitchenPhase4Adapter.__new__(KitchenPhase4Adapter)
    adapter.dispatcher = Dispatcher()
    adapter.successful_inspection_history = []
    action = {"operator": "PICK", "arguments": ["object_0001"]}
    result = adapter._prepare_pick_access(action)
    assert not result["success"]
    assert result["failure_code"] == "REGION_CLOSED"


def test_kitchen_inspected_closed_pick_prepares_access_without_changing_plan():
    class Dispatcher:
        inventory_by_id = {
            "object_0001": {"source_context": {"source_container": "C2"}}
        }

        def __init__(self):
            self.open_regions = {"B1"}

        def physically_open_containers(self):
            return set(self.open_regions)

        def close_container(self, region):
            self.open_regions.discard(region)
            return {"success": True}

        def open_container(self, region):
            self.open_regions.add(region)
            return {"success": True}

    action = {
        "action_index": 1,
        "action_instance_id": "fact_001_pick",
        "operator": "PICK",
        "arguments": ["object_0001"],
    }
    adapter = KitchenPhase4Adapter.__new__(KitchenPhase4Adapter)
    adapter.dispatcher = Dispatcher()
    adapter.successful_inspection_history = ["C2", "B1"]
    adapter.expected_actions = [dict(action)]
    result = adapter._prepare_pick_access(action)
    assert result["success"]
    assert result["conflicting_region"] == "B1"
    assert result["physical_close_verified"]
    assert result["physical_open_verified"]
    assert adapter.expected_actions == [action]
    assert adapter.dispatcher.open_regions == {"C2"}


def test_workshop_failed_physical_postcheck_does_not_apply_symbolic_state():
    class State:
        def __init__(self):
            self.applied = []

        def check(self, action, assignment):
            return True, None

        def apply(self, action):
            self.applied.append(action)

        def to_dict(self):
            return {"applied": len(self.applied)}

    class Dispatcher:
        held_object = None
        active_grasp_weld = -1

        def execute(self, action, state):
            return {"success": True, "grasp_weld_active": False}

    adapter = WorkshopPhase4Adapter.__new__(WorkshopPhase4Adapter)
    adapter.by_id = {
        "object_0001": ResolvedEntity(
            "object_0001", "OBJECT", "workshop_power_driver"
        )
    }
    adapter.state = State()
    adapter.controller_state = State()
    adapter.assignment = object()
    adapter.dispatcher = Dispatcher()
    adapter.scene = SimpleNamespace(
        data=SimpleNamespace(eq_active=np.zeros(1, dtype=bool)),
        state=SimpleNamespace(joint_repaired=False),
    )
    adapter.successful_actions = 0
    result = adapter.execute_action({
        "action_index": 1,
        "action_instance_id": "fact_001_pick",
        "operator": "PICK",
        "arguments": ["object_0001"],
    })
    assert result.failure == (
        ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
    )
    assert adapter.state.applied == []
    assert adapter.controller_state.applied == []


def test_workshop_verified_physical_postcheck_commits_both_states():
    class State:
        def __init__(self):
            self.applied = []

        def check(self, action, assignment):
            return True, None

        def apply(self, action):
            self.applied.append(action)

        def to_dict(self):
            return {"applied": len(self.applied)}

    class Dispatcher:
        held_object = "workshop_power_driver"
        active_grasp_weld = 0

        def execute(self, action, state):
            return {"success": True, "grasp_weld_active": True}

    adapter = WorkshopPhase4Adapter.__new__(WorkshopPhase4Adapter)
    adapter.by_id = {
        "object_0001": ResolvedEntity(
            "object_0001", "OBJECT", "workshop_power_driver"
        )
    }
    adapter.state = State()
    adapter.controller_state = State()
    adapter.assignment = object()
    adapter.dispatcher = Dispatcher()
    adapter.scene = SimpleNamespace(
        data=SimpleNamespace(eq_active=np.ones(1, dtype=bool)),
        state=SimpleNamespace(joint_repaired=False),
    )
    adapter.successful_actions = 0
    result = adapter.execute_action({
        "action_index": 1,
        "action_instance_id": "fact_001_pick",
        "operator": "PICK",
        "arguments": ["object_0001"],
    })
    assert result.success
    assert len(adapter.state.applied) == 1
    assert len(adapter.controller_state.applied) == 1


def test_kitchen_terminal_verifier_requires_exact_actions_held_state_and_open_regions():
    class Dispatcher:
        def physically_open_containers(self):
            return {"D1"}

    actions = [{
        "action_index": 1,
        "action_instance_id": "fact_001_pick",
        "operator": "PICK",
        "arguments": ["object_0001"],
    }, {
        "action_index": 2,
        "action_instance_id": "fact_002_place",
        "operator": "PLACE",
        "arguments": ["object_0001", "countertop"],
    }]
    adapter = KitchenPhase4Adapter.__new__(KitchenPhase4Adapter)
    adapter.dispatcher = Dispatcher()
    adapter.expected_inspected_regions = ("D1",)
    adapter.expected_actions = actions
    adapter.by_id = {"object_0001": object()}
    adapter.successful_actions = [
        {"action": action, "post_check": {"success": True}}
        for action in actions
    ]
    adapter.successful_inspection_history = ["D1"]
    adapter._held_planner_id = lambda: None
    result = adapter.final_verification()
    assert result["performed"] is True
    assert result["success"] is True
    adapter._held_planner_id = lambda: "object_0001"
    assert adapter.final_verification()["success"] is False


def test_common_kitchen_failure_mapping_is_normalized():
    assert classify_planner_failure(
        "source region C2 is closed",
        infrastructure_failure=ExecutionFailure.PRECONDITION_STATE_FAILURE.value,
        operator="PICK",
    ) == "REGION_CLOSED"


def test_benchmark_open_relaxes_minor_contact_but_rejects_gross_penetration():
    minor, minor_gross = benchmark_open_verified(
        opened_enough=True, strict_mode=False,
        strict_penetration_observed=True,
        minimum_furniture_clearance_m=-0.004,
        minimum_drawer_shell_clearance_m=-0.003,
    )
    gross, gross_flag = benchmark_open_verified(
        opened_enough=True, strict_mode=False,
        strict_penetration_observed=True,
        minimum_furniture_clearance_m=-0.060,
        minimum_drawer_shell_clearance_m=None,
    )
    assert minor and not minor_gross
    assert not gross and gross_flag


def test_strict_living_execution_never_modifies_post_release_damping():
    assert post_release_dynamics_modification_enabled(False) is False
    assert post_release_dynamics_modification_enabled(True) is True


def test_living_room_unresolved_argument_is_reported_fail_closed():
    resolved, unresolved = resolve_living_room_action_arguments(
        ["object_0001", "missing_region"],
        {
            "object_0001": {
                "generic_object_id": "object_0001",
                "backend_body": "living_room_payload_1",
            }
        },
        {},
    )
    assert [row["planner_id"] for row in resolved] == ["object_0001"]
    assert unresolved == ["missing_region"]
    with pytest.raises(ValueError, match="unresolved arguments"):
        validate_living_room_plan_ids(
            [{
                "action_instance_id": "fact_001_place",
                "arguments": ["object_0001", "missing_region"],
            }],
            {"objects": {"object_0001": {}}},
            {"regions": {}},
        )
