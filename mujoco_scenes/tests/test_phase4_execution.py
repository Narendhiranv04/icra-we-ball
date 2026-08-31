import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest
import mujoco

from mujoco_scenes.phase4_execution import (
    ActionExecutionResult,
    ExecutionFailure,
    Phase4Executor,
    Phase4EntityMappingError,
    ResolvedEntity,
    UpstreamPhase3Blocked,
    load_phase3_handoff,
    audit_strict_telemetry,
)
from mujoco_scenes.phase4_workshop_entities import (
    WorkshopEntityResolutionError,
    resolve_workshop_entities,
    workshop_body_world_geometry_aabb_center,
)
from mujoco_scenes.phase4_workshop import (
    WorkshopPhase4Adapter,
    strict_workshop_place_block,
    resolve_workshop_entities_for_execution,
)
from mujoco_scenes.workshop_ground_truth_execution import (
    WorkshopExecutionDispatcher,
)
from mujoco_scenes.living_room_mobile_execution import (
    post_release_dynamics_modification_enabled,
)
from mujoco_scenes.phase4_living_room import (
    resolve_living_room_action_arguments,
    validate_living_room_plan_ids,
)


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
        {"actions": actions, "validation": {"status": "VALID"}},
    )
    _write(
        tmp_path / "action_sequence/replay_validation.json",
        {"status": "VALID"},
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


def test_ready_handoff_requires_manifest_declared_result(tmp_path):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["artifacts"].pop("result")
    _write(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="result artifact"):
        load_phase3_handoff(tmp_path)


@pytest.mark.parametrize("domain", ["kitchen", "living_room"])
def test_non_workshop_handoff_requires_explicit_replay(tmp_path, domain):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["domain"] = domain
    manifest["artifacts"].pop("replay_validation")
    _write(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="only for Workshop"):
        load_phase3_handoff(tmp_path)


def test_workshop_embedded_replay_is_explicitly_allowed(tmp_path):
    _handoff(tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["domain"] = "workshop"
    manifest["artifacts"].pop("replay_validation")
    _write(tmp_path / "run_manifest.json", manifest)
    handoff = load_phase3_handoff(tmp_path)
    assert handoff.replay_validation_source == (
        "WORKSHOP_EMBEDDED_FINAL_PLAN_VALIDATION"
    )


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


def test_executor_replays_inspection_before_immutable_plan(tmp_path):
    handoff = _handoff(tmp_path, inspected_regions=("D1", "C2"))
    adapter = _Adapter()
    result = Phase4Executor(handoff, adapter).run()
    assert result["inspection_execution"]["regions"] == ["D1", "C2"]
    assert result["inspection_execution"]["actions_completed"] == 2
    assert adapter.calls == list(handoff.actions)
    assert result["strict_execution"] is True
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


def test_executor_rejects_success_with_forbidden_strict_telemetry(tmp_path):
    result = Phase4Executor(
        _handoff(tmp_path), _AssistedTelemetryAdapter()
    ).run()
    assert not result["success"]
    assert result["failure"] == "STRICT_EXECUTION_TELEMETRY_VIOLATION"
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


def test_strict_workshop_never_invokes_legacy_fixture_places():
    insertion = strict_workshop_place_block({
        "operator": "PLACE",
        "arguments": ["object_0001", "workshop_frame_joint"],
    }, "object_0001")
    assert insertion["status"] == "STRICT_PHYSICAL_INSERTION_UNAVAILABLE"
    assert insertion["no_legacy_insertion_invoked"] is True
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
    assert surface["status"] == "STRICT_PHYSICAL_SURFACE_PLACE_UNAVAILABLE"
    assert surface["no_legacy_surface_place_invoked"] is True

    class _State:
        def check(self, action, assignment):
            return True, None

        def to_dict(self):
            return {}

    class _Dispatcher:
        def execute(self, action, state):
            raise AssertionError("legacy dispatcher must not be invoked")

    adapter = WorkshopPhase4Adapter.__new__(WorkshopPhase4Adapter)
    adapter.by_id = {
        "object_0001": ResolvedEntity(
            "object_0001", "OBJECT", "workshop_medium_phillips_screw"
        ),
        "object_0002": ResolvedEntity(
            "object_0002", "OBJECT", "workshop_power_driver"
        ),
    }
    adapter.planner_fastener = "object_0001"
    adapter.state = _State()
    adapter.assignment = object()
    adapter.dispatcher = _Dispatcher()
    for index, arguments in enumerate((
        ["object_0001", "workshop_frame_joint"],
        ["object_0002", "workshop_frame_joint"],
        ["object_0002", "MAIN_WORKBENCH_ZONE"],
    ), start=1):
        result = adapter.execute_action({
            "action_index": index,
            "action_instance_id": f"fact_{index:03d}_place",
            "operator": "PLACE",
            "arguments": arguments,
        })
        assert not result.success
        assert result.failure in {
            ExecutionFailure.CONTROLLER_FAILURE.value,
            ExecutionFailure.PRECONDITION_STATE_FAILURE.value,
        }


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
