import json
from pathlib import Path

import pytest

from mujoco_scenes.phase4_execution import (
    ActionExecutionResult,
    ExecutionFailure,
    Phase4Executor,
    UpstreamPhase3Blocked,
    load_phase3_handoff,
)
from mujoco_scenes.phase4_workshop_entities import (
    WorkshopEntityResolutionError,
    resolve_workshop_entities,
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
            "artifacts": {"final_plan": "action_sequence/action_plan.json"},
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
    return load_phase3_handoff(tmp_path)


def test_load_phase3_handoff_preserves_exact_plan_contract(tmp_path):
    handoff = _handoff(tmp_path)
    assert handoff.source == "GT_FUNCTIONAL_SPEC_ONLY"
    assert handoff.assignment == {"tool": "object_0001"}
    assert handoff.actions[0]["arguments"] == ["object_0001"]
    assert set(handoff.artifact_sha256) == {"manifest", "grounding", "plan"}


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
    assert result["direct_task_state_fallback_used"] is True
    assert result["strict_telemetry_verification"]["violations"][0]["flag"] == (
        "direct_payload_pose_write"
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
